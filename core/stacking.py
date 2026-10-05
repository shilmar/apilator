# core/stacking.py
import os
import gc
import cv2
import numpy as np
import rawpy
import tifffile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from astropy.io import fits
from core.gpu_backend import is_gpu_enabled

try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        HAS_GPU = True
    else:
        HAS_GPU = False
except Exception:
    HAS_GPU = False


def load_image_as_float32(filepath: str) -> np.ndarray:
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.nef', '.cr2', '.cr3', '.arw', '.dng']:
        with rawpy.imread(filepath) as raw:
            rgb = raw.postprocess(
                gamma=(2.222, 4.5),
                no_auto_bright=True,
                output_bps=16,
                use_camera_wb=True
            )
            return (rgb.astype(np.float32) / 65535.0)

    elif ext in ['.fits', '.fit', '.fts']:
        with fits.open(filepath) as hdul:
            data = hdul[0].data.astype(np.float32)
            if data.ndim == 2:
                data = np.stack([data] * 3, axis=-1)
            elif data.ndim == 3 and data.shape[0] in [3, 4]:
                data = np.transpose(data[:3], (1, 2, 0))
            d_min, d_max = data.min(), data.max()
            return (data - d_min) / (d_max - d_min + 1e-8)

    elif ext in ['.tif', '.tiff']:
        data = tifffile.imread(filepath).astype(np.float32)
        if data.ndim == 2:
            data = np.stack([data] * 3, axis=-1)
        if data.max() <= 1.05 and data.min() >= 0.0:
            return np.clip(data, 0.0, 1.0)
        max_val = 65535.0 if data.max() > 255.0 else 255.0
        return np.clip(data / max_val, 0.0, 1.0)

    else:
        bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"No se pudo leer el archivo: {filepath}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def create_master_dark(dark_paths: list) -> np.ndarray:
    if not dark_paths:
        return None

    loaded_darks = []
    for path in dark_paths:
        img = load_image_as_float32(path)
        loaded_darks.append(img)

    if len(loaded_darks) == 1:
        return loaded_darks[0]

    stack = np.stack(loaded_darks, axis=0)
    del loaded_darks
    gc.collect()

    master_dark = np.median(stack, axis=0).astype(np.float32)
    del stack
    gc.collect()

    return master_dark


def calibrate_light(light_img: np.ndarray, master_dark: np.ndarray = None) -> np.ndarray:
    if master_dark is None:
        return light_img
    return np.maximum(0.0, light_img - master_dark)


def estimate_frame_background_dome(img_rgb: np.ndarray, sky_mask: np.ndarray = None) -> np.ndarray:
    h, w, c = img_rgb.shape
    scale = max(1, min(h, w) // 120)
    sh, sw = max(16, h // scale), max(16, w // scale)

    small_img = cv2.resize(img_rgb, (sw, sh), interpolation=cv2.INTER_AREA)

    if sky_mask is not None:
        sm_mask = cv2.resize(np.squeeze(sky_mask).astype(np.float32), (sw, sh), interpolation=cv2.INTER_NEAREST)
        k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        valid_sky = cv2.erode((sm_mask > 0.5).astype(np.uint8), k_erode) > 0
    else:
        valid_sky = np.ones((sh, sw), dtype=bool)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bg_low = np.zeros_like(small_img)

    for ch in range(c):
        channel = small_img[..., ch]
        opened = cv2.morphologyEx(channel, cv2.MORPH_OPEN, k_open)
        sigma = max(5.0, min(sh, sw) / 10.0)
        bg_low[..., ch] = cv2.GaussianBlur(opened, (0, 0), sigmaX=sigma, sigmaY=sigma)

    bg_full = cv2.resize(bg_low, (w, h), interpolation=cv2.INTER_LINEAR)
    return bg_full


def preprocess_subframe_lp(
    frame_rgb: np.ndarray,
    method: str = "standard",
    strength: float = 0.5,
    sky_mask: np.ndarray = None,
    ref_stats: dict = None
) -> np.ndarray:
    if strength <= 1e-4 or method in ["standard", "min_rejection"]:
        return frame_rgb

    h, w, c = frame_rgb.shape

    if method == "sequator_subtraction":
        bg_dome = estimate_frame_background_dome(frame_rgb, sky_mask=sky_mask)
        pedestal = np.percentile(frame_rgb, 5, axis=(0, 1))
        corrected = frame_rgb - (bg_dome * strength) + (pedestal * strength)

        if sky_mask is not None:
            m = cv2.resize(np.squeeze(sky_mask).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)[..., np.newaxis]
            return np.clip(corrected * m + frame_rgb * (1.0 - m), 0.0, 1.0).astype(np.float32)
        return np.clip(corrected, 0.0, 1.0).astype(np.float32)

    elif method == "local_norm" and ref_stats is not None:
        curr_median = np.median(frame_rgb, axis=(0, 1))
        target_median = ref_stats.get('median', curr_median)
        diff = (target_median - curr_median) * strength
        corrected = frame_rgb + diff

        if sky_mask is not None:
            m = cv2.resize(np.squeeze(sky_mask).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)[..., np.newaxis]
            return np.clip(corrected * m + frame_rgb * (1.0 - m), 0.0, 1.0).astype(np.float32)
        return np.clip(corrected, 0.0, 1.0).astype(np.float32)

    return frame_rgb


def detect_sky_stars(image_rgb: np.ndarray, sky_mask: np.ndarray = None, max_stars: int = 3500):
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    if sky_mask is not None:
        mask_u8 = ((sky_mask > 0.4) * 255).astype(np.uint8)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        mask_clean = cv2.erode(mask_u8, kernel_erode)
    else:
        mask_clean = np.full((h, w), 255, dtype=np.uint8)

    masked_gray = cv2.bitwise_and(gray, gray, mask=mask_clean)
    stretched = np.clip(np.power(masked_gray, 0.6) * 255.0, 0, 255).astype(np.uint8)

    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    stars_isolated = cv2.morphologyEx(stretched, cv2.MORPH_TOPHAT, kernel_tophat)
    _, stars_thresholded = cv2.threshold(stars_isolated, 15, 255, cv2.THRESH_TOZERO)

    detector = cv2.ORB_create(
        nfeatures=max_stars,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=12,
        fastThreshold=6
    )
    kps, descs = detector.detectAndCompute(stars_thresholded, mask=mask_clean)
    return kps, descs, cv2.NORM_HAMMING


def refine_star_centroids(gray_img: np.ndarray, keypoints: list) -> np.ndarray:
    if not keypoints:
        return np.empty((0, 1, 2), dtype=np.float32)

    pts = np.float32([kp.pt for kp in keypoints]).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.005)
    refined_pts = cv2.cornerSubPix(gray_img, pts, (5, 5), (-1, -1), criteria)
    return refined_pts


def save_frame_float32(filepath: str, img_float32: np.ndarray):
    raw_data = np.ascontiguousarray(img_float32, dtype=np.float32)
    with open(filepath, "wb") as f:
        f.write(raw_data.tobytes())
    del raw_data


def align_single_light_task(args: tuple) -> dict:
    (
        idx, path, total_lights, temp_dir, mode, master_dark,
        ref_kps_pts, ref_desc, norm_type, ref_gray,
        w, h, sky_mask, lp_method, lp_strength, ref_stats,
        use_ram_buffer
    ) = args

    filename = os.path.basename(path)
    res = {
        "idx": idx,
        "filename": filename,
        "success": False,
        "error": None,
        "sky_data": None,
        "gnd_data": None,
        "inliers": 0,
        "dx": 0.0,
        "dy": 0.0
    }

    try:
        raw_frame = load_image_as_float32(path)
        calibrated_frame = calibrate_light(raw_frame, master_dark)

        curr_kp, curr_desc, _ = detect_sky_stars(calibrated_frame, sky_mask=sky_mask)
        if curr_desc is None or len(curr_kp) < 15:
            raise RuntimeError(f"Solo se detectaron {len(curr_kp) if curr_kp else 0} estrellas.")

        bf = cv2.BFMatcher(norm_type, crossCheck=False)
        matches = bf.knnMatch(ref_desc, curr_desc, k=2)
        good = [m for m, n in matches if len((m, n)) == 2 and m.distance < 0.78 * n.distance]

        if len(good) < 10:
            raise RuntimeError(f"Correspondencias insuficientes con la referencia ({len(good)} pares).")

        curr_gray = cv2.cvtColor((calibrated_frame * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        ref_matched_kps = [cv2.KeyPoint(ref_kps_pts[m.queryIdx][0], ref_kps_pts[m.queryIdx][1], 1.0) for m in good]
        curr_matched_kps = [curr_kp[m.trainIdx] for m in good]

        dst_pts = refine_star_centroids(ref_gray, ref_matched_kps)
        src_pts = refine_star_centroids(curr_gray, curr_matched_kps)

        del curr_gray, curr_kp, curr_desc, matches, good

        H_matrix, inliers = cv2.findHomography(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=5000,
            confidence=0.995
        )

        if H_matrix is None:
            H_aff, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
            if H_aff is None:
                raise RuntimeError("Fallo RANSAC en homografía/afín.")
            H_matrix = np.vstack([H_aff, [0.0, 0.0, 1.0]])

        num_inliers = int(np.sum(inliers)) if inliers is not None else 0
        del dst_pts, src_pts, inliers

        if sky_mask is not None:
            sm = sky_mask if sky_mask.shape[:2] == (h, w) else cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            sm_3d = sm[..., np.newaxis] if sm.ndim == 2 else sm
            sky_median_val = np.median(calibrated_frame[sm > 0.5]) if np.any(sm > 0.5) else 0.05
            clean_sky_frame = (calibrated_frame * sm_3d) + (sky_median_val * (1.0 - sm_3d))
        else:
            clean_sky_frame = calibrated_frame

        # Regreso exacto al borderValue=(0, 0, 0) de la 0.5.5 (Cero NaNs)
        # Relleno por reflexión en bordes para evitar franjas negras y conservar dimensiones completas
        warped = cv2.warpPerspective(
            clean_sky_frame, H_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT
        )
        del clean_sky_frame, calibrated_frame

        warped_processed = preprocess_subframe_lp(
            warped,
            method=lp_method,
            strength=lp_strength,
            sky_mask=sky_mask,
            ref_stats=ref_stats
        )
        del warped

        p_sky = os.path.join(temp_dir, f"sky_{idx-1:04d}.bin")
        save_frame_float32(p_sky, warped_processed)
        res["sky_data"] = p_sky

        if mode == "fixed_tripod":
            p_gnd = os.path.join(temp_dir, f"gnd_{idx-1:04d}.bin")
            save_frame_float32(p_gnd, raw_frame)
            res["gnd_data"] = p_gnd

        del warped_processed, raw_frame

        res["success"] = True
        res["inliers"] = num_inliers
        res["dx"] = float(H_matrix[0, 2])
        res["dy"] = float(H_matrix[1, 2])
        gc.collect()

    except Exception as exc:
        res["error"] = str(exc)

    return res


# =========================================================================
# MOTOR MATEMÁTICO EXACTO DE LA VERSIÓN 0.5.5 CON CORRECCIÓN DE MÁRGENES
# =========================================================================

def _process_chunk_gpu(sub_stack_np: np.ndarray, kappa: float, lp_method: str = "standard", lp_strength: float = 0.5) -> np.ndarray:
    n_frames, actual_rows, w, c = sub_stack_np.shape
    chunk_result = np.zeros((actual_rows, w, c), dtype=np.float32)

    sub_stack_gpu = cp.asarray(sub_stack_np)

    if lp_method == "min_rejection" and lp_strength > 1e-4:
        target_p = max(5.0, 50.0 - (lp_strength * 40.0))
        for ch in range(c):
            ch_data = sub_stack_gpu[:, :, :, ch].reshape((n_frames, -1))
            res = cp.percentile(ch_data, target_p, axis=0)
            chunk_result[:, :, ch] = cp.asnumpy(res).reshape((actual_rows, w))
        del sub_stack_gpu
        cp.get_default_memory_pool().free_all_blocks()
        return chunk_result

    # Motor original 0.5.5 exacto
    upper_tol = kappa + 0.8
    for ch in range(c):
        ch_data = sub_stack_gpu[:, :, :, ch].reshape((n_frames, -1))
        
        med = cp.median(ch_data, axis=0)
        abs_diff = cp.abs(ch_data - med)
        mad = cp.median(abs_diff, axis=0)
        sigma = 1.4826 * mad + 1e-6

        low = med - 0.4 * sigma
        high = med + upper_tol * sigma
        
        valid = (ch_data >= low) & (ch_data <= high)

        filtered = cp.where(valid, ch_data, cp.nan)
        res = cp.nanmean(filtered, axis=0)

        nan_mask = cp.isnan(res)
        if cp.any(nan_mask):
            res[nan_mask] = med[nan_mask]

        chunk_result[:, :, ch] = cp.asnumpy(res).reshape((actual_rows, w))

    del sub_stack_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return chunk_result


def _process_single_chunk_cpu(args):
    frames_source, y_start, y_end, w, c, kappa, lp_method, lp_strength = args
    actual_rows = y_end - y_start
    n_frames = len(frames_source)
    is_ram_mode = isinstance(frames_source[0], np.ndarray)

    sub_stack = np.empty((n_frames, actual_rows, w, c), dtype=np.float32)

    if is_ram_mode:
        for i, arr in enumerate(frames_source):
            sub_stack[i] = arr[y_start:y_end]
    else:
        bytes_per_row = w * c * 4
        current_read_bytes = actual_rows * bytes_per_row
        offset = y_start * bytes_per_row
        for i, f in enumerate(frames_source):
            with open(f, "rb") as fp:
                fp.seek(offset)
                raw_bytes = fp.read(current_read_bytes)
                sub_stack[i] = np.frombuffer(raw_bytes, dtype=np.float32).reshape((actual_rows, w, c))

    chunk_result = np.empty((actual_rows, w, c), dtype=np.float32)

    if lp_method == "min_rejection" and lp_strength > 1e-4:
        target_p = max(5.0, 50.0 - (lp_strength * 40.0))
        for ch in range(c):
            channel_data = sub_stack[:, :, :, ch].reshape((n_frames, -1))
            res = np.percentile(channel_data, target_p, axis=0)
            chunk_result[:, :, ch] = res.reshape((actual_rows, w))
        del sub_stack
        return y_start, y_end, chunk_result

    upper_tol = kappa + 0.8
    for ch in range(c):
        channel_data = sub_stack[:, :, :, ch].reshape((n_frames, -1))
        
        med = np.median(channel_data, axis=0)
        abs_diff = np.abs(channel_data - med)
        mad = np.median(abs_diff, axis=0)
        sigma = 1.4826 * mad + 1e-6

        low = med - 0.4 * sigma
        high = med + upper_tol * sigma

        valid = (channel_data >= low) & (channel_data <= high)
        
        counts = np.sum(valid, axis=0)
        sums = np.sum(np.where(valid, channel_data, 0.0), axis=0)
        
        fallback = counts == 0
        counts[fallback] = 1
        res = sums / counts
        res[fallback] = med[fallback]

        chunk_result[:, :, ch] = res.reshape((actual_rows, w))

    del sub_stack
    return y_start, y_end, chunk_result


def parallel_stream_stack(
    frames_source: list, 
    shape: tuple, 
    chunk_rows: int = 200, 
    kappa: float = 2.2, 
    max_workers: int = None,
    lp_method: str = "standard",
    lp_strength: float = 0.5
) -> np.ndarray:
    h, w, c = shape
    stacked_out = np.zeros((h, w, c), dtype=np.float32)
    is_ram_mode = isinstance(frames_source[0], np.ndarray)

    tasks = []
    for y in range(0, h, chunk_rows):
        y_end = min(y + chunk_rows, h)
        tasks.append((frames_source, y, y_end, w, c, kappa, lp_method, lp_strength))

    if max_workers is None:
        max_workers = max(1, min(4, (os.cpu_count() or 4) - 1))

    if is_ram_mode:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for y_start, y_end, chunk_data in executor.map(_process_single_chunk_cpu, tasks):
                stacked_out[y_start:y_end, :, :] = chunk_data
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            for y_start, y_end, chunk_data in executor.map(_process_single_chunk_cpu, tasks):
                stacked_out[y_start:y_end, :, :] = chunk_data

    gc.collect()
    return stacked_out


def stream_stack_auto(
    frames_source: list, 
    shape: tuple, 
    chunk_rows: int = 200, 
    kappa: float = 2.2,
    lp_method: str = "standard",
    lp_strength: float = 0.5,
    use_gpu: bool = None
) -> np.ndarray:
    h, w, c = shape
    n_frames = len(frames_source)
    is_ram_mode = isinstance(frames_source[0], np.ndarray)

    if use_gpu is None:
        use_gpu = is_gpu_enabled()

    if use_gpu and HAS_GPU:
        try:
            stacked_out = np.zeros((h, w, c), dtype=np.float32)
            bytes_per_row = w * c * 4

            for y in range(0, h, chunk_rows):
                y_end = min(y + chunk_rows, h)
                actual_rows = y_end - y

                if is_ram_mode:
                    sub_stack = np.empty((n_frames, actual_rows, w, c), dtype=np.float32)
                    for i in range(n_frames):
                        sub_stack[i] = frames_source[i][y:y_end]
                else:
                    offset = y * bytes_per_row
                    current_read_bytes = actual_rows * bytes_per_row
                    block_frames = []
                    for f in frames_source:
                        with open(f, "rb") as fp:
                            fp.seek(offset)
                            raw_bytes = fp.read(current_read_bytes)
                            frame_chunk = np.frombuffer(raw_bytes, dtype=np.float32).reshape((actual_rows, w, c))
                            block_frames.append(frame_chunk)
                    sub_stack = np.stack(block_frames, axis=0)
                    del block_frames

                stacked_out[y:y_end, :, :] = _process_chunk_gpu(
                    sub_stack, 
                    kappa=kappa, 
                    lp_method=lp_method, 
                    lp_strength=lp_strength
                )
                del sub_stack

            return stacked_out
        except Exception as exc:
            print(f"[CuPy Fallback]: {exc}")
            if HAS_GPU:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

    return parallel_stream_stack(
        frames_source, 
        shape, 
        chunk_rows=chunk_rows, 
        kappa=kappa,
        lp_method=lp_method,
        lp_strength=lp_strength
    )