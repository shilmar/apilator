# core/stacking.py
import os
import gc
import cv2
import numpy as np
import rawpy
import tifffile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from astropy.io import fits

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

    first = load_image_as_float32(dark_paths[0])
    h, w, c = first.shape
    del first

    chunk_rows = 1000
    master_dark = np.zeros((h, w, c), dtype=np.float32)

    for y in range(0, h, chunk_rows):
        y_end = min(y + chunk_rows, h)
        chunks = []
        for path in dark_paths:
            img = load_image_as_float32(path)
            chunks.append(img[y:y_end, :, :])
            del img
        master_dark[y:y_end, :, :] = np.median(np.stack(chunks, axis=0), axis=0)
        del chunks
        gc.collect()

    return master_dark


def calibrate_light(light_img: np.ndarray, master_dark: np.ndarray = None) -> np.ndarray:
    if master_dark is None:
        return light_img
    return np.maximum(0.0, light_img - master_dark)


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


def register_consecutive_homography(prev_kp, prev_desc, prev_gray, curr_img, sky_mask=None, norm_type=cv2.NORM_HAMMING):
    curr_kp, curr_desc, _ = detect_sky_stars(curr_img, sky_mask=sky_mask)
    if curr_desc is None or len(curr_kp) < 15:
        raise RuntimeError(f"Solo se detectaron {len(curr_kp) if curr_kp else 0} estrellas en la toma.")

    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    matches = bf.knnMatch(prev_desc, curr_desc, k=2)

    good = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.80 * n.distance:
                good.append(m)

    if len(good) < 12:
        raise RuntimeError(f"Correspondencias insuficientes entre tomas ({len(good)} pares encontrados).")

    curr_gray = cv2.cvtColor((curr_img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    prev_matched_kps = [prev_kp[m.queryIdx] for m in good]
    curr_matched_kps = [curr_kp[m.trainIdx] for m in good]

    dst_pts = refine_star_centroids(prev_gray, prev_matched_kps)
    src_pts = refine_star_centroids(curr_gray, curr_matched_kps)

    H_matrix, inliers = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.5,
        maxIters=4000,
        confidence=0.99
    )

    if H_matrix is None:
        H_aff, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        if H_aff is None:
            raise RuntimeError("Fallo en la estimación de homografía/afín con RANSAC.")
        H_matrix = np.vstack([H_aff, [0.0, 0.0, 1.0]])

    num_inliers = int(np.sum(inliers)) if inliers is not None else 0
    return H_matrix, num_inliers, curr_kp, curr_desc, curr_gray


def _process_single_chunk(args):
    file_paths, y_start, y_end, w, c, kappa = args
    actual_rows = y_end - y_start
    bytes_per_row = w * c * 4
    current_read_bytes = actual_rows * bytes_per_row
    offset = y_start * bytes_per_row

    n_frames = len(file_paths)
    block_frames = []

    for f in file_paths:
        with open(f, "rb") as fp:
            fp.seek(offset)
            raw_bytes = fp.read(current_read_bytes)
            frame_chunk = np.frombuffer(raw_bytes, dtype=np.float32).reshape((actual_rows, w, c))
            block_frames.append(frame_chunk)

    sub_stack = np.stack(block_frames, axis=0)
    del block_frames

    chunk_result = np.zeros((actual_rows, w, c), dtype=np.float32)

    for ch in range(c):
        channel_data = sub_stack[:, :, :, ch].reshape((n_frames, -1))
        med = np.median(channel_data, axis=0)
        abs_diff = np.abs(channel_data - med)
        mad = np.median(abs_diff, axis=0)
        sigma = 1.4826 * mad + 1e-6

        low = med - 2.5 * sigma
        high = med + (kappa + 0.8) * sigma
        valid = (channel_data >= low) & (channel_data <= high)

        filtered = np.where(valid, channel_data, np.nan)
        with np.errstate(all='ignore'):
            res = np.nanmean(filtered, axis=0)

        nan_mask = np.isnan(res)
        if np.any(nan_mask):
            res[nan_mask] = med[nan_mask]

        chunk_result[:, :, ch] = res.reshape((actual_rows, w))

    del sub_stack
    return y_start, y_end, chunk_result


def parallel_stream_stack(file_paths: list, shape: tuple, chunk_rows: int = 800, kappa: float = 1.8, max_workers: int = None) -> np.ndarray:
    h, w, c = shape
    stacked_out = np.zeros((h, w, c), dtype=np.float32)

    tasks = []
    for y in range(0, h, chunk_rows):
        y_end = min(y + chunk_rows, h)
        tasks.append((file_paths, y, y_end, w, c, kappa))

    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        for y_start, y_end, chunk_data in executor.map(_process_single_chunk, tasks):
            stacked_out[y_start:y_end, :, :] = chunk_data

    gc.collect()
    return stacked_out


def _process_chunk_gpu(sub_stack_np: np.ndarray, kappa: float) -> np.ndarray:
    n_frames, actual_rows, w, c = sub_stack_np.shape
    chunk_result = np.zeros((actual_rows, w, c), dtype=np.float32)

    sub_stack_gpu = cp.asarray(sub_stack_np)

    for ch in range(c):
        ch_data = sub_stack_gpu[:, :, :, ch].reshape((n_frames, -1))
        med = cp.median(ch_data, axis=0)
        abs_diff = cp.abs(ch_data - med)
        mad = cp.median(abs_diff, axis=0)
        sigma = 1.4826 * mad + 1e-6

        low = med - 2.5 * sigma
        high = med + (kappa + 0.8) * sigma
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


def stream_stack_auto(file_paths: list, shape: tuple, chunk_rows: int = 1000, kappa: float = 1.8) -> np.ndarray:
    h, w, c = shape

    if HAS_GPU:
        try:
            stacked_out = np.zeros((h, w, c), dtype=np.float32)
            bytes_per_row = w * c * 4
            n_frames = len(file_paths)

            for y in range(0, h, chunk_rows):
                y_end = min(y + chunk_rows, h)
                actual_rows = y_end - y
                current_read_bytes = actual_rows * bytes_per_row
                offset = y * bytes_per_row

                block_frames = []
                for f in file_paths:
                    with open(f, "rb") as fp:
                        fp.seek(offset)
                        raw_bytes = fp.read(current_read_bytes)
                        frame_chunk = np.frombuffer(raw_bytes, dtype=np.float32).reshape((actual_rows, w, c))
                        block_frames.append(frame_chunk)

                sub_stack = np.stack(block_frames, axis=0)
                del block_frames

                stacked_out[y:y_end, :, :] = _process_chunk_gpu(sub_stack, kappa)
                del sub_stack

            return stacked_out
        except Exception:
            pass

    return parallel_stream_stack(file_paths, shape, chunk_rows=chunk_rows, kappa=kappa)