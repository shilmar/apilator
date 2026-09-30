# gui/worker.py
import os
import gc
import shutil
import tempfile
import cv2
import numpy as np
import tifffile
from PySide6.QtCore import QThread, Signal

from core.stacking import (
    load_image_as_float32, 
    detect_sky_stars, 
    refine_star_centroids,
    stream_stack_auto,
    create_master_dark,
    calibrate_light,
    HAS_GPU
)
from core.graxpert_bridge import run_graxpert_background_extraction
from core.starnet_bridge import run_starnet


def save_frame_float32(filepath: str, img_float32: np.ndarray):
    raw_data = np.ascontiguousarray(img_float32, dtype=np.float32)
    with open(filepath, "wb") as f:
        f.write(raw_data.tobytes())
    del raw_data


class StackingWorker(QThread):
    progress_changed = Signal(int)
    status_changed = Signal(str)
    finished_success = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        temp_dir = None
        try:
            lights = self.config.get("lights", [])
            darks = self.config.get("darks", [])
            sky_mask = self.config.get("mask", None)
            mode = self.config.get("mode", "fixed_tripod")
            kappa = float(self.config.get("kappa", 2.2))
            output_path = self.config.get("output_path", "resultado_dual_32bit.tiff")

            if len(lights) < 2:
                self.error_occurred.emit("Se necesitan al menos 2 tomas de luz para apilar.")
                return

            self.status_changed.emit("Iniciando preparación del entorno temporal...")
            self.progress_changed.emit(2)

            temp_dir = tempfile.mkdtemp(prefix="astro_gui_")

            master_dark = None
            if darks:
                self.status_changed.emit(f"Generando Master Dark a partir de {len(darks)} tomas...")
                master_dark = create_master_dark(darks)
                self.status_changed.emit("-> Master Dark generado y listo para calibración.")

            ref_path = lights[0]
            self.status_changed.emit(f"Cargando toma de referencia: {os.path.basename(ref_path)}")
            
            # Cargar imagen base pura
            ref_raw = load_image_as_float32(ref_path)
            h, w, c = ref_raw.shape

            if sky_mask is not None and sky_mask.shape != (h, w):
                sky_mask = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_LINEAR)

            # Para el cielo se calibra con dark; si no hay dark, pasa íntegra
            ref_sky = calibrate_light(ref_raw, master_dark)

            self.status_changed.emit("Extrayendo estrellas de la toma de referencia...")
            ref_kp, ref_desc, norm_type = detect_sky_stars(ref_sky, sky_mask=sky_mask)
            ref_gray = cv2.cvtColor((ref_sky * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            self.status_changed.emit(f"-> Estrellas base detectadas: {len(ref_kp)}")

            if ref_desc is None or len(ref_kp) < 15:
                raise RuntimeError("No se detectaron suficientes estrellas en la toma de referencia para alinear.")

            sky_temp_files = []
            ground_temp_files = []

            # Guardar referencia 0 para el cielo
            p_sky_0 = os.path.join(temp_dir, "sky_0.bin")
            save_frame_float32(p_sky_0, ref_sky)
            sky_temp_files.append(p_sky_0)

            # Guardar referencia 0 para el suelo (la imagen original sin recortar pedestales)
            if mode == "fixed_tripod":
                p_gnd_0 = os.path.join(temp_dir, "gnd_0.bin")
                save_frame_float32(p_gnd_0, ref_raw)
                ground_temp_files.append(p_gnd_0)

            del ref_raw, ref_sky
            gc.collect()

            # Matcher configurado para emparejar contra la referencia fija
            bf = cv2.BFMatcher(norm_type, crossCheck=False)
            total_lights = len(lights)
            discarded_count = 0

            for idx, path in enumerate(lights[1:], start=2):
                if self._is_cancelled:
                    self.status_changed.emit("Cancelado por el usuario.")
                    return

                pct = int(5 + (idx / total_lights) * 45)
                self.progress_changed.emit(pct)
                filename = os.path.basename(path)

                raw_frame = load_image_as_float32(path)
                calibrated_frame = calibrate_light(raw_frame, master_dark)

                try:
                    curr_kp, curr_desc, _ = detect_sky_stars(calibrated_frame, sky_mask=sky_mask)
                    if curr_desc is None or len(curr_kp) < 15:
                        raise RuntimeError(f"Solo se detectaron {len(curr_kp) if curr_kp else 0} estrellas.")

                    matches = bf.knnMatch(ref_desc, curr_desc, k=2)
                    good = [m for m, n in matches if len((m, n)) == 2 and m.distance < 0.78 * n.distance]

                    if len(good) < 10:
                        raise RuntimeError(f"Correspondencias insuficientes con la referencia ({len(good)} pares).")

                    curr_gray = cv2.cvtColor((calibrated_frame * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    dst_pts = refine_star_centroids(ref_gray, [ref_kp[m.queryIdx] for m in good])
                    src_pts = refine_star_centroids(curr_gray, [curr_kp[m.trainIdx] for m in good])

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

                    warped = cv2.warpPerspective(
                        calibrated_frame, H_matrix, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT
                    )

                    p_sky = os.path.join(temp_dir, f"sky_{idx-1}.bin")
                    save_frame_float32(p_sky, warped)
                    sky_temp_files.append(p_sky)

                    if mode == "fixed_tripod":
                        p_gnd = os.path.join(temp_dir, f"gnd_{idx-1}.bin")
                        save_frame_float32(p_gnd, raw_frame)
                        ground_temp_files.append(p_gnd)

                    dx, dy = H_matrix[0, 2], H_matrix[1, 2]
                    self.status_changed.emit(
                        f"[{idx}/{total_lights}] {filename} -> OK ({num_inliers} inliers | Subpíxel | dx: {dx:.1f}px, dy: {dy:.1f}px)"
                    )
                    del warped

                except Exception as e:
                    discarded_count += 1
                    self.status_changed.emit(f"[DESCARTADA] [{idx}/{total_lights}] {filename} -> {e}")

                del raw_frame, calibrated_frame
                gc.collect()

            if discarded_count > 0:
                self.status_changed.emit(f"-> Resumen: {discarded_count} toma(s) descartada(s).")

            if len(sky_temp_files) < 2:
                raise RuntimeError("No se pudieron alinear suficientes tomas con la referencia.")

            backend_label = "GPU CUDA (NVIDIA)" if HAS_GPU else "CPU Multi-Core"

            if self._is_cancelled: return
            self.status_changed.emit(f"Apilando Cielo ({len(sky_temp_files)} tomas) con [{backend_label}] (Kappa={kappa:.1f})...")
            self.progress_changed.emit(55)
            sky_stacked = stream_stack_auto(sky_temp_files, (h, w, c), chunk_rows=800, kappa=kappa)

            if sky_mask is not None and mode == "fixed_tripod":
                if self._is_cancelled: return
                self.status_changed.emit(f"Apilando Suelo ({len(ground_temp_files)} tomas) con [{backend_label}] (Kappa={kappa:.1f})...")
                self.progress_changed.emit(75)
                ground_stacked = stream_stack_auto(ground_temp_files, (h, w, c), chunk_rows=800, kappa=kappa)

                self.status_changed.emit("Componiendo imagen final de 32 bits con máscara suavizada...")
                self.progress_changed.emit(92)

                ksize = int(max(15, (min(h, w) // 150) | 1))
                if ksize % 2 == 0:
                    ksize += 1
                smooth_mask = cv2.GaussianBlur(sky_mask.astype(np.float32), (ksize, ksize), sigmaX=ksize / 3.0)
                mask_3d = np.repeat(smooth_mask[..., np.newaxis], 3, axis=2)

                final_composite = (sky_stacked * mask_3d) + (ground_stacked * (1.0 - mask_3d))
                del sky_stacked, ground_stacked, mask_3d, smooth_mask
            else:
                final_composite = sky_stacked

            self.status_changed.emit(f"Guardando resultado de 32 bits en: {output_path}...")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass

            tifffile.imwrite(output_path, final_composite.astype(np.float32), compression='zlib')

            self.progress_changed.emit(100)
            self.status_changed.emit("¡Apilado completado exitosamente!")
            self.finished_success.emit(output_path)

        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)


class GraXpertWorker(QThread):
    finished_success = Signal(np.ndarray)
    error_occurred = Signal(str)
    status_changed = Signal(str)

    def __init__(self, image_rgb: np.ndarray, sky_mask: np.ndarray = None, smoothing: float = 0.5):
        super().__init__()
        self.image_rgb = image_rgb
        self.sky_mask = sky_mask
        self.smoothing = smoothing

    def run(self):
        try:
            self.status_changed.emit("Invocando GraXpert AI para extracción de fondo...")
            corrected = run_graxpert_background_extraction(
                self.image_rgb, 
                sky_mask=self.sky_mask, 
                smoothing=self.smoothing
            )
            self.status_changed.emit("Extracción de gradiente completada con éxito.")
            self.finished_success.emit(corrected)
        except Exception as e:
            self.error_occurred.emit(str(e))


class StarNetWorker(QThread):
    status_changed = Signal(str)
    finished_success = Signal(object, object)
    error_occurred = Signal(str)

    def __init__(self, img_rgb: np.ndarray, sky_mask: np.ndarray = None, stride: int = 256, custom_exe: str = None):
        super().__init__()
        self.img_rgb = img_rgb
        self.sky_mask = sky_mask
        self.stride = stride
        self.custom_exe = custom_exe

    def run(self):
        try:
            starless, stars = run_starnet(
                self.img_rgb,
                sky_mask=self.sky_mask,
                stride=self.stride,
                starnet_exe=self.custom_exe,
                log_callback=self.status_changed.emit
            )
            self.finished_success.emit(starless, stars)
        except Exception as exc:
            self.error_occurred.emit(str(exc))