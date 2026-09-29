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
    register_consecutive_homography, 
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
                self.status_changed.emit("-> Master Dark generado y listo para sustracción térmica.")

            ref_path = lights[0]
            self.status_changed.emit(f"Cargando y calibrando referencia: {os.path.basename(ref_path)}")
            ref_img = load_image_as_float32(ref_path)
            ref_img = calibrate_light(ref_img, master_dark)
            h, w, c = ref_img.shape

            if sky_mask is not None and sky_mask.shape != (h, w):
                sky_mask = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_LINEAR)

            self.status_changed.emit("Extrayendo estrellas de referencia con precisión subpíxel...")
            prev_kp, prev_desc, norm_type = detect_sky_stars(ref_img, sky_mask=sky_mask)
            prev_gray = cv2.cvtColor((ref_img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            self.status_changed.emit(f"-> Estrellas base detectadas: {len(prev_kp)}")

            sky_temp_files = []
            ground_temp_files = []

            p_sky_0 = os.path.join(temp_dir, "sky_0.bin")
            save_frame_float32(p_sky_0, ref_img)
            sky_temp_files.append(p_sky_0)

            if mode == "fixed_tripod":
                p_gnd_0 = os.path.join(temp_dir, "gnd_0.bin")
                save_frame_float32(p_gnd_0, ref_img)
                ground_temp_files.append(p_gnd_0)

            del ref_img
            gc.collect()

            accumulated_H = np.eye(3, dtype=np.float64)
            total_lights = len(lights)
            discarded_count = 0

            for idx, path in enumerate(lights[1:], start=2):
                if self._is_cancelled:
                    self.status_changed.emit("Cancelado por el usuario.")
                    return

                pct = int(5 + (idx / total_lights) * 45)
                self.progress_changed.emit(pct)
                filename = os.path.basename(path)

                img = load_image_as_float32(path)
                img = calibrate_light(img, master_dark)

                try:
                    step_H, inliers, curr_kp, curr_desc, curr_gray = register_consecutive_homography(
                        prev_kp, prev_desc, prev_gray, img, sky_mask=sky_mask, norm_type=norm_type
                    )

                    accumulated_H = accumulated_H @ step_H

                    warped = cv2.warpPerspective(
                        img, accumulated_H, (w, h), 
                        flags=cv2.INTER_CUBIC, 
                        borderMode=cv2.BORDER_REFLECT
                    )

                    p_sky = os.path.join(temp_dir, f"sky_{idx-1}.bin")
                    save_frame_float32(p_sky, warped)
                    sky_temp_files.append(p_sky)

                    # Guardar suelo solo si la toma es válida
                    if mode == "fixed_tripod":
                        p_gnd = os.path.join(temp_dir, f"gnd_{idx-1}.bin")
                        save_frame_float32(p_gnd, img)
                        ground_temp_files.append(p_gnd)

                    total_dx = accumulated_H[0, 2]
                    total_dy = accumulated_H[1, 2]
                    self.status_changed.emit(
                        f"[{idx}/{total_lights}] {filename} -> OK ({inliers} est. | Subpíxel H | dx: {total_dx:.1f}px, dy: {total_dy:.1f}px)"
                    )

                    prev_kp = curr_kp
                    prev_desc = curr_desc
                    prev_gray = curr_gray
                    del warped

                except Exception as e:
                    discarded_count += 1
                    self.status_changed.emit(
                        f"[DESCARTADA] [{idx}/{total_lights}] {filename} -> Motivo: {e}"
                    )
                    # Intentar re-anclar keypoints para no perder el hilo si la siguiente toma es buena
                    try:
                        kps_fail, desc_fail, _ = detect_sky_stars(img, sky_mask=sky_mask)
                        if desc_fail is not None and len(kps_fail) >= 20:
                            prev_kp = kps_fail
                            prev_desc = desc_fail
                            prev_gray = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    except Exception:
                        pass

                del img
                gc.collect()

            if discarded_count > 0:
                self.status_changed.emit(f"-> Resumen: {discarded_count} toma(s) descartada(s) por problemas de alineación.")

            if len(sky_temp_files) < 2:
                raise RuntimeError("No se pudieron alinear suficientes tomas estelares consecutivas.")

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

                self.status_changed.emit("Componiendo imagen final de 32 bits con máscara...")
                self.progress_changed.emit(92)
                mask_3d = np.repeat(sky_mask[..., np.newaxis], 3, axis=2)
                final_composite = (sky_stacked * mask_3d) + (ground_stacked * (1.0 - mask_3d))
                del sky_stacked, ground_stacked, mask_3d
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
    finished_success = Signal(object, object)  # (starless_img, stars_img)
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