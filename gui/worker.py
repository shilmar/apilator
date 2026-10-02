"""
gui/worker.py - Hilos secundarios de ejecución (QThread) para el pipeline de Apilator,
incluyendo apilado dual subpíxel, extracción de gradiente GraXpert y StarNet++.
"""

import os
import gc
import shutil
import tempfile
import time
import cv2
import numpy as np
import tifffile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from PySide6.QtCore import QThread, Signal

from core.stacking import (
    load_image_as_float32, 
    detect_sky_stars, 
    refine_star_centroids,
    stream_stack_auto,
    create_master_dark,
    calibrate_light,
    preprocess_subframe_lp,          
    HAS_GPU,
    align_single_light_task,
    save_frame_float32
)

from core.graxpert_bridge import run_graxpert_background_extraction
from core.starnet_bridge import run_starnet


def evaluate_storage_mode(strategy: str, num_frames: int, h: int, w: int, c: int, has_ground: bool) -> bool:
    """
    Determina si usar memoria RAM (True) o Caché en Disco .bin (False).
    En modo auto, usa RAM solo si la estimación total ocupa menos del 60% de la RAM disponible.
    """
    if strategy == "disk":
        return False
    if strategy == "ram":
        return True

    # Estrategia 'auto':
    try:
        import psutil
        free_bytes = psutil.virtual_memory().available
        bytes_per_frame = h * w * c * 4  # float32
        multiplier = 2 if has_ground else 1
        estimated_total_bytes = num_frames * bytes_per_frame * multiplier
        return estimated_total_bytes < (free_bytes * 0.60)
    except Exception:
        return False


class StackingWorker(QThread):
    progress_changed = Signal(int)
    status_changed = Signal(str)
    finished_success = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        temp_dir = None
        t_global_start = time.perf_counter()
        try:
            lights = self.config.get("lights", [])
            darks = self.config.get("darks", [])
            sky_mask = self.config.get("mask", None)
            mode = self.config.get("mode", "fixed_tripod")
            kappa = float(self.config.get("kappa", 2.2))
            output_path = self.config.get("output_path", "resultado_dual_32bit.tiff")

            lp_method = self.config.get("lp_method", "standard")
            lp_strength = float(self.config.get("lp_strength", 0.5))
            use_gpu = self.config.get("use_gpu", True) and HAS_GPU
            
            # Salvaguarda estricta: entre 1 y 4 hilos de proceso
            cpu_workers = min(4, max(1, int(self.config.get("cpu_workers", 4))))
            storage_strategy = self.config.get("storage_strategy", "auto")

            if len(lights) < 2:
                self.error_occurred.emit("Se necesitan al menos 2 tomas de luz para apilar.")
                return

            self.status_changed.emit("Iniciando preparación del entorno temporal...")
            self.progress_changed.emit(1)

            temp_dir = tempfile.mkdtemp(prefix="astro_gui_")

            # ============================================================
            # FASE 1/4: CALIBRACIÓN / MASTER DARK (Rango 2% -> 10%)
            # ============================================================
            master_dark = None
            if darks:
                t0_darks = time.perf_counter()
                self.status_changed.emit(f"[Fase 1/4] Generando Master Dark a partir de {len(darks)} tomas...")
                self.progress_changed.emit(3)
                master_dark = create_master_dark(darks)
                dt_darks = time.perf_counter() - t0_darks
                self.status_changed.emit(f"-> Master Dark completado en {dt_darks:.1f}s.")
            else:
                self.status_changed.emit("[Fase 1/4] Sin tomas Dark. Omitiendo calibración térmica.")

            self.progress_changed.emit(10)

            # ============================================================
            # FASE 2/4: REFERENCIA Y ALINEACIÓN DE LIGHTS (Rango 10% -> 60%)
            # ============================================================
            t0_ref = time.perf_counter()
            ref_path = lights[0]
            self.status_changed.emit(f"[Fase 2/4] Preparando toma de referencia: {os.path.basename(ref_path)}")
            
            ref_raw = load_image_as_float32(ref_path)
            h, w, c = ref_raw.shape

            if sky_mask is not None and sky_mask.shape != (h, w):
                sky_mask = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_LINEAR)

            ref_sky = calibrate_light(ref_raw, master_dark)
            ref_stats = {"median": np.median(ref_sky, axis=(0, 1))}

            self.status_changed.emit("[Fase 2/4] Extrayendo estrellas de la referencia...")
            ref_kp, ref_desc, norm_type = detect_sky_stars(ref_sky, sky_mask=sky_mask)
            ref_gray = cv2.cvtColor((ref_sky * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            self.status_changed.emit(f"-> Estrellas base detectadas: {len(ref_kp)}")

            if ref_desc is None or len(ref_kp) < 15:
                raise RuntimeError("No se detectaron suficientes estrellas en la toma de referencia para alinear.")

            has_ground = (sky_mask is not None and mode == "fixed_tripod")
            use_ram_buffer = evaluate_storage_mode(
                strategy=storage_strategy,
                num_frames=len(lights),
                h=h, w=w, c=c,
                has_ground=has_ground
            )
            storage_label = "Memoria RAM (Ultra-rápido)" if use_ram_buffer else "Caché en Disco (.bin)"
            self.status_changed.emit(f"-> Estrategia de almacenamiento activa: [{storage_label}]")

            sky_frames_collection = []
            ground_frames_collection = []

            ref_sky_processed = preprocess_subframe_lp(
                ref_sky,
                method=lp_method,
                strength=lp_strength,
                sky_mask=sky_mask,
                ref_stats=ref_stats
            )

            if use_ram_buffer:
                sky_frames_collection.append(ref_sky_processed)
                if mode == "fixed_tripod":
                    ground_frames_collection.append(ref_raw.copy())
            else:
                p_sky_0 = os.path.join(temp_dir, "sky_0000.bin")
                save_frame_float32(p_sky_0, ref_sky_processed)
                sky_frames_collection.append(p_sky_0)

                if mode == "fixed_tripod":
                    p_gnd_0 = os.path.join(temp_dir, "gnd_0000.bin")
                    save_frame_float32(p_gnd_0, ref_raw)
                    ground_frames_collection.append(p_gnd_0)

            del ref_raw, ref_sky, ref_sky_processed
            gc.collect()

            dt_ref = time.perf_counter() - t0_ref
            self.status_changed.emit(f"-> Referencia lista en {dt_ref:.1f}s.")
            self.progress_changed.emit(15)

            # Sub-fase de alineación concurrente (15% -> 60%)
            t0_align = time.perf_counter()
            total_lights = len(lights)
            discarded_count = 0
            to_align_count = total_lights - 1

            ref_kps_pts = [(kp.pt[0], kp.pt[1]) for kp in ref_kp]

            tasks = []
            for idx, path in enumerate(lights[1:], start=2):
                task_args = (
                    idx, path, total_lights, temp_dir, mode, master_dark,
                    ref_kps_pts, ref_desc, norm_type, ref_gray,
                    w, h, sky_mask, lp_method, lp_strength, ref_stats,
                    use_ram_buffer
                )
                tasks.append(task_args)

            aligned_results = []
            completed_count = 0
            ctx = mp.get_context("spawn")

            with ProcessPoolExecutor(max_workers=cpu_workers, mp_context=ctx) as executor:
                futures = [executor.submit(align_single_light_task, t) for t in tasks]

                for future in as_completed(futures):
                    if self._is_cancelled:
                        executor.shutdown(wait=False, cancel_futures=True)
                        self.status_changed.emit("Cancelado por el usuario.")
                        return

                    completed_count += 1
                    pct_align = 15 + int((completed_count / max(1, to_align_count)) * 45)
                    self.progress_changed.emit(min(60, pct_align))

                    res = future.result()
                    aligned_results.append(res)

                    if res["success"]:
                        self.status_changed.emit(
                            f"[Fase 2/4] [{res['idx']}/{total_lights}] {res['filename']} -> OK "
                            f"({res['inliers']} inliers | Subpíxel | dx: {res['dx']:.1f}px, dy: {res['dy']:.1f}px)"
                        )
                    else:
                        discarded_count += 1
                        self.status_changed.emit(
                            f"[DESCARTADA] [{res['idx']}/{total_lights}] {res['filename']} -> {res['error']}"
                        )

            aligned_results.sort(key=lambda r: r["idx"])
            for res in aligned_results:
                if res["success"]:
                    sky_frames_collection.append(res["sky_data"])
                    if res["gnd_data"] is not None:
                        ground_frames_collection.append(res["gnd_data"])

            dt_align = time.perf_counter() - t0_align
            self.status_changed.emit(
                f"-> Alineación completada ({to_align_count} tomas en {dt_align:.1f}s | {dt_align / max(1, to_align_count):.2f}s/toma)."
            )
            if discarded_count > 0:
                self.status_changed.emit(f"-> Resumen: {discarded_count} toma(s) descartada(s).")

            if len(sky_frames_collection) < 2:
                raise RuntimeError("No se pudieron alinear suficientes tomas con la referencia.")

            self.progress_changed.emit(60)

            # ============================================================
            # FASE 3/4: INTEGRACIÓN MATEMÁTICA (Rango 60% -> 95%)
            # ============================================================
            backend_label = "GPU CUDA (NVIDIA)" if use_gpu else "CPU Multi-Core"
            sky_target_pct = 78 if has_ground else 95

            if self._is_cancelled: return
            self.status_changed.emit(
                f"[Fase 3/4] Apilando Cielo ({len(sky_frames_collection)} tomas) con [{backend_label}] (Kappa={kappa:.1f}, LP={lp_method})..."
            )
            self.progress_changed.emit(63)

            t0_stack_sky = time.perf_counter()
            sky_stacked = stream_stack_auto(
                sky_frames_collection, 
                (h, w, c), 
                chunk_rows=800, 
                kappa=kappa,
                lp_method=lp_method,
                lp_strength=lp_strength,
                use_gpu=use_gpu
            )
            dt_stack_sky = time.perf_counter() - t0_stack_sky
            self.status_changed.emit(f"-> Apilado de cielo completado en {dt_stack_sky:.1f}s.")
            self.progress_changed.emit(sky_target_pct)

            final_composite = sky_stacked

            # Suelo: 78% -> 95%
            if has_ground:
                if self._is_cancelled: return
                self.status_changed.emit(f"[Fase 3/4] Apilando Suelo ({len(ground_frames_collection)} tomas) con [{backend_label}] (Kappa={kappa:.1f})...")
                self.progress_changed.emit(80)

                t0_stack_gnd = time.perf_counter()
                ground_stacked = stream_stack_auto(
                    ground_frames_collection, 
                    (h, w, c), 
                    chunk_rows=800, 
                    kappa=kappa,
                    use_gpu=use_gpu
                )
                dt_stack_gnd = time.perf_counter() - t0_stack_gnd
                self.status_changed.emit(f"-> Apilado de suelo completado en {dt_stack_gnd:.1f}s.")
                self.progress_changed.emit(95)

                # ============================================================
                # FASE 4/4: COMPOSICIÓN Y GUARDADO (Rango 95% -> 100%)
                # ============================================================
                t0_comp = time.perf_counter()
                self.status_changed.emit("[Fase 4/4] Componiendo imagen final de 32 bits sin discontinuidades...")
                self.progress_changed.emit(96)

                raw_mask = np.clip(sky_mask.astype(np.float32), 0.0, 1.0)

                # Transición suave microscópica (sigma 0.8 - 1.0) para que Claridad no cree halos
                smooth_mask = cv2.GaussianBlur(raw_mask, (5, 5), sigmaX=0.8)
                if smooth_mask.ndim == 2:
                    mask_3d = smooth_mask[..., np.newaxis]
                else:
                    mask_3d = smooth_mask

                # Fusión lineal continua
                base_composite = (sky_stacked * mask_3d) + (ground_stacked * (1.0 - mask_3d))

                # Protección de silueta sin bordes duros:
                # En la franja de transición (donde mask_3d está entre 0.05 y 0.95), 
                # si el suelo estático es más oscuro (ramas/hojas), permitimos que preserve
                # la silueta natural sin crear un escalón de contraste artificial
                transition_zone = (mask_3d > 0.02) & (mask_3d < 0.90)
                final_composite = np.where(
                    transition_zone & (ground_stacked < base_composite),
                    ground_stacked * (1.0 - mask_3d * 0.5) + base_composite * (mask_3d * 0.5),
                    base_composite
                )

                del ground_stacked, mask_3d, smooth_mask, base_composite
                dt_comp = time.perf_counter() - t0_comp
                self.status_changed.emit(f"-> Composición completada en {dt_comp:.2f}s.")
            else:
                self.status_changed.emit("[Fase 4/4] Modo sin suelo. Saltando composición de máscara.")

            if final_composite is not sky_stacked:
                del sky_stacked

            # Guardado TIFF final
            t0_io = time.perf_counter()
            self.progress_changed.emit(98)
            self.status_changed.emit(f"[Fase 4/4] Guardando TIFF de 32 bits en: {os.path.basename(output_path)}...")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass

            tifffile.imwrite(output_path, final_composite.astype(np.float32), compression='zlib')
            dt_io = time.perf_counter() - t0_io
            self.status_changed.emit(f"-> Guardado TIFF completado en {dt_io:.1f}s.")

            dt_total = time.perf_counter() - t_global_start
            m, s = divmod(dt_total, 60)
            self.progress_changed.emit(100)
            self.status_changed.emit(f"¡Apilado completado exitosamente en {int(m)}m {s:.1f}s!")
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

    def __init__(self, image_rgb: np.ndarray, sky_mask: np.ndarray = None, smoothing: float = 0.5, parent=None):
        super().__init__(parent)
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

    def __init__(self, img_rgb: np.ndarray, sky_mask: np.ndarray = None, stride: int = 256, custom_exe: str = None, parent=None):
        super().__init__(parent)
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