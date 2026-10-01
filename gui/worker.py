# gui/worker.py
import os
import gc
import shutil
import tempfile
import time
import cv2
import numpy as np
import tifffile
import multiprocessing as mp

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

from concurrent.futures import ProcessPoolExecutor, as_completed


from core.graxpert_bridge import run_graxpert_background_extraction
from core.starnet_bridge import run_starnet


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

            lp_method = self.config.get("lp_method", "standard")
            lp_strength = float(self.config.get("lp_strength", 0.5))
            use_gpu = self.config.get("use_gpu", True) and HAS_GPU
            cpu_workers = int(self.config.get("cpu_workers", 4))


            if len(lights) < 2:
                self.error_occurred.emit("Se necesitan al menos 2 tomas de luz para apilar.")
                return


            self.status_changed.emit("Iniciando preparación del entorno temporal...")
            self.progress_changed.emit(2)

            temp_dir = tempfile.mkdtemp(prefix="astro_gui_")
            # --- ETAPA 1: MASTER DARK ---
            master_dark = None
            if darks:
                t0_darks = time.perf_counter()
                self.status_changed.emit(f"Generando Master Dark a partir de {len(darks)} tomas...")
                master_dark = create_master_dark(darks)
                dt_darks = time.perf_counter() - t0_darks
                self.status_changed.emit(f"-> Master Dark completado en {dt_darks:.1f}s.")

            # --- ETAPA 2: REFERENCIA Y DETECCIÓN BASE ---
            t0_ref = time.perf_counter()
            ref_path = lights[0]
            self.status_changed.emit(f"Cargando toma de referencia: {os.path.basename(ref_path)}")
            
            ref_raw = load_image_as_float32(ref_path)
            h, w, c = ref_raw.shape

            if sky_mask is not None and sky_mask.shape != (h, w):
                sky_mask = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_LINEAR)

            ref_sky = calibrate_light(ref_raw, master_dark)
            ref_stats = {"median": np.median(ref_sky, axis=(0, 1))}

            self.status_changed.emit("Extrayendo estrellas de la toma de referencia...")
            ref_kp, ref_desc, norm_type = detect_sky_stars(ref_sky, sky_mask=sky_mask)
            ref_gray = cv2.cvtColor((ref_sky * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            self.status_changed.emit(f"-> Estrellas base detectadas: {len(ref_kp)}")

            if ref_desc is None or len(ref_kp) < 15:
                raise RuntimeError("No se detectaron suficientes estrellas en la toma de referencia para alinear.")

            sky_temp_files = []
            ground_temp_files = []

            ref_sky_processed = preprocess_subframe_lp(
                ref_sky,
                method=lp_method,
                strength=lp_strength,
                sky_mask=sky_mask,
                ref_stats=ref_stats
            )

            p_sky_0 = os.path.join(temp_dir, "sky_0000.bin")
            save_frame_float32(p_sky_0, ref_sky_processed)
            sky_temp_files.append(p_sky_0)

            if mode == "fixed_tripod":
                p_gnd_0 = os.path.join(temp_dir, "gnd_0000.bin")
                save_frame_float32(p_gnd_0, ref_raw)
                ground_temp_files.append(p_gnd_0)

            del ref_raw, ref_sky, ref_sky_processed
            gc.collect()

            dt_ref = time.perf_counter() - t0_ref
            self.status_changed.emit(f"-> Preparación de referencia completada en {dt_ref:.1f}s.")

            # --- ETAPA 3: ALINEACIÓN PARALELA DE LIGHTS ---
            t0_align = time.perf_counter()
            total_lights = len(lights)
            discarded_count = 0

            # Convertir KeyPoints a tuplas (x, y) serializables en multiproceso
            ref_kps_pts = [(kp.pt[0], kp.pt[1]) for kp in ref_kp]

            tasks = []
            for idx, path in enumerate(lights[1:], start=2):
                task_args = (
                    idx, path, total_lights, temp_dir, mode, master_dark,
                    ref_kps_pts, ref_desc, norm_type, ref_gray,
                    w, h, sky_mask, lp_method, lp_strength, ref_stats
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
                    pct = int(5 + (completed_count / (total_lights - 1)) * 45)
                    self.progress_changed.emit(pct)

                    res = future.result()
                    aligned_results.append(res)

                    if res["success"]:
                        self.status_changed.emit(
                            f"[{res['idx']}/{total_lights}] {res['filename']} -> OK "
                            f"({res['inliers']} inliers | Subpíxel | dx: {res['dx']:.1f}px, dy: {res['dy']:.1f}px)"
                        )
                    else:
                        discarded_count += 1
                        self.status_changed.emit(
                            f"[DESCARTADA] [{res['idx']}/{total_lights}] {res['filename']} -> {res['error']}"
                        )

            # Ordenar por índice para preservar el orden temporal de las tomas
            aligned_results.sort(key=lambda r: r["idx"])
            for res in aligned_results:
                if res["success"]:
                    sky_temp_files.append(res["p_sky"])
                    if res["p_gnd"]:
                        ground_temp_files.append(res["p_gnd"])

            dt_align = time.perf_counter() - t0_align
            self.status_changed.emit(
                f"-> Alineación paralela ({total_lights - 1} tomas con {cpu_workers} hilos): "
                f"{dt_align:.1f}s (media efectiva: {dt_align / max(1, total_lights - 1):.2f}s/toma)."
            )

            if discarded_count > 0:
                self.status_changed.emit(f"-> Resumen: {discarded_count} toma(s) descartada(s).")

            if len(sky_temp_files) < 2:
                raise RuntimeError("No se pudieron alinear suficientes tomas con la referencia.")

            # --- ETAPA 4: INTEGRACIÓN MATEMÁTICA (CIELO Y SUELO) ---
            backend_label = "GPU CUDA (NVIDIA)" if use_gpu else "CPU Multi-Core"

            if self._is_cancelled: return
            self.status_changed.emit(
                f"Apilando Cielo ({len(sky_temp_files)} tomas) con [{backend_label}] (Kappa={kappa:.1f}, LP={lp_method})..."
            )
            self.progress_changed.emit(55)

            t0_stack_sky = time.perf_counter()
            sky_stacked = stream_stack_auto(
                sky_temp_files, 
                (h, w, c), 
                chunk_rows=800, 
                kappa=kappa,
                lp_method=lp_method,
                lp_strength=lp_strength,
                use_gpu=use_gpu
            )
            dt_stack_sky = time.perf_counter() - t0_stack_sky
            self.status_changed.emit(f"-> Apilado de cielo finalizado en {dt_stack_sky:.1f}s.")

            final_composite = sky_stacked

            if sky_mask is not None and mode == "fixed_tripod":
                if self._is_cancelled: return
                self.status_changed.emit(f"Apilando Suelo ({len(ground_temp_files)} tomas) con [{backend_label}] (Kappa={kappa:.1f})...")
                self.progress_changed.emit(75)

                t0_stack_gnd = time.perf_counter()
                ground_stacked = stream_stack_auto(
                    ground_temp_files, 
                    (h, w, c), 
                    chunk_rows=800, 
                    kappa=kappa,
                    use_gpu=use_gpu
                )
                dt_stack_gnd = time.perf_counter() - t0_stack_gnd
                self.status_changed.emit(f"-> Apilado de suelo finalizado en {dt_stack_gnd:.1f}s.")

                # --- ETAPA 5: COMPOSICIÓN Y BLENDING ---
                t0_comp = time.perf_counter()
                self.status_changed.emit("Componiendo imagen final de 32 bits con máscara suavizada...")
                self.progress_changed.emit(92)

                ksize = int(max(15, (min(h, w) // 150) | 1))
                if ksize % 2 == 0:
                    ksize += 1
                smooth_mask = cv2.GaussianBlur(sky_mask.astype(np.float32), (ksize, ksize), sigmaX=ksize / 3.0)
                mask_3d = np.repeat(smooth_mask[..., np.newaxis], 3, axis=2)

                final_composite = (sky_stacked * mask_3d) + (ground_stacked * (1.0 - mask_3d))
                del ground_stacked, mask_3d, smooth_mask
                dt_comp = time.perf_counter() - t0_comp
                self.status_changed.emit(f"-> Composición completada en {dt_comp:.2f}s.")

            if final_composite is not sky_stacked:
                del sky_stacked

            # --- ETAPA 6: ESCRITURA EN DISCO ---
            t0_io = time.perf_counter()
            self.status_changed.emit(f"Guardando resultado de 32 bits en: {output_path}...")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass

            tifffile.imwrite(output_path, final_composite.astype(np.float32), compression='zlib')
            dt_io = time.perf_counter() - t0_io
            self.status_changed.emit(f"-> Guardado TIFF completado en {dt_io:.1f}s.")

            self.progress_changed.emit(100)
            self.status_changed.emit("¡Apilado completado exitosamente!")
            self.finished_success.emit(output_path)

        except Exception as exc:
            import traceback
            print(f"\n[EXCEPCION EN WORKER]: {exc}", flush=True)
            traceback.print_exc()
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