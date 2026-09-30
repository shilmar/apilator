# core/graxpert_bridge.py
import os
import sys
import shutil
import tempfile
import subprocess
import cv2
import numpy as np
import tifffile
from astropy.io import fits


def _find_graxpert_executable():
    py_dir = os.path.dirname(sys.executable)
    scripts_graxpert = os.path.join(py_dir, "Scripts", "graxpert.exe")
    if os.path.exists(scripts_graxpert):
        return scripts_graxpert
    
    direct_graxpert = os.path.join(py_dir, "graxpert.exe")
    if os.path.exists(direct_graxpert):
        return direct_graxpert

    found = shutil.which("graxpert")
    return found if found else "graxpert"


def _read_image_generic(filepath: str) -> np.ndarray:
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in ['.fits', '.fit', '.fts']:
        with fits.open(filepath) as hdul:
            data = hdul[0].data.astype(np.float32)
            if data.ndim == 2:
                data = np.stack([data] * 3, axis=-1)
            elif data.ndim == 3:
                if data.shape[0] in [3, 4]:
                    data = np.transpose(data[:3], (1, 2, 0))
            return data
    else:
        data = tifffile.imread(filepath).astype(np.float32)
        if data.max() > 1.0:
            data /= 65535.0 if data.dtype == np.uint16 else 1.0
        if data.ndim == 2:
            data = np.stack([data] * 3, axis=-1)
        return data


def _project_sky_gradient_downwards(image_rgb: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
    """
    Extiende el gradiente del cielo proyectando verticalmente el valor
    del horizonte hacia el fondo del suelo. Evita discontinuidades y arcos artificiales.
    """
    h, w, c = image_rgb.shape
    sky_bin = (sky_mask > 0.4).astype(np.uint8)

    # Erosionar máscara de cielo para no capturar ramas ni siluetas terrestres
    k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    clean_sky = cv2.erode(sky_bin, k_erode)

    prep = image_rgb.copy()

    # Para cada columna, encontrar el píxel de cielo válido más bajo y rellenar hacia abajo
    for col in range(w):
        sky_indices = np.where(clean_sky[:, col] > 0)[0]
        if len(sky_indices) > 0:
            last_sky_row = sky_indices[-1]
            last_val = image_rgb[last_sky_row, col, :]
            prep[last_sky_row:, col, :] = last_val
        else:
            # Fallback si toda la columna es suelo
            prep[:, col, :] = np.median(image_rgb[clean_sky > 0], axis=0) if np.any(clean_sky > 0) else 0.1

    # Desenfoque gaussiano horizontal/vertical para suavizar la base sintética
    prep_blurred = cv2.GaussianBlur(prep, (51, 51), sigmaX=30.0, sigmaY=30.0)
    
    # Componer: cielo real donde hay cielo limpio, proyección suave donde hay tierra/árboles
    mask_3d = np.repeat(clean_sky[..., np.newaxis].astype(np.float32), 3, axis=2)
    result = (image_rgb * mask_3d) + (prep_blurred * (1.0 - mask_3d))
    return np.ascontiguousarray(result, dtype=np.float32)


def run_graxpert_background_extraction(
    image_rgb: np.ndarray, 
    sky_mask: np.ndarray = None, 
    smoothing: float = 0.5,
    algorithm: str = "Spline"  # "Spline" o "RBF" son mucho más estables en paisaje que "AI"
) -> np.ndarray:
    h, w, c = image_rgb.shape
    temp_dir = tempfile.mkdtemp(prefix="graxpert_run_")

    try:
        in_tiff = os.path.join(temp_dir, "input.tif")
        out_folder = os.path.join(temp_dir, "output")
        os.makedirs(out_folder, exist_ok=True)

        if sky_mask is not None:
            prep_img = _project_sky_gradient_downwards(image_rgb, sky_mask)
        else:
            prep_img = image_rgb.copy()

        tifffile.imwrite(in_tiff, prep_img.astype(np.float32))

        graxpert_bin = _find_graxpert_executable()
        
        # Parámetros explícitos: algoritmo suave, corrección por sustracción
        cmd = [
            graxpert_bin,
            "-cli",
            "-cmd", "background-extraction",
            "-smoothing", str(float(np.clip(smoothing, 0.2, 1.0))),
            "-correction", "Subtraction",
            "-output", out_folder,
            in_tiff
        ]

        # Si el ejecutable admite -algorithm, forzar Spline o RBF
        # Si se usa AI en paisaje, la red suele fallar en las cúpulas de luz
        try:
            cmd_with_algo = cmd.copy()
            cmd_with_algo.extend(["-algorithm", algorithm])
            res = subprocess.run(
                cmd_with_algo, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            if res.returncode != 0:
                # Si la versión de GraXpert instalada no reconoce el flag -algorithm, ejecutar estándar
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if res.returncode != 0:
            err_msg = res.stderr if res.stderr else res.stdout
            raise RuntimeError(f"GraXpert devolvió error (código {res.returncode}):\n{err_msg}")

        actual_output = None
        candidates = []
        for root, _, files in os.walk(temp_dir):
            for f in files:
                full_p = os.path.join(root, f)
                lower_f = f.lower()
                if (lower_f.endswith(('.tif', '.tiff', '.fits', '.fit')) 
                        and os.path.abspath(full_p) != os.path.abspath(in_tiff)):
                    candidates.append(full_p)

        for cand in candidates:
            if "output" in cand.lower() or "graxpert" in cand.lower():
                actual_output = cand
                break

        if not actual_output and candidates:
            actual_output = candidates[0]

        if not actual_output or not os.path.exists(actual_output):
            raise RuntimeError(f"No se halló el archivo de salida en {temp_dir}.")

        corrected_full = _read_image_generic(actual_output)
        
        # Proteger contra artefactos de desbordamiento (underflow/overflow) de GraXpert
        corrected_full = np.nan_to_num(corrected_full, nan=0.0, posinf=1.0, neginf=0.0)
        corrected_full = np.clip(corrected_full, 0.0, 1.0)

        if corrected_full.shape[:2] != (h, w):
            corrected_full = cv2.resize(corrected_full, (w, h), interpolation=cv2.INTER_CUBIC)

        if sky_mask is not None:
            # Fusión progresiva con la imagen base mediante máscara desenfocada
            ksize = int(max(25, (min(h, w) // 100) | 1))
            if ksize % 2 == 0:
                ksize += 1
            smooth_mask = cv2.GaussianBlur(sky_mask.astype(np.float32), (ksize, ksize), sigmaX=ksize / 3.0)
            mask_3d = np.repeat(smooth_mask[..., np.newaxis], 3, axis=2)
            
            # El suelo se conserva íntegro y el cielo recibe la corrección limpia
            final_composite = (corrected_full * mask_3d) + (image_rgb * (1.0 - mask_3d))
        else:
            final_composite = corrected_full

        return np.clip(final_composite, 0.0, 1.0)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)