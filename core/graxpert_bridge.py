# core/graxpert_bridge.py
"""
core/graxpert_bridge.py - Puente de invocación para GraXpert CLI (extracción de gradiente de fondo).
"""
from typing import Optional
import os
import shutil
import subprocess
import sys
import tempfile
import cv2
import numpy as np
import tifffile
from astropy.io import fits


def _find_graxpert_executable() -> str:
    """Localiza el ejecutable de GraXpert en el entorno virtual activo o en el sistema."""
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
    """Lee un archivo FITS o TIFF y retorna un array normalizado float32 [0.0, 1.0] con 3 canales."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in ['.fits', '.fit', '.fts']:
        with fits.open(filepath) as hdul:
            data = hdul[0].data.astype(np.float32)
            if data.ndim == 2:
                data = np.stack([data] * 3, axis=-1)
            elif data.ndim == 3 and data.shape[0] in [3, 4]:
                data = np.transpose(data[:3], (1, 2, 0))
            d_min, d_max = data.min(), data.max()
            return (data - d_min) / (d_max - d_min + 1e-8)
    else:
        data = tifffile.imread(filepath).astype(np.float32)
        if data.ndim == 2:
            data = np.stack([data] * 3, axis=-1)
        if data.max() > 1.05:
            max_v = 65535.0 if data.dtype == np.uint16 else 255.0
            data /= max_v
        return np.clip(data, 0.0, 1.0)


def _project_sky_gradient_downwards(image_rgb: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
    """
    Extiende el gradiente del cielo proyectando verticalmente el valor del horizonte
    hacia el suelo. Evita discontinuidades y arcos artificiales generados por la IA en la tierra.
    """
    h, w, c = image_rgb.shape
    sky_bin = (sky_mask > 0.4).astype(np.uint8)

    # Erosionar máscara de cielo para no capturar ramas ni siluetas terrestres
    k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    clean_sky = cv2.erode(sky_bin, k_erode)

    prep = image_rgb.copy()

    # Rellenar verticalmente hacia abajo desde el último píxel de cielo válido
    for col in range(w):
        sky_indices = np.where(clean_sky[:, col] > 0)[0]
        if len(sky_indices) > 0:
            last_sky_row = sky_indices[-1]
            prep[last_sky_row:, col, :] = image_rgb[last_sky_row, col, :]
        else:
            prep[:, col, :] = np.median(image_rgb[clean_sky > 0], axis=0) if np.any(clean_sky > 0) else 0.1

    prep_blurred = cv2.GaussianBlur(prep, (51, 51), sigmaX=30.0, sigmaY=30.0)
    mask_3d = clean_sky[..., np.newaxis].astype(np.float32)
    result = (image_rgb * mask_3d) + (prep_blurred * (1.0 - mask_3d))
    return np.ascontiguousarray(result, dtype=np.float32)


def run_graxpert_background_extraction(
    image_rgb: np.ndarray, 
    sky_mask: Optional[np.ndarray] = None, 
    smoothing: float = 0.5,
    algorithm: str = "Spline"
) -> np.ndarray:
    """
    Ejecuta GraXpert por CLI para extraer y restar el gradiente de fondo.
    Protege el suelo mediante proyección de horizonte si se proporciona sky_mask.
    """
    h, w, _ = image_rgb.shape
    temp_dir = tempfile.mkdtemp(prefix="graxpert_run_")

    try:
        in_tiff = os.path.join(temp_dir, "input.tif")
        out_folder = os.path.join(temp_dir, "output")
        os.makedirs(out_folder, exist_ok=True)

        prep_img = _project_sky_gradient_downwards(image_rgb, sky_mask) if sky_mask is not None else image_rgb
        tifffile.imwrite(in_tiff, prep_img.astype(np.float32))

        graxpert_bin = _find_graxpert_executable()
        smooth_val = str(float(np.clip(smoothing, 0.2, 1.0)))

        cmd = [
            graxpert_bin,
            "-cli",
            "-cmd", "background-extraction",
            "-smoothing", smooth_val,
            "-correction", "Subtraction",
            "-output", out_folder,
            in_tiff
        ]

        try:
            cmd_with_algo = cmd.copy()
            cmd_with_algo.extend(["-algorithm", algorithm])
            res = subprocess.run(cmd_with_algo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if res.returncode != 0:
            err_msg = res.stderr if res.stderr else res.stdout
            raise RuntimeError(f"GraXpert devolvió error (código {res.returncode}):\n{err_msg}")

        # Buscar el archivo generado dentro de out_folder
        candidates = [
            os.path.join(out_folder, f) for f in os.listdir(out_folder)
            if f.lower().endswith(('.tif', '.tiff', '.fits', '.fit'))
        ]

        if not candidates:
            raise RuntimeError(f"GraXpert finalizó pero no generó archivos en {out_folder}.")

        actual_output = candidates[0]
        corrected_full = _read_image_generic(actual_output)
        corrected_full = np.nan_to_num(corrected_full, nan=0.0, posinf=1.0, neginf=0.0)
        corrected_full = np.clip(corrected_full, 0.0, 1.0)

        if corrected_full.shape[:2] != (h, w):
            corrected_full = cv2.resize(corrected_full, (w, h), interpolation=cv2.INTER_CUBIC)

        if sky_mask is not None:
            ksize = int(max(25, (min(h, w) // 100) | 1))
            if ksize % 2 == 0:
                ksize += 1
            smooth_mask = cv2.GaussianBlur(sky_mask.astype(np.float32), (ksize, ksize), sigmaX=ksize / 3.0)
            mask_3d = smooth_mask[..., np.newaxis]
            final_composite = (corrected_full * mask_3d) + (image_rgb * (1.0 - mask_3d))
        else:
            final_composite = corrected_full

        return np.clip(final_composite, 0.0, 1.0).astype(np.float32)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)