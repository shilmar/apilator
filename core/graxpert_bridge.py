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


def run_graxpert_background_extraction(
    image_rgb: np.ndarray, 
    sky_mask: np.ndarray = None, 
    smoothing: float = 0.5
) -> np.ndarray:
    h, w, c = image_rgb.shape
    temp_dir = tempfile.mkdtemp(prefix="graxpert_run_")

    try:
        in_tiff = os.path.join(temp_dir, "input.tif")
        out_folder = os.path.join(temp_dir, "output")
        os.makedirs(out_folder, exist_ok=True)

        if sky_mask is not None:
            sky_pixels = image_rgb[sky_mask > 0.5]
            sky_fill = (
                np.median(sky_pixels, axis=0) 
                if len(sky_pixels) > 0 
                else np.array([0.1, 0.1, 0.1], dtype=np.float32)
            )
            mask_3d = np.repeat(sky_mask[..., np.newaxis], 3, axis=2)
            prep_img = (image_rgb * mask_3d) + (sky_fill * (1.0 - mask_3d))
        else:
            prep_img = image_rgb.copy()

        tifffile.imwrite(in_tiff, prep_img.astype(np.float32))

        graxpert_bin = _find_graxpert_executable()
        cmd = [
            graxpert_bin,
            "-cli",
            "-cmd", "background-extraction",
            "-output", out_folder,
            in_tiff
        ]

        res = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )

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

        if corrected_full.shape[:2] != (h, w):
            corrected_full = cv2.resize(corrected_full, (w, h), interpolation=cv2.INTER_CUBIC)

        if sky_mask is not None:
            mask_3d = np.repeat(sky_mask[..., np.newaxis], 3, axis=2)
            final_composite = (corrected_full * mask_3d) + (image_rgb * (1.0 - mask_3d))
        else:
            final_composite = corrected_full

        return np.clip(final_composite, 0.0, 1.0)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)