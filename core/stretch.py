# core/stretch.py
import numpy as np

def asinh_stretch(image_float: np.ndarray, stretch: float = 3.0, black_point: float = 0.0) -> np.ndarray:
    img = np.maximum(0.0, image_float - black_point)
    if stretch <= 0.0:
        return np.clip(img, 0.0, 1.0)
    stretched = np.arcsinh(img * stretch) / np.arcsinh(stretch)
    return np.clip(stretched, 0.0, 1.0)

def auto_mtf_stretch(image_float: np.ndarray, target_bg: float = 0.25) -> np.ndarray:
    med = float(np.median(image_float))
    abs_diff = np.abs(image_float - med)
    mad = float(np.median(abs_diff))
    
    shadows = max(0.0, med - 2.8 * (1.4826 * mad))
    x = med - shadows
    if x <= 0:
        m = 0.5
    else:
        m = (x * (target_bg - 1.0)) / (x * (2.0 * target_bg - 1.0) - target_bg)
        m = float(np.clip(m, 0.001, 0.999))

    normalized = np.clip((image_float - shadows) / (1.0 - shadows + 1e-7), 0.0, 1.0)
    out = ((m - 1.0) * normalized) / ((2.0 * m - 1.0) * normalized - m)
    return np.clip(out, 0.0, 1.0)

def stretch_display_image(img_float: np.ndarray, exposure_boost: float = 1.8, contrast: float = 1.2) -> np.ndarray:
    lifted = np.clip(img_float * exposure_boost, 0.0, 1.0)
    x = lifted - 0.5
    curve = 1.0 / (1.0 + np.exp(-contrast * 6.0 * x))
    c_min = 1.0 / (1.0 + np.exp(contrast * 3.0))
    c_max = 1.0 / (1.0 + np.exp(-contrast * 3.0))
    stretched = (curve - c_min) / (c_max - c_min + 1e-7)
    return np.clip(stretched, 0.0, 1.0)

def manual_stretch(image_float: np.ndarray, black_point: float = 0.0, stretch_factor: float = 5.0) -> np.ndarray:
    bp = np.clip(black_point, 0.0, 0.95)
    normalized = np.maximum(0.0, image_float - bp) / (1.0 - bp + 1e-7)
    if stretch_factor <= 0.1:
        return np.clip(normalized, 0.0, 1.0)
    stretched = np.arcsinh(normalized * stretch_factor) / np.arcsinh(stretch_factor)
    return np.clip(stretched, 0.0, 1.0)