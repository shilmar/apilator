# core/stretch.py
import numpy as np

def calculate_mtf_params(img_float: np.ndarray, target_background: float = 0.20):
    """
    Calcula el punto negro estadístico y el punto de medios tonos 'm' de la curva MTF.
    """
    sample = img_float[::4, ::4]
    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med)))
    
    # Punto negro sugerido: justo bajo el ruido de fondo
    bp = float(np.clip(med - 1.5 * mad, 0.0, 0.95))
    
    # Estimación analítica del parámetro 'm' de MTF para llevar el fondo a target_background
    norm_med = max(1e-6, (med - bp) / max(1e-6, 1.0 - bp))
    denom = norm_med * (2.0 * target_background - 1.0) - target_background
    if abs(denom) < 1e-7:
        m = 0.5
    else:
        m = (norm_med * (target_background - 1.0)) / denom
    
    m = float(np.clip(m, 0.0001, 0.9999))
    return bp, m

def mtf_curve(x: np.ndarray, m: float) -> np.ndarray:
    """Función de transferencia de medios tonos estándar (Midtone Transfer Function)."""
    # MTF(x, m) = (m - 1) * x / ((2m - 1) * x - m)
    num = (m - 1.0) * x
    den = (2.0 * m - 1.0) * x - m
    return np.where(np.abs(den) > 1e-7, num / den, 0.0)

def auto_mtf_stretch(img_float: np.ndarray, target_background: float = 0.20):
    bp, m = calculate_mtf_params(img_float, target_background)
    stretched = manual_stretch(img_float, black_point=bp, midtone=m)
    return stretched, bp, m

def manual_stretch(img_float: np.ndarray, black_point: float = 0.0, midtone: float = 0.1):
    # 1. Recorte y re-escalado del punto negro
    clipped = np.clip((img_float - black_point) / max(1e-6, 1.0 - black_point), 0.0, 1.0)
    # 2. Aplicación directa de curva MTF con el 'm' elegido
    stretched = mtf_curve(clipped, max(1e-5, min(0.9999, midtone)))
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)