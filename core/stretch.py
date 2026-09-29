# core/stretch.py
import numpy as np

import cv2  

def adjust_saturation_dual(
    img_rgb: np.ndarray, 
    sat_sky: float = 1.0, 
    sat_ground: float = 1.0, 
    sky_mask: np.ndarray = None
) -> np.ndarray:
    """
    Ajusta la saturación en espacio HSV en float32 de forma independiente para cielo y suelo.
    sat_sky, sat_ground: 1.0 es original, 0.0 es escala de grises, 2.0 es el doble de saturación.
    """
    if sat_sky == 1.0 and (sat_ground == 1.0 or sky_mask is None):
        return img_rgb

    # Conversión a HSV en rango [0.0, 1.0]
    hsv = cv2.cvtColor(np.clip(img_rgb, 0.0, 1.0), cv2.COLOR_RGB2HSV)

    if sky_mask is None or sat_sky == sat_ground:
        # Ajuste global directo
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_sky, 0.0, 1.0)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # 1. Crear versión saturada para cielo y versión saturada para suelo
    hsv_sky = hsv.copy()
    hsv_sky[..., 1] = np.clip(hsv_sky[..., 1] * sat_sky, 0.0, 1.0)
    rgb_sky = cv2.cvtColor(hsv_sky, cv2.COLOR_HSV2RGB)

    hsv_gnd = hsv.copy()
    hsv_gnd[..., 1] = np.clip(hsv_gnd[..., 1] * sat_ground, 0.0, 1.0)
    rgb_gnd = cv2.cvtColor(hsv_gnd, cv2.COLOR_HSV2RGB)

    # 2. Suavizado de máscara para evitar artefactos o halos en la silueta (horizonte/árboles)
    h, w = img_rgb.shape[:2]
    if sky_mask.shape != (h, w):
        mask_aligned = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        mask_aligned = sky_mask

    # Desenfoque gaussiano suave adaptado a la resolución (kernel impar ~15-25 px)
    ksize = int(max(15, (min(h, w) // 150) | 1))
    smooth_mask = cv2.GaussianBlur(mask_aligned, (ksize, ksize), sigmaX=ksize / 3.0)
    smooth_mask = np.repeat(smooth_mask[..., np.newaxis], 3, axis=2)

    # 3. Mezcla lineal continua
    blended = (rgb_sky * smooth_mask) + (rgb_gnd * (1.0 - smooth_mask))
    return np.clip(blended, 0.0, 1.0)

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
    
def apply_white_balance(img_rgb: np.ndarray, temp_factor: float = 0.0, tint_factor: float = 0.0) -> np.ndarray:
    """
    Aplica balance de blancos multiplicativo sobre float32 lineal.
    - temp_factor: [-1.0, 1.0] (negativo enfría/azul, positivo calienta/rojo)
    - tint_factor: [-1.0, 1.0] (negativo tira a verde, positivo tira a magenta)
    """
    if temp_factor == 0.0 and tint_factor == 0.0:
        return img_rgb

    # Canales RGB base
    r_gain = 1.0 + (temp_factor * 0.5) + (tint_factor * 0.25)
    g_gain = 1.0 - (tint_factor * 0.5)
    b_gain = 1.0 - (temp_factor * 0.5) + (tint_factor * 0.25)

    gains = np.array([r_gain, g_gain, b_gain], dtype=np.float32)
    # Normalizar para que la luminancia promedio no sufra saltos bruscos
    gains /= np.mean(gains)

    balanced = img_rgb * gains
    return np.clip(balanced, 0.0, 1.0)