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
    
def adjust_vibrance(img_rgb: np.ndarray, vibrance: float = 0.0) -> np.ndarray:
    """
    Ajusta la intensidad (Vibrance) de forma selectiva.
    vibrance va de -1.0 a 1.0 (0.0 = neutro).
    Satura más los tonos tenues y protege los tonos ya saturados.
    """
    if abs(vibrance) < 1e-4:
        return img_rgb

    # Trabajamos en espacio HSV
    hsv = cv2.cvtColor(np.clip(img_rgb, 0.0, 1.0), cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]

    if vibrance > 0:
        # Cuanto menos saturado esté el píxel, más incremento recibe
        factor = 1.0 + vibrance * (1.0 - sat)
        hsv[..., 1] = np.clip(sat * factor, 0.0, 1.0)
    else:
        # Reducción suave
        hsv[..., 1] = np.clip(sat * (1.0 + vibrance), 0.0, 1.0)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def adjust_contrast(img_rgb: np.ndarray, contrast: float = 0.0, pivot: float = None) -> np.ndarray:
    """
    Ajuste de contraste sigmoidal en 'S' sobre espacio estirado [0.0, 1.0].
    - contrast > 0: curva en S (sombras caen, luces suben, aumenta el relieve).
    - contrast < 0: comprime el rango (sombras suben, luces bajan).
    """
    if abs(contrast) < 1e-4:
        return img_rgb

    # Si no se indica pivote, se usa la mediana real de la imagen estirada
    if pivot is None:
        pivot = float(np.median(img_rgb))
    
    pivot = np.clip(pivot, 0.05, 0.85)

    # Factor de ganancia (alfa)
    # contrast de -1.0 a +1.0
    alpha = contrast * 2.5

    # Función sigmoide cuadrática normalizada
    # delta va de -pivot a (1 - pivot)
    delta = img_rgb - pivot

    # Aplicamos una función tangente hiperbólica normalizada
    # Esto asegura que f(pivot) = pivot, f(0) >= 0 y f(1) <= 1
    # sin cambiar el nivel medio de brillo
    if contrast >= 0:
        factor = np.tanh(alpha * delta) / np.tanh(alpha * max(pivot, 1.0 - pivot) + 1e-6)
        # Atenuación en los extremos para no recortar a blanco o negro duro
        scale = np.where(delta < 0, pivot, 1.0 - pivot)
        out = pivot + factor * scale
    else:
        # Suavizado de contraste
        gain = 1.0 / (1.0 - alpha * 0.5)
        out = pivot + delta * (1.0 + contrast * 0.7)

    return np.clip(out, 0.0, 1.0).astype(np.float32)
    
# --- Ecualizador Tonal de 4 Bandas (Starless) ---
def adjust_tonal_bands(
    img_rgb: np.ndarray,
    blacks: float = 0.0,
    shadows: float = 0.0,
    highlights: float = 0.0,
    whites: float = 0.0
) -> np.ndarray:
    """
    Ecualizador de 4 bandas actuando sobre la imagen ya estirada (perceptual [0.0, 1.0]):
    - blacks: [-1.0, 1.0] ajusta el anclaje del fondo profundo (0.0 a 0.25).
    - shadows: [-1.0, 1.0] modula el polvo y nebulosas tenues (0.15 a 0.50).
    - highlights: [-1.0, 1.0] modula estructuras brillantes (0.45 a 0.80).
    - whites: [-1.0, 1.0] expande o comprime los picos de luminosidad (0.70 a 1.0).
    """
    if all(abs(v) < 1e-4 for v in (blacks, shadows, highlights, whites)):
        return img_rgb

    # 1. Luminancia perceptiva (segura y delimitada en 0..1)
    lum = 0.2126 * img_rgb[..., 0] + 0.7152 * img_rgb[..., 1] + 0.0722 * img_rgb[..., 2]
    lum = np.clip(lum, 0.0, 1.0)

    # 2. Pesos polinómicos suaves y continuos (sin funciones trigonométricas ni potencias fraccionarias)
    # A) Negros: máximo en 0.0, decrece hasta 0.25
    w_blacks = np.clip((0.25 - lum) / 0.25, 0.0, 1.0) ** 2

    # B) Sombras: campana suave entre 0.10 y 0.55 (pico en ~0.30)
    w_shadows = np.clip((lum - 0.10) / 0.20, 0.0, 1.0) * np.clip((0.55 - lum) / 0.25, 0.0, 1.0) * 4.0

    # C) Altas Luces: campana entre 0.40 y 0.85 (pico en ~0.65)
    w_highlights = np.clip((lum - 0.40) / 0.25, 0.0, 1.0) * np.clip((0.85 - lum) / 0.20, 0.0, 1.0) * 4.0

    # D) Blancos: sube a partir de 0.65 hasta 1.0
    w_whites = np.clip((lum - 0.65) / 0.35, 0.0, 1.0) ** 2

    # 3. Deltas tonales (con escala de fuerza equilibrada e intuitiva)
    delta_lum = (
        (blacks * 0.12 * w_blacks) +
        (shadows * 0.18 * w_shadows) +
        (highlights * 0.18 * w_highlights) +
        (whites * 0.15 * w_whites)
    )

    new_lum = np.clip(lum + delta_lum, 0.0, 1.0)

    # 4. Modulación cromática segura (evitando cualquier división por cero)
    eps = 1e-6
    ratio = (new_lum + eps) / (lum + eps)
    out_rgb = img_rgb * ratio[..., np.newaxis]

    return np.clip(out_rgb, 0.0, 1.0).astype(np.float32)

# --- Reducción de Ruido Multimodelo (Starless) ---

def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Implementación rápida en NumPy del Guided Filter."""
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(guide, -1, ksize)
    mean_p = cv2.boxFilter(src, -1, ksize)
    mean_Ip = cv2.boxFilter(guide * src, -1, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide * guide, -1, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)

    q = mean_a * guide + mean_b
    return q


def apply_denoise(
    img_rgb: np.ndarray,
    strength: float = 0.0,
    method: str = "Bilateral",
    is_full_res: bool = False
) -> np.ndarray:
    """
    Aplica reducción de ruido en img_rgb (float32 [0.0, 1.0]).
    - strength: [0.0, 1.0] (0% a 100%)
    - method: 'Bilateral', 'Guided Filter', o 'NL-Means'
    - is_full_res: si es True, escala el radio del kernel para resolución completa.
    """
    if strength <= 1e-4:
        return img_rgb

    scale_factor = 2.5 if is_full_res else 1.0
    u8 = (np.clip(img_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    if method == "Bilateral":
        # d: diámetro del vecindario de píxeles
        d = int(max(3, round((5 + 4 * strength) * scale_factor)))
        sigma_color = float(strength * 75.0)
        sigma_space = float((strength * 8.0 + 2.0) * scale_factor)
        filtered = cv2.bilateralFilter(u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        return (filtered.astype(np.float32) / 255.0)

    elif method == "Guided Filter":
        radius = int(max(2, round((2 + 5 * strength) * scale_factor)))
        # Regularización epsilon: controla cuánto suaviza las variaciones pequeñas
        eps = float((0.001 + 0.04 * strength) ** 2)
        gray_guide = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        channels = []
        for c in range(3):
            ch_denoised = _guided_filter(gray_guide, img_rgb[..., c], radius=radius, eps=eps)
            channels.append(ch_denoised)
        out = np.stack(channels, axis=-1)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    elif method == "NL-Means":
        # fastNlMeansDenoisingColored opera en uint8
        h_lum = float(strength * 18.0)
        h_col = float(strength * 18.0)
        t_size = 7
        s_size = 15 if not is_full_res else 21
        filtered = cv2.fastNlMeansDenoisingColored(
            u8, None,
            h=h_lum, hColor=h_col,
            templateWindowSize=t_size, searchWindowSize=s_size
        )
        return (filtered.astype(np.float32) / 255.0)

    return img_rgb
    
def apply_curve_lut(img_rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Aplica una curva tonal mediante tabla de consulta (LUT de 256 elementos [0..1]).
    Rendimiento sub-milisegundo vía cv2.LUT.
    """
    if lut is None or img_rgb is None:
        return img_rgb

    # Escalado a uint8 -> mapeo LUT -> recuperación float32
    u8 = (np.clip(img_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    lut_u8 = (np.clip(lut, 0.0, 1.0) * 255.0).astype(np.uint8)

    mapped_u8 = cv2.LUT(u8, lut_u8)
    return (mapped_u8.astype(np.float32) / 255.0)