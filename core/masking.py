# core/masking.py
"""
core/masking.py - Algoritmos de segmentación y refinamiento de máscaras cielo/suelo.
Implementa segmentación híbrida mediante GrabCut asistido por luminancia y refinado guiado.
"""
from typing import Optional
import cv2
import numpy as np


def refine_mask_guided(
    image_rgb: np.ndarray, 
    user_scribbles: np.ndarray,
    feather_radius: int = 7,
    iterations: int = 3
) -> np.ndarray:
    """
    Refina la máscara cielo/suelo preservando ramas finas y siluetas complejas mediante:
    1. Análisis fotométrico entre trazos de cielo y suelo para calcular el umbral de silueta.
    2. GrabCut iterativo asistido por máscara en resolución adaptativa.
    3. Recuperación de micro-estructuras oscuras (ramas/follaje) y suavizado perimetral guiado.

    Parámetros:
        image_rgb (np.ndarray): Imagen base normalizada float32 [0.0, 1.0].
        user_scribbles (np.ndarray): Matriz uint8 (0: no marcado, 1: suelo, 2: cielo).
        feather_radius (int): Radio para el desenfoque gaussiano del perímetro (en píxeles).
        iterations (int): Número de iteraciones del algoritmo GrabCut.

    Retorna:
        np.ndarray: Máscara continua float32 [0.0, 1.0] (1.0 = cielo, 0.0 = suelo).
    """
    h, w = user_scribbles.shape[:2]
    
    # Si falta trazo de cielo (2) o de suelo (1), retorna cielo completo por defecto
    if not (np.any(user_scribbles == 1) and np.any(user_scribbles == 2)):
        return np.ones((h, w), dtype=np.float32)

    # 1. Luminancia normalizada
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(np.clip(image_rgb, 0.0, 1.0), cv2.COLOR_RGB2GRAY)
    else:
        gray = np.clip(image_rgb, 0.0, 1.0)

    # 2. Análisis fotométrico de las semillas del usuario
    sky_samples = gray[user_scribbles == 2]
    gnd_samples = gray[user_scribbles == 1]

    sky_min = float(np.percentile(sky_samples, 5)) if len(sky_samples) > 0 else 0.2
    gnd_max = float(np.percentile(gnd_samples, 95)) if len(gnd_samples) > 0 else 0.1

    # Umbral de separación de silueta entre suelo y cielo
    split_thresh = (sky_min * 0.4) + (gnd_max * 0.6)

    # 3. Inicializar matriz de probabilidades de GrabCut
    gc_mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)

    # Fijar semillas seguras del usuario
    gc_mask[user_scribbles == 1] = cv2.GC_BGD  # Suelo seguro
    gc_mask[user_scribbles == 2] = cv2.GC_FGD  # Cielo seguro

    # Clasificar como probable fondo las estructuras oscuras que contrastan con el cielo
    dark_silhouette = (gray <= split_thresh) & (user_scribbles == 0)
    gc_mask[dark_silhouette] = cv2.GC_PR_BGD

    # 4. Redimensionado conservando detalle para optimizar el rendimiento de GrabCut
    max_dim = 1600.0
    scale = min(1.0, max_dim / max(h, w))
    
    if scale < 1.0:
        small_w = int(w * scale)
        small_h = int(h * scale)
        small_rgb_8u = (np.clip(cv2.resize(image_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA), 0.0, 1.0) * 255.0).astype(np.uint8)
        small_mask = cv2.resize(gc_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    else:
        small_rgb_8u = (np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        small_mask = gc_mask.copy()

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # GrabCut iterativo asistido
    cv2.grabCut(
        small_rgb_8u, 
        small_mask, 
        None, 
        bgd_model, 
        fgd_model, 
        iterCount=max(1, iterations), 
        mode=cv2.GC_INIT_WITH_MASK
    )

    # Probable o seguro cielo = 1.0; resto = 0.0
    refined_small = np.where(
        (small_mask == cv2.GC_FGD) | (small_mask == cv2.GC_PR_FGD), 1.0, 0.0
    ).astype(np.float32)

    # Reescalar la máscara base a la resolución nativa
    if scale < 1.0:
        mask_full = cv2.resize(refined_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        mask_full = refined_small

    # Forzar que los píxeles más oscuros que el umbral de silueta queden como suelo (ramas finas)
    tree_branches = (gray <= (split_thresh * 1.15)) & (user_scribbles != 2)
    mask_full[tree_branches] = 0.0

    # 5. Suavizado y refinado guiado de bordes
    k = max(1, feather_radius)
    if k % 2 == 0:
        k += 1

    if k > 1:
        feathered = cv2.GaussianBlur(mask_full, (k, k), sigmaX=k / 3.0)
        # Mantener las semillas manuales del usuario con peso estricto
        feathered[user_scribbles == 2] = 1.0
        feathered[user_scribbles == 1] = 0.0
        mask_final = feathered
    else:
        mask_final = mask_full

    return np.clip(mask_final, 0.0, 1.0).astype(np.float32)