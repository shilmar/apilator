# core/masking.py
import cv2
import numpy as np

def refine_mask_guided(
    image_rgb: np.ndarray, 
    user_scribbles: np.ndarray,
    feather_radius: int = 7,
    iterations: int = 3
) -> np.ndarray:
    """
    Refina la máscara cielo/suelo preservando ramitas y siluetas complejas mediante:
    1. Estimación fotométrica de umbral entre cielo y vegetación.
    2. GrabCut asistido por luminancia (sin restricciones arbitrarias de altura).
    3. Refinado de bordes guiado en alta resolución.
    """
    h, w = user_scribbles.shape[:2]
    
    # Si falta trazo de cielo (2) o de suelo (1), retorna cielo completo
    if not (np.any(user_scribbles == 1) and np.any(user_scribbles == 2)):
        return np.ones((h, w), dtype=np.float32)

    # Convertir a escala de grises / luminancia
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(np.clip(image_rgb, 0.0, 1.0), cv2.COLOR_RGB2GRAY)
    else:
        gray = np.clip(image_rgb, 0.0, 1.0)

    # 1. Análisis fotométrico de las semillas del usuario
    sky_samples = gray[user_scribbles == 2]
    gnd_samples = gray[user_scribbles == 1]

    sky_min = np.percentile(sky_samples, 5) if len(sky_samples) > 0 else 0.2
    gnd_max = np.percentile(gnd_samples, 95) if len(gnd_samples) > 0 else 0.1

    # Umbral de separación de silueta entre suelo y cielo
    split_thresh = (sky_min * 0.4) + (gnd_max * 0.6)

    # 2. Inicializar matriz de GrabCut
    gc_mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)

    # Si el usuario dibujó una franja verde, todo lo que quede por encima del píxel verde más bajo
    # en cada columna (o un margen de seguridad) se declara cielo seguro sin coste computacional
    sky_ys = np.where(user_scribbles == 2)[0]
    #if len(sky_ys) > 0:
    #    min_sky_y = np.min(sky_ys)
    #    # Todo lo que esté claramente por encima de donde empieza el cielo se marca seguro
    #    gc_mask[:max(0, min_sky_y - 10), :] = cv2.GC_FGD
    # test

    # Marcar semillas del usuario
    gc_mask[user_scribbles == 1] = cv2.GC_BGD  # Suelo seguro
    gc_mask[user_scribbles == 2] = cv2.GC_FGD  # Cielo seguro

    # Clasificar automáticamente estructuras oscuras (ramas, troncos) que contrastan con el fondo
    dark_silhouette = (gray <= split_thresh) & (user_scribbles == 0)
    gc_mask[dark_silhouette] = cv2.GC_PR_BGD

    # Redimensionado conservando detalle para GrabCut
    max_dim = 1600.0
    scale = min(1.0, max_dim / max(h, w))
    small_w = int(w * scale)
    small_h = int(h * scale)

    small_rgb_8u = (cv2.resize(image_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA) * 255.0).astype(np.uint8)
    small_mask = cv2.resize(gc_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # GrabCut iterativo asistido
    cv2.grabCut(
        small_rgb_8u, small_mask, None, 
        bgd_model, fgd_model, 
        iterCount=max(1, iterations), 
        mode=cv2.GC_INIT_WITH_MASK
    )

    # Probable o seguro cielo = 1.0; resto = 0.0
    refined_small = np.where(
        (small_mask == cv2.GC_FGD) | (small_mask == cv2.GC_PR_FGD), 1.0, 0.0
    ).astype(np.float32)

    # Reescalar la máscara base a la resolución nativa
    mask_full = cv2.resize(refined_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Forzar que los píxeles más oscuros que el umbral de silueta queden como suelo (ramas finas)
    tree_branches = (gray <= (split_thresh * 1.15)) & (user_scribbles != 2)
    mask_full[tree_branches] = 0.0

    # 3. Refinado guiado de bordes para evitar artefactos duros
    k = max(1, feather_radius)
    if k % 2 == 0:
        k += 1

    if k > 1:
        # Suavizado suave en el límite sin difuminar la estructura interior
        feathered = cv2.GaussianBlur(mask_full, (k, k), sigmaX=k / 3.0)
        # Mantener los trazos manuales y el interior del follaje intactos
        feathered[user_scribbles == 2] = 1.0
        feathered[user_scribbles == 1] = 0.0
        mask_final = feathered
    else:
        mask_final = mask_full

    # Erosionar ligeramente el cielo (1 a 3 px) para que el suelo se coma cualquier borde dudoso
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #mask_final = cv2.erode(mask_final, kernel, iterations=1)

    return np.clip(mask_final, 0.0, 1.0).astype(np.float32)