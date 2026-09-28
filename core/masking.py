# core/masking.py
import cv2
import numpy as np

def refine_mask_guided(image_rgb: np.ndarray, user_scribbles: np.ndarray) -> np.ndarray:
    h, w = user_scribbles.shape[:2]
    
    if not (np.any(user_scribbles == 1) and np.any(user_scribbles == 2)):
        return np.ones((h, w), dtype=np.float32)

    gc_mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
    gc_mask[user_scribbles == 1] = cv2.GC_BGD
    gc_mask[user_scribbles == 2] = cv2.GC_FGD

    ground_y_indices = np.where(user_scribbles == 1)[0]
    highest_ground_y = np.min(ground_y_indices) if len(ground_y_indices) > 0 else h // 2
    
    safe_sky_zone = max(0, highest_ground_y - int(h * 0.05))
    gc_mask[:safe_sky_zone, :] = cv2.GC_FGD

    scale = min(1.0, 1200.0 / max(h, w))
    small_w = int(w * scale)
    small_h = int(h * scale)

    small_rgb = cv2.resize(image_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_rgb_8u = (np.clip(small_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    small_mask = cv2.resize(gc_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(small_rgb_8u, small_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)

    refined_small = np.where(
        (small_mask == cv2.GC_FGD) | (small_mask == cv2.GC_PR_FGD), 1.0, 0.0
    ).astype(np.float32)

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    refined_small = cv2.morphologyEx(refined_small, cv2.MORPH_CLOSE, kernel_clean)

    refined = cv2.resize(refined_small, (w, h), interpolation=cv2.INTER_LINEAR)
    feathered = cv2.GaussianBlur(refined, (9, 9), 2.0)
    
    return np.clip(feathered, 0.0, 1.0)