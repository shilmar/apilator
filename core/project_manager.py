# core/project_manager.py
import json
import os
import cv2
import numpy as np

class ProjectManager:
    def __init__(self):
        self.reset()

    def reset(self):
        self.project_path = None
        self.data = {
            "version": "1.1",
            "mode": "fixed_tripod",
            "mask_path": "",
            "light_frames": [],
            "dark_frames": [],
            "parameters": {
                "kappa": 2.2,
                "black_point": 0.0,
                "stretch_factor": 5.0
            }
        }

    def save(self, filepath: str, mask_array: np.ndarray = None) -> bool:
        self.project_path = os.path.abspath(filepath)
        project_dir = os.path.dirname(self.project_path)
        base_name = os.path.splitext(os.path.basename(self.project_path))[0]

        if mask_array is not None:
            mask_filename = f"{base_name}_mask.png"
            mask_full_path = os.path.join(project_dir, mask_filename)
            u16_mask = (np.clip(mask_array, 0.0, 1.0) * 65535.0).astype(np.uint16)
            cv2.imwrite(mask_full_path, u16_mask)
            self.data["mask_path"] = mask_full_path
        else:
            self.data["mask_path"] = ""

        with open(self.project_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)
        return True

    def load(self, filepath: str):
        self.project_path = os.path.abspath(filepath)
        with open(self.project_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        mask_data = None
        mask_path = self.data.get("mask_path", "")
        if mask_path and os.path.exists(mask_path):
            raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if raw is not None:
                if raw.ndim == 3:
                    raw = raw[..., 0]
                max_v = 65535.0 if raw.dtype == np.uint16 else 255.0
                mask_data = np.clip(raw.astype(np.float32) / max_v, 0.0, 1.0)

        return self.data, mask_data