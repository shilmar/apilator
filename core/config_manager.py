# core/config_manager.py
import os
import json

CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.json"))

DEFAULT_CONFIG = {
    "starnet_exe": "",
    "graxpert_smoothing": 0.5,
    "default_kappa": 2.2,
    "preview_max_dim": 1600,
    "export_format_default": "TIFF 16-bit (*.tif *.tiff)"
}

def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(data)
        return cfg
    except Exception:
        return DEFAULT_CONFIG.copy()

def save_config(cfg: dict):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] No se pudo guardar la configuración: {e}")

def get_config_val(key: str, default=None):
    cfg = load_config()
    return cfg.get(key, default)

def set_config_val(key: str, val):
    cfg = load_config()
    cfg[key] = val
    save_config(cfg)