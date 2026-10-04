# core/config_manager.py
"""
core/config_manager.py - Gestor de configuración persistente en formato JSON.
"""
from typing import Any, Dict
import json
import os

CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.json"))

DEFAULT_CONFIG: Dict[str, Any] = {
    "starnet_exe": "",
    "graxpert_smoothing": 0.5,
    "default_kappa": 2.2,
    "preview_max_dim": 1600,
    "export_format_default": "TIFF 16-bit (*.tif *.tiff)",
    "cpu_workers": max(1, (os.cpu_count() or 4) - 1)
}


def load_config() -> Dict[str, Any]:
    """Carga la configuración desde config.json o crea una por defecto si no existe."""
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


def save_config(cfg: Dict[str, Any]) -> None:
    """Guarda el diccionario de configuración en config.json."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] No se pudo guardar la configuración: {e}")


def get_config_val(key: str, default: Any = None) -> Any:
    """Obtiene el valor de una clave de configuración específica."""
    cfg = load_config()
    return cfg.get(key, default)


def set_config_val(key: str, val: Any) -> None:
    """Actualiza una clave concreta y guarda los cambios en disco."""
    cfg = load_config()
    cfg[key] = val
    save_config(cfg)