# core/gpu_backend.py
"""
core/gpu_backend.py - Detección e inicialización de aceleración GPU (CuPy/CUDA).
"""
import numpy as np

_CUPY_AVAILABLE = False
_GPU_DEVICE_NAME = ""
_cp = None

try:
    import cupy as _cp
    num_devices = _cp.cuda.runtime.getDeviceCount()
    if num_devices > 0:
        props = _cp.cuda.runtime.getDeviceProperties(0)
        gpu_raw_name = props.get("name", b"GPU NVIDIA")
        if isinstance(gpu_raw_name, bytes):
            _GPU_DEVICE_NAME = gpu_raw_name.decode("utf-8", errors="ignore")
        else:
            _GPU_DEVICE_NAME = str(gpu_raw_name)

        _CUPY_AVAILABLE = True
except Exception as e:
    _CUPY_AVAILABLE = False
    _GPU_DEVICE_NAME = ""
    _cp = None

_USE_GPU = _CUPY_AVAILABLE


def is_cupy_installed() -> bool:
    """Indica si CuPy está instalado y detecta hardware CUDA funcional."""
    return _CUPY_AVAILABLE


def get_gpu_name() -> str:
    """Retorna el nombre comercial del dispositivo CUDA detectado."""
    return _GPU_DEVICE_NAME


def set_gpu_enabled(enabled: bool) -> None:
    """Habilita o deshabilita el uso de la GPU (si está disponible en hardware)."""
    global _USE_GPU
    _USE_GPU = bool(enabled and _CUPY_AVAILABLE)


def is_gpu_enabled() -> bool:
    """Comprueba si el pipeline debe utilizar aceleración GPU."""
    return _USE_GPU


def get_array_module(arr=None):
    """
    Retorna el módulo matemático correspondiente (cupy si la GPU está activa,
    o numpy en caso contrario). Si arr se suministra, puede inspeccionarse su tipo.
    """
    if _USE_GPU and _cp is not None:
        if arr is not None:
            return _cp.get_array_module(arr)
        return _cp
    return np