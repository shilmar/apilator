# core/gpu_backend.py
import numpy as np

_CUPY_AVAILABLE = False
_GPU_DEVICE_NAME = ""

try:
    import cupy as cp
    num_devices = cp.cuda.runtime.getDeviceCount()
    if num_devices > 0:
        # Obtener el nombre de la GPU mediante el runtime de CUDA
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_raw_name = props.get("name", b"GPU NVIDIA")
        if isinstance(gpu_raw_name, bytes):
            _GPU_DEVICE_NAME = gpu_raw_name.decode("utf-8", errors="ignore")
        else:
            _GPU_DEVICE_NAME = str(gpu_raw_name)

        _CUPY_AVAILABLE = True
except Exception as e:
    print(f"[gpu_backend] Falló la inicialización de CuPy/CUDA: {e}")
    _CUPY_AVAILABLE = False
    _GPU_DEVICE_NAME = ""

_USE_GPU = _CUPY_AVAILABLE

def is_cupy_installed() -> bool:
    return _CUPY_AVAILABLE

def get_gpu_name() -> str:
    return _GPU_DEVICE_NAME

def set_gpu_enabled(enabled: bool):
    global _USE_GPU
    _USE_GPU = bool(enabled and _CUPY_AVAILABLE)

def is_gpu_enabled() -> bool:
    return _USE_GPU

def get_array_module(arr=None):
    """Retorna cupy si la aceleración GPU está activa, o numpy en caso contrario."""
    if _USE_GPU:
        import cupy as cp
        return cp
    return np