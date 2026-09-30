# Apilator

**Apilator** es una herramienta especializada de posprocesado y apilado para **astrofotografía de paisaje (nightscapes)**. Permite desacoplar el movimiento del cielo respecto al horizonte terrestre, aplicar integración estadística Kappa-Sigma en coma flotante de 32 bits y editar la imagen final en tiempo real con asistencia de redes neuronales (GraXpert y StarNet++ v2).

---

## Novedades de la Versión 0.3.2

- **Alineación Robusta de Referencia Fija:** Registro estelar directo contra el fotograma base mediante homografía subpíxel RANSAC, eliminando la acumulación de deriva en el cielo.
- **Protección Térmica del Suelo:** Desacoplo de la sustracción de *Master Dark* en la máscara terrestre para evitar el empastado de sombras, junto con un suavizado gaussiano de frontera para transiciones naturales en el horizonte.
- **Editor de Curvas con Histograma Logarítmico:** Sustitución de controles fijos por un widget interactivo de curvas tonales con evaluación acelerada por LUT (Look-Up Table) y visualización en tiempo real del fondo de cielo.
- **Compatibilidad con Archivos RAW:** Soporte directo para formatos RAW (`.NEF`, `.CR2`, `.CR3`, `.ARW`, `.DNG`) tanto en el apilador como en el panel de revelado.
- **Limpieza de Arquitectura:** Eliminación de controles deprecados y desacoplo de eventos de refresco en la interfaz.

---

## Características Principales

### 1. Motor de Apilado Dual (Stacker)
- **Modo Trípode Fijo:** Separación de componentes de cielo y suelo mediante máscaras manuales o refinadas.
- **Alineación Subpíxel:** Detección de estrellas con ORB y refinamiento por `cornerSubPix`.
- **Integración Estadística:** Algoritmo Sigma-Clipping adaptativo basado en MAD (*Median Absolute Deviation*) con aceleración por GPU (CUDA / CuPy) o procesamiento multicore en CPU.
- **Gestión de Calibración:** Generación y aplicación de *Master Dark* para supresión de ruido térmico.

### 2. Panel de Revelado (Developer)
- **Control de Estrellas con StarNet++ AI:** Separación en capas independientes de fondo (*starless*) y estrellas en espacio lineal.
- **Reducción de Gradientes con GraXpert AI:** Extracción y neutralización del fondo atmosférico y contaminación lumínica.
- **Reducción de Ruido Multimodelo:** Filtros Bilateral, Guided Filter y Non-Local Means específicos para la capa de cielo.
- **Flujo de Revelado Perceptual:** Balance de blancos fino, saturación diferencial (cielo/suelo), intensidad (*vibrance*), estirado MTF analítico y contraste sigmoidal con preservación del fondo.
- **Visualizador Interactivo:** Arquitectura basada en buffers proxy para previsualización fluida a 60 FPS.

### 3. Configuración y Entorno (Settings)
- ** Detección automática y manual de ejecutables CLI externos.
- ** Ajustes de rendimiento de memoria, hilos y tamaño del proxy visual.

---



---

## Estructura del Proyecto
```text
apilator/
├── config.json              # Configuración persistente del usuario
├── run_app.py               # Punto de entrada de la aplicación
├── core/
│   ├── config_manager.py    # Carga y almacenamiento de ajustes JSON
│   ├── graxpert_bridge.py   # Conector CLI con GraXpert AI
│   ├── starnet_bridge.py    # Conector y parser CLI con StarNet++ v2
│   ├── stacking.py          # Motor de registro, RANSAC y apilado Kappa-Sigma
│   └── stretch.py           # Algoritmos MTF, Balance de Blancos y Saturación
└── gui/
    ├── canvas.py            # Visor interactivo QGraphicsView acelerado
    ├── main_window.py       # Ventana principal y gestión de pestañas
    ├── tab_developer.py     # Pestaña de revelado y composición de capas
    ├── tab_settings.py      # Pestaña de configuración de rutas y parámetros
    ├── tab_stacker.py       # Pestaña de apilado dual cielo/suelo
    └── worker.py            # Hilos de ejecución en segundo plano (QThread)
```

## Instalacion y Requisitos
## Requisitos del Sistema

- **Python:** 3.10 o superior.
- **Dependencias Principales:**
  - `PySide6` (interfaz gráfica basada en Qt)
  - `numpy`, `scipy` (cálculo numérico y splines)
  - `opencv-python` (visión por computador y transformaciones geométricas)
  - `rawpy` (decodificación de archivos RAW de cámara)
  - `tifffile` (lectura y escritura de imágenes TIFF de alta profundidad)
  - `astropy` (gestión de archivos FITS astronómicos)
  - *(Opcional)* `cupy` (aceleración por GPU NVIDIA CUDA)
- **Binarios Externos:**
  - Ejecutables de StarNet++ CLI y GraXpert configurados en el entorno.

---


### 1. Clonar el repositorio
```cmd
git clone https://github.com/shilmar/apilator.git
cd apilator
```

### 2. Instalar dependencias
pip install -r requirements.txt

O instalando manualmente los paquetes requeridos:
pip install PySide6 numpy opencv-python rawpy tifffile astropy graxpert

### 3. Aceleracion GPU (Opcional - NVIDIA CUDA)
Si dispones de una tarjeta grafica NVIDIA, puedes habilitar el procesamiento acelerado instalando la version de CuPy adecuada a tus controladores CUDA:
* CUDA 12.x: pip install cupy-cuda12x
* CUDA 11.x: pip install cupy-cuda11x

(Si CuPy no esta presente, el motor utiliza automaticamente todos los nucleos logicos de la CPU).

---

## Uso de la Aplicacion

Inicia la herramienta ejecutando:
python run_app.py

Flujo de Trabajo Recomendado

    Pestaña 1 - Apilador:
        Carga las tomas de luz (Lights) y las tomas oscuras (Darks).
        Dibuja la máscara para delimitar el cielo y el suelo si estás trabajando en modo trípode fijo.
        Ejecuta Iniciar Apilado Dual. El resultado se enviará automáticamente al revelador.

    Pestaña 2 - Revelador:
        Ajusta el estirado inicial mediante el Auto-Estirado MTF.
        Ejecuta GraXpert AI para eliminar viñeteo o gradientes lumínicos.
        Ejecuta StarNet++ AI si deseas modular o reducir el tamaño de las estrellas de forma independiente al fondo.
        Modela los contrastes y las estructuras de nebulosidad usando el Editor de Curvas.
        Exporta el resultado final en TIFF (16 o 32 bits) o JPEG.
---

## Licencia

Este proyecto esta bajo la Licencia MIT.
