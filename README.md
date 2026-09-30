# Apilator

**Apilator** es una herramienta especializada de posprocesado y apilado para **astrofotografía de paisaje (nightscapes)**. Permite desacoplar el movimiento del cielo respecto al horizonte terrestre, aplicar integración estadística Kappa-Sigma en coma flotante de 32 bits y editar la imagen final en tiempo real con asistencia de redes neuronales (GraXpert y StarNet++ v2).

---

## Novedades destacadas de la Versión 0.4

### 1. Reducción de Contaminación Lumínica en el Apilado
- **Sustracción de Domo (Sequator-Style):** Modelado analítico y sustracción del pedestal de baja frecuencia fija al encuadre antes de la alineación y el apilado. Neutraliza cúpulas lumínicas urbanas preservando el contraste galáctico sin artefactos en el horizonte.
- **Rechazo Asimétrico (Min-Sigma):** Fusión adaptativa en la pila del cielo orientada a percentiles inferiores para descartar la dispersión lumínica fija durante ráfagas prolongadas.
- **Normalización Local Fotométrica:** Compensación de variaciones de pedestal y transparencia atmosférica entre tomas individuales.
- **Selector y Deslizador de Fuerza (0–100%):** Integración completa en la UI del apilador para conmutar dinámicamente entre algoritmos de apilado.

### 2. Revelado Starless de Alta Dinámica
- **Claridad y Borrar Neblina (Dehaze) Ampliados:** Rango extendido a $\pm 2.00$ con curvas de respuesta progresiva para extraer estructuras de polvo sin colapsar tonos medios.
- **Editor de Curvas con Histograma en Tiempo Real:** Control tonal fino con visualización interactiva del histograma sobre la capa procesada.
- **Pipeline de Capas Independiente:** Procesamiento no destructivo del fondo sin estrellas (StarNet++) con recombinación configurable de estrellas.
- **Extracción de Gradientes Híbrida:** Modelo Polinómico Cuadrático nativo optimizado para suelo y cielo terrestre junto a integración con GraXpert AI.

---

## Estructura del Pipeline

1. **Calibración y Alineación:**
   - Detección de estrellas mediante ORB + refinamiento subpíxel de centroides.
   - Alineación robusta por homografía RANSAC.
   - Calibración por Master Dark y máscara guiada cielo/suelo.

2. **Apilado Híbrido (CPU Multi-core / GPU CUDA):**
   - Apilado independiente de cielo (estrellas alineadas) y suelo estático (sin distorsión).
   - Rejection por Kappa-Sigma / MAD y opciones antipolución integradas.

3. **Revelado Especializado:**
   - Balance de blancos de precisión ($\pm 0.500$).
   - Saturación diferencial y vibrance.
   - Estirado MTF automático y manual.
   - Separación estelar StarNet++ en espacio lineal de 32 bits.
   - Exportación multiformato (TIFF 16-bit, TIFF 32-bit Float, JPEG 8-bit).
   
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
