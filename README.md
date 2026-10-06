# Apilator

**Apilator** es una herramienta especializada de posprocesado y apilado para **astrofotografía de paisaje (nightscapes)**. Permite desacoplar el movimiento del cielo respecto al horizonte terrestre, aplicar integración estadística Kappa-Sigma en coma flotante de 32 bits y editar la imagen final en tiempo real con asistencia de redes neuronales (GraXpert y StarNet++ v2).

---

## Características Principales

### 1. Calibración Astronómica Rigurosa
- **Pipeline Completo de Calibración:** Pestañas dedicadas y soporte nativo para **Lights**, **Darks**, **Flats** y **Bias**.
- **Sustracción de Pedestal Electrónico y Térmico:** Eliminación del ruido térmico y píxeles calientes mediante Master Dark, y del ruido de lectura/offset mediante Master Bias.
- **Aplanado de Campo y Viñeteo (Master Flat):** Corrección lineal de sombras de motas de polvo y caída de luz periférica del tren óptico, con normalización independiente por canal para preservar el balance de color.
- **Persistencia de Proyecto (.mwstack):** Guardado y restauración completa del estado de la sesión (listas de archivos de calibración, parámetros de integración y máscaras calculadas).

### 2. Apilado Diferencial Cielo / Suelo
- **Alineación Estelar Robusta:** Detección de estrellas y emparejamiento homográfico/afín de precisión subpíxel sobre datos lineales de 32 bits.
- **Integración Kappa-Sigma Pura:** Eliminación drástica y limpia de trazas de satélites, estelas de aviones y artefactos transitorios mediante rechazo por desviación absoluta respecto a la mediana (MAD) sin elevar el ruido de fondo.
- **Conservación Íntegra del Sensor (Sin Auto-Crop):** Relleno periférico reflectivo (`BORDER_REFLECT`) que neutraliza los marcos y escalones oscuros provocados por la rotación del campo estelar, conservando el 100% de la resolución nativa original.
- **Motor Optimizado por Bloques:** Procesamiento segmentado en franjas (200 filas) que previene desbordamientos de memoria RAM (`MemoryError`) y cuellos de botella de transferencia GPU en lotes pesados.
- **Modos de Suelo Flexibles:** Integración dual completa (cielo alineado + suelo estático), uso del suelo de la toma de referencia o composición directa desde una exposición dedicada en la pestaña Suelo.
- **Antipolución Lumínica:** Algoritmos selectivos de atenuación de gradientes de fondo (Sustracción de Domo, Rechazo Asimétrico Min-Sigma y Normalización Local).

### 3. Segmentación y Máscaras Interactivas
- **GrabCut Guiado Multiescala:** Delineación asistida del horizonte y perfiles complejos (árboles, relieve montañoso, estructuras).
- **Edición Continua no Destructiva:** Permite refinar la máscara, seguir corrigiendo con trazos de pincel en tiempo real y recalcular sin reiniciar el trabajo.
- **Control Paramétrico Fino:** Ajuste del radio de difusión/suavizado gaussiano del borde (feathering) e iteraciones del modelo.
- **Zoom 1:1 Nativo con Clic Derecho:** Inspección píxel a píxel del sensor original para retoque milimétrico de bordes.

### 4. Revelador Astrofotográfico Avanzado
- **Canal Lineal Puro:** Pipeline optimizado en coma flotante (float32), preservando la linealidad fotométrica hasta la compresión tonal final.
- **Separación de Estrellas con StarNet++:** Extracción de capas Starless y Stars-only en espacio lineal, permitiendo trabajar nebulosas y polvo galáctico sin hinchar las estrellas.
- **Estructura Multiescala por Ondículas (À Trous / B-Spline):** Realce selectivo de filamentos de gas y bandas de absorción de la Vía Láctea, junto con atenuación de la capa residual de fondo.
- **Extracción de Gradientes:** Motor dual integrado con GraXpert AI y ajuste polinómico cuadrático para corregir gradientes luminosos.
- **Reducción de Ruido Adaptativa:** Múltiples métodos para el fondo (Filtro Bilateral, Filtro Guiado y Non-Local Means) respetando los límites de las estrellas.
- **Balance de Color Fino y Tonalidad:** Curvas interactivas con histograma integrado en tiempo real, balance de temperatura/tinte calibrado ($\pm0.250$), control de vibranza y saturación diferencial cielo/suelo.

### 5. Configuración y Rendimiento
- **Aceleración por GPU Dual:** Soporte automático para NVIDIA CUDA mediante CuPy y conmutación transparente a CPU multinúcleo en equipos sin GPU dedicada.
- **Estrategias de Memoria Configurables:** Modos automático, memoria RAM intermedia de alta velocidad o volcado temporal a disco SSD para equipos con recursos limitados.
- **Detección Automática de Binarios:** Localización y validación de ejecutables externos de StarNet++ CLI y GraXpert.

---

## Estructura del Proyecto

```text
apilator/
├── config.json              # Configuración persistente del usuario
├── run_app.py               # Punto de entrada de la aplicación
├── core/
│   ├── config_manager.py    # Carga y almacenamiento de ajustes JSON
│   ├── gpu_backend.py       # Detección y gestión de aceleración NVIDIA CUDA / CuPy
│   ├── graxpert_bridge.py   # Conector CLI con GraXpert AI
│   ├── masking.py           # Algoritmos de segmentación y refinado guiado de máscaras
│   ├── project_manager.py   # Serialización y persistencia de proyectos (.mwstack)
│   ├── starnet_bridge.py    # Conector y parser CLI con StarNet++ v2
│   ├── stacking.py          # Motor de calibración (Dark/Flat/Bias), alineación y apilado
│   └── stretch.py           # Algoritmos MTF, ondículas À Trous, balance y tono
└── gui/
    ├── canvas.py            # Visor interactivo QGraphicsView acelerado con zoom 1:1
    ├── main_window.py       # Ventana principal y gestión de pestañas maestras
    ├── tab_developer.py     # Pestaña de revelado y composición de capas
    ├── tab_settings.py      # Pestaña de configuración de rutas y parámetros
    ├── tab_stacker.py       # Pestaña de apilado dual, calibración y máscaras
    └── worker.py            # Orquestador de tareas en segundo plano multihilo (QThread)
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
  - `imagecodecs` (códecs extendidos de compresión de imagen)
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
pip install numpy scipy opencv-python PySide6 tifffile rawpy matplotlib astropy imagecodecs

### 3. Aceleracion GPU (Opcional - NVIDIA CUDA)
Si dispones de una tarjeta grafica NVIDIA, puedes habilitar el procesamiento acelerado instalando la version de CuPy adecuada a tus controladores CUDA:
* CUDA 12.x: pip install cupy-cuda12x
* CUDA 11.x: pip install cupy-cuda11x

(Si CuPy no esta presente, el motor utiliza automaticamente todos los nucleos logicos de la CPU).

---

## Uso de la Aplicacion

Inicia la herramienta ejecutando:
python run_app.py

## Flujo de Trabajo Recomendado
### 1. Flujo de Calibración y Apilado
* Abre la pestaña 1. Apilador (Stacker).
* Carga tus tomas de luz en Lights y haz doble clic sobre la toma que servirá como base de encuadre.
* (Opcional) Carga tomas en Darks, Flats y Bias para corrección de ruido térmico, viñeteo óptico y offset.
* Dibuja los trazos guía sobre la vista previa: Verde para el cielo y Rojo para el suelo.
* Haz clic en Refinar Automática para generar la máscara. Inspecciona los bordes manteniendo pulsado el botón derecho del ratón para ver la imagen al 100% de resolución nativa.
* Selecciona el Modo de Captura (Trípode Fijo o Star Tracker), el tratamiento del suelo deseado y el Factor Kappa.
* Pulsa INICIAR APILADO DUAL.

### 2. Flujo de Revelado
* Al concluir el apilado, la imagen lineal de 32 bits y su máscara calculada se cargarán automáticamente en la pestaña 2. Revelador / Editor.
* Neutraliza gradientes residuales con el motor Polinómico o GraXpert AI.
* Ejecuta StarNet++ para separar el fondo galáctico de las estrellas.
* Aplica realce de gas y polvo molecular mediante los controles de Estructura Multiescala (Ondículas) sin deformar el perfil estelar.
* Ajusta curvas, temperatura de color, tinte y saturación diferencial cielo/suelo.
* Exporta el resultado final en formato TIFF 16-bit, TIFF 32-bit float o JPEG.

## Hoja de Ruta
[x] Apilado diferencial cielo/suelo con alineación estelar por homografía.
[x] Rechazo estadístico Kappa-Sigma (MAD) libre de trazas de satélites y aviones.
[x] Pipeline completo de calibración con Darks, Flats y Bias.
[x] Conservación íntegra de resolución nativa mediante BORDER_REFLECT.
[x] Gestión de proyectos y sesiones en disco (.mwstack).
[x] Segmentación asistida GrabCut y zoom nativo 1:1.
[x] Descomposición y realce multiescala mediante ondículas À Trous.
[x] Integración de StarNet++ v2 y GraXpert AI.
[ ] Procesado por lotes para secuencias de timelapse.
[ ] Rutina específica de apilado y alineación para eclipses solares.
[ ] Módulo de composición panorámica para mosaicos nocturnos.
[ ] Exportación de perfiles de color ICC embebidos (sRGB / AdobeRGB / ProPhoto).
---

## Licencia

Este proyecto esta bajo la Licencia MIT.
