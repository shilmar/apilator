# Apilator

**Apilator** es una herramienta especializada de posprocesado, apilado y revelado para **astrofotografía de paisaje (nightscapes)** y **eclipses (solares y lunares)**. Permite desacoplar el movimiento del cielo respecto al horizonte terrestre, aplicar integración estadística Kappa-Sigma en coma flotante de 32 bits, procesar series de bracketing de eclipses con alineación subpíxel de limbo y filtro NRGF continuo, y editar la imagen final en tiempo real con asistencia de redes neuronales (GraXpert y StarNet++ v2).

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

### 5. Módulo Especializado de Eclipses (Solar / Lunar)
- **Ingesta Automática de Bracketing y Metadatos:** Carga de series de exposición completa (RAW, TIFF, FITS) con extracción de tiempos de obturación reales, ISO y apertura, ordenando automáticamente las tomas.
- **Detección Subpíxel de Limbo Inmune a Protuberancias:** Algoritmo por flujo de gradiente radial perpendicular (`optimize_center_circular_flux`) basado en la mediana de 360 rayos angulares, insensible a protuberancias cromosféricas, fulguraciones o cuentas de Baily, ubicando el centro real con precisión < 0.1 píxeles.
- **Alineación de Lotes con Radio Físico Anclado:** Detección en lote que fija el radio astronómico real de la toma de referencia para toda la serie y ajusta los desplazamientos afines con interpolación Lanczos4.
- **Fusión Fotométrica HDR Lineal (32-bit):** Ponderación por tiempos reales de exposición con corte suave de saturación y estirado no lineal Asinh para revelar desde la cromosfera y protuberancias hasta la corona externa tenue.
- **Filtro NRGF (Normalized Radial Gradient Filter) Continuo:**
  - Interpolación continua subpíxel (`np.interp`) sin discretización de radios, eliminando por completo cualquier artefacto de bandas concéntricas.
  - Modulación dinámica desinhibida para estirar y contrastar filamentos coronales ("hilos") hasta 3.5x–4.0x radios solares.
  - Compuerta radial suave (*Radial Gate*) que transiciona gradualmente al cielo de fondo profundo para mantenerlo negro aterciopelado sin ruido.
  - Filtro bilateral tangencial de alta frecuencia para realce selectivo de líneas de campo magnético solar.
- **Exportación Dual 16-bit / 32-bit:** Exportación directa a TIFF de 16 bits optimizada (`uint16`, zlib, `photometric='rgb'`) compatible de forma nativa con Photoshop y Lightroom sin mapeos forzados de tono, y TIFF de 32 bits flotante para archivo maestro.

### 6. Configuración, Rendimiento y Vitrina de Resultados
- **Interfaz Moderna Unificada:** Paneles laterales oscuros con desplazamiento independiente mediante `QScrollArea`, consolas de registro (`txt_log`) fijas en la base en todos los módulos y diseño responsivo optimizado para pantallas compactas y monitores de alta resolución.
- **Vitrina de Resultados Integrada (Showcase):** Panel visual en la pestaña de Configuración con visor responsivo antialias que exhibe astrofotografía real de paisaje procesada de principio a fin con Apilator, acreditación de autoría (*Fotografía y Procesado: Shilmar*) e inspección a resolución nativa.
- **Aceleración por GPU Dual:** Soporte automático para NVIDIA CUDA mediante CuPy y conmutación transparente a CPU multinúcleo en equipos sin GPU dedicada.
- **Estrategias de Memoria Configurables:** Modos automático, memoria RAM intermedia de alta velocidad o volcado temporal a disco SSD para equipos con recursos limitados.
- **Detección Automática de Binarios:** Localización y validación dinámica de ejecutables externos de StarNet++ CLI y GraXpert con indicadores en tiempo real de disponibilidad.

---

## Estructura del Proyecto

```text
apilator/
├── config.json              # Configuración persistente del usuario
├── run_app.py               # Punto de entrada de la aplicación
├── core/
│   ├── config_manager.py    # Carga y almacenamiento de ajustes JSON
│   ├── eclipse.py           # Detección de limbo, alineación subpíxel, fusión HDR y NRGF continuo
│   ├── gpu_backend.py       # Detección y gestión de aceleración NVIDIA CUDA / CuPy
│   ├── graxpert_bridge.py   # Conector CLI con GraXpert AI
│   ├── masking.py           # Algoritmos de segmentación y refinado guiado de máscaras
│   ├── project_manager.py   # Serialización y persistencia de proyectos (.mwstack)
│   ├── starnet_bridge.py    # Conector y parser CLI con StarNet++ v2
│   ├── stacking.py          # Motor de calibración (Dark/Flat/Bias), alineación y apilado
│   └── stretch.py           # Algoritmos MTF, ondículas À Trous, balance y tono
└── gui/
    ├── assets/              # Recursos gráficos y fotografía de demostración
    │   └── showcase_milkyway.jpg
    ├── canvas.py            # Visor interactivo QGraphicsView acelerado con zoom 1:1
    ├── main_window.py       # Ventana principal y gestión de pestañas maestras
    ├── tab_developer.py     # Pestaña de revelado y composición de capas
    ├── tab_eclipse.py       # Pestaña de procesado integral de eclipses solares y lunares
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
* Abre la pestaña **1. Apilador (Stacker)**.
* Carga tus tomas de luz en Lights y haz doble clic sobre la toma que servirá como base de encuadre.
* (Opcional) Carga tomas en Darks, Flats y Bias para corrección de ruido térmico, viñeteo óptico y offset.
* Dibuja los trazos guía sobre la vista previa: Verde para el cielo y Rojo para el suelo.
* Haz clic en Refinar Automática para generar la máscara. Inspecciona los bordes manteniendo pulsado el botón derecho del ratón para ver la imagen al 100% de resolución nativa.
* Selecciona el Modo de Captura (Trípode Fijo o Star Tracker), el tratamiento del suelo deseado y el Factor Kappa.
* Pulsa INICIAR APILADO DUAL.

### 2. Flujo de Revelado
* Al concluir el apilado, la imagen lineal de 32 bits y su máscara calculada se cargarán automáticamente en la pestaña **2. Revelador / Editor**.
* Neutraliza gradientes residuales con el motor Polinómico o GraXpert AI.
* Ejecuta StarNet++ para separar el fondo galáctico de las estrellas.
* Aplica realce de gas y polvo molecular mediante los controles de Estructura Multiescala (Ondículas) sin deformar el perfil estelar.
* Ajusta curvas, temperatura de color, tinte y saturación diferencial cielo/suelo.
* Exporta el resultado final en formato TIFF 16-bit, TIFF 32-bit float o JPEG.

### 3. Flujo de Procesado de Eclipses (Solar / Lunar)
* Abre la pestaña **3. Eclipses (Solar / Lunar)**.
* Selecciona la modalidad deseada con el conmutador interactivo (**Solar** o **Lunar**).
* Pulsa **Cargar Serie de Bracketing** para importar todas las exposiciones (RAW, TIFF o FITS). El sistema leerá automáticamente tiempos de obturación e ISO y ordenará las tomas de menor a mayor exposición.
* Haz doble clic sobre una toma con buena definición del limbo para fijarla como referencia.
* Pulsa **Detectar Limbo en Todo el Lote**: el sistema detectará el centro y radio físico en la referencia y optimizará el centro subpíxel (`< 0.1 px`) en cada fotograma del lote manteniendo el radio anclado.
* Pulsa **Alinear y Fusionar Bracketing (HDR)** para generar el compuesto lineal de 32 bits con compresión Asinh.
* Ajusta los parámetros del filtro **NRGF**: alcance de la corona (por defecto 3.5x), mezcla de filamentos (50%) y filtro tangencial para estirar y contrastar los filamentos coronales sin quemar las protuberancias.
* Pulsa **Guardar TIFF 16-bit** para abrir directamente en Photoshop/Lightroom sin cuadros de diálogo de mapeo de tono, **Guardar TIFF 32-bit** para procesado HDR de alta fidelidad, o envíalo directamente al **Revelador / Editor**.

## Hoja de Ruta
[x] Apilado diferencial cielo/suelo con alineación estelar por homografía.
[x] Rechazo estadístico Kappa-Sigma (MAD) libre de trazas de satélites y aviones.
[x] Pipeline completo de calibración con Darks, Flats y Bias.
[x] Conservación íntegra de resolución nativa mediante BORDER_REFLECT.
[x] Gestión de proyectos y sesiones en disco (.mwstack).
[x] Segmentación asistida GrabCut y zoom nativo 1:1.
[x] Descomposición y realce multiescala mediante ondículas À Trous.
[x] Integración de StarNet++ v2 y GraXpert AI.
[x] Rutina específica de apilado y alineación para eclipses solares y lunares (Módulo Eclipses con NRGF continuo y detección subpíxel).
[ ] Procesado por lotes para secuencias de timelapse.
[ ] Módulo de composición panorámica para mosaicos nocturnos.
[ ] Exportación de perfiles de color ICC embebidos (sRGB / AdobeRGB / ProPhoto).
---

## Licencia

Este proyecto esta bajo la Licencia MIT.
