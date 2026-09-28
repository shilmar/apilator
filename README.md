# Milky Way Dual Stacker & Astro Studio

Herramienta de escritorio en Python / PySide6 diseñada para astrofotografía de paisaje nocturno (nightscape). Implementa un flujo de trabajo desacoplado en dos módulos: Apilado Dual Diferencial (cielo y suelo independientes) y Revelador / Postprocesado con integración de GraXpert AI en coma flotante de 32 bits nativos.

---

## Novedades de esta versión

### 1. Interfaz y Experiencia de Usuario
- **Arquitectura Modular por Pestañas:** Separación entre el flujo de apilado (*Stacker*) y las herramientas de revelado/postprocesado (*Editor*).
- **Lienzo Reactivo con Máscara Guiada:** Dibujo y ajuste interactivo de máscara cielo/suelo con redimensionado adaptativo y preservación de aspecto.
- **Microajuste MTF en Tiempo Real:** Previsualización no destructiva del estirado de histograma para inspección rápida de tomas lineales.
- **Gestión de Proyectos (`.mwstack`):** Guardado y restauración completa de sesiones (rutas de archivos, parámetros de apilado, máscaras y configuración).

### 2. Motor de Procesado y Alineación
- **Filtrado Morfológico Top-Hat:** Aislamiento robusto de fuentes estelares sobre gradientes galácticos y fondos lineales oscuros.
- **Refinamiento Subpíxel Selectivo:** Cálculo de centroides estelares con precisión subpíxel (`cornerSubPix`) sobre pares verificados por RANSAC.
- **Alineación Proyectiva Consecutiva:** Encadenamiento de homografías entre tomas contiguas para absorber la perspectiva sideral sin derivas geométricas.
- **Rechazo Kappa-Sigma Asimétrico:** Clipping MAD con tolerancia superior adaptativa para preservar la señal estelar frente a artefactos y satélites.
- **Streaming en 32-bit Float:** Procesado por bloques horizontales para mantener un consumo de memoria constante en lotes extensos.
- **Soporte de Calibración:** Generación y sustracción térmica de Master Dark en coma flotante.

---

## Caracteristicas Principales

### 1. Modulo Apilador Dual (StackerTab)
* Calibracion termica: Generacion por bloques de memoria (chunked streaming) de Master Dark y sustraccion termica lineal.
* Separacion Cielo / Suelo: Segmentacion interactiva mediante trazos guiados con refinamiento morfologico y algoritmo GrabCut.
* Alineacion estelar de alta precision:
  * Deteccion de estrellas mediante filtros Top-Hat morfologicos y descriptores ORB.
  * Refinamiento subpixel de centroides con cornerSubPix.
  * Estimacion robusta de matrices de homografia/afines mediante RANSAC con soporte para rotacion de campo.
* Apilado estadistico con rechazo:
  * Algoritmo de rechazo Kappa-Sigma basado en MAD (Median Absolute Deviation).
  * Streaming binario a disco para apilar grandes secuencias sin saturar la memoria RAM.
  * Modos soportados: Tripode Fijo (cielo alineado + suelo estatico compuesto) y Star Tracker (seguimiento ecuatorial).
* Gestion de tomas granular: Seleccion interactiva de la toma de referencia (REF) y eliminacion individual de tomas descartadas sin reiniciar la sesion ni la mascara.
* Persistencia de sesion: Guardado y carga de proyectos (.mwstack / .json) preservando listas de archivos, ajustes y mascaras asociadas.

### 2. Modulo Revelador / Editor (DeveloperTab)
* Procesamiento en 32 bits: Carga directa de imagenes maestras en formato TIFF lineal de 32 bits (float32) o archivos FITS astronomicos.
* Correccion de gradientes con GraXpert AI: Extraccion de contaminacion luminica mediante red neuronal con proteccion de la mascara de suelo y guardado automatico con sufijo _graxpert.tiff.
* Ajustes tonales no destructivos:
  * Algoritmo Auto-MTF (Midtone Transfer Function).
  * Estirado hiperbolico Asinh con control en tiempo real de punto negro.
* Exportacion final: Guardado del revelado en TIFF de 16 bits sin perdida o JPEG de alta calidad para publicacion.

---

## Estructura del Proyecto
```text
apilador_astro/
├── run_app.py               # Punto de entrada de la aplicacion
├── requirements.txt         # Dependencias del entorno
├── core/
│   ├── __init__.py
│   ├── project_manager.py   # Gestion de sesiones y serializacion JSON/PNG
│   ├── stacking.py          # Logica de carga, calibracion, homografia y apilado
│   ├── masking.py           # Algoritmos de segmentacion y refinado GrabCut
│   ├── stretch.py           # Funciones de estirado no lineal (Asinh, MTF)
│   └── graxpert_bridge.py   # Pasarela CLI con GraXpert y soporte FITS/TIFF
└── gui/
    ├── __init__.py
    ├── canvas.py            # Visor grafico interactivo y sistema de trazos
    ├── tab_stacker.py       # Pestana del Apilador Dual
    ├── tab_developer.py     # Pestana del Revelador / Editor
    ├── main_window.py       # Ventana principal contenedora
    └── worker.py            # Hilos secundarios en segundo plano (QThread)
```

## Instalacion y Requisitos

### Requisitos Previos
* Python 3.10 o superior (recomendado 3.11 / 3.12 / 3.13).
* Windows 10 / 11 de 64 bits.

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

1. Pestana 1 (Apilador):
   * Anade las tomas de luz (Lights) y las tomas oscuras de calibracion (Darks).
   * Haz doble clic sobre la toma que desees usar como referencia visual y de alineacion.
   * Traza marcas verdes en el cielo y rojas en el suelo, y pulsa "Refinar Automatica".
   * Pulsa "INICIAR APILADO DUAL" y elige la ruta de guardado del TIFF maestro de 32 bits.
   * Al finalizar, el resultado se transferira automaticamente al modulo de revelado.

2. Pestana 2 (Revelador):
   * Abre un archivo TIFF/FITS existente o trabaja con el resultado recien apilado.
   * Aplica GraXpert AI para eliminar gradientes de contaminacion luminica.
   * Ajusta el punto negro y el factor Asinh hasta obtener el contraste y detalle deseados.
   * Exporta el resultado final en 16 bits o JPG mediante "Exportar Imagen Revelada...".

---

## Licencia

Este proyecto esta bajo la Licencia MIT.
