# Apilator

**Apilator** es una herramienta especializada de posprocesado y apilado para **astrofotografía de paisaje (nightscapes)**. Permite desacoplar el movimiento del cielo respecto al horizonte terrestre, aplicar integración estadística Kappa-Sigma en coma flotante de 32 bits y editar la imagen final en tiempo real con asistencia de redes neuronales (GraXpert y StarNet++ v2).

---

## Características Principales

### 1. Apilado Diferencial Cielo / Suelo
- **Alineación Estelar Robusta:** Detección y emparejamiento de estrellas mediante transformaciones afines y descriptores clave sobre datos en coma flotante de 32 bits.
- **Apilado de Ruido Mínimo:** Integración selectiva para el cielo (alineado) y el suelo (estático).
- **Rechazo Estadístico Min-Sigma:** Eliminación eficaz de trazas de satélites, estelas de aviones y artefactos transitorios sin degradar la relación señal/ruido (SNR).
- **Antipolución Lumínica:** Corrección de gradientes y cúpulas de luz artificial inspirada en algoritmos de tipo Sequator.

### 2. Segmentación y Máscaras Interactivas
- **GrabCut Guiado Multiescala:** Delineación precisa del horizonte y perfiles complejos (árboles, relieve montañoso, estructuras).
- **Edición Continua no Destructiva:** Permite refinar la máscara, seguir corrigiendo con trazos de pincel en tiempo real y recalcular sin reiniciar el trabajo.
- **Control Paramétrico Fino:** Ajuste del radio de difusión/suavizado gaussiano del borde (feathering) e iteraciones del modelo.
- **Zoom 1:1 Nativo con Clic Derecho:** Inspección pixel a pixel del sensor original para retoque milimétrico de bordes.

### 3. Revelador Astrofotográfico Avanzado
- **Canal Lineal Puro:** Pipeline optimizado en coma flotante (float32), preservando la linealidad fotométrica hasta la compresión tonal final.
- **Separación de Estrellas con StarNet++:** Extracción de capas Starless y Stars-only en espacio lineal, permitiendo trabajar nebulosas y polvo galáctico sin hinchar las estrellas.
- **Estructura Multiescala por Ondículas (À Trous / B-Spline):** Realce selectivo de filamentos de gas y bandas de absorción de la Vía Láctea, junto con atenuación de la capa residual de fondo.
- **Extracción de Gradientes:** Motor dual integrado con GraXpert AI y ajuste polinómico cuadrático para corregir viñeteo y gradientes luminosos.
- **Reducción de Ruido Adaptativa:** Múltiples métodos para el fondo (Filtro Bilateral, Filtro Guiado y Non-Local Means) respetando los límites de las estrellas.
- **Balance de Color Fino y Tonalidad:** Curvas interactivas con histograma integrado en tiempo real, balance de temperatura/tinte calibrado ($\pm0.250$), control de vibranza y saturación diferencial cielo/suelo.

### 4. Configuración y Entorno
- ** Detección automática y manual de ejecutables CLI externos.
- ** Ajustes de rendimiento y tamaño del proxy visual.   
---

## Estructura del Proyecto
```text
apilator/
├── config.json              # Configuración persistente del usuario
├── run_app.py               # Punto de entrada de la aplicación
├── core/
│   ├── config_manager.py    # Carga y almacenamiento de ajustes JSON
│   ├── graxpert_bridge.py   # Conector CLI con GraXpert AI
│   ├── masking.py           # 
│   ├── project_manager.py   # 
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
  - `imagecodecs` ()
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

Flujo de Trabajo Recomendado
	Flujo de Apilado:
        Ve a la pestaña Apilador.
        Carga tu serie de tomas (RAW o TIFF) y selecciona la toma base de referencia.
        Dibuja los trazos base: Verde para el cielo y Rojo para el suelo.
        Haz clic en Refinar Automática para obtener la máscara inicial. Usa el clic derecho para inspeccionar al 100% y pulir zonas difíciles.
        Configura las opciones de rechazo Min-Sigma y pulsa Apilar Tomas.
    Flujo de Revelado:
        La imagen resultante se enviará automáticamente a la pestaña Revelador (o puedes abrir un archivo TIFF/FITS existente).
        Neutraliza gradientes con el motor Polinómico o GraXpert.
        Ejecuta StarNet++ para desacoplar el fondo de las estrellas.
        Aplica realce de gas y polvo galáctico mediante los deslizadores de Estructura Multiescala (Ondículas).
        Ajusta curvas, balance fino y saturación, e inspecciona cualquier zona en escala nativa (clic derecho).
        Haz clic en Exportar Imagen Revelada (soporta TIFF 16-bit, TIFF 32-bit float y JPEG).

Hoja de Ruta

    Apilado con alineación estelar y rechazo Min-Sigma.
    Máscaras guiadas interactivas y zoom 1:1 nativo.
    Integración de StarNet++ y GraXpert AI.
    Descomposición y realce multiescala mediante ondículas.
    Módulo de composición panorámica para mosaicos nocturnos.
    Exportación de perfiles de color ICC embebidos (sRGB / AdobeRGB / ProPhoto).
---

## Licencia

Este proyecto esta bajo la Licencia MIT.
