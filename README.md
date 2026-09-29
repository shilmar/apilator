# Apilator

**Apilator** es una herramienta especializada de posprocesado y apilado para **astrofotografía de paisaje (nightscapes)**. Permite desacoplar el movimiento del cielo respecto al horizonte terrestre, aplicar integración estadística Kappa-Sigma en coma flotante de 32 bits y editar la imagen final en tiempo real con asistencia de redes neuronales (GraXpert y StarNet++ v2).

---

## Novedades en la Versión 0.3.1

* **Integración de StarNet++ v2 CLI**: Separación neuronal de estrellas y fondo nebuloso (*starless*) en espacio lineal nativo (`--linear`) protegiendo siluetas terrestres[cite: 1].
* **Control interactivo de estrellas**: Deslizador dinámico de intensidad/reducción estelar (0% a 150%) y conmutador visual de capas (*Compuesta*, *Solo Fondo*, *Solo Estrellas*).
* **Pestaña de Configuración y Ajustes**: Persistencia de rutas externas (`starnet++.exe` / `starnet2.exe`), factor Kappa base y resolución de proxies en `config.json`.
* **Motor de previsualización a 60 fps**: Pipeline basado en proxies escalados con memoria contigua en C (`np.ascontiguousarray`), eliminando retrasos al interactuar con deslizadores.
* **Exportación multiformato nativa**: Soporte en cuadro de diálogo para **TIFF de 32 bits Float**, **TIFF de 16 bits** (RGB compatible con Windows/Photoshop) y **JPEG**.
* **Motor Kappa-Sigma optimizado**: Integración estadística acelerada vectorizada en bloques horizontales de baja huella de RAM.

---

## Características Principales

### 1. Apilador Dual Cielo / Suelo (Stacker)
* **Detección y alineación estelar**: Extracción morfológica de estrellas (Top-Hat + centroides subpíxel) con registro afín robusto basado en RANSAC.
* **Separación de horizonte**: Generación de máscaras binarias y desenfoque adaptativo (*feathering*) para aislar el suelo estático del cielo en rotación.
* **Rechazo Kappa-Sigma vectorizado**: Supresión de trazas de satélites, aviones y ruido térmico/cósmico en buffers de 32 bits (`float32`).

### 2. Revelador y Procesado de Color (Developer)
* **Eliminación de gradientes por IA (GraXpert)**: Neutralización del fondo astronómico protegiendo el primer plano terrestre.
* **Separación y reducción de estrellas por IA (StarNet++ v2)**: Extracción en segundo plano con control de opacidad en tiempo real.
* **Balance de blancos astrofotográfico**: Ajuste directo de Temperatura (Azul/Ámbar) y Tinte (Verde/Magenta) en espacio lineal.
* **Saturación cromática diferencial**: Controles desacoplados de saturación para el cielo y el suelo mediante máscaras gaussianas.
* **Curva MTF interactiva**: Algoritmo de función de transferencia de medios tonos automático y manual.

### 3. Configuración y Entorno (Settings)
* Detección automática y manual de ejecutables CLI externos.
* Ajustes de rendimiento de memoria, hilos y tamaño del proxy visual.

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

* **Sistema Operativo**: Windows 10/11, Linux o macOS.
* **Python**: 3.10 o superior.
* **Dependencias principales**:
  * `PySide6`
  * `numpy`
  * `opencv-python`
  * `tifffile`

*(Opcional para módulos de IA)*:
* **GraXpert** (versión CLI o ejecutable en PATH).
* **StarNet++ v2 CLI** (especificar ruta al ejecutable en la pestaña de Configuración).

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
    Apilado:
        Cargar las tomas en la pestaña 1. Apilador.
        Definir la máscara de separación cielo/suelo.
        Ejecutar el apilado dual. Al concluir, el resultado en float32 se transferirá automáticamente a la pestaña de revelado.
    Corrección de Gradientes:
        En la pestaña 2. Revelador, pulsar Eliminar Gradientes con GraXpert si hay contaminación lumínica residual.
    Control de Estrellas:
        Ejecutar Separar Estrellas con StarNet AI para aislar el campo estelar.
        Reducir la intensidad al 40%-60% para resaltar las estructuras de la Vía Láctea.
    Color y Curva:
        Ajustar balance de blancos, saturación selectiva para el cielo y estirado MTF.
    Exportación:
        Guardar en TIFF de 32 bits si se va a continuar la edición en Photoshop o PixInsight, o en TIFF de 16 bits / JPEG para entrega final.
---

## Licencia

Este proyecto esta bajo la Licencia MIT.
