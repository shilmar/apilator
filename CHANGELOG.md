# Changelog

Todas las novedades, mejoras y correcciones notables de **Apilator** se documentan en este archivo.
El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/) y este proyecto se adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [0.6.2] - 2026-10-09

### Añadido
* **Nuevo Módulo Dedicado de Trazas de Estrellas (Startrails / Circumpolares):** Integración completa como **Pestaña 4** en la ventana principal, permitiendo la generación de trazas estelares de alta fidelidad fotométrica con streaming eficiente en memoria $O(1)$ sin importar el número ni tamaño de las tomas RAW/TIFF.
* **Algoritmo de Máximo Estándar (Lighten Clásico):** Composición progresiva de luminancia máxima para trazas estelares continuas.
* **Algoritmo de Efecto Cometa / Estela Progresiva (Comet / Meteor Fade):**
  - Factor de longitud de estela ajustable porcentualmente (5% a 100%).
  - Modulación de curva de decaimiento matemático: lineal uniforme, cosenoidal suave y exponencial rápido.
  - 3 modos de dirección temporal: hacia atrás (*backward*, cola en tomas pasadas), hacia adelante (*forward*, cola en tomas futuras) y simétrico / ambos sentidos (*bidirectional*, cabeza brillante central con colas afiladas hacia pasado y futuro).
  - Cota mínima de luminancia de fondo (*min_floor*) para evitar cielos artificialmente oscuros.
* **Fusión de Suelo Limpio (Anti-Ruido) con Máscara de Horizonte:**
  - Separación completa entre cielo de trazas y terreno estático.
  - Modos de suelo seleccionables: promedio temporal de toda la serie (reducción drástica del ruido térmico y sombras en el terreno) o toma de referencia fija.
  - Conexión e importación directa en un solo clic de la máscara de horizonte calculada en el Apilador (`🖌️ Usar Máscara del Apilador`) o carga desde archivos externos (PNG, TIFF, FITS).
  - Suavizado gaussiano de borde de máscara ajustable (*feathering*).
* **Supresión Automática de Trazas de Satélites y Aviones (Anti-Trazas Transitorias):**
  - Detección adaptativa temporal con búfer deslizante inteligente y doble referencia estadística ($\min(I_A, I_B)$) en toda la serie, incluyendo fotogramas iniciales y finales.
  - Cierre morfológico lineal que reconecta las pulsaciones discontinuas de luces estroboscópicas de aeronaves en trazas continuas.
  - Discriminación geométrica por elongación lineal (`cv2.minAreaRect`), diferenciando matemáticamente trazas rectas ($\text{elongación} \ge 2.8 - 4.0$) de estrellas en rotación o cúmulos estelares circulares ($1.0 - 1.8$).
  - Exclusión automática del suelo y vegetación en movimiento mediante la máscara de horizonte, evitando falsos positivos provocados por ramas o hierba mecidas por el viento.
  - Inpainting y sustitución acelerada por caja envolvente (*bounding box slicing* local) con tiempo de cómputo inferior a $0.15\text{ s}$ por toma de 24.5 MP.
  - Selector de sensibilidad ajustable: Baja, Media (por defecto) y Alta.
* **Exportación y Flujo de Trabajo Directo:**
  - Botones dedicados para exportación en **TIFF 16-bit** (compatible con Photoshop y Lightroom sin mapeos de tono) y **TIFF 32-bit Float** (rango dinámico HDR maestro).
  - Botón de transferencia directa al **Revelador / Editor** (`➡️ Enviar al Revelador / Editor`) con carga instantánea de imagen y máscara.

### Modificado
* **Mejoras en el Visor y Canvas Interactivo (`gui/canvas.py`):**
  - Desacoplamiento de la visualización de máscara de la herramienta de dibujo interactivo: ahora la superposición de máscara coloreada (verde cielo / rojo suelo) es visible en cualquier módulo con solo activar la casilla correspondiente.
  - Auto-reescalado dinámico de la máscara a la resolución exacta de la imagen base cargada en el visor.
* **Organización y Nomenclatura de Pestañas Principales:**
  1. `1. Apilador (Stacker)`
  2. `2. Revelador / Editor`
  3. `3. Eclipses (Solar / Lunar)`
  4. `4. Trazas de Estrellas (Startrails)`
  5. `5. Configuración / Ajustes`
* **Actualización de Versión:** Incremento general a **v0.6.2**.

---

## [0.6.1] - 2026-10-08

### Añadido
* **Vitrina de Resultados y Galería en Configuración:**
  - Nueva columna derecha en la pestaña de Configuración con visor responsivo antialias (`ShowcaseViewer`) que presenta una fotografía real de la Vía Láctea procesada íntegramente con Apilator (`showcase_milkyway.jpg`), adaptándose fluidamente al tamaño de ventana manteniendo su relación de aspecto.
  - Acreditación de autoría fotográfica integrada (**📷 Fotografía y Procesado: Shilmar**) con insignias técnicas de motor (`32-bit HDR`, `StarNet++ AI`, `MTF Asinh`, `Alineación Subpíxel`).
  - Botón de acceso directo (`🔍 Ver imagen completa`) para abrir la toma en alta resolución en el visor predeterminado del sistema operativo.
* **Monitor Dinámico de Estado para StarNet++ CLI:** En el panel de configuración, StarNet++ ahora valida en tiempo real la existencia del archivo ejecutable mostrando un distintivo dinámico en verde (`✓ StarNet++ detectado y listo`) o advertencia en ámbar.

### Modificado
* **Unificación Estética Global e Interfaz Moderna:**
  - Rediseño del panel izquierdo del **Apilador (Stacker)** con contenedor `QScrollArea`, consola de actividad inferior y barra de progreso fijas (siempre visibles sin scroll), e iconografía enriquecida (`⭐ Lights`, `🌑 Darks`, `⚪ Flats`, `⚙️ Bias`, insignia `📌 REF`).
  - Rediseño del panel izquierdo del **Revelador / Editor** con consola de actividad fija abajo, reordenación lógica de controles en 7 secciones secuenciales de revelado y emparejamiento en 2 columnas de deslizadores complementarios (Temperatura/Tinte, Saturación Cielo/Suelo, Claridad/Neblina, Ondículas/Atenuación, Contraste/Ruido, Sombras/Punto Negro).
  - Ajuste de espaciado y desahogo visual en el Revelador para una interacción cómoda y ergonómica.
  - Fijación de la barra de progreso y consola de log en la base del módulo de **Eclipses**.
  - Rediseño en división de 2 columnas de la pestaña de **Configuración** con paneles oscuros estilizados, campos de texto oscuros e interactivos y botón de guardado destacado.
* **Actualización de Versión:** Incremento general a **v0.6.1**.

---

## [0.6.0] - 2026-10-08

### Añadido
* **Nuevo Módulo Especializado de Eclipses (Solar / Lunar):** Integración completa como **Pestaña 3** en la ventana principal, permitiendo el procesado integral de series de bracketing de eclipses solares (corona, cromosfera y protuberancias) y eclipses lunares.
* **Selector de Modalidad Compacto (Boolean Toggle):** Interruptor interactivo tipo píldora booleana estilizado para alternar al instante entre Eclipse Solar (modo cromosfera/corona) y Eclipse Lunar (modo disco reflectivo), optimizando más de 50 px de espacio vertical en el panel.
* **Ingesta Inteligente de Bracketing y Parser EXIF:** Extracción automática de metadatos reales de tiempo de obturación, ISO y apertura desde formatos RAW propietarios (`.nef`, `.cr2`, `.cr3`, `.arw`, `.dng`) y formatos estándar (`.tif`, `.tiff`, `.fits`), ordenando el lote automáticamente por tiempo de exposición.
* **Detección Subpíxel de Limbo Inmune a Protuberancias (`optimize_center_circular_flux`):**
  - Detección topológica inicial del hueco lunar y escáner radial subpíxel en dos pasadas.
  - Algoritmo de optimización de flujo de gradiente radial perpendicular basado en estadística de mediana sobre 360 rayos angulares. Proporciona inmunidad matemática total frente a grandes protuberancias de hidrógeno-alfa, fulguraciones o cuentas de Baily, ubicando el centro `(cx, cy)` con precisión inferior a 0.1 px en la base real de la silueta lunar.
* **Alineación de Lote con Radio Físico Anclado:** Durante la detección en lote (*Detectar Limbo en Todo el Lote*), el radio lunar se ancla de forma exacta al diámetro físico de la toma de referencia para toda la ráfaga, eliminando cualquier encogimiento o fluctuación del radio y registrando la deriva con interpolación afín de alta fidelidad Lanczos4.
* **Fusión Fotométrica HDR Lineal de 32 bits:** Combinación de flujo lineal ponderada por tiempos de exposición reales con umbralización suave de saturación al 99.9995% y estirado no lineal Asinh para levantar la corona media y externa sin quemar protuberancias.
* **Filtro NRGF (Normalized Radial Gradient Filter) de Rango Dinámico Ampliado:**
  - Interpolación continua subpíxel (`np.interp`) sobre los perfiles radiales $\mu(r)$ y $\sigma(r)$, erradicando al 100% cualquier efecto de bandas concéntricas o escalonamientos discretos.
  - Modulación de filamentos liberada de amortiguación excesiva por luminancia, permitiendo el estiramiento y contraste completo de los filamentos coronales magnéticos ("los hilos") hasta más de 3.5 a 4.0 radios solares.
  - Compuerta radial suave (*Radial Gate*) mediante función *smoothstep* que desvanece suavemente la modulación antes del límite de la corona activa, garantizando un cielo de fondo negro aterciopelado sin elevar el grano del sensor.
* **Filtro Bilateral Tangencial de Alta Frecuencia:** Realce selectivo de filamentos finos a lo largo de las líneas de campo coronal.
* **Exportación Dual en TIFF de 16 bits y 32 bits:**
  - Botón dedicado **Guardar TIFF 16-bit**: convierte linealmente de `float32` a entero `uint16` sin pérdida con compresión `zlib` y metadatos `photometric='rgb'`, permitiendo abrir la imagen directamente en **Photoshop o Lightroom** como un TIFF estándar de 16 bits sin disparar cuadros de diálogo de mapeo tonal ni conversiones extrañas.
  - Botón dedicado **Guardar TIFF 32-bit**: exporta en coma flotante `float32` con rango dinámico lineal HDR completo para procesado científico.
* **Puente Directo con el Revelador / Editor:** Botón *Enviar al Revelador / Editor* que transfiere la imagen 32-bit procesada directamente a la Pestaña 2 y conmuta la vista de forma inmediata.

### Modificado
* **Organización y Nomenclatura de Pestañas Principales:**
  1. `1. Apilador (Stacker)`
  2. `2. Revelador / Editor`
  3. `3. Eclipses (Solar / Lunar)`
  4. `4. Configuración / Ajustes`
* **Persistencia de Selección en Tabla de Bracketing:** Selección bidireccional estable que preserva la fila activa y actualiza inmediatamente el canvas interactivo y los controles de coordenadas sin desfases.
* **Valores Predeterminados del Filtro NRGF:** Alcance radial por defecto actualizado a **3.5x** y mezcla de filamentos al **50%**.

---

## [0.5.12] - 2026-10-06
### Añadido
* Motor de gestión de memoria ultra-eficiente para apilado de lotes masivos.
* Pipeline nativo CPU multi-hilo optimizado por franjas de procesamiento (chunks).
* Eliminación de cuellos de botella en la comunicación entre procesos (IPC).

## [0.5.11] - 2026-10-05
### Añadido
* Validación preventiva de orientación de fotogramas.
* Suite unificada de StarNet++ con manejo robusto de excepciones y formatos FITS/TIFF.

## [0.5.10] - 2026-10-04
### Añadido
* Pipeline completo de calibración astronómica con pestañas independientes para Flats y Bias.
* Refinamiento guiado de bordes en segmentación con visualizador 1:1 nativo.
