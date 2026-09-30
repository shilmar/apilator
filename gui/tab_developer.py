# gui/tab_developer.py
import os
import cv2
import numpy as np
import tifffile
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QGroupBox, QSlider, QSplitter,
    QDoubleSpinBox, QTextEdit, QComboBox, QSpinBox,
    QScrollArea, QFrame  # <-- Asegúrate de incluir estos dos
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from core.stacking import load_image_as_float32
from core.stretch import (
    calculate_mtf_params, manual_stretch, 
    apply_white_balance, adjust_saturation_dual,
    adjust_vibrance, adjust_contrast,
    apply_denoise, apply_curve_lut  # <-- Reemplazamos adjust_tonal_bands por apply_curve_lut
)
from gui.widgets.curve_widget import CurveWidget  # <-- Importamos el nuevo widget
from gui.canvas import MaskCanvas
from gui.worker import GraXpertWorker, StarNetWorker


class DeveloperTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_32bit = None       # Imagen nativa base (float32)[cite: 2]
        self.image_starless = None    # Capa sin estrellas (float32)[cite: 2]
        self.image_stars = None       # Capa de solo estrellas (float32)[cite: 2]

        # Buffers proxy para previsualización interactiva a 60 fps[cite: 2]
        self.preview_proxy = None     
        self.proxy_starless = None
        self.proxy_stars = None
        self.proxy_mask = None        

        self.current_mask = None      
        self.active_filepath = None
        self.gx_worker = None
        self.sn_worker = None

        # Parámetros del revelador
        self.temp_val = 0.0          # Rango: -0.500 a +0.500
        self.tint_val = 0.0          # Rango: -0.500 a +0.500
        self.sat_sky_val = 1.0       # 0.00x a 2.50x[cite: 1]
        self.sat_gnd_val = 1.0       # 0.00x a 2.50x[cite: 1]
        self.vibrance_val = 0.0      # -1.00 a +1.00
        self.contrast_val = 0.0      # -1.00 a +1.00
        self.star_intensity = 1.0    # 1.0 = 100%, 0.0 = Starless puro[cite: 2]
        self.view_layer_mode = 0     # 0: Compuesta, 1: Solo Fondo, 2: Solo Estrellas[cite: 2]
        self.contrast_starless_val = 0.0  # -1.0 a +1.0 (solo para el fondo sin estrellas)
        # Parámetros tonales específicos para la capa Starless ([-1.0, 1.0])
        self.starless_blacks = 0.0
        self.starless_shadows = 0.0
        self.starless_highlights = 0.0
        self.starless_whites = 0.0

        # Parámetros de Reducción de Ruido
        self.denoise_strength = 0.0
        self.denoise_method = "Bilateral"
        
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # 1. Visor interactivo
        self.canvas = MaskCanvas(self, enable_masking=False)

        # 2. Panel interno de controles
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(6)

        # Carga externa
        btn_open_img = QPushButton("Abrir Imagen (TIFF / FITS / RAW)...")
        btn_open_img.setStyleSheet("font-weight: bold; padding: 6px;")
        btn_open_img.clicked.connect(self.open_image_file)
        left_layout.addWidget(btn_open_img)

        # Módulo GraXpert Compacto
        grp_graxpert = QGroupBox("GraXpert AI - Fondo y Gradientes")
        gx_layout = QHBoxLayout(grp_graxpert)
        gx_layout.setContentsMargins(8, 8, 8, 8)
        gx_layout.setSpacing(6)

        gx_layout.addWidget(QLabel("Suavizado:"))
        self.spin_smoothing = QDoubleSpinBox()
        self.spin_smoothing.setRange(0.01, 1.0)
        self.spin_smoothing.setValue(0.5)
        self.spin_smoothing.setSingleStep(0.05)
        self.spin_smoothing.setFixedWidth(65)
        gx_layout.addWidget(self.spin_smoothing)

        self.btn_graxpert = QPushButton("Eliminar Gradientes")
        self.btn_graxpert.setStyleSheet("font-weight: bold; background-color: #3b4252; color: #eceff4; padding: 4px;")
        self.btn_graxpert.clicked.connect(self.run_graxpert)
        gx_layout.addWidget(self.btn_graxpert)

        left_layout.addWidget(grp_graxpert)

        # Módulo StarNet AI - Encabezado Compacto
        grp_starnet = QGroupBox("StarNet++ AI - Control de Estrellas")
        sn_layout = QVBoxLayout(grp_starnet)
        sn_layout.setContentsMargins(8, 8, 8, 8)
        sn_layout.setSpacing(5)

        row_sn_action = QHBoxLayout()
        row_sn_action.addWidget(QLabel("Paso:"))
        self.spin_stride = QSpinBox()
        self.spin_stride.setRange(64, 512)
        self.spin_stride.setSingleStep(64)
        self.spin_stride.setValue(256)
        self.spin_stride.setFixedWidth(60)
        row_sn_action.addWidget(self.spin_stride)

        self.btn_starnet = QPushButton("Separar Estrellas con StarNet")
        self.btn_starnet.setStyleSheet("font-weight: bold; background-color: #434c5e; color: #eceff4; padding: 4px;")
        self.btn_starnet.clicked.connect(self.run_starnet)
        row_sn_action.addWidget(self.btn_starnet)
        sn_layout.addLayout(row_sn_action)

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("Capa visible:"))
        self.combo_layer = QComboBox()
        self.combo_layer.addItems(["Compuesta (Normal)", "Solo Fondo (Starless)", "Solo Estrellas"])
        self.combo_layer.setEnabled(False)
        self.combo_layer.currentIndexChanged.connect(self.on_layer_mode_changed)
        row_mode.addWidget(self.combo_layer)
        sn_layout.addLayout(row_mode)

        row_star_slider = QHBoxLayout()
        row_star_slider.addWidget(QLabel("Intensidad Estrellas:"))
        self.lbl_star_intensity = QLabel("100%")
        row_star_slider.addWidget(self.lbl_star_intensity)
        sn_layout.addLayout(row_star_slider)

        self.slider_stars = QSlider(Qt.Horizontal)
        self.slider_stars.setRange(0, 150)
        self.slider_stars.setValue(100)
        self.slider_stars.setEnabled(False)
        self.slider_stars.valueChanged.connect(self.on_star_intensity_changed)
        sn_layout.addWidget(self.slider_stars)

        # Contraste específico para la capa de fondo (Starless) con rango +-1.000
        row_sn_cnt = QHBoxLayout()
        row_sn_cnt.addWidget(QLabel("Contraste Fondo (Starless):"))
        self.lbl_contrast_starless = QLabel("0.000")
        row_sn_cnt.addWidget(self.lbl_contrast_starless)
        sn_layout.addLayout(row_sn_cnt)

        self.slider_contrast_starless = QSlider(Qt.Horizontal)
        self.slider_contrast_starless.setRange(-1000, 1000)  # <-- De -1000 a +1000
        self.slider_contrast_starless.setValue(0)
        self.slider_contrast_starless.setEnabled(False)
        self.slider_contrast_starless.valueChanged.connect(self.on_contrast_starless_changed)
        sn_layout.addWidget(self.slider_contrast_starless)
        
        # --- Reducción de Ruido (Denoise) ---
        lbl_dn = QLabel("<b>Reducción de Ruido (Fondo):</b>")
        sn_layout.addWidget(lbl_dn)

        row_dn_mode = QHBoxLayout()
        row_dn_mode.addWidget(QLabel("Método:"))
        self.combo_denoise = QComboBox()
        self.combo_denoise.addItems(["Bilateral", "Guided Filter", "NL-Means"])
        self.combo_denoise.setEnabled(False)
        self.combo_denoise.currentTextChanged.connect(self.on_denoise_method_changed)
        row_dn_mode.addWidget(self.combo_denoise)
        sn_layout.addLayout(row_dn_mode)

        row_dn = QHBoxLayout()
        row_dn.addWidget(QLabel("Fuerza Denoise:"))
        self.lbl_denoise = QLabel("0%")
        row_dn.addWidget(self.lbl_denoise)
        sn_layout.addLayout(row_dn)
        
        self.slider_denoise = QSlider(Qt.Horizontal)
        self.slider_denoise.setRange(0, 100)
        self.slider_denoise.setValue(0)
        self.slider_denoise.setEnabled(False)
        self.slider_denoise.valueChanged.connect(self.on_denoise_changed)
        sn_layout.addWidget(self.slider_denoise)
        
        left_layout.addWidget(grp_starnet)
        
        # --- Editor de Curvas con Histograma de Fondo ---
        grp_curves = QGroupBox("Curvas de Tono (Fondo e Histograma)")
        curves_layout = QVBoxLayout(grp_curves)
        curves_layout.setContentsMargins(6, 6, 6, 6)
        curves_layout.setSpacing(4)

        self.curve_widget = CurveWidget()
        self.curve_widget.curveChanged.connect(self.update_stretch_preview)
        curves_layout.addWidget(self.curve_widget)

        row_crv_btn = QHBoxLayout()
        self.btn_reset_curve = QPushButton("Resetear Curva")
        self.btn_reset_curve.setStyleSheet("font-size: 11px; padding: 2px;")
        self.btn_reset_curve.clicked.connect(self.curve_widget.reset_curve)
        row_crv_btn.addWidget(self.btn_reset_curve)
        curves_layout.addLayout(row_crv_btn)

        sn_layout.addWidget(grp_curves)
              
        # Módulo Balance de Blancos (Control Fino +-0.500)
        grp_wb = QGroupBox("Balance de Blancos (Precisión Fina)")
        wb_layout = QVBoxLayout(grp_wb)
        wb_layout.setSpacing(3)

        row_temp = QHBoxLayout()
        row_temp.addWidget(QLabel("Temp (Frío / Cálido):"))
        self.lbl_temp_val = QLabel("0.000")
        row_temp.addWidget(self.lbl_temp_val)
        wb_layout.addLayout(row_temp)

        # Rango -500 a +500 mapeado a -0.500 a +0.500
        self.slider_temp = QSlider(Qt.Horizontal)
        self.slider_temp.setRange(-500, 500)
        self.slider_temp.setValue(0)
        self.slider_temp.valueChanged.connect(self.on_temp_changed)
        wb_layout.addWidget(self.slider_temp)

        row_tint = QHBoxLayout()
        row_tint.addWidget(QLabel("Tinte (Verde / Magenta):"))
        self.lbl_tint_val = QLabel("0.000")
        row_tint.addWidget(self.lbl_tint_val)
        wb_layout.addLayout(row_tint)

        self.slider_tint = QSlider(Qt.Horizontal)
        self.slider_tint.setRange(-500, 500)
        self.slider_tint.setValue(0)
        self.slider_tint.valueChanged.connect(self.on_tint_changed)
        wb_layout.addWidget(self.slider_tint)

        btn_reset_wb = QPushButton("Restablecer Balance")
        btn_reset_wb.clicked.connect(self.reset_wb)
        wb_layout.addWidget(btn_reset_wb)
        left_layout.addWidget(grp_wb)

        # Módulo Saturación e Intensidad (Vibrance)
        grp_sat = QGroupBox("Color: Saturación e Intensidad")
        sat_layout = QVBoxLayout(grp_sat)
        sat_layout.setSpacing(3)

        row_sky = QHBoxLayout()
        row_sky.addWidget(QLabel("Saturación Cielo:"))
        self.lbl_sat_sky = QLabel("1.00x")
        row_sky.addWidget(self.lbl_sat_sky)
        sat_layout.addLayout(row_sky)

        self.slider_sat_sky = QSlider(Qt.Horizontal)
        self.slider_sat_sky.setRange(0, 250)
        self.slider_sat_sky.setValue(100)
        self.slider_sat_sky.valueChanged.connect(self.on_sat_sky_changed)
        sat_layout.addWidget(self.slider_sat_sky)

        row_gnd = QHBoxLayout()
        row_gnd.addWidget(QLabel("Saturación Suelo:"))
        self.lbl_sat_gnd = QLabel("1.00x")
        row_gnd.addWidget(self.lbl_sat_gnd)
        sat_layout.addLayout(row_gnd)

        self.slider_sat_gnd = QSlider(Qt.Horizontal)
        self.slider_sat_gnd.setRange(0, 250)
        self.slider_sat_gnd.setValue(100)
        self.slider_sat_gnd.valueChanged.connect(self.on_sat_gnd_changed)
        sat_layout.addWidget(self.slider_sat_gnd)

        # Control Vibrance
        row_vib = QHBoxLayout()
        row_vib.addWidget(QLabel("Intensidad (Vibrance):"))
        self.lbl_vibrance = QLabel("0.00")
        row_vib.addWidget(self.lbl_vibrance)
        sat_layout.addLayout(row_vib)

        self.slider_vibrance = QSlider(Qt.Horizontal)
        self.slider_vibrance.setRange(-100, 100)
        self.slider_vibrance.setValue(0)
        self.slider_vibrance.valueChanged.connect(self.on_vibrance_changed)
        sat_layout.addWidget(self.slider_vibrance)

        btn_reset_sat = QPushButton("Restablecer Color")
        btn_reset_sat.clicked.connect(self.reset_saturation)
        sat_layout.addWidget(btn_reset_sat)
        left_layout.addWidget(grp_sat)

        # Módulo MTF y Contraste
        grp_stretch = QGroupBox("Estirado Tonal y Contraste")
        stretch_layout = QVBoxLayout(grp_stretch)
        stretch_layout.setSpacing(4)

        btn_auto_mtf = QPushButton("Auto-Estirado MTF")
        btn_auto_mtf.clicked.connect(self.apply_auto_mtf)
        stretch_layout.addWidget(btn_auto_mtf)

        row_bp = QHBoxLayout()
        row_bp.addWidget(QLabel("Punto Negro:"))
        self.spin_bp = QDoubleSpinBox()
        self.spin_bp.setDecimals(5)
        self.spin_bp.setRange(0.0, 0.20000)
        self.spin_bp.setSingleStep(0.00010)
        self.spin_bp.setValue(0.0)
        self.spin_bp.valueChanged.connect(self.on_spin_bp_changed)
        row_bp.addWidget(self.spin_bp)
        stretch_layout.addLayout(row_bp)

        self.slider_bp = QSlider(Qt.Horizontal)
        self.slider_bp.setRange(0, 2000)
        self.slider_bp.setValue(0)
        self.slider_bp.valueChanged.connect(self.on_slider_bp_changed)
        stretch_layout.addWidget(self.slider_bp)

        row_mtf = QHBoxLayout()
        row_mtf.addWidget(QLabel("Medios Tonos (m):"))
        self.spin_mtf = QDoubleSpinBox()
        self.spin_mtf.setDecimals(5)
        self.spin_mtf.setRange(0.00010, 0.50000)
        self.spin_mtf.setSingleStep(0.00050)
        self.spin_mtf.setValue(0.10000)
        self.spin_mtf.valueChanged.connect(self.on_spin_mtf_changed)
        row_mtf.addWidget(self.spin_mtf)
        stretch_layout.addLayout(row_mtf)

        self.slider_mtf = QSlider(Qt.Horizontal)
        self.slider_mtf.setRange(1, 2000)
        self.slider_mtf.setValue(400)
        self.slider_mtf.valueChanged.connect(self.on_slider_mtf_changed)
        stretch_layout.addWidget(self.slider_mtf)

        # Control Contraste
        row_cnt = QHBoxLayout()
        row_cnt.addWidget(QLabel("Contraste:"))
        self.lbl_contrast = QLabel("0.00")
        row_cnt.addWidget(self.lbl_contrast)
        stretch_layout.addLayout(row_cnt)

        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(-100, 100)
        self.slider_contrast.setValue(0)
        self.slider_contrast.valueChanged.connect(self.on_contrast_changed)
        stretch_layout.addWidget(self.slider_contrast)

        btn_reset = QPushButton("Restablecer Tono")
        btn_reset.clicked.connect(self.reset_sliders)
        stretch_layout.addWidget(btn_reset)
        left_layout.addWidget(grp_stretch)

        # Exportación
        btn_export = QPushButton("Exportar Imagen Revelada...")
        btn_export.setFixedHeight(38)
        btn_export.setStyleSheet("font-weight: bold; background-color: #2e6648; color: white;")
        btn_export.clicked.connect(self.export_image)
        left_layout.addWidget(btn_export)

        # Registro
        left_layout.addWidget(QLabel("Registro del Revelador:"))
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(95)
        self.txt_log.setStyleSheet(
            "background-color: #141414; color: #d0d0d0; "
            "font-family: Consolas, monospace; font-size: 11px; "
            "border: 1px solid #333333; border-radius: 4px; padding: 4px;"
        )
        left_layout.addWidget(self.txt_log)
        left_layout.addStretch()

        # AHORA:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setWidget(left_panel)

        # Los anchos se fijan aquí, en el contenedor del scroll
        scroll_area.setMinimumWidth(380)
        scroll_area.setMaximumWidth(450)

        # Estilo para la barra de desplazamiento
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background: #1e1e1e;
                width: 8px;
                margin: 0px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #4a4a4a;
                min-height: 25px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #606060;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }            
        """)

        splitter.addWidget(scroll_area)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 10)
        layout.addWidget(splitter)

    def log_message(self, text: str):
        self.txt_log.append(text)
        self.txt_log.moveCursor(QTextCursor.End)

    # --- Generación de Proxies ligeros ---
    def _generate_preview_proxy(self):
        if self.image_32bit is None:
            self.preview_proxy = None
            self.proxy_starless = None
            self.proxy_stars = None
            self.proxy_mask = None
            return

        h, w = self.image_32bit.shape[:2]
        max_dim = 1600.0
        scale = min(1.0, max_dim / max(h, w))

        if scale < 1.0:
            pw = (int(w * scale) // 4) * 4
            ph = (int(h * scale) // 4) * 4
            self.preview_proxy = np.ascontiguousarray(
                cv2.resize(self.image_32bit, (pw, ph), interpolation=cv2.INTER_AREA), dtype=np.float32
            )
            if self.image_starless is not None:
                self.proxy_starless = np.ascontiguousarray(
                    cv2.resize(self.image_starless, (pw, ph), interpolation=cv2.INTER_AREA), dtype=np.float32
                )
                self.proxy_stars = np.ascontiguousarray(
                    cv2.resize(self.image_stars, (pw, ph), interpolation=cv2.INTER_AREA), dtype=np.float32
                )

            if self.current_mask is not None:
                m_small = cv2.resize(self.current_mask, (pw, ph), interpolation=cv2.INTER_LINEAR)
                ksize = int(max(7, (min(ph, pw) // 150) | 1))
                if ksize % 2 == 0: ksize += 1
                sm = cv2.GaussianBlur(m_small, (ksize, ksize), sigmaX=ksize / 3.0)
                self.proxy_mask = np.ascontiguousarray(np.repeat(sm[..., np.newaxis], 3, axis=2), dtype=np.float32)
            else:
                self.proxy_mask = None
        else:
            self.preview_proxy = np.ascontiguousarray(self.image_32bit, dtype=np.float32)
            self.proxy_starless = self.image_starless.copy() if self.image_starless is not None else None
            self.proxy_stars = self.image_stars.copy() if self.image_stars is not None else None
            if self.current_mask is not None:
                ksize = int(max(15, (min(h, w) // 150) | 1))
                if ksize % 2 == 0: ksize += 1
                sm = cv2.GaussianBlur(self.current_mask, (ksize, ksize), sigmaX=ksize / 3.0)
                self.proxy_mask = np.ascontiguousarray(np.repeat(sm[..., np.newaxis], 3, axis=2), dtype=np.float32)
            else:
                self.proxy_mask = None

    def load_image_direct(self, filepath: str, mask: np.ndarray = None):
        self.active_filepath = filepath
        self.current_mask = mask
        self.image_starless = None
        self.image_stars = None
        self.combo_layer.setEnabled(False)
        self.slider_stars.setEnabled(False)
        self.log_message(f"Cargando imagen: {os.path.basename(filepath)}...")
        self.slider_contrast_starless.setEnabled(False)
        self.slider_contrast_starless.setValue(0)
        self.contrast_starless_val = 0.0
        self.lbl_contrast_starless.setText("0.00")        
        # En lugar de los sliders antiguos de bandas tonales:
        if hasattr(self, 'curve_widget'):
            self.curve_widget.reset_curve()
        self.combo_denoise.setEnabled(False)
        
        try:
            self.image_32bit = load_image_as_float32(filepath)
            self._generate_preview_proxy()
            self.apply_auto_mtf()
            self.log_message(f"Imagen en memoria ({self.image_32bit.shape[1]}x{self.image_32bit.shape[0]} px). Vista acelerada activa.")
        except Exception as e:
            self.log_message(f"[ERROR] No se pudo cargar: {e}")
            QMessageBox.critical(self, "Error", f"Fallo al abrir archivo:\n{e}")

    def open_image_file(self):
        filtros = (
            "Imágenes Astronómicas y RAW (*.tif *.tiff *.fits *.fit *.nef *.cr2 *.cr3 *.arw *.dng);;"
            "Archivos RAW (*.nef *.cr2 *.cr3 *.arw *.dng);;"
            "Archivos TIFF (*.tif *.tiff);;"
            "Archivos FITS (*.fits *.fit)"
        )
        p, _ = QFileDialog.getOpenFileName(
            self, "Abrir Imagen de Astronomía o RAW", "",
            filtros
        )
        if p:
            self.load_image_direct(p)

    # --- Composición Dinámica de Capas ---
    def _compose_active_base(self, for_export: bool = False) -> np.ndarray:
        if for_export:
            starless, stars, base = self.image_starless, self.image_stars, self.image_32bit
            mask_3d = None
            if self.current_mask is not None and base is not None:
                h_f, w_f = base.shape[:2]
                m_f = cv2.resize(self.current_mask, (w_f, h_f), interpolation=cv2.INTER_LINEAR) if self.current_mask.shape[:2] != (h_f, w_f) else self.current_mask
                mask_3d = np.repeat(m_f[..., np.newaxis], 3, axis=2)
        else:
            starless, stars, base = self.proxy_starless, self.proxy_stars, self.preview_proxy
            mask_3d = self.proxy_mask

        if base is None:
            return None

        # Si aún no se ha ejecutado StarNet, base directa
        if starless is None or stars is None:
            return base

        processed_starless = starless

        # 1. Reducción de Ruido (Denoise) en Starless lineal
        if self.denoise_strength > 1e-4:
            denoised = apply_denoise(
                processed_starless,
                strength=self.denoise_strength,
                method=self.denoise_method,
                is_full_res=for_export
            )
            if mask_3d is not None:
                processed_starless = (denoised * mask_3d) + (processed_starless * (1.0 - mask_3d))
            else:
                processed_starless = denoised

        # 2. Mezcla de capas
        if self.view_layer_mode == 0:  # Compuesta
            composed = processed_starless + (stars * self.star_intensity)
            return np.clip(composed, 0.0, 1.0)
        elif self.view_layer_mode == 1:  # Solo Fondo
            return processed_starless
        else:  # Solo Estrellas
            return stars

    # --- Cadena de Revelado Compartida ---
    def _apply_pipeline_on_image(self, target_img: np.ndarray, precomputed_mask: np.ndarray = None) -> np.ndarray:
        if target_img is None:
            return None

        # 1. Balance de blancos
        img = apply_white_balance(target_img, self.temp_val, self.tint_val)

        # 2. Saturación diferencial cielo / suelo
        if self.sat_sky_val != 1.0 or self.sat_gnd_val != 1.0:
            hsv = cv2.cvtColor(np.clip(img, 0.0, 1.0), cv2.COLOR_RGB2HSV)
            if precomputed_mask is None or self.sat_sky_val == self.sat_gnd_val:
                hsv[..., 1] = np.clip(hsv[..., 1] * self.sat_sky_val, 0.0, 1.0)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            else:
                hsv_sky = hsv.copy()
                hsv_sky[..., 1] = np.clip(hsv_sky[..., 1] * self.sat_sky_val, 0.0, 1.0)
                rgb_sky = cv2.cvtColor(hsv_sky, cv2.COLOR_HSV2RGB)

                hsv_gnd = hsv.copy()
                hsv_gnd[..., 1] = np.clip(hsv_gnd[..., 1] * self.sat_gnd_val, 0.0, 1.0)
                rgb_gnd = cv2.cvtColor(hsv_gnd, cv2.COLOR_HSV2RGB)

                img = (rgb_sky * precomputed_mask) + (rgb_gnd * (1.0 - precomputed_mask))

        # 3. Intensidad (Vibrance selectivo)
        if abs(self.vibrance_val) > 1e-4:
            img = adjust_vibrance(img, self.vibrance_val)

        # 4. Estirado MTF (la imagen pasa a espacio perceptual visible [0.0, 1.0])
        bp_val = self.spin_bp.value()
        m_val = self.spin_mtf.value()
        stretched = manual_stretch(img, black_point=bp_val, midtone=m_val)

        # 5. Editor de Curvas Tonales (Aplica la LUT del CurveWidget)
        if not self.curve_widget.is_identity():
            curved = apply_curve_lut(stretched, self.curve_widget.get_lut())
            if precomputed_mask is not None:
                stretched = (curved * precomputed_mask) + (stretched * (1.0 - precomputed_mask))
            else:
                stretched = curved

        # 6. Contraste específico del Fondo (Starless)
        if abs(self.contrast_starless_val) > 1e-4:
            if precomputed_mask is not None and np.any(precomputed_mask > 0.5):
                sky_median = float(np.median(stretched[precomputed_mask > 0.5]))
            else:
                sky_median = float(np.median(stretched))

            sn_contrasted = adjust_contrast(stretched, self.contrast_starless_val, pivot=sky_median)
            if precomputed_mask is not None:
                stretched = (sn_contrasted * precomputed_mask) + (stretched * (1.0 - precomputed_mask))
            else:
                stretched = sn_contrasted

        # 7. Contraste Global
        if abs(self.contrast_val) > 1e-4:
            stretched = adjust_contrast(stretched, self.contrast_val, pivot=None)

        return np.ascontiguousarray(stretched, dtype=np.float32)

    def update_stretch_preview(self):
        base_to_render = self._compose_active_base(for_export=False)
        if base_to_render is None:
            return

        if self.view_layer_mode == 2:  # Solo Estrellas
            stretched = manual_stretch(base_to_render, black_point=self.spin_bp.value(), midtone=self.spin_mtf.value())
        else:
            stretched = self._apply_pipeline_on_image(base_to_render, precomputed_mask=self.proxy_mask)

        if stretched is not None:
            self.canvas.load_image(base_to_render, display_stretched=stretched)
            # Actualiza el histograma de fondo de la curva
            self.curve_widget.set_histogram_from_image(stretched)

    # --- Callbacks StarNet ---
    def run_starnet(self):
        if self.image_32bit is None:
            QMessageBox.warning(self, "Aviso", "Carga o apila una imagen primero.")
            return

        self.btn_starnet.setEnabled(False)
        self.log_message("=== INICIANDO SEPARACIÓN STARNET++ AI ===")
        stride = self.spin_stride.value()

        self.sn_worker = StarNetWorker(
            self.image_32bit, 
            sky_mask=self.current_mask, 
            stride=stride
        )
        self.sn_worker.status_changed.connect(self.log_message)
        self.sn_worker.finished_success.connect(self.on_starnet_success)
        self.sn_worker.error_occurred.connect(self.on_starnet_error)
        self.sn_worker.start()

    def on_starnet_success(self, starless_img: np.ndarray, stars_img: np.ndarray):
        self.btn_starnet.setEnabled(True)
        self.image_starless = np.ascontiguousarray(starless_img, dtype=np.float32)
        self.image_stars = np.ascontiguousarray(stars_img, dtype=np.float32)

        self.combo_layer.setEnabled(True)
        self.slider_stars.setEnabled(True)
        self.slider_contrast_starless.setEnabled(True)  # <-- Activar aquí
        # Habilitar los nuevos controles
        #self.slider_sn_blacks.setEnabled(True)
        #self.slider_sn_shadows.setEnabled(True)
        #self.slider_sn_highlights.setEnabled(True)
        #self.slider_sn_whites.setEnabled(True)
        self.combo_denoise.setEnabled(True)
        self.slider_denoise.setEnabled(True)

        self._generate_preview_proxy()
        self.update_stretch_preview()
        self.log_message("[STARNET] Estrellas separadas en lineal. Capas activadas.")

    def on_starnet_error(self, err_msg: str):
        self.btn_starnet.setEnabled(True)
        self.log_message(f"[ERROR STARNET] {err_msg}")
        QMessageBox.warning(self, "Error en StarNet++", err_msg)

    def on_layer_mode_changed(self, idx: int):
        self.view_layer_mode = idx
        self.slider_stars.setEnabled(idx == 0)
        self.update_stretch_preview()

    def on_star_intensity_changed(self, val: int):
        self.star_intensity = val / 100.0
        self.lbl_star_intensity.setText(f"{val}%")
        self.update_stretch_preview()

    # --- Callbacks Balance de Blancos Fino (+-0.500) ---
    def on_temp_changed(self, val: int):
        self.temp_val = val / 1000.0
        self.lbl_temp_val.setText(f"{self.temp_val:+.3f}")
        self.update_stretch_preview()

    def on_tint_changed(self, val: int):
        self.tint_val = val / 1000.0
        self.lbl_tint_val.setText(f"{self.tint_val:+.3f}")
        self.update_stretch_preview()

    def reset_wb(self):
        self.slider_temp.blockSignals(True)
        self.slider_tint.blockSignals(True)
        self.slider_temp.setValue(0)
        self.slider_tint.setValue(0)
        self.temp_val = 0.0
        self.tint_val = 0.0
        self.lbl_temp_val.setText("0.000")
        self.lbl_tint_val.setText("0.000")
        self.slider_temp.blockSignals(False)
        self.slider_tint.blockSignals(False)
        self.update_stretch_preview()

    # --- Callbacks Color e Intensidad ---
    def on_sat_sky_changed(self, val: int):
        self.sat_sky_val = val / 100.0
        self.lbl_sat_sky.setText(f"{self.sat_sky_val:.2f}x")
        self.update_stretch_preview()

    def on_sat_gnd_changed(self, val: int):
        self.sat_gnd_val = val / 100.0
        self.lbl_sat_gnd.setText(f"{self.sat_gnd_val:.2f}x")
        self.update_stretch_preview()

    def on_vibrance_changed(self, val: int):
        self.vibrance_val = val / 100.0
        self.lbl_vibrance.setText(f"{self.vibrance_val:+.2f}")
        self.update_stretch_preview()

    def reset_saturation(self):
        self.slider_sat_sky.blockSignals(True)
        self.slider_sat_gnd.blockSignals(True)
        self.slider_vibrance.blockSignals(True)
        self.slider_sat_sky.setValue(100)
        self.slider_sat_gnd.setValue(100)
        self.slider_vibrance.setValue(0)
        self.sat_sky_val = 1.0
        self.sat_gnd_val = 1.0
        self.vibrance_val = 0.0
        self.lbl_sat_sky.setText("1.00x")
        self.lbl_sat_gnd.setText("1.00x")
        self.lbl_vibrance.setText("0.00")
        self.slider_sat_sky.blockSignals(False)
        self.slider_sat_gnd.blockSignals(False)
        self.slider_vibrance.blockSignals(False)
        self.update_stretch_preview()

    # --- Callbacks Tono y Contraste ---
    def on_contrast_changed(self, val: int):
        self.contrast_val = val / 100.0
        self.lbl_contrast.setText(f"{self.contrast_val:+.2f}")
        self.update_stretch_preview()

    def on_slider_bp_changed(self, val: int):
        bp = (val / 2000.0) * 0.20
        self.spin_bp.blockSignals(True)
        self.spin_bp.setValue(bp)
        self.spin_bp.blockSignals(False)
        self.update_stretch_preview()

    def on_spin_bp_changed(self, val: float):
        slider_val = int(np.clip((val / 0.20) * 2000.0, 0, 2000))
        self.slider_bp.blockSignals(True)
        self.slider_bp.setValue(slider_val)
        self.slider_bp.blockSignals(False)
        self.update_stretch_preview()

    def on_slider_mtf_changed(self, val: int):
        m = max(0.0001, (val / 2000.0) * 0.50)
        self.spin_mtf.blockSignals(True)
        self.spin_mtf.setValue(m)
        self.spin_mtf.blockSignals(False)
        self.update_stretch_preview()

    def on_spin_mtf_changed(self, val: float):
        slider_val = int(np.clip((val / 0.50) * 2000.0, 1, 2000))
        self.slider_mtf.blockSignals(True)
        self.slider_mtf.setValue(slider_val)
        self.slider_mtf.blockSignals(False)
        self.update_stretch_preview()

    def apply_auto_mtf(self):
        target = self.preview_proxy if self.preview_proxy is not None else self.image_32bit
        if target is None:
            return

        bp_suggested, m_suggested = calculate_mtf_params(target, target_background=0.20)
        self.sync_controls(bp_suggested, m_suggested)
        self.update_stretch_preview()
        self.log_message(f"Auto-MTF aplicado (Punto Negro: {bp_suggested:.5f}, MTF m: {m_suggested:.5f}).")

    def sync_controls(self, bp: float, m: float):
        self.spin_bp.blockSignals(True)
        self.slider_bp.blockSignals(True)
        self.spin_mtf.blockSignals(True)
        self.slider_mtf.blockSignals(True)

        self.spin_bp.setValue(bp)
        self.slider_bp.setValue(int(np.clip((bp / 0.20) * 2000.0, 0, 2000)))

        self.spin_mtf.setValue(m)
        self.slider_mtf.setValue(int(np.clip((m / 0.50) * 2000.0, 1, 2000)))

        self.spin_bp.blockSignals(False)
        self.slider_bp.blockSignals(False)
        self.spin_mtf.blockSignals(False)
        self.slider_mtf.blockSignals(False)

    def reset_sliders(self):
        self.slider_contrast.blockSignals(True)
        self.slider_contrast.setValue(0)
        self.contrast_val = 0.0
        self.lbl_contrast.setText("0.00")
        self.slider_contrast.blockSignals(False)

        if self.image_32bit is not None:
            self.apply_auto_mtf()
        else:
            self.sync_controls(0.0, 0.10)

    # --- GraXpert ---
    def run_graxpert(self):
        if self.image_32bit is None:
            QMessageBox.warning(self, "Aviso", "Carga o apila una imagen primero.")
            return

        self.btn_graxpert.setEnabled(False)
        self.log_message("=== INICIANDO GRAXPERT AI ===")
        smoothing = self.spin_smoothing.value()

        self.gx_worker = GraXpertWorker(self.image_32bit, sky_mask=self.current_mask, smoothing=smoothing)
        self.gx_worker.status_changed.connect(self.log_message)
        self.gx_worker.finished_success.connect(self.on_graxpert_success)
        self.gx_worker.error_occurred.connect(self.on_graxpert_error)
        self.gx_worker.start()

    def on_graxpert_success(self, corrected_img: np.ndarray):
        self.btn_graxpert.setEnabled(True)
        self.image_32bit = np.ascontiguousarray(corrected_img, dtype=np.float32)
        self.image_starless = None
        self.image_stars = None
        self.combo_layer.setEnabled(False)
        self.slider_stars.setEnabled(False)
        self._generate_preview_proxy()
        self.update_stretch_preview()
        self.slider_contrast_starless.setEnabled(False)
        self.slider_contrast_starless.setValue(0)
        self.contrast_starless_val = 0.0
        self.lbl_contrast_starless.setText("0.00")
        self.log_message("[GRAXPERT] Fondo neutralizado y aplicado.")
        #for s in (self.slider_sn_blacks, self.slider_sn_shadows,
        #          self.slider_sn_highlights, self.slider_sn_whites, self.slider_denoise):
        #    s.setEnabled(False)
        #    s.setValue(0)
        self.combo_denoise.setEnabled(False)

    def on_graxpert_error(self, err_msg: str):
        self.btn_graxpert.setEnabled(True)
        self.log_message(f"[ERROR GRAXPERT] {err_msg}")
        QMessageBox.warning(self, "Error en GraXpert", err_msg)

    def on_contrast_starless_changed(self, val: int):
        self.contrast_starless_val = val / 1000.0  # Mapea -1000..1000 a -1.000..+1.000
        self.lbl_contrast_starless.setText(f"{self.contrast_starless_val:+.3f}")
        self.update_stretch_preview()
    
    #def on_sn_blacks_changed(self, val: int):
    #    self.starless_blacks = val / 100.0
    #    self.lbl_sn_blacks.setText(f"{self.starless_blacks:+.2f}")
    #    self.update_stretch_preview()
    #
    #def on_sn_shadows_changed(self, val: int):
    #    self.starless_shadows = val / 100.0
    #    self.lbl_sn_shadows.setText(f"{self.starless_shadows:+.2f}")
    #    self.update_stretch_preview()
    #
    #def on_sn_highlights_changed(self, val: int):
    #    self.starless_highlights = val / 100.0
    #    self.lbl_sn_highlights.setText(f"{self.starless_highlights:+.2f}")
    #    self.update_stretch_preview()
    #
    #def on_sn_whites_changed(self, val: int):
    #    self.starless_whites = val / 100.0
    #    self.lbl_sn_whites.setText(f"{self.starless_whites:+.2f}")
    #    self.update_stretch_preview()

    def on_denoise_method_changed(self, text: str):
        self.denoise_method = text
        self.update_stretch_preview()

    def on_denoise_changed(self, val: int):
        self.denoise_strength = val / 100.0
        self.lbl_denoise.setText(f"{val}%")
        self.update_stretch_preview()
    
    # --- Exportación Multiformato en Resolución Completa ---
    def export_image(self):
        if self.image_32bit is None:
            return

        filtros = (
            "TIFF 32-bit Float (*.tif *.tiff);;"
            "TIFF 16-bit (*.tif *.tiff);;"
            "JPEG (*.jpg *.jpeg)"
        )

        p, selected_filter = QFileDialog.getSaveFileName(
            self, "Exportar Revelado", "revelado_final.tif", filtros
        )
        if not p:
            return

        h_full, w_full = self.image_32bit.shape[:2]
        self.log_message(f"Exportando imagen completa ({w_full}x{h_full} px)...")

        # 1. Composición de base nativa
        full_base = self._compose_active_base(for_export=True)

        # 2. Máscara diferencial completa
        full_mask = None
        if self.current_mask is not None and (self.sat_sky_val != self.sat_gnd_val):
            if self.current_mask.shape[:2] != (h_full, w_full):
                m_full = cv2.resize(self.current_mask, (w_full, h_full), interpolation=cv2.INTER_LINEAR)
            else:
                m_full = self.current_mask
            ksize = int(max(15, (min(h_full, w_full) // 150) | 1))
            if ksize % 2 == 0: ksize += 1
            sm = cv2.GaussianBlur(m_full, (ksize, ksize), sigmaX=ksize / 3.0)
            full_mask = np.ascontiguousarray(np.repeat(sm[..., np.newaxis], 3, axis=2), dtype=np.float32)

        # 3. Procesar cadena sobre la base compuesta
        processed = self._apply_pipeline_on_image(full_base, precomputed_mask=full_mask)

        # 4. Guardar archivo
        ext = os.path.splitext(p)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            bgr8 = cv2.cvtColor((processed * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(p, bgr8, [cv2.IMWRITE_JPEG_QUALITY, 96])
            desc = "JPEG 8-bit"
        elif "32-bit" in selected_filter:
            out32 = np.clip(processed, 0.0, 1.0).astype(np.float32)
            tifffile.imwrite(p, out32, compression='zlib', photometric='rgb')
            desc = "TIFF 32-bit float"
        else:
            u16 = (np.clip(processed, 0.0, 1.0) * 65535.0).astype(np.uint16)
            tifffile.imwrite(p, u16, compression='zlib', photometric='rgb')
            desc = "TIFF 16-bit"

        self.log_message(f"Imagen guardada en: {os.path.basename(p)} ({desc})")
        QMessageBox.information(
            self, "Exportación", 
            f"Guardada con éxito en resolución completa:\n{p}\n\nFormato: {desc} ({w_full}x{h_full} px)"
        )