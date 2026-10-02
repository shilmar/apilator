# gui/tab_developer.py
import os
import gc
import cv2
import numpy as np
import tifffile
from PySide6.QtWidgets import (
    QWidget, 
    QVBoxLayout, 
    QHBoxLayout, 
    QPushButton, 
    QLabel,
    QFileDialog, 
    QMessageBox, 
    QGroupBox, 
    QSlider, 
    QSplitter,
    QDoubleSpinBox, 
    QTextEdit, 
    QComboBox, 
    QSpinBox,
    QScrollArea, 
    QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from core.stacking import load_image_as_float32
from core.stretch import (
    calculate_mtf_params, 
    manual_stretch, 
    apply_white_balance, 
    adjust_saturation_dual, 
    extract_background_polynomial,
    adjust_vibrance, 
    adjust_contrast, 
    apply_light_pollution_gradient,
    apply_denoise,
    apply_clarity,
    apply_dehaze,
    apply_curve_lut,
    realce_multiescala_ondiculas
)
from gui.widgets.curve_widget import CurveWidget
from gui.canvas import MaskCanvas
from gui.worker import GraXpertWorker, StarNetWorker


class DeveloperTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_32bit = None       # Imagen nativa base (float32)
        self.image_starless = None    # Capa sin estrellas (float32)
        self.image_stars = None       # Capa de solo estrellas (float32)

        # Buffers proxy para previsualización interactiva a 60 fps
        self.preview_proxy = None     
        self.proxy_starless = None
        self.proxy_stars = None
        self.proxy_mask = None        

        self.current_mask = None      
        self.active_filepath = None
        self.gx_worker = None
        self.sn_worker = None
        
        self.is_zoomed_100 = False
        self.zoom_rel_coords = (0.5, 0.5)
        
        self.multiescala_estructura = 0.0   # -1.0 a +1.0
        self.multiescala_fondo = 0.0        # 0.0 a 1.0

        self.lp_reduction = 0.0  # Rango 0.0 a 1.0 (Reducción contaminación lumínica)
        self.clarity_starless_val = 0.0
        self.dehaze_starless_val = 0.0
        
        # Parámetros del revelador
        self.temp_val = 0.0          # Rango: -0.500 a +0.500
        self.tint_val = 0.0          # Rango: -0.500 a +0.500
        self.sat_sky_val = 1.0       # 0.00x a 2.50x
        self.sat_gnd_val = 1.0       # 0.00x a 2.50x
        self.vibrance_val = 0.0      # -1.00 a +1.00
        self.contrast_val = 0.0      # -1.00 a +1.00
        
        # Parámetros de Tono del Suelo 
        self.gnd_ev_val = 0.0        # -2.00 EV a +3.00 EV
        self.gnd_shadows_val = 0.0   # 0.0 a 1.0 (0% a 100%)
        self.gnd_bp_val = 0.0        # -0.050 a +0.100
        
        self.star_intensity = 1.0    # 1.0 = 100%, 0.0 = Starless puro
        self.view_layer_mode = 0     # 0: Compuesta, 1: Solo Fondo, 2: Solo Estrellas
        self.contrast_starless_val = 0.0  # -1.0 a +1.0 (solo para el fondo sin estrellas)

        # Parámetros de Reducción de Ruido
        #self.denoise_strength = 0.0
        self.denoise_method = "Bilateral"
        
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # 1. Visor interactivo
        self.canvas = MaskCanvas(self, enable_masking=False)
        self.canvas.external_zoom_handler = True
        self.canvas.zoom_toggled.connect(self.on_canvas_zoom_toggled)

        # 2. Panel interno de controles (irá dentro del scroll)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(6)

        # Carga externa
        btn_open_img = QPushButton("Abrir Imagen (TIFF / FITS / RAW)...")
        btn_open_img.setStyleSheet("font-weight: bold; padding: 6px;")
        btn_open_img.clicked.connect(self.open_image_file)
        left_layout.addWidget(btn_open_img)

        # --- Extracción de Fondo y Gradientes ---
        grp_bg = QGroupBox("Extracción de Fondo y Gradientes")
        layout_bg = QVBoxLayout(grp_bg)

        row_engine = QHBoxLayout()
        row_engine.addWidget(QLabel("Motor:"))
        self.combo_bg_engine = QComboBox()
        self.combo_bg_engine.addItems(["Polinómico (Paisaje)", "GraXpert AI"])
        row_engine.addWidget(self.combo_bg_engine)
        layout_bg.addLayout(row_engine)

        row_grax = QHBoxLayout()
        self.lbl_smoothing = QLabel("Suavizado:")
        row_grax.addWidget(self.lbl_smoothing)
        self.spin_smoothing = QDoubleSpinBox()
        self.spin_smoothing.setRange(0.1, 1.0)
        self.spin_smoothing.setSingleStep(0.05)
        self.spin_smoothing.setValue(0.50)
        row_grax.addWidget(self.spin_smoothing)

        self.btn_extract_bg = QPushButton("Eliminar Gradientes")
        self.btn_extract_bg.clicked.connect(self.on_extract_background_clicked)
        row_grax.addWidget(self.btn_extract_bg)
        layout_bg.addLayout(row_grax)

        left_layout.addWidget(grp_bg)

        # Control de Atenuación de Cúpula de Contaminación Lumínica
        grp_lp = QGroupBox("Contaminación Lumínica")
        layout_lp = QVBoxLayout(grp_lp)
        row_lp = QHBoxLayout()
        row_lp.addWidget(QLabel("Atenuar Cúpula de Luz:"))
        self.lbl_lp = QLabel("0%")
        row_lp.addWidget(self.lbl_lp)
        layout_lp.addLayout(row_lp)

        self.slider_lp = QSlider(Qt.Horizontal)
        self.slider_lp.setRange(0, 100)
        self.slider_lp.setValue(0)
        self.slider_lp.valueChanged.connect(self.on_lp_changed)
        layout_lp.addWidget(self.slider_lp)
        left_layout.addWidget(grp_lp)

        # --- StarNet++ AI Control de Estrellas ---
        grp_starnet = QGroupBox("StarNet++ AI - Control de Estrellas")
        layout_starnet = QVBoxLayout(grp_starnet)

        row_sn = QHBoxLayout()
        row_sn.addWidget(QLabel("Paso:"))
        self.spin_stride = QSpinBox()
        self.spin_stride.setRange(64, 512)
        self.spin_stride.setSingleStep(64)
        self.spin_stride.setValue(256)
        row_sn.addWidget(self.spin_stride)

        self.btn_starnet = QPushButton("Separar Estrellas con StarNet")
        self.btn_starnet.clicked.connect(self.run_starnet)
        row_sn.addWidget(self.btn_starnet)
        layout_starnet.addLayout(row_sn)

        row_view = QHBoxLayout()
        row_view.addWidget(QLabel("Capa visible:"))
        self.combo_layer = QComboBox()
        self.combo_layer.addItems(["Compuesta (Normal)", "Solo Fondo (Starless)", "Solo Estrellas"])
        self.combo_layer.setEnabled(False)
        self.combo_layer.currentIndexChanged.connect(self.on_layer_mode_changed)
        row_view.addWidget(self.combo_layer)
        layout_starnet.addLayout(row_view)

        row_stars = QHBoxLayout()
        row_stars.addWidget(QLabel("Intensidad Estrellas:"))
        self.lbl_stars = QLabel("100%")
        row_stars.addWidget(self.lbl_stars)
        layout_starnet.addLayout(row_stars)

        self.slider_stars = QSlider(Qt.Horizontal)
        self.slider_stars.setRange(0, 200)
        self.slider_stars.setValue(100)
        self.slider_stars.setEnabled(False)
        self.slider_stars.valueChanged.connect(self.on_stars_slider_changed)
        layout_starnet.addWidget(self.slider_stars)

        row_cs = QHBoxLayout()
        row_cs.addWidget(QLabel("Contraste Fondo (Starless):"))
        self.lbl_contrast_starless = QLabel("0.00")
        row_cs.addWidget(self.lbl_contrast_starless)
        layout_starnet.addLayout(row_cs)

        self.slider_contrast_starless = QSlider(Qt.Horizontal)
        self.slider_contrast_starless.setRange(-100, 100)
        self.slider_contrast_starless.setValue(0)
        self.slider_contrast_starless.setEnabled(False)
        self.slider_contrast_starless.valueChanged.connect(self.on_contrast_starless_changed)
        layout_starnet.addWidget(self.slider_contrast_starless)

        row_clarity = QHBoxLayout()
        row_clarity.addWidget(QLabel("Claridad Fondo:"))
        self.lbl_clarity = QLabel("0.00")
        row_clarity.addWidget(self.lbl_clarity)
        layout_starnet.addLayout(row_clarity)

        self.slider_clarity = QSlider(Qt.Horizontal)
        self.slider_clarity.setRange(-200, 200)
        self.slider_clarity.setValue(0)
        self.slider_clarity.setEnabled(False)
        self.slider_clarity.valueChanged.connect(self.on_clarity_changed)
        layout_starnet.addWidget(self.slider_clarity)

        row_dehaze = QHBoxLayout()
        row_dehaze.addWidget(QLabel("Borrar Neblina:"))
        self.lbl_dehaze = QLabel("0.00")
        row_dehaze.addWidget(self.lbl_dehaze)
        layout_starnet.addLayout(row_dehaze)

        self.slider_dehaze = QSlider(Qt.Horizontal)
        self.slider_dehaze.setRange(-200, 200)
        self.slider_dehaze.setValue(0)
        self.slider_dehaze.setEnabled(False)
        self.slider_dehaze.valueChanged.connect(self.on_dehaze_changed)
        layout_starnet.addWidget(self.slider_dehaze)

        left_layout.addWidget(grp_starnet)
        
        # --- Realce Multiescala por Ondículas ---
        grp_ondiculas = QGroupBox("Estructura Multiescala (Ondículas)")
        layout_ondiculas = QVBoxLayout(grp_ondiculas)

        row_ond_est = QHBoxLayout()
        row_ond_est.addWidget(QLabel("Estructura Galáctica:"))
        self.lbl_ond_est = QLabel("0.00")
        row_ond_est.addWidget(self.lbl_ond_est)
        layout_ondiculas.addLayout(row_ond_est)

        self.slider_ond_est = QSlider(Qt.Horizontal)
        self.slider_ond_est.setRange(-100, 100)
        self.slider_ond_est.setValue(0)
        self.slider_ond_est.setEnabled(False)
        self.slider_ond_est.valueChanged.connect(self.on_ond_estructura_changed)
        layout_ondiculas.addWidget(self.slider_ond_est)

        row_ond_bg = QHBoxLayout()
        row_ond_bg.addWidget(QLabel("Atenuar Fondo Residual:"))
        self.lbl_ond_bg = QLabel("0%")
        row_ond_bg.addWidget(self.lbl_ond_bg)
        layout_ondiculas.addLayout(row_ond_bg)

        self.slider_ond_bg = QSlider(Qt.Horizontal)
        self.slider_ond_bg.setRange(0, 100)
        self.slider_ond_bg.setValue(0)
        self.slider_ond_bg.setEnabled(False)
        self.slider_ond_bg.valueChanged.connect(self.on_ond_fondo_changed)
        layout_ondiculas.addWidget(self.slider_ond_bg)

        left_layout.addWidget(grp_ondiculas)
        
        # --- Reducción de Ruido (Fondo) ---
        grp_dn = QGroupBox("Reducción de Ruido (Fondo)")
        layout_dn = QVBoxLayout(grp_dn)

        row_dn = QHBoxLayout()
        row_dn.addWidget(QLabel("Fuerza Denoise:"))
        self.lbl_denoise = QLabel("0%")
        row_dn.addWidget(self.lbl_denoise)
        layout_dn.addLayout(row_dn)
        
        self.slider_denoise = QSlider(Qt.Horizontal)
        self.slider_denoise.setRange(0, 100)
        self.slider_denoise.setValue(0)
        self.slider_denoise.setEnabled(False)
        self.slider_denoise.valueChanged.connect(self.on_denoise_changed)
        layout_dn.addWidget(self.slider_denoise)

        left_layout.addWidget(grp_dn)
        
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

        left_layout.addWidget(grp_curves)
              
        # Módulo Balance de Blancos
        grp_wb = QGroupBox("Balance de Blancos (Precisión Fina)")
        wb_layout = QVBoxLayout(grp_wb)
        wb_layout.setSpacing(3)

        row_temp = QHBoxLayout()
        row_temp.addWidget(QLabel("Temp (Frío / Cálido):"))
        self.lbl_temp_val = QLabel("0.000")
        row_temp.addWidget(self.lbl_temp_val)
        wb_layout.addLayout(row_temp)

        self.slider_temp = QSlider(Qt.Horizontal)
        self.slider_temp.setRange(-250, 250)
        self.slider_temp.setValue(0)
        self.slider_temp.valueChanged.connect(self.on_temp_changed)
        wb_layout.addWidget(self.slider_temp)

        row_tint = QHBoxLayout()
        row_tint.addWidget(QLabel("Tinte (Verde / Magenta):"))
        self.lbl_tint_val = QLabel("0.000")
        row_tint.addWidget(self.lbl_tint_val)
        wb_layout.addLayout(row_tint)

        self.slider_tint = QSlider(Qt.Horizontal)
        self.slider_tint.setRange(-250, 250)
        self.slider_tint.setValue(0)
        self.slider_tint.valueChanged.connect(self.on_tint_changed)
        wb_layout.addWidget(self.slider_tint)

        btn_reset_wb = QPushButton("Restablecer Balance")
        btn_reset_wb.clicked.connect(self.reset_wb)
        wb_layout.addWidget(btn_reset_wb)
        left_layout.addWidget(grp_wb)

        # --- NUEVO: Módulo Ajuste Tonal de Suelo ---
        grp_gnd_tone = QGroupBox("Ajuste Tonal Suelo (Paisaje)")
        gnd_tone_layout = QVBoxLayout(grp_gnd_tone)
        gnd_tone_layout.setSpacing(3)

        # 1. Exposición Suelo (EV)
        row_gnd_ev = QHBoxLayout()
        row_gnd_ev.addWidget(QLabel("Exposición Suelo:"))
        self.lbl_gnd_ev = QLabel("0.00 EV")
        row_gnd_ev.addWidget(self.lbl_gnd_ev)
        gnd_tone_layout.addLayout(row_gnd_ev)

        self.slider_gnd_ev = QSlider(Qt.Horizontal)
        self.slider_gnd_ev.setRange(-200, 300)  # -2.00 EV a +3.00 EV
        self.slider_gnd_ev.setValue(0)
        self.slider_gnd_ev.valueChanged.connect(self.on_gnd_ev_changed)
        gnd_tone_layout.addWidget(self.slider_gnd_ev)

        # 2. Recuperar Sombras Suelo
        row_gnd_sh = QHBoxLayout()
        row_gnd_sh.addWidget(QLabel("Recuperar Sombras:"))
        self.lbl_gnd_sh = QLabel("0%")
        row_gnd_sh.addWidget(self.lbl_gnd_sh)
        gnd_tone_layout.addLayout(row_gnd_sh)

        self.slider_gnd_sh = QSlider(Qt.Horizontal)
        self.slider_gnd_sh.setRange(0, 100)  # 0% a 100%
        self.slider_gnd_sh.setValue(0)
        self.slider_gnd_sh.valueChanged.connect(self.on_gnd_shadows_changed)
        gnd_tone_layout.addWidget(self.slider_gnd_sh)

        # 3. Punto Negro Suelo
        row_gnd_bp = QHBoxLayout()
        row_gnd_bp.addWidget(QLabel("Punto Negro Suelo:"))
        self.lbl_gnd_bp = QLabel("0.000")
        row_gnd_bp.addWidget(self.lbl_gnd_bp)
        gnd_tone_layout.addLayout(row_gnd_bp)

        self.slider_gnd_bp = QSlider(Qt.Horizontal)
        self.slider_gnd_bp.setRange(-50, 100)  # -0.050 a +0.100
        self.slider_gnd_bp.setValue(0)
        self.slider_gnd_bp.valueChanged.connect(self.on_gnd_bp_changed)
        gnd_tone_layout.addWidget(self.slider_gnd_bp)

        btn_reset_gnd = QPushButton("Restablecer Tono Suelo")
        btn_reset_gnd.clicked.connect(self.reset_gnd_tone)
        gnd_tone_layout.addWidget(btn_reset_gnd)

        left_layout.addWidget(grp_gnd_tone)

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
        left_layout.addStretch()

        # Configuración Scroll para la zona de controles
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setWidget(left_panel)
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

        # Contenedor Izquierdo Principal (Scroll de controles arriba + Log estático abajo)
        left_container = QWidget()
        left_container_layout = QVBoxLayout(left_container)
        left_container_layout.setContentsMargins(0, 0, 0, 0)
        left_container_layout.setSpacing(6)
        left_container.setMinimumWidth(380)
        left_container.setMaximumWidth(520)

        # Scroll arriba con factor de expansión 1
        left_container_layout.addWidget(scroll_area, 1)

        # Log fijo abajo (sin factor de expansión)
        lbl_log = QLabel("Registro del Revelador:")
        lbl_log.setStyleSheet("font-weight: bold; font-size: 11px; margin-left: 10px; margin-top: 4px;")
        left_container_layout.addWidget(lbl_log, 0)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(150)
        self.txt_log.setStyleSheet(
            "background-color: #141414; color: #d0d0d0; "
            "font-family: Consolas, monospace; font-size: 11px; "
            "border: 1px solid #333333; border-radius: 4px; padding: 4px; "
            "margin-left: 10px; margin-right: 10px; margin-bottom: 10px;"
        )
        left_container_layout.addWidget(self.txt_log, 0)

        splitter.addWidget(left_container)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 10)
        
        splitter.setSizes([450, 1400])
        
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

        clean_mask = None
        if self.current_mask is not None:
            clean_mask = np.squeeze(self.current_mask)
            if clean_mask.ndim > 2:
                clean_mask = clean_mask[..., 0]
            clean_mask = clean_mask.astype(np.float32)
            if clean_mask.shape != (h, w):
                clean_mask = cv2.resize(clean_mask, (w, h), interpolation=cv2.INTER_LINEAR)

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

            if clean_mask is not None:
                m_small = cv2.resize(clean_mask, (pw, ph), interpolation=cv2.INTER_LINEAR)
                ksize = int(max(7, (min(ph, pw) // 150) | 1))
                if ksize % 2 == 0: 
                    ksize += 1
                sm = cv2.GaussianBlur(m_small, (ksize, ksize), sigmaX=ksize / 3.0)
                self.proxy_mask = np.ascontiguousarray(np.repeat(sm[..., np.newaxis], 3, axis=2), dtype=np.float32)
            else:
                self.proxy_mask = None
        else:
            self.preview_proxy = np.ascontiguousarray(self.image_32bit, dtype=np.float32)
            self.proxy_starless = self.image_starless.copy() if self.image_starless is not None else None
            self.proxy_stars = self.image_stars.copy() if self.image_stars is not None else None
            if clean_mask is not None:
                ksize = int(max(15, (min(h, w) // 150) | 1))
                if ksize % 2 == 0: 
                    ksize += 1
                sm = cv2.GaussianBlur(clean_mask, (ksize, ksize), sigmaX=ksize / 3.0)
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
        
        self.reset_all_parameters()
        
        self.log_message(f"Cargando imagen: {os.path.basename(filepath)}...")
        self.slider_contrast_starless.setEnabled(False)
        self.slider_contrast_starless.setValue(0)
        self.contrast_starless_val = 0.0
        self.lbl_contrast_starless.setText("0.00")        
        if hasattr(self, 'curve_widget'):
            self.curve_widget.reset_curve()
        #self.combo_denoise.setEnabled(False)
        self.slider_ond_est.setEnabled(False)
        self.slider_ond_bg.setEnabled(False)
        
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
    def _compose_active_base(self, for_export: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Retorna (base_a_procesar, mascara_correspondiente).
        Si no es exportación y el zoom 100% está activo, recorta la región nativa
        del sensor que cabe en el tamaño actual del canvas.
        """
        if for_export:
            starless, stars, base = self.image_starless, self.image_stars, self.image_32bit
            mask_3d = None
            if self.current_mask is not None and base is not None:
                h_f, w_f = base.shape[:2]
                m_f = cv2.resize(self.current_mask, (w_f, h_f), interpolation=cv2.INTER_LINEAR) if self.current_mask.shape[:2] != (h_f, w_f) else self.current_mask
                mask_3d = np.repeat(m_f[..., np.newaxis], 3, axis=2)
            crop_active = False
        elif self.is_zoomed_100 and self.image_32bit is not None:
            # Recorte nativo 1:1 directo de la imagen de resolución completa
            h_f, w_f = self.image_32bit.shape[:2]
            cw = min(w_f, self.canvas.width())
            ch = min(h_f, self.canvas.height())
            
            rx, ry = self.zoom_rel_coords
            cx = int(rx * w_f)
            cy = int(ry * h_f)

            x1 = max(0, min(w_f - cw, cx - cw // 2))
            y1 = max(0, min(h_f - ch, cy - ch // 2))
            x2 = x1 + cw
            y2 = y1 + ch

            base = self.image_32bit[y1:y2, x1:x2].copy()
            starless = self.image_starless[y1:y2, x1:x2].copy() if self.image_starless is not None else None
            stars = self.image_stars[y1:y2, x1:x2].copy() if self.image_stars is not None else None

            if self.current_mask is not None:
                m_sub = self.current_mask[y1:y2, x1:x2]
                mask_3d = np.repeat(m_sub[..., np.newaxis], 3, axis=2)
            else:
                mask_3d = None
            crop_active = True
        else:
            # Vista general usando el proxy optimizado
            starless, stars, base = self.proxy_starless, self.proxy_stars, self.preview_proxy
            mask_3d = self.proxy_mask
            crop_active = False

        if base is None:
            return None, None

        if starless is None or stars is None:
            return base, mask_3d

        processed_starless = starless.copy()

        # 1. Reducción de Ruido
        if self.denoise_strength > 1e-4:
            denoised = apply_denoise(
                processed_starless,
                strength=self.denoise_strength,
                is_full_res=(for_export or crop_active)
            )
            if mask_3d is not None:
                processed_starless = (denoised * mask_3d) + (processed_starless * (1.0 - mask_3d))
            else:
                processed_starless = denoised

        # 2. Claridad y Borrar Neblina
        if abs(self.clarity_starless_val) > 1e-4:
            clarified = apply_clarity(processed_starless, strength=self.clarity_starless_val)
            if mask_3d is not None:
                processed_starless = (clarified * mask_3d) + (processed_starless * (1.0 - mask_3d))
            else:
                processed_starless = clarified

        if abs(self.dehaze_starless_val) > 1e-4:
            dehazed = apply_dehaze(processed_starless, strength=self.dehaze_starless_val)
            if mask_3d is not None:
                processed_starless = (dehazed * mask_3d) + (processed_starless * (1.0 - mask_3d))
            else:
                processed_starless = dehazed

        # 3. Ondículas sobre la capa de fondo
        if abs(self.multiescala_estructura) > 1e-4 or self.multiescala_fondo > 1e-4:
            processed_starless = realce_multiescala_ondiculas(
                processed_starless,
                fuerza_estructura=self.multiescala_estructura,
                reduccion_fondo=self.multiescala_fondo,
                sky_mask=mask_3d[..., 0] if mask_3d is not None else None
            )

        # 4. Mezcla de capas
        if self.view_layer_mode == 0:
            composed = np.clip(processed_starless + (stars * self.star_intensity), 0.0, 1.0)
            return composed, mask_3d
        elif self.view_layer_mode == 1:
            return processed_starless, mask_3d
        else:
            return stars, mask_3d

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

        # Atenuación de Cúpula de Luz
        if self.lp_reduction > 1e-4:
            mask_2d = precomputed_mask[..., 0] if precomputed_mask is not None else None
            img = apply_light_pollution_gradient(img, strength=self.lp_reduction, height_ratio=0.50, sky_mask=mask_2d)

        # === NUEVO: Ajuste Tonal Diferencial para el Suelo ===
        if precomputed_mask is not None:
            has_gnd_tone_change = (
                abs(self.gnd_ev_val) > 1e-4 or 
                self.gnd_shadows_val > 1e-4 or 
                abs(self.gnd_bp_val) > 1e-4
            )
            if has_gnd_tone_change:
                # Extraemos el suelo actual
                gnd_part = img.copy()

                # a) Exposición diferencial en escala EV
                if abs(self.gnd_ev_val) > 1e-4:
                    gnd_part = gnd_part * (2.0 ** self.gnd_ev_val)

                # b) Recuperación de sombras suave (sin quemar medios tonos ni altas luces)
                if self.gnd_shadows_val > 1e-4:
                    lift_curve = (1.0 - np.clip(gnd_part, 0.0, 1.0)) ** 2
                    gnd_part = gnd_part * (1.0 + self.gnd_shadows_val * lift_curve)

                # c) Ajuste de punto negro / pedestal específico para el suelo
                if abs(self.gnd_bp_val) > 1e-4:
                    if self.gnd_bp_val < 0.0:
                        # Hacia la derecha (gnd_bp_val < 0): añade pedestal de luz a las sombras
                        gnd_part = np.clip(gnd_part - self.gnd_bp_val, 0.0, 1.0)
                    else:
                        # Hacia la izquierda (gnd_bp_val > 0): recorta/oscurece sombras
                        gnd_part = np.clip((gnd_part - self.gnd_bp_val) / max(1e-4, 1.0 - self.gnd_bp_val), 0.0, 1.0)

                # Fusionamos respetando la máscara: precomputed_mask=1 (cielo), 0 (suelo)
                img = (img * precomputed_mask) + (gnd_part * (1.0 - precomputed_mask))

        # 4. Estirado MTF (espacio visible [0.0, 1.0])
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
        base_to_render, active_mask = self._compose_active_base(for_export=False)
        if base_to_render is None:
            return

        if self.view_layer_mode == 2:
            stretched = manual_stretch(base_to_render, black_point=self.spin_bp.value(), midtone=self.spin_mtf.value())
        else:
            stretched = self._apply_pipeline_on_image(base_to_render, precomputed_mask=active_mask)

        if stretched is not None:
            self.canvas.load_image(base_to_render, display_stretched=stretched)
            # Solo actualizamos histograma en vista completa para no sesgarlo por el recorte
            if not self.is_zoomed_100:
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
        self.slider_contrast_starless.setEnabled(True) 
        self.slider_clarity.setEnabled(True)
        self.slider_dehaze.setEnabled(True)
        #self.combo_denoise.setEnabled(True)
        self.slider_denoise.setEnabled(True)
        self.slider_ond_est.setEnabled(True)
        self.slider_ond_bg.setEnabled(True)

        self._generate_preview_proxy()
        self.update_stretch_preview()
        self.log_message("[STARNET] Estrellas separadas en lineal. Capas activadas.")
        
    def on_stars_slider_changed(self, val: int):
        self.star_intensity = val / 100.0
        self.lbl_stars.setText(f"{val}%")
        self.update_stretch_preview()

    def on_starnet_error(self, err_msg: str):
        self.btn_starnet.setEnabled(True)
        self.log_message(f"[ERROR STARNET] {err_msg}")
        QMessageBox.warning(self, "Error en StarNet++", err_msg)

    def on_layer_mode_changed(self, idx: int):
        self.view_layer_mode = idx
        self.slider_stars.setEnabled(idx == 0)
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
            
    def on_canvas_zoom_toggled(self, is_zoomed: bool, rel_x: float, rel_y: float):
        self.is_zoomed_100 = is_zoomed
        if is_zoomed:
            self.zoom_rel_coords = (rel_x, rel_y)
        self.update_stretch_preview()

    # --- Extracción de Fondo ---
    def on_extract_background_clicked(self):
        if self.image_32bit is None:
            return

        engine = self.combo_bg_engine.currentText()

        if "Polinómico" in engine:
            self.log_message("[FONDO] Extrayendo gradiente con modelo Polinómico Cuadrático...")
            try:
                corrected = extract_background_polynomial(
                    self.image_32bit, 
                    sky_mask=self.current_mask, 
                    degree=2
                )
                self.on_bg_extraction_success(corrected)
                self.log_message("[FONDO] Gradiente polinómico neutralizado con éxito.")
            except Exception as e:
                self.log_message(f"[ERROR FONDO] Fallo en cálculo polinómico: {e}")
        else:
            self.btn_extract_bg.setEnabled(False)
            self.log_message("=== INICIANDO GRAXPERT ===")
            self.gx_worker = GraXpertWorker(
                self.image_32bit,
                sky_mask=self.current_mask,
                smoothing=self.spin_smoothing.value()
            )
            self.gx_worker.status_changed.connect(self.log_message)
            self.gx_worker.finished_success.connect(self.on_bg_extraction_success)
            self.gx_worker.error_occurred.connect(self.on_bg_extraction_error)
            self.gx_worker.start()

    def on_bg_extraction_success(self, corrected_img: np.ndarray):
        self.btn_extract_bg.setEnabled(True)
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
        self.slider_denoise.setEnabled(False)
        self.slider_denoise.setValue(0)
        #self.combo_denoise.setEnabled(False)
        self.slider_ond_est.setEnabled(False)
        self.slider_ond_bg.setEnabled(False)

    def on_bg_extraction_error(self, err_msg: str):
        self.btn_extract_bg.setEnabled(True)
        self.log_message(f"[ERROR GRAXPERT] {err_msg}")
        QMessageBox.warning(self, "Error en GraXpert", err_msg)

    def on_lp_changed(self, val: int):
        self.lp_reduction = val / 100.0
        self.lbl_lp.setText(f"{val}%")
        self.update_stretch_preview()

    def on_contrast_starless_changed(self, val: int):
        self.contrast_starless_val = val / 100.0
        self.lbl_contrast_starless.setText(f"{self.contrast_starless_val:+.2f}")
        self.update_stretch_preview()
        
    def on_clarity_changed(self, val: int):
        self.clarity_starless_val = val / 100.0  # Mapea -200..200 a -2.00..+2.00
        self.lbl_clarity.setText(f"{self.clarity_starless_val:+.2f}")
        self.update_stretch_preview()

    def on_dehaze_changed(self, val: int):
        self.dehaze_starless_val = val / 100.0  # Mapea -200..200 a -2.00..+2.00
        self.lbl_dehaze.setText(f"{self.dehaze_starless_val:+.2f}")
        self.update_stretch_preview()
    
    #def on_denoise_method_changed(self, text: str):
    #    self.denoise_method = text
    #    self.update_stretch_preview()

    def on_denoise_changed(self, val: int):
        self.denoise_strength = val / 100.0
        self.lbl_denoise.setText(f"{val}%")
        self.update_stretch_preview()
        
    def on_ond_estructura_changed(self, val: int):
        self.multiescala_estructura = val / 100.0
        self.lbl_ond_est.setText(f"{self.multiescala_estructura:+.2f}")
        self.update_stretch_preview()

    def on_ond_fondo_changed(self, val: int):
        self.multiescala_fondo = val / 100.0
        self.lbl_ond_bg.setText(f"{val}%")
        self.update_stretch_preview()
    
    # --- Callbacks Ajuste Tonal Suelo ---
    def on_gnd_ev_changed(self, val: int):
        self.gnd_ev_val = val / 100.0
        self.lbl_gnd_ev.setText(f"{self.gnd_ev_val:+.2f} EV")
        self.update_stretch_preview()

    def on_gnd_shadows_changed(self, val: int):
        self.gnd_shadows_val = val / 100.0
        self.lbl_gnd_sh.setText(f"{val}%")
        self.update_stretch_preview()

    def on_gnd_bp_changed(self, val: int):
        self.gnd_bp_val = - (val / 1000.0)
        display_val = val / 1000.0
        self.lbl_gnd_bp.setText(f"{display_val:+.3f}")
        self.update_stretch_preview()

    def reset_gnd_tone(self):
        self.slider_gnd_ev.blockSignals(True)
        self.slider_gnd_sh.blockSignals(True)
        self.slider_gnd_bp.blockSignals(True)

        self.slider_gnd_ev.setValue(0)
        self.slider_gnd_sh.setValue(0)
        self.slider_gnd_bp.setValue(0)

        self.gnd_ev_val = 0.0
        self.gnd_shadows_val = 0.0
        self.gnd_bp_val = 0.0

        self.lbl_gnd_ev.setText("0.00 EV")
        self.lbl_gnd_sh.setText("0%")
        self.lbl_gnd_bp.setText("0.000")

        self.slider_gnd_ev.blockSignals(False)
        self.slider_gnd_sh.blockSignals(False)
        self.slider_gnd_bp.blockSignals(False)

        self.update_stretch_preview()
    
    # --- Exportación Multiformato en Resolución Completa ---
    def export_image(self):
        if self.image_32bit is None:
            return

        filtros = (
            "TIFF 16-bit (*.tif *.tiff);;"
            "TIFF 32-bit Float (*.tif *.tiff);;"
            "JPEG (*.jpg *.jpeg)"
        )

        p, selected_filter = QFileDialog.getSaveFileName(
            self, "Exportar Revelado", "revelado_final.tif", filtros
        )
        if not p:
            return

        p = os.path.abspath(os.path.normpath(str(p).strip()))

        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError as err:
                QMessageBox.critical(
                    self, "Error de Exportación",
                    f"No se puede sobreescribir el archivo destino porque está abierto en otro programa:\n{p}\n\nDetalle: {err}"
                )
                return

        h_full, w_full = self.image_32bit.shape[:2]
        self.log_message(f"Exportando imagen completa ({w_full}x{h_full} px)...")

        #full_base = self._compose_active_base(for_export=True)
        full_base, _ = self._compose_active_base(for_export=True)

        full_mask = None
        if self.current_mask is not None:
            clean_m = np.squeeze(self.current_mask)
            if clean_m.ndim > 2:
                clean_m = clean_m[..., 0]
            clean_m = clean_m.astype(np.float32)

            if clean_m.shape != (h_full, w_full):
                clean_m = cv2.resize(clean_m, (w_full, h_full), interpolation=cv2.INTER_LINEAR)

            ksize = int(max(15, (min(h_full, w_full) // 150) | 1))
            if ksize % 2 == 0:
                ksize += 1
            sm = cv2.GaussianBlur(clean_m, (ksize, ksize), sigmaX=ksize / 3.0)
            full_mask = np.ascontiguousarray(np.repeat(sm[..., np.newaxis], 3, axis=2), dtype=np.float32)

        processed = self._apply_pipeline_on_image(full_base, precomputed_mask=full_mask)

        ext = os.path.splitext(p)[1].lower()
        try:
            if ext in ['.jpg', '.jpeg']:
                bgr8 = cv2.cvtColor((np.clip(processed, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(p, bgr8, [cv2.IMWRITE_JPEG_QUALITY, 96])
                desc = "JPEG 8-bit"
            elif "32-bit" in selected_filter:
                out32 = np.ascontiguousarray(np.clip(processed, 0.0, 1.0), dtype=np.float32)
                tifffile.imwrite(p, out32, compression='zlib', photometric='rgb')
                desc = "TIFF 32-bit float"
            else:
                u16 = np.ascontiguousarray((np.clip(processed, 0.0, 1.0) * 65535.0), dtype=np.uint16)
                tifffile.imwrite(p, u16, compression='zlib', photometric='rgb')
                desc = "TIFF 16-bit"

            self.log_message(f"Imagen guardada en: {os.path.basename(p)} ({desc})")
            QMessageBox.information(
                self, "Exportación",
                f"Guardada con éxito en resolución completa:\n{p}\n\nFormato: {desc} ({w_full}x{h_full} px)"
            )
        except Exception as e:
            self.log_message(f"[ERROR EXPORT] Fallo al escribir el archivo: {e}")
            QMessageBox.critical(self, "Error de Exportación", f"Fallo al escribir el archivo:\n{e}")
            
    def reset_all_parameters(self):
        """Devuelve todos los controles y variables del revelador a su estado neutro original."""
        # 1. Variables internas
        self.lp_reduction = 0.0
        self.clarity_starless_val = 0.0
        self.dehaze_starless_val = 0.0
        self.multiescala_estructura = 0.0
        self.multiescala_fondo = 0.0
        self.temp_val = 0.0
        self.tint_val = 0.0
        self.sat_sky_val = 1.0
        self.sat_gnd_val = 1.0
        self.vibrance_val = 0.0
        self.contrast_val = 0.0
        self.contrast_starless_val = 0.0
        self.star_intensity = 1.0
        self.view_layer_mode = 0
        self.denoise_strength = 0.0
        self.gnd_ev_val = 0.0
        self.gnd_shadows_val = 0.0
        self.gnd_bp_val = 0.0
        #self.denoise_method = "Bilateral"

        # 2. Bloquear señales de la UI para evitar recálculos en cadena
        widgets_to_block = [
            self.slider_lp, self.slider_stars, self.slider_contrast_starless,
            self.slider_clarity, self.slider_dehaze, self.slider_ond_est,
            self.slider_ond_bg, self.slider_denoise, 
            self.slider_temp, self.slider_tint, self.slider_sat_sky,
            self.slider_sat_gnd, self.slider_vibrance, self.slider_contrast,
            self.combo_layer, self.spin_bp, self.slider_bp,
            self.spin_mtf, self.slider_mtf,
            self.slider_gnd_ev, self.slider_gnd_sh, self.slider_gnd_bp,
        ]
        #self.combo_denoise,
        for w in widgets_to_block:
            w.blockSignals(True)

        # 3. Restaurar valores y etiquetas de la UI
        self.slider_lp.setValue(0)
        self.lbl_lp.setText("0%")

        self.combo_layer.setCurrentIndex(0)
        self.combo_layer.setEnabled(False)

        self.slider_stars.setValue(100)
        self.lbl_stars.setText("100%")
        self.slider_stars.setEnabled(False)

        self.slider_contrast_starless.setValue(0)
        self.lbl_contrast_starless.setText("0.00")
        self.slider_contrast_starless.setEnabled(False)

        self.slider_clarity.setValue(0)
        self.lbl_clarity.setText("0.00")
        self.slider_clarity.setEnabled(False)

        self.slider_dehaze.setValue(0)
        self.lbl_dehaze.setText("0.00")
        self.slider_dehaze.setEnabled(False)

        self.slider_ond_est.setValue(0)
        self.lbl_ond_est.setText("0.00")
        self.slider_ond_est.setEnabled(False)

        self.slider_ond_bg.setValue(0)
        self.lbl_ond_bg.setText("0%")
        self.slider_ond_bg.setEnabled(False)

        #self.combo_denoise.setCurrentIndex(0)
        #self.combo_denoise.setEnabled(False)
        self.slider_denoise.setValue(0)
        self.lbl_denoise.setText("0%")
        self.slider_denoise.setEnabled(False)

        if hasattr(self, 'curve_widget'):
            self.curve_widget.reset_curve()

        self.slider_temp.setValue(0)
        self.lbl_temp_val.setText("0.000")
        self.slider_tint.setValue(0)
        self.lbl_tint_val.setText("0.000")

        self.slider_sat_sky.setValue(100)
        self.lbl_sat_sky.setText("1.00x")
        self.slider_sat_gnd.setValue(100)
        self.lbl_sat_gnd.setText("1.00x")

        self.slider_vibrance.setValue(0)
        self.lbl_vibrance.setText("0.00")

        self.slider_contrast.setValue(0)
        self.lbl_contrast.setText("0.00")

        self.slider_gnd_ev.setValue(0)
        self.lbl_gnd_ev.setText("0.00 EV")
        self.slider_gnd_sh.setValue(0)
        self.lbl_gnd_sh.setText("0%")
        self.slider_gnd_bp.setValue(0)
        self.lbl_gnd_bp.setText("0.000")

        self.spin_bp.setValue(0.0)
        self.slider_bp.setValue(0)
        self.spin_mtf.setValue(0.10)
        self.slider_mtf.setValue(400)
        
        self.is_zoomed_100 = False
        self.zoom_rel_coords = (0.5, 0.5)
        if hasattr(self, 'canvas'):
            self.canvas.is_zoomed = False

        # 4. Desbloquear señales
        for w in widgets_to_block:
            w.blockSignals(False)
            
    def clear_session(self):
        """
        Descarga por completo la sesión del revelador:
        vacía buffers de 32 bits, proxies, máscaras, canvas y fuerza gc.collect().
        """
        # 1. Resetear todos los sliders y variables de parámetros
        self.reset_all_parameters()

        # 2. Desvincular y liberar buffers pesados de imagen nativa
        self.image_32bit = None
        self.image_starless = None
        self.image_stars = None

        # 3. Desvincular y liberar proxies acelerados
        self.preview_proxy = None
        self.proxy_starless = None
        self.proxy_stars = None
        self.proxy_mask = None

        # 4. Estado de proyecto y máscaras
        self.current_mask = None
        self.active_filepath = None

        # 5. Deshabilitar controles dependientes de StarNet y procesado
        self.combo_layer.setEnabled(False)
        self.slider_stars.setEnabled(False)
        self.slider_contrast_starless.setEnabled(False)
        self.slider_clarity.setEnabled(False)
        self.slider_dehaze.setEnabled(False)
        #self.combo_denoise.setEnabled(False)
        self.slider_denoise.setEnabled(False)
        self.slider_ond_est.setEnabled(False)
        self.slider_ond_bg.setEnabled(False)

        # 6. Limpiar lienzo interactivo
        if hasattr(self, 'canvas') and self.canvas is not None:
            self.canvas.orig_rgb = None
            self.canvas.base_pixmap = None
            self.canvas.display_stretched = None
            if hasattr(self.canvas, 'scribble_pixmap') and self.canvas.scribble_pixmap is not None:
                self.canvas.scribble_pixmap.fill(Qt.transparent)
            if hasattr(self.canvas, 'refined_overlay'):
                self.canvas.refined_overlay = None
            self.canvas.update()

        # 7. Limpiar registro de logs
        self.txt_log.clear()

        # 8. Recolección forzada de memoria RAM
        gc.collect()