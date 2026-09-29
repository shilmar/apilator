# gui/tab_developer.py
import os
import cv2
import numpy as np
import tifffile
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QGroupBox, QSlider, QSplitter,
    QDoubleSpinBox, QTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor

from core.stacking import load_image_as_float32
from core.stretch import (
    calculate_mtf_params, manual_stretch, 
    apply_white_balance, adjust_saturation_dual
)
from gui.canvas import MaskCanvas
from gui.worker import GraXpertWorker


class DeveloperTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_32bit = None       # Imagen nativa float32 a resolución completa
        self.preview_proxy = None     # Proxy de previsualización rápida
        self.proxy_mask = None        # Máscara suavizada adaptada al tamaño del proxy
        self.current_mask = None      # Máscara original a resolución nativa
        self.active_filepath = None
        self.gx_worker = None

        # Parámetros del revelador
        self.temp_val = 0.0
        self.tint_val = 0.0
        self.sat_sky_val = 1.0
        self.sat_gnd_val = 1.0

        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # 1. Visor interactivo
        self.canvas = MaskCanvas(self, enable_masking=False)

        # 2. Panel lateral de controles
        left_panel = QWidget()
        left_panel.setMinimumWidth(350)
        left_panel.setMaximumWidth(410)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(6)

        # Carga externa
        btn_open_img = QPushButton("Abrir Imagen (TIFF / FITS)...")
        btn_open_img.setStyleSheet("font-weight: bold; padding: 6px;")
        btn_open_img.clicked.connect(self.open_image_file)
        left_layout.addWidget(btn_open_img)

        # Módulo GraXpert
        grp_graxpert = QGroupBox("GraXpert AI - Fondo y Gradientes")
        gx_layout = QVBoxLayout(grp_graxpert)
        gx_layout.addWidget(QLabel("Factor de Suavizado (Smoothing):"))
        self.spin_smoothing = QDoubleSpinBox()
        self.spin_smoothing.setRange(0.01, 1.0)
        self.spin_smoothing.setValue(0.5)
        self.spin_smoothing.setSingleStep(0.05)
        gx_layout.addWidget(self.spin_smoothing)

        self.btn_graxpert = QPushButton("Eliminar Gradientes con GraXpert")
        self.btn_graxpert.setStyleSheet("font-weight: bold; background-color: #3b4252; color: #eceff4; padding: 5px;")
        self.btn_graxpert.clicked.connect(self.run_graxpert)
        gx_layout.addWidget(self.btn_graxpert)
        left_layout.addWidget(grp_graxpert)

        # Módulo Balance de Blancos
        grp_wb = QGroupBox("Balance de Blancos")
        wb_layout = QVBoxLayout(grp_wb)
        wb_layout.setSpacing(3)

        row_temp = QHBoxLayout()
        row_temp.addWidget(QLabel("Temp (Frío / Cálido):"))
        self.lbl_temp_val = QLabel("0.00")
        row_temp.addWidget(self.lbl_temp_val)
        wb_layout.addLayout(row_temp)

        self.slider_temp = QSlider(Qt.Horizontal)
        self.slider_temp.setRange(-100, 100)
        self.slider_temp.setValue(0)
        self.slider_temp.valueChanged.connect(self.on_temp_changed)
        wb_layout.addWidget(self.slider_temp)

        row_tint = QHBoxLayout()
        row_tint.addWidget(QLabel("Tinte (Verde / Magenta):"))
        self.lbl_tint_val = QLabel("0.00")
        row_tint.addWidget(self.lbl_tint_val)
        wb_layout.addLayout(row_tint)

        self.slider_tint = QSlider(Qt.Horizontal)
        self.slider_tint.setRange(-100, 100)
        self.slider_tint.setValue(0)
        self.slider_tint.valueChanged.connect(self.on_tint_changed)
        wb_layout.addWidget(self.slider_tint)

        btn_reset_wb = QPushButton("Restablecer Balance")
        btn_reset_wb.clicked.connect(self.reset_wb)
        wb_layout.addWidget(btn_reset_wb)
        left_layout.addWidget(grp_wb)

        # Módulo Saturación
        grp_sat = QGroupBox("Saturación de Color")
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

        btn_reset_sat = QPushButton("Restablecer Saturación")
        btn_reset_sat.clicked.connect(self.reset_saturation)
        sat_layout.addWidget(btn_reset_sat)
        left_layout.addWidget(grp_sat)

        # Módulo MTF
        grp_stretch = QGroupBox("Estirado Tonal (Curva MTF)")
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

        btn_reset = QPushButton("Restablecer Curva")
        btn_reset.clicked.connect(self.reset_sliders)
        stretch_layout.addWidget(btn_reset)
        left_layout.addWidget(grp_stretch)

        # Botón de exportación
        btn_export = QPushButton("Exportar Imagen Revelada...")
        btn_export.setFixedHeight(40)
        btn_export.setStyleSheet("font-weight: bold; background-color: #2e6648; color: white;")
        btn_export.clicked.connect(self.export_image)
        left_layout.addWidget(btn_export)

        # Log
        left_layout.addWidget(QLabel("Registro del Revelador:"))
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(110)
        self.txt_log.setStyleSheet(
            "background-color: #141414; color: #d0d0d0; "
            "font-family: Consolas, monospace; font-size: 11px; "
            "border: 1px solid #333333; border-radius: 4px; padding: 4px;"
        )
        left_layout.addWidget(self.txt_log)
        left_layout.addStretch()

        splitter.addWidget(left_panel)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 10)
        layout.addWidget(splitter)

    def log_message(self, text: str):
        self.txt_log.append(text)
        self.txt_log.moveCursor(QTextCursor.End)

    # --- Generación de proxy para previsualización a 60 fps ---
    def _generate_preview_proxy(self):
        if self.image_32bit is None:
            self.preview_proxy = None
            self.proxy_mask = None
            return

        h, w = self.image_32bit.shape[:2]
        max_dim = 1600.0
        scale = min(1.0, max_dim / max(h, w))

        if scale < 1.0:
            pw = (int(w * scale) // 4) * 4
            ph = (int(h * scale) // 4) * 4
            resized = cv2.resize(self.image_32bit, (pw, ph), interpolation=cv2.INTER_AREA)
            self.preview_proxy = np.ascontiguousarray(resized, dtype=np.float32)

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
        self.log_message(f"Cargando imagen: {os.path.basename(filepath)}...")
        try:
            self.image_32bit = load_image_as_float32(filepath)
            self._generate_preview_proxy()
            self.apply_auto_mtf()
            self.log_message(f"Imagen en memoria ({self.image_32bit.shape[1]}x{self.image_32bit.shape[0]} px). Vista acelerada activa.")
        except Exception as e:
            self.log_message(f"[ERROR] No se pudo cargar: {e}")
            QMessageBox.critical(self, "Error", f"Fallo al abrir archivo:\n{e}")

    def open_image_file(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Abrir Imagen de Astronomía", "",
            "Astro Images (*.tif *.tiff *.fits *.fit)"
        )
        if p:
            self.load_image_direct(p)

    # --- Pipeline compartido ---
    def _apply_pipeline_on_image(self, target_img: np.ndarray, precomputed_mask: np.ndarray = None) -> np.ndarray:
        if target_img is None:
            return None

        # 1. Balance de blancos
        img = apply_white_balance(target_img, self.temp_val, self.tint_val)

        # 2. Saturación diferencial
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

        # 3. Estirado MTF
        bp_val = self.spin_bp.value()
        m_val = self.spin_mtf.value()
        stretched = manual_stretch(img, black_point=bp_val, midtone=m_val)
        return np.ascontiguousarray(stretched, dtype=np.float32)

    def _apply_full_pipeline(self) -> np.ndarray:
        target = self.preview_proxy if self.preview_proxy is not None else self.image_32bit
        return self._apply_pipeline_on_image(target, precomputed_mask=self.proxy_mask)

    def update_stretch_preview(self):
        target = self.preview_proxy if self.preview_proxy is not None else self.image_32bit
        if target is None:
            return
        stretched = self._apply_full_pipeline()
        if stretched is not None:
            self.canvas.load_image(target, display_stretched=stretched)

    # --- Callbacks de Sliders ---
    def on_temp_changed(self, val: int):
        self.temp_val = val / 100.0
        self.lbl_temp_val.setText(f"{self.temp_val:+.2f}")
        self.update_stretch_preview()

    def on_tint_changed(self, val: int):
        self.tint_val = val / 100.0
        self.lbl_tint_val.setText(f"{self.tint_val:+.2f}")
        self.update_stretch_preview()

    def reset_wb(self):
        self.slider_temp.blockSignals(True)
        self.slider_tint.blockSignals(True)
        self.slider_temp.setValue(0)
        self.slider_tint.setValue(0)
        self.temp_val = 0.0
        self.tint_val = 0.0
        self.lbl_temp_val.setText("0.00")
        self.lbl_tint_val.setText("0.00")
        self.slider_temp.blockSignals(False)
        self.slider_tint.blockSignals(False)
        self.update_stretch_preview()

    def on_sat_sky_changed(self, val: int):
        self.sat_sky_val = val / 100.0
        self.lbl_sat_sky.setText(f"{self.sat_sky_val:.2f}x")
        self.update_stretch_preview()

    def on_sat_gnd_changed(self, val: int):
        self.sat_gnd_val = val / 100.0
        self.lbl_sat_gnd.setText(f"{self.sat_gnd_val:.2f}x")
        self.update_stretch_preview()

    def reset_saturation(self):
        self.slider_sat_sky.blockSignals(True)
        self.slider_sat_gnd.blockSignals(True)
        self.slider_sat_sky.setValue(100)
        self.slider_sat_gnd.setValue(100)
        self.sat_sky_val = 1.0
        self.sat_gnd_val = 1.0
        self.lbl_sat_sky.setText("1.00x")
        self.lbl_sat_gnd.setText("1.00x")
        self.slider_sat_sky.blockSignals(False)
        self.slider_sat_gnd.blockSignals(False)
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
        self._generate_preview_proxy()
        self.update_stretch_preview()
        self.log_message("[GRAXPERT] Fondo neutralizado y aplicado.")

    def on_graxpert_error(self, err_msg: str):
        self.btn_graxpert.setEnabled(True)
        self.log_message(f"[ERROR GRAXPERT] {err_msg}")
        QMessageBox.warning(self, "Error en GraXpert", err_msg)

    # --- Exportación única y en resolución completa ---
    def export_image(self):
        if self.image_32bit is None:
            return

        # Filtros con opciones de 32 bits, 16 bits y JPEG
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

        # 1. Preparar la máscara nativa para la imagen completa si hay saturación diferencial
        full_mask = None
        if self.current_mask is not None and (self.sat_sky_val != self.sat_gnd_val):
            if self.current_mask.shape[:2] != (h_full, w_full):
                m_full = cv2.resize(self.current_mask, (w_full, h_full), interpolation=cv2.INTER_LINEAR)
            else:
                m_full = self.current_mask
            ksize = int(max(15, (min(h_full, w_full) // 150) | 1))
            if ksize % 2 == 0:
                ksize += 1
            sm = cv2.GaussianBlur(m_full, (ksize, ksize), sigmaX=ksize / 3.0)
            full_mask = np.ascontiguousarray(np.repeat(sm[..., np.newaxis], 3, axis=2), dtype=np.float32)

        # 2. Aplicar la cadena de revelado sobre self.image_32bit a tamaño nativo
        processed = self._apply_pipeline_on_image(self.image_32bit, precomputed_mask=full_mask)

        # 3. Guardar según el formato/profundidad elegida
        ext = os.path.splitext(p)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            bgr8 = cv2.cvtColor((processed * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(p, bgr8, [cv2.IMWRITE_JPEG_QUALITY, 96])
            desc_formato = "JPEG 8-bit"
        elif "32-bit" in selected_filter:
            # TIFF float32 nativo estándar con compresión zlib sin pérdidas
            out32 = np.clip(processed, 0.0, 1.0).astype(np.float32)
            tifffile.imwrite(p, out32, compression='zlib', photometric='rgb')
            desc_formato = "TIFF 32-bit float"
        else:
            # TIFF uint16 clásico compatible con todos los visores
            u16 = (np.clip(processed, 0.0, 1.0) * 65535.0).astype(np.uint16)
            tifffile.imwrite(p, u16, compression='zlib', photometric='rgb')
            desc_formato = "TIFF 16-bit"

        self.log_message(f"Imagen guardada en: {os.path.basename(p)} ({desc_formato})")
        QMessageBox.information(
            self, "Exportación", 
            f"Guardada con éxito en resolución completa:\n{p}\n\nFormato: {desc_formato} ({w_full}x{h_full} px)"
        )