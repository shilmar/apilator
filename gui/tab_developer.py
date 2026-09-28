# gui/tab_developer.py
import os
import cv2
import numpy as np
import tifffile
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QGroupBox, QSlider, QSplitter,
    QDoubleSpinBox, QTextEdit, QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor

from core.stacking import load_image_as_float32
from core.stretch import auto_mtf_stretch, manual_stretch
from gui.canvas import MaskCanvas
from gui.worker import GraXpertWorker


class DeveloperTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_32bit = None
        self.current_mask = None
        self.active_filepath = None
        self.gx_worker = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # 1. Visor interactivo (sin pincel de máscara en modo revelado)
        self.canvas = MaskCanvas(self, enable_masking=False)

        # 2. Panel lateral de edición
        left_panel = QWidget()
        left_panel.setMinimumWidth(350)
        left_panel.setMaximumWidth(410)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

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
        self.btn_graxpert.setStyleSheet("font-weight: bold; background-color: #3b4252; color: #eceff4; padding: 6px;")
        self.btn_graxpert.clicked.connect(self.run_graxpert)
        gx_layout.addWidget(self.btn_graxpert)
        left_layout.addWidget(grp_graxpert)

        # Ajuste tonal y estirado
        grp_stretch = QGroupBox("Estirado Tonal (Non-Destructive)")
        stretch_layout = QVBoxLayout(grp_stretch)

        btn_auto_mtf = QPushButton("Auto-Estirado MTF")
        btn_auto_mtf.clicked.connect(self.apply_auto_mtf)
        stretch_layout.addWidget(btn_auto_mtf)

        self.lbl_bp = QLabel("Punto Negro: 0.000")
        stretch_layout.addWidget(self.lbl_bp)
        self.slider_bp = QSlider(Qt.Horizontal)
        self.slider_bp.setRange(0, 500)
        self.slider_bp.setValue(0)
        self.slider_bp.valueChanged.connect(self.update_stretch_preview)
        stretch_layout.addWidget(self.slider_bp)

        self.lbl_stretch = QLabel("Estirado (Asinh): 5.0")
        stretch_layout.addWidget(self.lbl_stretch)
        self.slider_asinh = QSlider(Qt.Horizontal)
        self.slider_asinh.setRange(1, 100)
        self.slider_asinh.setValue(10)
        self.slider_asinh.valueChanged.connect(self.update_stretch_preview)
        stretch_layout.addWidget(self.slider_asinh)

        btn_reset = QPushButton("Restablecer Ajustes")
        btn_reset.clicked.connect(self.reset_sliders)
        stretch_layout.addWidget(btn_reset)
        left_layout.addWidget(grp_stretch)

        # Exportación
        btn_export = QPushButton("Exportar Imagen Revelada...")
        btn_export.setFixedHeight(40)
        btn_export.setStyleSheet("font-weight: bold; background-color: #2e6648; color: white;")
        btn_export.clicked.connect(self.export_image)
        left_layout.addWidget(btn_export)

        # Log
        left_layout.addWidget(QLabel("Registro del Revelador:"))
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(120)
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

    def load_image_direct(self, filepath: str, mask: np.ndarray = None):
        self.active_filepath = filepath
        self.current_mask = mask
        self.log_message(f"Cargando imagen: {os.path.basename(filepath)}...")
        try:
            self.image_32bit = load_image_as_float32(filepath)
            self.update_stretch_preview()
            self.log_message(f"Imagen lista en memoria ({self.image_32bit.shape[1]}x{self.image_32bit.shape[0]} px).")
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

    def update_stretch_preview(self):
        if self.image_32bit is None: return
        bp_val = self.slider_bp.value() / 1000.0
        stretch_factor = self.slider_asinh.value() * 0.5
        self.lbl_bp.setText(f"Punto Negro: {bp_val:.3f}")
        self.lbl_stretch.setText(f"Estirado (Asinh): {stretch_factor:.1f}")

        stretched = manual_stretch(self.image_32bit, black_point=bp_val, stretch_factor=stretch_factor)
        self.canvas.load_image(self.image_32bit, display_stretched=stretched)

    def apply_auto_mtf(self):
        if self.image_32bit is None: return
        stretched = auto_mtf_stretch(self.image_32bit)
        self.canvas.load_image(self.image_32bit, display_stretched=stretched)
        self.log_message("Aplicado Auto-MTF Stretch al visor.")

    def reset_sliders(self):
        self.slider_bp.setValue(0)
        self.slider_asinh.setValue(10)
        self.update_stretch_preview()

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
        self.image_32bit = corrected_img
        self.update_stretch_preview()
        self.log_message("[GRAXPERT] Fondo neutralizado y aplicado al visor.")

        if self.active_filepath:
            base, ext = os.path.splitext(self.active_filepath)
            out_auto = f"{base}_graxpert{ext if ext else '.tiff'}"
        else:
            out_auto = os.path.abspath("resultado_graxpert_32bit.tiff")

        try:
            tifffile.imwrite(out_auto, corrected_img.astype(np.float32), compression='zlib')
            self.log_message(f"[GRAXPERT] Guardado archivo 32-bit en: {out_auto}")
            QMessageBox.information(self, "GraXpert AI", f"Gradiente corregido.\nArchivo guardado en:\n{out_auto}")
        except Exception as e:
            self.log_message(f"[AVISO] Error al autoguardar: {e}")

    def on_graxpert_error(self, err_msg: str):
        self.btn_graxpert.setEnabled(True)
        self.log_message(f"[ERROR GRAXPERT] {err_msg}")
        QMessageBox.warning(self, "Error en GraXpert", err_msg)

    def export_image(self):
        if self.image_32bit is None: return
        p, _ = QFileDialog.getSaveFileName(
            self, "Exportar Revelado", "revelado_final.tif",
            "TIFF 16-bit (*.tif *.tiff);;JPEG (*.jpg *.jpeg)"
        )
        if not p: return

        bp_val = self.slider_bp.value() / 1000.0
        stretch_factor = self.slider_asinh.value() * 0.5
        processed = manual_stretch(self.image_32bit, black_point=bp_val, stretch_factor=stretch_factor)

        ext = os.path.splitext(p)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            bgr = cv2.cvtColor((processed * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(p, bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
        else:
            u16 = (processed * 65535.0).astype(np.uint16)
            tifffile.imwrite(p, u16, compression='zlib')

        self.log_message(f"Imagen guardada en: {os.path.basename(p)}")
        QMessageBox.information(self, "Exportación", f"Guardado con éxito en:\n{p}")