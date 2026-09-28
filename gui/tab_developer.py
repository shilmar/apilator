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
        grp_stretch = QGroupBox("Estirado Tonal (Curva MTF)")
        stretch_layout = QVBoxLayout(grp_stretch)
        stretch_layout.setSpacing(6)

        btn_auto_mtf = QPushButton("Auto-Estirado MTF")
        btn_auto_mtf.clicked.connect(self.apply_auto_mtf)
        stretch_layout.addWidget(btn_auto_mtf)

        # Control Punto Negro (Label + SpinBox numérico en fila, Slider debajo)
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
        self.slider_bp.setRange(0, 2000)  # Mapea 0.0 a 0.20 con 2000 pasos
        self.slider_bp.setValue(0)
        self.slider_bp.valueChanged.connect(self.on_slider_bp_changed)
        stretch_layout.addWidget(self.slider_bp)

        # Control Medios Tonos MTF (Label + SpinBox numérico en fila, Slider debajo)
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
        self.slider_mtf.setRange(1, 2000)  # Mapea 0.0001 a 0.50 con 2000 pasos
        self.slider_mtf.setValue(400)
        self.slider_mtf.valueChanged.connect(self.on_slider_mtf_changed)
        stretch_layout.addWidget(self.slider_mtf)

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
            # En lugar de ir al estirado manual plano, aplicamos Auto-MTF de cortesía
            self.apply_auto_mtf()
            self.log_message(f"Imagen lista en memoria ({self.image_32bit.shape[1]}x{self.image_32bit.shape[0]} px). Vista Auto-MTF aplicada.")
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

    # --- Eventos de Cambio Bidireccional ---

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

    # --- Actualización del Render y Sincronización ---

    def update_stretch_preview(self):
        if self.image_32bit is None:
            return
        bp_val = self.spin_bp.value()
        m_val = self.spin_mtf.value()

        stretched = manual_stretch(self.image_32bit, black_point=bp_val, midtone=m_val)
        self.canvas.load_image(self.image_32bit, display_stretched=stretched)

    def apply_auto_mtf(self):
        if self.image_32bit is None:
            return
        stretched, bp_suggested, m_suggested = auto_mtf_stretch(self.image_32bit)
        self.canvas.load_image(self.image_32bit, display_stretched=stretched)
        self.sync_controls(bp_suggested, m_suggested)
        self.log_message(f"Auto-MTF aplicado (Punto Negro: {bp_suggested:.5f}, MTF m: {m_suggested:.5f}).")

    def sync_controls(self, bp: float, m: float):
        # Bloquear todas las señales para evitar disparar cálculos redundantes
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

        bp_val = self.spin_bp.value()
        m_val = self.spin_mtf.value()
        processed = manual_stretch(self.image_32bit, black_point=bp_val, midtone=m_val)

        ext = os.path.splitext(p)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            bgr = cv2.cvtColor((processed * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(p, bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
        else:
            u16 = (processed * 65535.0).astype(np.uint16)
            tifffile.imwrite(p, u16, compression='zlib')

        self.log_message(f"Imagen guardada en: {os.path.basename(p)}")
        QMessageBox.information(self, "Exportación", f"Guardado con éxito en:\n{p}")