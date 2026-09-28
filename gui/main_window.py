# gui/main_window.py
import os
import json
import cv2
import numpy as np
import tifffile
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QTabWidget, QListWidget, QProgressBar,
    QMessageBox, QGroupBox, QRadioButton, QSlider, QDoubleSpinBox,
    QComboBox, QSplitter, QTextEdit, QSizePolicy, QCheckBox
)
from PySide6.QtCore import Qt, QPoint, QRectF, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor, QAction, QTextCursor

from core.project_manager import ProjectManager
from core.masking import refine_mask_guided
from core.stacking import load_image_as_float32
from core.stretch import asinh_stretch, auto_mtf_stretch, stretch_display_image, manual_stretch
from gui.worker import StackingWorker, GraXpertWorker


class MaskCanvas(QWidget):
    brush_size_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.orig_rgb = None
        self.base_pixmap = None
        self.scribble_pixmap = None
        self.refined_overlay = None

        self.show_mask_overlay = True
        self.brush_mode = 2            # 2: Cielo, 1: Suelo
        self.screen_brush_radius = 20
        self.last_img_pt = None
        self.drawing = False
        self.update_brush_cursor()

    def update_brush_cursor(self):
        d = max(6, self.screen_brush_radius * 2)
        cursor_pix = QPixmap(d + 4, d + 4)
        cursor_pix.fill(Qt.transparent)
        p = QPainter(cursor_pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(0, 0, 0, 230), 2))
        p.drawEllipse(2, 2, d, d)
        p.setPen(QPen(QColor(255, 255, 255, 230), 1))
        p.drawEllipse(2, 2, d, d)
        center = d // 2 + 2
        p.setPen(QPen(QColor(255, 255, 255, 240), 1))
        p.drawPoint(center, center)
        p.end()
        self.setCursor(QCursor(cursor_pix, center, center))

    def set_brush_radius(self, radius: int):
        self.screen_brush_radius = max(5, min(120, radius))
        self.update_brush_cursor()

    def set_mask_visible(self, visible: bool):
        self.show_mask_overlay = visible
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta != 0:
            step = 3 if delta > 0 else -3
            new_r = max(5, min(120, self.screen_brush_radius + step))
            if new_r != self.screen_brush_radius:
                self.set_brush_radius(new_r)
                self.brush_size_changed.emit(self.screen_brush_radius)
            event.accept()

    def load_image(self, rgb_float: np.ndarray, display_stretched: np.ndarray = None):
        self.orig_rgb = rgb_float.astype(np.float32)
        h, w = self.orig_rgb.shape[:2]

        if display_stretched is None:
            display = np.clip(np.power(self.orig_rgb, 0.5) * 255.0, 0, 255).astype(np.uint8)
        else:
            display = (np.clip(display_stretched, 0.0, 1.0) * 255.0).astype(np.uint8)

        display_contiguous = np.ascontiguousarray(display)
        qimg = QImage(display_contiguous.data, w, h, 3 * w, QImage.Format_RGB888).copy()

        self.base_pixmap = QPixmap.fromImage(qimg)
        if self.scribble_pixmap is None or self.scribble_pixmap.size() != self.base_pixmap.size():
            self.scribble_pixmap = QPixmap(w, h)
            self.scribble_pixmap.fill(Qt.transparent)
        self.update()

    def get_render_rect(self):
        if not self.base_pixmap or self.base_pixmap.isNull():
            return QRectF(), 1.0
        w_w, h_w = self.width(), self.height()
        w_i, h_i = self.base_pixmap.width(), self.base_pixmap.height()
        scale = min(w_w / w_i, h_w / h_i)
        tw, th = w_i * scale, h_i * scale
        return QRectF((w_w - tw) / 2.0, (h_w - th) / 2.0, tw, th), scale

    def widget_to_image_coords(self, pt: QPoint):
        rect, scale = self.get_render_rect()
        if not rect.contains(pt) or scale <= 0:
            return None
        return QPoint(int((pt.x() - rect.x()) / scale), int((pt.y() - rect.y()) / scale))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.base_pixmap and not self.base_pixmap.isNull():
            self.drawing = True
            ipt = self.widget_to_image_coords(event.position().toPoint())
            if ipt:
                self.last_img_pt = ipt
                self.paint_on_scribble(ipt, ipt)

    def mouseMoveEvent(self, event):
        if self.drawing and self.base_pixmap and not self.base_pixmap.isNull():
            ipt = self.widget_to_image_coords(event.position().toPoint())
            if ipt and self.last_img_pt:
                self.paint_on_scribble(self.last_img_pt, ipt)
                self.last_img_pt = ipt

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_img_pt = None

    def paint_on_scribble(self, p1: QPoint, p2: QPoint):
        _, scale = self.get_render_rect()
        if scale <= 0 or not self.scribble_pixmap: return
        native_d = max(2, int((self.screen_brush_radius * 2) / scale))
        painter = QPainter(self.scribble_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(40, 240, 60, 180) if self.brush_mode == 2 else QColor(240, 40, 40, 180)
        painter.setPen(QPen(color, native_d, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if p1 == p2:
            painter.drawPoint(p1)
        else:
            painter.drawLine(p1, p2)
        painter.end()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(18, 18, 18))
        if not self.base_pixmap or self.base_pixmap.isNull():
            painter.setPen(QColor(130, 130, 130))
            painter.drawText(self.rect(), Qt.AlignCenter, "Carga tomas de luz para previsualizar aquí.")
            return

        rect, _ = self.get_render_rect()
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(rect, self.base_pixmap, QRectF(self.base_pixmap.rect()))

        if self.show_mask_overlay:
            if self.refined_overlay and not self.refined_overlay.isNull():
                painter.drawPixmap(rect, self.refined_overlay, QRectF(self.refined_overlay.rect()))
            elif self.scribble_pixmap and not self.scribble_pixmap.isNull():
                painter.drawPixmap(rect, self.scribble_pixmap, QRectF(self.scribble_pixmap.rect()))

    def get_scribbles_matrix(self):
        if not self.scribble_pixmap or self.scribble_pixmap.isNull(): return None
        qimg = self.scribble_pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        arr = np.frombuffer(qimg.constBits(), np.uint8).reshape((qimg.height(), qimg.width(), 4))
        scribbles = np.zeros((qimg.height(), qimg.width()), dtype=np.uint8)
        has_stroke = arr[..., 3] > 30
        scribbles[(arr[..., 0] > arr[..., 1]) & has_stroke] = 1
        scribbles[(arr[..., 1] > arr[..., 0]) & has_stroke] = 2
        return scribbles

    def set_refined_mask(self, mask_float: np.ndarray):
        h, w = mask_float.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = ((1.0 - mask_float) * 230).astype(np.uint8)
        rgba[..., 2] = (mask_float * 230).astype(np.uint8)
        rgba[..., 3] = 95

        rgba_contiguous = np.ascontiguousarray(rgba)
        qimg = QImage(rgba_contiguous.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        self.refined_overlay = QPixmap.fromImage(qimg)
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Apilador Astro - Vía Láctea Dual Stack")
        self.resize(1520, 940)

        self.project_mgr = ProjectManager()
        self.lights_list = []
        self.darks_list = []
        self.computed_mask = None
        self.final_result_32bit = None
        self.last_stacked_output_path = None
        self.worker = None
        self.gx_worker = None

        self._setup_ui()
        self._setup_menu()

    def _setup_ui(self):
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)

        # 1. Canvas
        self.canvas = MaskCanvas(self)
        self.canvas.brush_size_changed.connect(self.on_canvas_brush_changed)

        # 2. Panel lateral
        left_panel = QWidget()
        left_panel.setMinimumWidth(350)
        left_panel.setMaximumWidth(410)
        left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(6)

        # Pestañas Lights / Darks
        self.tabs_files = QTabWidget()
        
        self.list_lights = QListWidget()
        self.list_lights.itemDoubleClicked.connect(self.on_light_double_clicked)
        self.tabs_files.addTab(self.list_lights, "Lights")

        self.list_darks = QListWidget()
        self.tabs_files.addTab(self.list_darks, "Darks")
        left_layout.addWidget(self.tabs_files, stretch=1)

        btn_box = QHBoxLayout()
        btn_add = QPushButton("Añadir Tomas...")
        btn_add.clicked.connect(self.add_current_tab_files)
        btn_clear = QPushButton("Limpiar Pestaña")
        btn_clear.clicked.connect(self.clear_current_tab_files)
        btn_box.addWidget(btn_add)
        btn_box.addWidget(btn_clear)
        left_layout.addLayout(btn_box)

        # Parámetros de apilado
        grp_settings = QGroupBox("Parámetros de Apilado")
        set_layout = QVBoxLayout(grp_settings)
        set_layout.addWidget(QLabel("Modo de Captura:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Trípode Fijo (Suelo Estático)", "Star Tracker (Seguimiento)"])
        set_layout.addWidget(self.combo_mode)

        set_layout.addWidget(QLabel("Factor Kappa (MAD Rejection):"))
        self.spin_kappa = QDoubleSpinBox()
        self.spin_kappa.setRange(0.5, 5.0)
        self.spin_kappa.setValue(2.2)
        self.spin_kappa.setSingleStep(0.1)
        set_layout.addWidget(self.spin_kappa)
        left_layout.addWidget(grp_settings)

        # Herramientas de máscara
        grp_mask = QGroupBox("Máscara Cielo / Suelo")
        mask_layout = QVBoxLayout(grp_mask)

        self.chk_show_mask = QCheckBox("Mostrar Máscara en Visor")
        self.chk_show_mask.setChecked(True)
        self.chk_show_mask.toggled.connect(self.canvas.set_mask_visible)
        mask_layout.addWidget(self.chk_show_mask)

        self.rb_sky = QRadioButton("Pintar Cielo (Verde)")
        self.rb_ground = QRadioButton("Pintar Suelo (Rojo)")
        self.rb_sky.setChecked(True)
        self.rb_sky.toggled.connect(self.update_brush_mode)
        mask_layout.addWidget(self.rb_sky)
        mask_layout.addWidget(self.rb_ground)

        self.lbl_brush = QLabel("Tamaño de Cursor: 40 px")
        mask_layout.addWidget(self.lbl_brush)
        self.slider_brush = QSlider(Qt.Horizontal)
        self.slider_brush.setRange(5, 120)
        self.slider_brush.setValue(20)
        self.slider_brush.valueChanged.connect(self.update_brush_slider)
        mask_layout.addWidget(self.slider_brush)

        btn_refine = QPushButton("Refinar Máscara Automática")
        btn_refine.clicked.connect(self.refine_mask)
        mask_layout.addWidget(btn_refine)

        btn_mask_io = QHBoxLayout()
        btn_load_m = QPushButton("Cargar Máscara")
        btn_load_m.clicked.connect(self.load_mask)
        btn_save_m = QPushButton("Guardar Máscara")
        btn_save_m.clicked.connect(self.save_mask)
        btn_mask_io.addWidget(btn_load_m)
        btn_mask_io.addWidget(btn_save_m)
        mask_layout.addLayout(btn_mask_io)
        left_layout.addWidget(grp_mask)

        # Ajuste tonal y filtros
        grp_stretch = QGroupBox("Ajuste Tonal y Filtros")
        stretch_layout = QVBoxLayout(grp_stretch)

        btn_auto_mtf = QPushButton("Auto-Estirado MTF")
        btn_auto_mtf.clicked.connect(self.apply_auto_mtf)
        stretch_layout.addWidget(btn_auto_mtf)

        self.lbl_bp = QLabel("Punto Negro: 0.000")
        stretch_layout.addWidget(self.lbl_bp)
        self.slider_bp = QSlider(Qt.Horizontal)
        self.slider_bp.setRange(0, 500)
        self.slider_bp.setValue(0)
        self.slider_bp.valueChanged.connect(self.update_manual_stretch)
        stretch_layout.addWidget(self.slider_bp)

        self.lbl_stretch = QLabel("Estirado (Asinh): 5.0")
        stretch_layout.addWidget(self.lbl_stretch)
        self.slider_asinh = QSlider(Qt.Horizontal)
        self.slider_asinh.setRange(1, 100)
        self.slider_asinh.setValue(10)
        self.slider_asinh.valueChanged.connect(self.update_manual_stretch)
        stretch_layout.addWidget(self.slider_asinh)

        btn_reset_stretch = QPushButton("Restablecer Ajustes")
        btn_reset_stretch.clicked.connect(self.reset_stretch_sliders)
        stretch_layout.addWidget(btn_reset_stretch)

        # GraXpert AI
        self.btn_graxpert = QPushButton("Eliminar Gradientes con GraXpert (AI)")
        self.btn_graxpert.setStyleSheet(
            "font-weight: bold; background-color: #3b4252; color: #eceff4; padding: 6px;"
        )
        self.btn_graxpert.clicked.connect(self.run_graxpert_ai)
        stretch_layout.addWidget(self.btn_graxpert)

        btn_export_stretched = QPushButton("Exportar Imagen Revelada...")
        btn_export_stretched.setStyleSheet("font-weight: bold; background-color: #2e6648; color: white;")
        btn_export_stretched.clicked.connect(self.export_stretched_image)
        stretch_layout.addWidget(btn_export_stretched)

        left_layout.addWidget(grp_stretch)

        # Registro de actividad
        left_layout.addWidget(QLabel("Registro de Actividad:"))
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(115)
        self.txt_log.setStyleSheet(
            "background-color: #141414; color: #d0d0d0; "
            "font-family: Consolas, monospace; font-size: 11px; "
            "border: 1px solid #333333; border-radius: 4px; padding: 4px;"
        )
        left_layout.addWidget(self.txt_log)

        # Barra de progreso y botón de inicio
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        self.btn_run = QPushButton("INICIAR APILADO DUAL")
        self.btn_run.setFixedHeight(45)
        self.btn_run.setStyleSheet(
            "font-weight: bold; font-size: 13px; background-color: #2b5c8f; color: white;"
        )
        self.btn_run.clicked.connect(self.start_stacking)
        left_layout.addWidget(self.btn_run)

        # Montaje en el Splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(self.canvas)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 10)

        self.setCentralWidget(main_splitter)
        self.log_message("Aplicación lista. Puedes añadir Lights y Darks.")

    def _setup_menu(self):
        menubar = self.menuBar()
        menu_proj = menubar.addMenu("Proyecto")

        act_open = QAction("Abrir Proyecto...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.open_project)
        menu_proj.addAction(act_open)

        act_save = QAction("Guardar Proyecto", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self.save_project)
        menu_proj.addAction(act_save)

        act_save_as = QAction("Guardar Proyecto Como...", self)
        act_save_as.triggered.connect(self.save_project_as)
        menu_proj.addAction(act_save_as)

    def log_message(self, text: str):
        self.txt_log.append(text)
        self.txt_log.moveCursor(QTextCursor.End)

    def add_current_tab_files(self):
        current_tab = self.tabs_files.currentIndex()
        tab_name = "Lights" if current_tab == 0 else "Darks"
        
        paths, _ = QFileDialog.getOpenFileNames(
            self, f"Seleccionar {tab_name}", "",
            "Astro Images (*.nef *.cr2 *.cr3 *.arw *.dng *.tif *.tiff *.fits)"
        )
        if not paths: return

        if current_tab == 0:
            first_add = len(self.lights_list) == 0
            for p in paths:
                if p not in self.lights_list:
                    self.lights_list.append(p)
                    self.list_lights.addItem(os.path.basename(p))
            self.log_message(f"Añadidos {len(paths)} Lights. Total: {len(self.lights_list)}")
            if first_add and self.lights_list:
                self.load_frame_to_canvas(self.lights_list[0])
        else:
            for p in paths:
                if p not in self.darks_list:
                    self.darks_list.append(p)
                    self.list_darks.addItem(os.path.basename(p))
            self.log_message(f"Añadidos {len(paths)} Darks. Total: {len(self.darks_list)}")

    def clear_current_tab_files(self):
        if self.tabs_files.currentIndex() == 0:
            self.lights_list.clear()
            self.list_lights.clear()
            self.canvas.base_pixmap = None
            self.canvas.update()
            self.log_message("Lista de Lights vaciada.")
        else:
            self.darks_list.clear()
            self.list_darks.clear()
            self.log_message("Lista de Darks vaciada.")

    def on_light_double_clicked(self, item):
        row = self.list_lights.row(item)
        if 0 <= row < len(self.lights_list):
            self.load_frame_to_canvas(self.lights_list[row])

    def load_frame_to_canvas(self, path: str):
        self.log_message(f"Cargando vista previa: {os.path.basename(path)}...")
        self.repaint()
        try:
            img = load_image_as_float32(path)
            self.canvas.load_image(img)
            self.log_message(f"Vista previa activa: {os.path.basename(path)}")
        except Exception as e:
            self.log_message(f"[ERROR] Al cargar {os.path.basename(path)}: {e}")
            QMessageBox.critical(self, "Error al cargar imagen", f"No se pudo cargar {path}:\n{e}")

    def update_brush_mode(self):
        self.canvas.brush_mode = 2 if self.rb_sky.isChecked() else 1

    def update_brush_slider(self, val):
        self.canvas.set_brush_radius(val)
        self.lbl_brush.setText(f"Tamaño de Cursor: {val * 2} px")

    def on_canvas_brush_changed(self, radius):
        self.slider_brush.blockSignals(True)
        self.slider_brush.setValue(radius)
        self.slider_brush.blockSignals(False)
        self.lbl_brush.setText(f"Tamaño de Cursor: {radius * 2} px")

    def refine_mask(self):
        if self.canvas.orig_rgb is None:
            QMessageBox.warning(self, "Aviso", "Abre primero una toma base.")
            return
        self.log_message("Refinando máscara cielo/suelo...")
        scribbles = self.canvas.get_scribbles_matrix()
        self.computed_mask = refine_mask_guided(self.canvas.orig_rgb, scribbles)
        self.canvas.set_refined_mask(self.computed_mask)
        self.log_message("Máscara refinada con éxito.")

    def save_mask(self):
        if self.computed_mask is None: return
        p, _ = QFileDialog.getSaveFileName(self, "Guardar Máscara", "mascara_cielo.png", "PNG (*.png)")
        if p:
            cv2.imwrite(p, (self.computed_mask * 65535.0).astype(np.uint16))
            self.log_message(f"Máscara guardada en: {os.path.basename(p)}")

    def load_mask(self):
        if self.canvas.orig_rgb is None: return
        p, _ = QFileDialog.getOpenFileName(self, "Cargar Máscara", "", "Imágenes (*.png *.tif *.tiff)")
        if not p: return
        raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if raw.ndim == 3: raw = raw[..., 0]
        h, w = self.canvas.orig_rgb.shape[:2]
        if raw.shape != (h, w): raw = cv2.resize(raw, (w, h))
        max_v = 65535.0 if raw.dtype == np.uint16 else 255.0
        self.computed_mask = np.clip(raw.astype(np.float32) / max_v, 0.0, 1.0)
        self.canvas.set_refined_mask(self.computed_mask)
        self.log_message(f"Máscara cargada desde: {os.path.basename(p)}")

    def apply_auto_mtf(self):
        target = self.final_result_32bit if self.final_result_32bit is not None else self.canvas.orig_rgb
        if target is None: return
        stretched = auto_mtf_stretch(target)
        self.canvas.load_image(target, display_stretched=stretched)
        self.log_message("Aplicado Auto-MTF Stretch al visor.")

    def update_manual_stretch(self):
        target = self.final_result_32bit if self.final_result_32bit is not None else self.canvas.orig_rgb
        if target is None: return

        bp_val = self.slider_bp.value() / 1000.0
        stretch_factor = self.slider_asinh.value() * 0.5

        self.lbl_bp.setText(f"Punto Negro: {bp_val:.3f}")
        self.lbl_stretch.setText(f"Estirado (Asinh): {stretch_factor:.1f}")

        stretched = manual_stretch(target, black_point=bp_val, stretch_factor=stretch_factor)
        self.canvas.load_image(target, display_stretched=stretched)

    def reset_stretch_sliders(self):
        self.slider_bp.blockSignals(True)
        self.slider_asinh.blockSignals(True)
        self.slider_bp.setValue(0)
        self.slider_asinh.setValue(10)
        self.slider_bp.blockSignals(False)
        self.slider_asinh.blockSignals(False)
        self.lbl_bp.setText("Punto Negro: 0.000")
        self.lbl_stretch.setText("Estirado (Asinh): 5.0")
        self.update_manual_stretch()

    def run_graxpert_ai(self):
        target = self.final_result_32bit if self.final_result_32bit is not None else self.canvas.orig_rgb
        if target is None:
            QMessageBox.warning(self, "Aviso", "Carga o apila una imagen primero.")
            return

        self.btn_graxpert.setEnabled(False)
        self.log_message("=== INICIANDO GRAXPERT AI ===")
        self.log_message("Procesando extracción de gradientes en segundo plano...")

        self.gx_worker = GraXpertWorker(target, sky_mask=self.computed_mask, smoothing=0.5)
        self.gx_worker.status_changed.connect(self.log_message)
        self.gx_worker.finished_success.connect(self.on_graxpert_success)
        self.gx_worker.error_occurred.connect(self.on_graxpert_error)
        self.gx_worker.start()

    def on_graxpert_success(self, corrected_img: np.ndarray):
        self.btn_graxpert.setEnabled(True)
        if self.final_result_32bit is not None:
            self.final_result_32bit = corrected_img
        else:
            self.canvas.orig_rgb = corrected_img

        self.update_manual_stretch()
        self.log_message("[GRAXPERT] Fondo neutralizado y aplicado al visor.")

        # Guardar automáticamente la versión de 32 bits con sufijo _graxpert
        if self.last_stacked_output_path:
            base, ext = os.path.splitext(self.last_stacked_output_path)
            auto_out_path = f"{base}_graxpert{ext if ext else '.tiff'}"
        else:
            auto_out_path = os.path.abspath("resultado_dual_32bit_graxpert.tiff")

        try:
            tifffile.imwrite(auto_out_path, corrected_img.astype(np.float32), compression='zlib')
            self.log_message(f"[GRAXPERT] Archivo 32-bit guardado automáticamente en: {auto_out_path}")
            QMessageBox.information(
                self, 
                "GraXpert AI Completado", 
                f"El gradiente de contaminación lumínica ha sido corregido con éxito.\n\n"
                f"Archivo maestro guardado en:\n{auto_out_path}"
            )
        except Exception as e:
            self.log_message(f"[ERROR] No se pudo guardar el archivo automático: {e}")
            QMessageBox.warning(self, "Aviso de Guardado", f"No se pudo guardar automáticamente el archivo:\n{e}")

    def on_graxpert_error(self, err_msg: str):
        self.btn_graxpert.setEnabled(True)
        self.log_message(f"[ERROR GRAXPERT] {err_msg}")
        QMessageBox.warning(self, "Error en GraXpert", f"Asegúrate de tener instalado GraXpert:\n\n{err_msg}")

    def export_stretched_image(self):
        target = self.final_result_32bit if self.final_result_32bit is not None else self.canvas.orig_rgb
        if target is None:
            QMessageBox.warning(self, "Aviso", "No hay ninguna imagen cargada o apilada para exportar.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Imagen Revelada", "resultado_revelado.tif",
            "TIFF 16-bit (*.tif *.tiff);;JPEG (*.jpg *.jpeg)"
        )
        if not out_path: return

        bp_val = self.slider_bp.value() / 1000.0
        stretch_factor = self.slider_asinh.value() * 0.5
        processed = manual_stretch(target, black_point=bp_val, stretch_factor=stretch_factor)

        ext = os.path.splitext(out_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            bgr = cv2.cvtColor((processed * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
        else:
            u16 = (processed * 65535.0).astype(np.uint16)
            tifffile.imwrite(out_path, u16, compression='zlib')

        self.log_message(f"Imagen revelada exportada con éxito en: {os.path.basename(out_path)}")
        QMessageBox.information(self, "Exportación Completada", f"Archivo guardado en:\n{out_path}")

    def save_project(self):
        if self.project_mgr.project_path:
            self._do_save(self.project_mgr.project_path)
        else:
            self.save_project_as()

    def save_project_as(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "Guardar Proyecto", "mi_sesion_astro.mwstack", "Astro Project (*.mwstack *.json)"
        )
        if not p: return
        self._do_save(p)

    def _do_save(self, filepath: str):
        bp_val = self.slider_bp.value() / 1000.0
        stretch_factor = self.slider_asinh.value() * 0.5

        self.project_mgr.data["light_frames"] = self.lights_list
        self.project_mgr.data["dark_frames"] = self.darks_list
        self.project_mgr.data["mode"] = "fixed_tripod" if self.combo_mode.currentIndex() == 0 else "star_tracker"
        self.project_mgr.data["parameters"] = {
            "kappa": self.spin_kappa.value(),
            "black_point": bp_val,
            "stretch_factor": stretch_factor
        }

        self.project_mgr.save(filepath, mask_array=self.computed_mask)
        self.setWindowTitle(f"Apilador Astro - [{os.path.basename(filepath)}]")
        self.log_message(f"Proyecto y máscara guardados con éxito en: {os.path.basename(filepath)}")

    def open_project(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Abrir Proyecto", "", "Astro Project (*.mwstack *.json)"
        )
        if not p: return

        try:
            data, loaded_mask = self.project_mgr.load(p)

            self.clear_current_tab_files()
            self.tabs_files.setCurrentIndex(1)
            self.clear_current_tab_files()
            self.tabs_files.setCurrentIndex(0)

            for fpath in data.get("light_frames", []):
                if os.path.exists(fpath):
                    self.lights_list.append(fpath)
                    self.list_lights.addItem(os.path.basename(fpath))

            for fpath in data.get("dark_frames", []):
                if os.path.exists(fpath):
                    self.darks_list.append(fpath)
                    self.list_darks.addItem(os.path.basename(fpath))

            mode = data.get("mode", "fixed_tripod")
            self.combo_mode.setCurrentIndex(0 if mode == "fixed_tripod" else 1)
            
            params = data.get("parameters", {})
            self.spin_kappa.setValue(params.get("kappa", 2.2))

            if self.lights_list:
                self.load_frame_to_canvas(self.lights_list[0])

            if loaded_mask is not None:
                self.computed_mask = loaded_mask
                self.canvas.set_refined_mask(self.computed_mask)
                self.chk_show_mask.setChecked(True)
                self.log_message("Máscara del proyecto restaurada en el visor.")

            bp_val = params.get("black_point", 0.0)
            st_val = params.get("stretch_factor", 5.0)
            self.slider_bp.setValue(int(bp_val * 1000.0))
            self.slider_asinh.setValue(int(st_val / 0.5))

            self.setWindowTitle(f"Apilador Astro - [{os.path.basename(p)}]")
            self.log_message(f"Proyecto cargado: {len(self.lights_list)} Lights, {len(self.darks_list)} Darks.")

        except Exception as e:
            self.log_message(f"[ERROR] Al abrir proyecto {os.path.basename(p)}: {e}")
            QMessageBox.critical(self, "Error al abrir proyecto", str(e))

    def start_stacking(self):
        if len(self.lights_list) < 2:
            QMessageBox.warning(self, "Aviso", "Añade al menos 2 tomas de luz (Lights) para apilar.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Resultado 32-bit", "resultado_dual_32bit.tiff", "TIFF (*.tiff *.tif)"
        )
        if not out_path: return

        self.last_stacked_output_path = out_path
        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_message("=== INICIANDO APILADO SUBPÍXEL CON CALIBRACIÓN ===")

        cfg = {
            "lights": self.lights_list,
            "darks": self.darks_list,
            "mask": self.computed_mask,
            "mode": "fixed_tripod" if self.combo_mode.currentIndex() == 0 else "star_tracker",
            "kappa": self.spin_kappa.value(),
            "output_path": out_path
        }

        self.worker = StackingWorker(cfg)
        self.worker.progress_changed.connect(self.progress_bar.setValue)
        self.worker.status_changed.connect(self.log_message)
        self.worker.finished_success.connect(self.on_stack_success)
        self.worker.error_occurred.connect(self.on_stack_error)
        self.worker.start()

    def on_stack_success(self, path):
        self.btn_run.setEnabled(True)
        self.last_stacked_output_path = path
        self.log_message(f"[COMPLETADO] Imagen guardada en: {path}")

        try:
            self.final_result_32bit = tifffile.imread(path)
            self.chk_show_mask.setChecked(False)
            self.update_manual_stretch()
            self.log_message("Resultado cargado en el visor con ajuste subpíxel e interpolación cúbica.")
        except Exception as e:
            self.log_message(f"[AVISO] No se pudo cargar en visor: {e}")

        QMessageBox.information(self, "Éxito", f"Apilado completado con éxito.\nGuardado en:\n{path}")

    def on_stack_error(self, err_msg):
        self.btn_run.setEnabled(True)
        self.log_message(f"[ERROR CRÍTICO] {err_msg}")
        QMessageBox.critical(self, "Error durante el apilado", f"Ocurrió un error:\n{err_msg}")