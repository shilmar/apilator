# gui/tab_stacker.py
import os
import cv2
import numpy as np
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QTabWidget, QListWidget, QListWidgetItem, QProgressBar,
    QMessageBox, QGroupBox, QRadioButton, QSlider, QDoubleSpinBox,
    QComboBox, QSplitter, QTextEdit, QSizePolicy, QCheckBox, QSpinBox
)
from PySide6.QtCore import Qt, Signal, QCoreApplication
from PySide6.QtGui import QTextCursor, QCursor

from core.project_manager import ProjectManager
from core.masking import refine_mask_guided
from core.stacking import load_image_as_float32
from gui.canvas import MaskCanvas
from gui.worker import StackingWorker
from core.gpu_backend import is_gpu_enabled
from core.config_manager import load_config


class FileListRow(QWidget):
    """Widget de fila con etiqueta de referencia y botón de borrado individual."""
    delete_requested = Signal(str, str)  # (tab_name, filepath)

    def __init__(self, filepath: str, tab_name: str, is_ref: bool = False, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.tab_name = tab_name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # Indicador de referencia / activo
        self.lbl_ref = QLabel("✓ REF" if is_ref else "     ")
        self.lbl_ref.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11px;")
        layout.addWidget(self.lbl_ref)

        # Nombre del archivo
        self.lbl_name = QLabel(os.path.basename(filepath))
        self.lbl_name.setToolTip(filepath)
        self.lbl_name.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self.lbl_name)

        # Botón de eliminar individual (Cruz roja)
        self.btn_del = QPushButton("✕")
        self.btn_del.setFixedSize(20, 20)
        self.btn_del.setToolTip("Eliminar esta toma de la lista")
        self.btn_del.setStyleSheet(
            "QPushButton { color: #ff5252; background: transparent; border: 1px solid #552222; border-radius: 3px; font-weight: bold; } "
            "QPushButton:hover { background: #d32f2f; color: white; }"
        )
        self.btn_del.clicked.connect(lambda: self.delete_requested.emit(self.tab_name, self.filepath))
        layout.addWidget(self.btn_del)

    def set_as_reference(self, is_ref: bool):
        self.lbl_ref.setText("✓ REF" if is_ref else "     ")


class StackerTab(QWidget):
    stacking_finished = Signal(str, object)
    session_title_changed = Signal(str)
    new_session_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_mgr = ProjectManager()
        self.lights_list = []
        self.darks_list = []
        self.flats_list = []
        self.bias_list = []
        self.ground_list = []
        self.computed_mask = None
        self.last_stacked_output_path = None
        self.current_ref_path = None
        self.worker = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # 1. Lienzo Interactivo
        self.canvas = MaskCanvas(self, enable_masking=True)
        self.canvas.brush_size_changed.connect(self.on_canvas_brush_changed)

        # 2. Panel lateral
        left_panel = QWidget()
        left_panel.setMinimumWidth(370)
        left_panel.setMaximumWidth(430)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(6)

        # SECCIÓN 1: Gestión de Sesión / Proyecto
        self.grp_project = QGroupBox("Sesión de Apilado (.mwstack)")
        proj_layout = QVBoxLayout(self.grp_project)
        
        self.lbl_session = QLabel("Sesión: Sin guardar")
        self.lbl_session.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        proj_layout.addWidget(self.lbl_session)

        btn_proj_row = QHBoxLayout()
        
        self.btn_new_sess = QPushButton("Nueva")
        self.btn_new_sess.setStyleSheet("font-weight: bold; color: #ffab91;")
        self.btn_new_sess.setToolTip("Reiniciar proyecto completo, vaciar tomas, máscaras y revelador")
        self.btn_new_sess.clicked.connect(self.on_new_session_clicked)
        btn_proj_row.addWidget(self.btn_new_sess)

        self.btn_open_sess = QPushButton("Cargar")
        self.btn_open_sess.clicked.connect(self.open_project)
        self.btn_save_sess = QPushButton("Guardar")
        self.btn_save_sess.clicked.connect(self.save_project)
        self.btn_save_as_sess = QPushButton("Guardar Como...")
        self.btn_save_as_sess.clicked.connect(self.save_project_as)

        btn_proj_row.addWidget(self.btn_open_sess)
        btn_proj_row.addWidget(self.btn_save_sess)
        btn_proj_row.addWidget(self.btn_save_as_sess)
        proj_layout.addLayout(btn_proj_row)
        left_layout.addWidget(self.grp_project)

        # SECCIÓN 2: Pestañas de Archivos (Lights / Darks / Flats / Suelo)
        self.tabs_files = QTabWidget()
        
        self.list_lights = QListWidget()
        self.list_lights.itemDoubleClicked.connect(self.on_light_double_clicked)
        self.tabs_files.addTab(self.list_lights, "Lights")

        self.list_darks = QListWidget()
        self.tabs_files.addTab(self.list_darks, "Darks")

        self.list_flats = QListWidget()
        self.tabs_files.addTab(self.list_flats, "Flats")

        self.list_bias = QListWidget()
        self.tabs_files.addTab(self.list_bias, "Bias")

        self.list_ground = QListWidget()
        self.list_ground.itemDoubleClicked.connect(self.on_ground_double_clicked)
        self.tabs_files.addTab(self.list_ground, "Suelo")
        
        left_layout.addWidget(self.tabs_files, stretch=1)

        btn_box = QHBoxLayout()
        self.btn_add_files = QPushButton("Añadir Tomas...")
        self.btn_add_files.clicked.connect(self.add_current_tab_files)
        self.btn_clear_files = QPushButton("Limpiar Pestaña")
        self.btn_clear_files.clicked.connect(self.clear_current_tab_files)
        btn_box.addWidget(self.btn_add_files)
        btn_box.addWidget(self.btn_clear_files)
        left_layout.addLayout(btn_box)

        # SECCIÓN 3: Parámetros de Apilado
        self.grp_settings = QGroupBox("Parámetros de Integración")
        set_layout = QVBoxLayout(self.grp_settings)
        set_layout.addWidget(QLabel("Modo de Captura:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Trípode Fijo (Suelo Estático)", "Star Tracker (Seguimiento)"])
        self.combo_mode.currentIndexChanged.connect(self._on_capture_mode_changed)
        set_layout.addWidget(self.combo_mode)

        # Tratamiento del Suelo
        set_layout.addWidget(QLabel("Tratamiento del Suelo:"))
        self.combo_ground = QComboBox()
        self.combo_ground.addItems([
            "Apilar Suelo Completo (Dual)",
            "Suelo de Referencia (Sin apilar)",
            "Usar Toma de Pestaña Suelo"
        ])
        set_layout.addWidget(self.combo_ground)

        set_layout.addWidget(QLabel("Factor Kappa (MAD Rejection):"))
        self.spin_kappa = QDoubleSpinBox()
        self.spin_kappa.setRange(0.5, 5.0)
        self.spin_kappa.setValue(2.2)
        self.spin_kappa.setSingleStep(0.1)
        set_layout.addWidget(self.spin_kappa)
        left_layout.addWidget(self.grp_settings)

        # Antipolución en Apilado
        self.grp_stack_lp = QGroupBox("Antipolución en Apilado")
        layout_stack_lp = QVBoxLayout(self.grp_stack_lp)

        row_lp_algo = QHBoxLayout()
        row_lp_algo.addWidget(QLabel("Algoritmo:"))
        self.combo_lp_algo = QComboBox()
        self.combo_lp_algo.addItems([
            "Estándar (Sin filtro)",
            "Sustracción de Domo (Sequator)",
            "Rechazo Asimétrico (Min-Sigma)",
            "Normalización Local"
        ])
        row_lp_algo.addWidget(self.combo_lp_algo)
        layout_stack_lp.addLayout(row_lp_algo)

        row_lp_str = QHBoxLayout()
        row_lp_str.addWidget(QLabel("Fuerza Antipolución:"))
        self.lbl_lp_str = QLabel("50%")
        row_lp_str.addWidget(self.lbl_lp_str)
        layout_stack_lp.addLayout(row_lp_str)

        self.slider_stack_lp = QSlider(Qt.Horizontal)
        self.slider_stack_lp.setRange(0, 100)
        self.slider_stack_lp.setValue(50)
        self.slider_stack_lp.valueChanged.connect(self._on_stack_lp_changed)
        layout_stack_lp.addWidget(self.slider_stack_lp)

        left_layout.addWidget(self.grp_stack_lp)

        # SECCIÓN 4: Máscara Cielo / Suelo
        self.grp_mask = QGroupBox("Máscara Cielo / Suelo")
        mask_layout = QVBoxLayout(self.grp_mask)

        row_mask_toggles = QHBoxLayout()
        self.chk_show_mask = QCheckBox("Mostrar Máscara")
        self.chk_show_mask.setChecked(True)
        self.chk_show_mask.toggled.connect(self.canvas.set_mask_visible)
        row_mask_toggles.addWidget(self.chk_show_mask)

        self.chk_show_strokes = QCheckBox("Mostrar Trazos")
        self.chk_show_strokes.setChecked(True)
        self.chk_show_strokes.toggled.connect(self.canvas.set_scribbles_visible)
        row_mask_toggles.addWidget(self.chk_show_strokes)
        mask_layout.addLayout(row_mask_toggles)

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

        # Controles de Refinamiento
        row_params = QHBoxLayout()
        row_params.addWidget(QLabel("Suavizado Borde:"))
        self.spin_feather = QSpinBox()
        self.spin_feather.setRange(1, 31)
        self.spin_feather.setSingleStep(2)
        self.spin_feather.setValue(7)
        row_params.addWidget(self.spin_feather)

        row_params.addWidget(QLabel("Pasos:"))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1, 8)
        self.spin_iter.setValue(3)
        row_params.addWidget(self.spin_iter)
        mask_layout.addLayout(row_params)

        btn_mask_actions = QHBoxLayout()
        self.btn_refine = QPushButton("Refinar Automática")
        self.btn_refine.clicked.connect(self.refine_mask)
        self.btn_clear_m = QPushButton("Limpiar Máscara")
        self.btn_clear_m.setStyleSheet("color: #ff9e80;")
        self.btn_clear_m.clicked.connect(self.clear_mask)
        btn_mask_actions.addWidget(self.btn_refine)
        btn_mask_actions.addWidget(self.btn_clear_m)
        mask_layout.addLayout(btn_mask_actions)

        btn_mask_io = QHBoxLayout()
        self.btn_load_m = QPushButton("Cargar PNG")
        self.btn_load_m.clicked.connect(self.load_mask)
        self.btn_save_m = QPushButton("Exportar PNG")
        self.btn_save_m.clicked.connect(self.save_mask)
        btn_mask_io.addWidget(self.btn_load_m)
        btn_mask_io.addWidget(self.btn_save_m)
        mask_layout.addLayout(btn_mask_io)
        left_layout.addWidget(self.grp_mask)
        
        # SECCIÓN 5: Registro y Ejecución
        left_layout.addWidget(QLabel("Registro de Actividad:"))
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(100)
        self.txt_log.setStyleSheet(
            "background-color: #141414; color: #d0d0d0; "
            "font-family: Consolas, monospace; font-size: 11px; "
            "border: 1px solid #333333; border-radius: 4px; padding: 4px;"
        )
        left_layout.addWidget(self.txt_log)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        self.btn_run = QPushButton("INICIAR APILADO DUAL")
        self.btn_run.setFixedHeight(45)
        self.btn_run.setStyleSheet(
            "font-weight: bold; font-size: 13px; background-color: #2b5c8f; color: white;"
        )
        self.btn_run.clicked.connect(self._on_btn_run_clicked)
        left_layout.addWidget(self.btn_run)

        splitter.addWidget(left_panel)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 10)
        layout.addWidget(splitter)

    def log_message(self, text: str):
        self.txt_log.append(text)
        self.txt_log.moveCursor(QTextCursor.End)

    def _on_capture_mode_changed(self, idx: int):
        """Si es star tracker, inhabilita el modo de apilar suelo de ráfaga."""
        is_fixed = (idx == 0)
        self.combo_ground.setEnabled(is_fixed)
        if not is_fixed:
            self.log_message("[MODO] Star Tracker activado: apilado restringido al cielo en movimiento.")

    # --- Reconstrucción de la lista visual ---
    def _refresh_list_view(self, tab_name: str):
        if tab_name == "Lights":
            widget_list = self.list_lights
            files_data = self.lights_list
        elif tab_name == "Darks":
            widget_list = self.list_darks
            files_data = self.darks_list
        elif tab_name == "Flats":
            widget_list = self.list_flats
            files_data = self.flats_list
        elif tab_name == "Bias":
            widget_list = self.list_bias
            files_data = self.bias_list
        else:
            widget_list = self.list_ground
            files_data = self.ground_list

        widget_list.clear()
        for idx, path in enumerate(files_data):
            item = QListWidgetItem(widget_list)
            is_ref = (tab_name == "Lights") and (path == self.current_ref_path)
            row_widget = FileListRow(path, tab_name=tab_name, is_ref=is_ref)
            row_widget.delete_requested.connect(self.delete_single_file)
            item.setSizeHint(row_widget.sizeHint())
            widget_list.setItemWidget(item, row_widget)

    def delete_single_file(self, tab_name: str, filepath: str):
        if tab_name == "Lights":
            file_list = self.lights_list
        elif tab_name == "Darks":
            file_list = self.darks_list
        elif tab_name == "Flats":
            file_list = self.flats_list
        elif tab_name == "Bias":
            file_list = self.bias_list
        else:
            file_list = self.ground_list

        if filepath in file_list:
            file_list.remove(filepath)
            self.log_message(f"Eliminada toma de {tab_name}: {os.path.basename(filepath)}")

            if tab_name == "Lights":
                if filepath == self.current_ref_path:
                    if self.lights_list:
                        self.load_frame_to_canvas(self.lights_list[0])
                    else:
                        self.current_ref_path = None
                        self.canvas.base_pixmap = None
                        self.canvas.update()
                self._refresh_list_view("Lights")
            elif tab_name == "Darks":
                self._refresh_list_view("Darks")
            elif tab_name == "Flats":
                self._refresh_list_view("Flats")
            elif tab_name == "Bias":
                self._refresh_list_view("Bias")
            else:
                self._refresh_list_view("Suelo")

    def add_current_tab_files(self):
        idx = self.tabs_files.currentIndex()
        tab_names = ["Lights", "Darks", "Flats", "Bias", "Suelo"]
        tab_name = tab_names[idx]

        paths, _ = QFileDialog.getOpenFileNames(
            self, f"Seleccionar {tab_name}", "",
            "Astro Images (*.nef *.cr2 *.cr3 *.arw *.dng *.tif *.tiff *.fits *.jpg *.png)"
        )
        if not paths: 
            return

        if idx == 0:
            first_add = len(self.lights_list) == 0
            for p in paths:
                if p not in self.lights_list:
                    self.lights_list.append(p)
            if first_add and self.lights_list:
                self.load_frame_to_canvas(self.lights_list[0])
            self._refresh_list_view("Lights")
            self.log_message(f"Añadidos {len(paths)} Lights. Total: {len(self.lights_list)}")

        elif idx == 1:
            for p in paths:
                if p not in self.darks_list:
                    self.darks_list.append(p)
            self._refresh_list_view("Darks")
            self.log_message(f"Añadidos {len(paths)} Darks. Total: {len(self.darks_list)}")

        elif idx == 2:
            for p in paths:
                if p not in self.flats_list:
                    self.flats_list.append(p)
            self._refresh_list_view("Flats")
            self.log_message(f"Añadidos {len(paths)} Flats. Total: {len(self.flats_list)}")
        elif idx == 3:  # Bias
            for p in paths:
                if p not in self.bias_list:
                    self.bias_list.append(p)
            self._refresh_list_view("Bias")
            self.log_message(f"Añadidos {len(paths)} Bias. Total: {len(self.bias_list)}")
        else:
            for p in paths:
                if p not in self.ground_list:
                    self.ground_list.append(p)
            self._refresh_list_view("Suelo")
            self.combo_ground.setCurrentIndex(2)
            self.log_message(f"Añadida(s) {len(paths)} toma(s) de Suelo. Modo suelo actualizado a 'Usar Toma de Pestaña Suelo'.")

    def clear_current_tab_files(self):
        idx = self.tabs_files.currentIndex()
        if idx == 0:
            self.lights_list.clear()
            self.list_lights.clear()
            self.current_ref_path = None
            self.canvas.base_pixmap = None
            self.canvas.update()
            self.log_message("Lista de Lights vaciada.")
        elif idx == 1:
            self.darks_list.clear()
            self.list_darks.clear()
            self.log_message("Lista de Darks vaciada.")
        elif idx == 2:
            self.flats_list.clear()
            self.list_flats.clear()
            self.log_message("Lista de Flats vaciada.")
        elif idx == 3:
            self.bias_list.clear()
            self.list_bias.clear()
            self.log_message("Lista de Bias vaciada.")
        else:
            self.ground_list.clear()
            self.list_ground.clear()
            self.log_message("Lista de Suelo vaciada.")

    def on_light_double_clicked(self, item):
        row = self.list_lights.row(item)
        if 0 <= row < len(self.lights_list):
            selected_path = self.lights_list[row]
            self.lights_list.remove(selected_path)
            self.lights_list.insert(0, selected_path)
            self.load_frame_to_canvas(selected_path)
            self._refresh_list_view("Lights")
            self.log_message(f"[REFERENCIA] Toma fijada como base: {os.path.basename(selected_path)}")

    def load_frame_to_canvas(self, path: str):
        self.current_ref_path = path
        self.log_message(f"Cargando toma de referencia: {os.path.basename(path)}...")
        self.setCursor(Qt.WaitCursor)
        QCoreApplication.processEvents()
        try:
            img = load_image_as_float32(path)
            self.canvas.load_image(img)
            self.log_message(f"Vista previa activa: {os.path.basename(path)}")
        except Exception as e:
            self.log_message(f"[ERROR] Al cargar {os.path.basename(path)}: {e}")
            QMessageBox.critical(self, "Error al cargar imagen", f"No se pudo cargar {path}:\n{e}")
        finally:
            self.unsetCursor()

    # --- Métodos de Máscara ---
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
        
        scribbles = self.canvas.get_scribbles_matrix()
        if scribbles is None or not (np.any(scribbles == 1) and np.any(scribbles == 2)):
            QMessageBox.warning(self, "Aviso", "Debes pintar al menos un trazo verde (cielo) y uno rojo (suelo).")
            return

        self.log_message("[MÁSCARA] Iniciando refinamiento automático...")
        self.setCursor(Qt.WaitCursor)
        QCoreApplication.processEvents()

        feather = self.spin_feather.value()
        iters = self.spin_iter.value()
        t_start = time.perf_counter()

        try:
            self.computed_mask = refine_mask_guided(
                self.canvas.orig_rgb, 
                scribbles,
                feather_radius=feather,
                iterations=iters
            )
            self.canvas.set_refined_mask(self.computed_mask)
            elapsed = time.perf_counter() - t_start
            self.log_message(f"✓ [MÁSCARA FINALIZADA] Refinamiento completado con éxito en {elapsed:.2f} s (Borde: {feather}px, Pasos: {iters}).")
        except Exception as e:
            self.log_message(f"[ERROR MÁSCARA] Falló el refinado: {e}")
            QMessageBox.critical(self, "Error", f"Fallo al refinar la máscara:\n{e}")
        finally:
            self.unsetCursor()

    def clear_mask(self):
        """Elimina todos los trazos y la máscara calculada para empezar de cero."""
        self.computed_mask = None
        if self.canvas.scribble_pixmap:
            self.canvas.scribble_pixmap.fill(Qt.transparent)
        self.canvas.refined_overlay = None
        self.canvas.update()
        self.log_message("Máscara y trazos eliminados por completo.")

    def save_mask(self):
        if self.computed_mask is None:
            return
        cfg = load_config()
        default_dir = cfg.get("masks_dir", os.getcwd())
        default_path = os.path.join(default_dir, "mascara_cielo.png")
        
        p, _ = QFileDialog.getSaveFileName(self, "Exportar Máscara PNG", default_path, "PNG (*.png)")
        if p:
            cv2.imwrite(p, (self.computed_mask * 65535.0).astype(np.uint16))
            self.log_message(f"Máscara exportada en: {os.path.basename(p)}")

    def load_mask(self):
        if self.canvas.orig_rgb is None:
            return
        cfg = load_config()
        default_dir = cfg.get("masks_dir", os.getcwd())
        
        p, _ = QFileDialog.getOpenFileName(self, "Cargar Máscara PNG", default_dir, "Imágenes (*.png *.tif *.tiff)")
        if not p:
            return
        raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if raw.ndim == 3:
            raw = raw[..., 0]
        h, w = self.canvas.orig_rgb.shape[:2]
        if raw.shape != (h, w):
            raw = cv2.resize(raw, (w, h))
        max_v = 65535.0 if raw.dtype == np.uint16 else 255.0
        self.computed_mask = np.clip(raw.astype(np.float32) / max_v, 0.0, 1.0)
        self.canvas.set_refined_mask(self.computed_mask)
        self.log_message(f"Máscara cargada desde: {os.path.basename(p)}")

    # --- Sesiones ---
    def save_project(self):
        if self.project_mgr.project_path:
            self._do_save(self.project_mgr.project_path)
        else:
            self.save_project_as()

    def save_project_as(self):
        cfg = load_config()
        default_dir = cfg.get("sessions_dir", os.getcwd())
        default_file = os.path.join(default_dir, "mi_sesion.mwstack")

        p, _ = QFileDialog.getSaveFileName(
            self, "Guardar Sesión de Apilado", default_file,
            "Astro Session (*.mwstack *.json)"
        )
        if not p:
            return
        self._do_save(p)

    def open_project(self):
        cfg = load_config()
        default_dir = cfg.get("sessions_dir", os.getcwd())

        p, _ = QFileDialog.getOpenFileName(
            self, "Cargar Sesión de Apilado", default_dir,
            "Astro Session (*.mwstack *.json)"
        )
        if not p:
            return

        try:
            data, loaded_mask = self.project_mgr.load(p)

            self.lights_list.clear()
            self.darks_list.clear()
            self.flats_list.clear()
            self.ground_list.clear()
            self.bias_list.clear()

            for fpath in data.get("light_frames", []):
                if os.path.exists(fpath):
                    self.lights_list.append(fpath)

            for fpath in data.get("dark_frames", []):
                if os.path.exists(fpath):
                    self.darks_list.append(fpath)

            for fpath in data.get("flat_frames", []):
                if os.path.exists(fpath):
                    self.flats_list.append(fpath)

            for fpath in data.get("bias_frames", []):
                if os.path.exists(fpath):
                    self.bias_list.append(fpath)

            for fpath in data.get("ground_frames", []):
                if os.path.exists(fpath):
                    self.ground_list.append(fpath)

            mode = data.get("mode", "fixed_tripod")
            self.combo_mode.setCurrentIndex(0 if mode == "fixed_tripod" else 1)
            self.combo_ground.setCurrentIndex(data.get("ground_mode_idx", 0))
            
            params = data.get("parameters", {})
            self.spin_kappa.setValue(params.get("kappa", 2.2))
            self.combo_lp_algo.setCurrentIndex(params.get("lp_method_idx", 0))
            self.slider_stack_lp.setValue(params.get("lp_strength", 50))

            if self.lights_list:
                self.load_frame_to_canvas(self.lights_list[0])

            self._refresh_list_view("Lights")
            self._refresh_list_view("Darks")
            self._refresh_list_view("Flats")
            self._refresh_list_view("Suelo")
            self._refresh_list_view("Bias")

            if loaded_mask is not None:
                self.computed_mask = loaded_mask
                self.canvas.set_refined_mask(self.computed_mask)
                self.chk_show_mask.setChecked(True)
                self.log_message("[PROYECTO] Máscara restaurada.")

            name = os.path.basename(p)
            self.lbl_session.setText(f"Sesión: {name}")
            self.session_title_changed.emit(name)
            self.log_message(
                f"[PROYECTO] Cargada: {len(self.lights_list)} Lights, "
                f"{len(self.darks_list)} Darks, {len(self.flats_list)} Flats, "
                f"{len(self.bias_list)} Bias, {len(self.ground_list)} Suelo."
            )

        except Exception as e:
            self.log_message(f"[ERROR] Al abrir sesión: {e}")
            QMessageBox.critical(self, "Error al abrir sesión", str(e))

    def _do_save(self, filepath: str):
        self.project_mgr.data["light_frames"] = self.lights_list
        self.project_mgr.data["dark_frames"] = self.darks_list
        self.project_mgr.data["flat_frames"] = self.flats_list
        self.project_mgr.data["bias_frames"] = self.bias_list
        self.project_mgr.data["ground_frames"] = self.ground_list
        self.project_mgr.data["mode"] = "fixed_tripod" if self.combo_mode.currentIndex() == 0 else "star_tracker"
        self.project_mgr.data["ground_mode_idx"] = self.combo_ground.currentIndex()
        self.project_mgr.data["parameters"] = {
            "kappa": self.spin_kappa.value(),
            "lp_method_idx": self.combo_lp_algo.currentIndex(),
            "lp_strength": self.slider_stack_lp.value()
        }

        self.project_mgr.save(filepath, mask_array=self.computed_mask)
        name = os.path.basename(filepath)
        self.lbl_session.setText(f"Sesión: {name}")
        self.session_title_changed.emit(name)
        self.log_message(f"[PROYECTO] Sesión guardada en: {name}")
        QMessageBox.information(self, "Sesión Guardada", f"Guardada en:\n{filepath}")

    def reset_session(self):
        self.lights_list.clear()
        self.darks_list.clear()
        self.flats_list.clear()
        self.ground_list.clear()
        self.list_lights.clear()
        self.list_darks.clear()
        self.list_flats.clear()
        self.bias_list.clear()
        self.list_bias.clear()      
        self.list_ground.clear()
        self.current_ref_path = None
        self.last_stacked_output_path = None

        self.project_mgr = ProjectManager()
        self.lbl_session.setText("Sesión: Sin guardar")
        self.session_title_changed.emit("")

        self.clear_mask()
        self.canvas.orig_rgb = None
        self.canvas.base_pixmap = None
        self.canvas.update()

        self.combo_mode.setCurrentIndex(0)
        self.combo_ground.setCurrentIndex(0)
        self.spin_kappa.setValue(2.2)
        self.combo_lp_algo.setCurrentIndex(0)
        self.slider_stack_lp.setValue(50)
        self.spin_feather.setValue(7)
        self.spin_iter.setValue(3)

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.txt_log.clear()

    def start_stacking(self):
        if len(self.lights_list) < 2:
            QMessageBox.warning(self, "Aviso", "Añade al menos 2 tomas de luz (Lights) para apilar.")
            return

        app_cfg = load_config()
        default_stacked_dir = app_cfg.get("stacked_dir", os.getcwd())
        default_file = os.path.join(default_stacked_dir, "resultado_dual_32bit.tiff")

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Resultado 32-bit", default_file, "TIFF (*.tiff *.tif)"
        )
        if not out_path: 
            return

        self._stack_start_time = time.perf_counter()
        self.last_stacked_output_path = out_path

        self._set_ui_busy(True)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("CANCELAR APILADO")
        self.btn_run.setStyleSheet(
            "font-weight: bold; font-size: 13px; background-color: #c62828; color: white;"
        )

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Iniciando... %p%")

        lp_mode_idx = self.combo_lp_algo.currentIndex()
        lp_methods = ["standard", "sequator_subtraction", "min_rejection", "local_norm"]
        selected_lp_method = lp_methods[lp_mode_idx]
        selected_lp_strength = self.slider_stack_lp.value() / 100.0

        cpu_workers = int(app_cfg.get("cpu_workers", max(1, (os.cpu_count() or 4) - 1)))

        try:
            saved_storage = self.window().tab_settings.combo_storage.currentData()
        except Exception:
            saved_storage = "auto"
            
        ground_modes = ["dual", "reference", "external"]
        selected_ground_mode = ground_modes[self.combo_ground.currentIndex()]

        external_path = None
        if selected_ground_mode == "external":
            if not self.ground_list:
                self._set_ui_busy(False)
                self._reset_run_button()
                QMessageBox.warning(
                    self, "Aviso", 
                    "Has seleccionado 'Usar Toma de Pestaña Suelo' pero la pestaña Suelo está vacía.\n"
                    "Añade una toma en la pestaña Suelo o cambia el modo de suelo."
                )
                return
            external_path = self.ground_list[0]

        cfg = {
            "lights": self.lights_list,
            "darks": self.darks_list,
            "flats": self.flats_list,
            "mask": self.computed_mask,
            "bias": self.bias_list,
            "mode": "fixed_tripod" if self.combo_mode.currentIndex() == 0 else "star_tracker",
            "ground_mode": selected_ground_mode,
            "external_ground_path": external_path,
            "kappa": self.spin_kappa.value(),
            "output_path": out_path,
            "lp_method": selected_lp_method,
            "lp_strength": selected_lp_strength,
            "use_gpu": is_gpu_enabled(),
            "storage_strategy": saved_storage,
            "cpu_workers": cpu_workers  
        }

        self.worker = StackingWorker(cfg)
        self.worker.progress_changed.connect(self.on_progress_changed)
        self.worker.status_changed.connect(self.log_message)
        self.worker.finished_success.connect(self.on_stack_success)
        self.worker.error_occurred.connect(self.on_stack_error)
        self.worker.cancelled.connect(self.on_stack_cancelled)
        self.worker.start()

    def on_stack_success(self, path):
        self._reset_run_button()

        if hasattr(self, '_stack_start_time') and self._stack_start_time is not None:
            elapsed = time.perf_counter() - self._stack_start_time
            mins = int(elapsed // 60)
            secs = elapsed % 60
            time_str = f"{mins}m {secs:.1f}s" if mins > 0 else f"{secs:.2f}s"
            self.log_message(f"[COMPLETADO] Tiempo total de apilado: {time_str}")
            self._stack_start_time = None

        self.log_message(f"[COMPLETADO] Imagen guardada en: {path}")
        QMessageBox.information(
            self, "Éxito", 
            f"Apilado completado con éxito.\nGuardado en:\n{path}\n\nTransfiriendo al Revelador..."
        )
        self.stacking_finished.emit(path, self.computed_mask)

    def on_stack_error(self, err_msg):
        self._reset_run_button()
        self._stack_start_time = None
        self.progress_bar.setFormat("Error")
        self.log_message(f"[ERROR CRÍTICO] {err_msg}")
        QMessageBox.critical(self, "Error durante el apilado", f"Ocurrió un error:\n{err_msg}")

    def on_stack_cancelled(self):
        """Maneja la finalización tras una interrupción manual."""
        self._reset_run_button()
        self._stack_start_time = None
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Apilado Cancelado")
        self.log_message("[INFO] Proceso detenido con éxito. Recursos y memoria liberados.")  

    def _on_stack_lp_changed(self, val: int):
        self.lbl_lp_str.setText(f"{val}%")
        
    def on_progress_changed(self, value: int):
        self.progress_bar.setValue(value)
        
        if hasattr(self, "_stack_start_time") and value > 3:
            elapsed = time.perf_counter() - self._stack_start_time
            total_est = (elapsed / value) * 100.0
            remaining = max(0.0, total_est - elapsed)
            
            rem_m, rem_s = divmod(int(remaining), 60)
            if rem_m > 0:
                eta_str = f"ETA: ~{rem_m}m {rem_s:02d}s"
            else:
                eta_str = f"ETA: ~{rem_s}s"
                
            self.progress_bar.setFormat(f"%p%  ({eta_str})")
        else:
            self.progress_bar.setFormat("%p%")

    def on_new_session_clicked(self):
        """Pide confirmación y resetea la sesión en todo el software."""
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Aviso", "No puedes reiniciar la sesión mientras el apilado está en ejecución.")
            return

        resp = QMessageBox.question(
            self, "Nueva Sesión",
            "¿Deseas reiniciar toda la sesión actual?\n\n"
            "Se vaciarán las listas de Lights/Darks, la máscara dibujada "
            "y se restablecerá el revelador a su estado inicial.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if resp != QMessageBox.Yes:
            return

        self.reset_session()
        self.new_session_requested.emit()
        self.log_message("[SESIÓN] Nueva sesión iniciada. Entorno restablecido.")
        
    def _set_ui_busy(self, busy: bool):
        """Bloquea o desbloquea controles secundarios durante el procesamiento."""
        self.grp_project.setEnabled(not busy)
        self.tabs_files.setEnabled(not busy)
        self.btn_add_files.setEnabled(not busy)
        self.btn_clear_files.setEnabled(not busy)
        self.grp_settings.setEnabled(not busy)        
        self.grp_stack_lp.setEnabled(not busy)
        self.grp_mask.setEnabled(not busy)

    def _on_btn_run_clicked(self):
        """Alterna entre iniciar el apilado o cancelarlo según el estado del worker."""
        if self.worker is not None and self.worker.isRunning():
            self.cancel_stacking()
        else:
            self.start_stacking()

    def cancel_stacking(self):
        """Solicita la detención inmediata del proceso."""
        if self.worker is not None and self.worker.isRunning():
            self.btn_run.setEnabled(False)
            self.btn_run.setText("CANCELANDO PROCESOS...")
            self.btn_run.setStyleSheet(
                "font-weight: bold; font-size: 13px; background-color: #555555; color: #aaaaaa;"
            )
            self.progress_bar.setFormat("Deteniendo... %p%")
            self.log_message("[AVISO] Solicitando parada de tareas y liberación de memoria...")
            self.worker.cancel()

    def _reset_run_button(self):
        """Devuelve el botón a su apariencia y comportamiento normal."""
        self.btn_run.setEnabled(True)
        self.btn_run.setText("INICIAR APILADO DUAL")
        self.btn_run.setStyleSheet(
            "font-weight: bold; font-size: 13px; background-color: #2b5c8f; color: white;"
        )
        self._set_ui_busy(False)
        
    def on_ground_double_clicked(self, item):
        row = self.list_ground.row(item)
        if 0 <= row < len(self.ground_list):
            path = self.ground_list[row]
            self.load_frame_to_canvas(path)
            self.log_message(f"[SUELO] Vista previa de suelo: {os.path.basename(path)}")