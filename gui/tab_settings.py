# gui/tab_settings.py
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QLineEdit, QPushButton, QFileDialog, QMessageBox, QDoubleSpinBox,
    QSpinBox
)
from core.config_manager import load_config, save_config

class SettingsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cfg = load_config()
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Grupo: Rutas de Herramientas Externas
        grp_paths = QGroupBox("Rutas de Herramientas Externas")
        paths_layout = QVBoxLayout(grp_paths)
        paths_layout.setSpacing(10)

        # StarNet++
        paths_layout.addWidget(QLabel("Ruta al ejecutable de StarNet++ (starnet++.exe):"))
        row_sn = QHBoxLayout()
        self.txt_starnet = QLineEdit(self.cfg.get("starnet_exe", ""))
        self.txt_starnet.setPlaceholderText("C:\\Program Files\\StarNetv2CLI_Win\\starnet++.exe")
        row_sn.addWidget(self.txt_starnet)

        btn_browse_sn = QPushButton("Examinar...")
        btn_browse_sn.clicked.connect(self._browse_starnet)
        row_sn.addWidget(btn_browse_sn)
        paths_layout.addLayout(row_sn)

        main_layout.addWidget(grp_paths)

        # Grupo: Valores por Defecto de Procesado
        grp_defaults = QGroupBox("Parámetros Generales y Rendimiento")
        def_layout = QVBoxLayout(grp_defaults)
        def_layout.setSpacing(10)

        # Kappa
        row_k = QHBoxLayout()
        row_k.addWidget(QLabel("Factor Kappa por defecto (Apilado):"))
        self.spin_kappa = QDoubleSpinBox()
        self.spin_kappa.setRange(0.5, 5.0)
        self.spin_kappa.setValue(float(self.cfg.get("default_kappa", 2.2)))
        self.spin_kappa.setSingleStep(0.1)
        row_k.addWidget(self.spin_kappa)
        row_k.addStretch()
        def_layout.addLayout(row_k)

        # Resolución máxima de Proxy
        row_px = QHBoxLayout()
        row_px.addWidget(QLabel("Dimensión máxima del proxy de previsualización (px):"))
        self.spin_proxy = QSpinBox()
        self.spin_proxy.setRange(800, 3840)
        self.spin_proxy.setSingleStep(200)
        self.spin_proxy.setValue(int(self.cfg.get("preview_max_dim", 1600)))
        row_px.addWidget(self.spin_proxy)
        row_px.addStretch()
        def_layout.addLayout(row_px)

        main_layout.addWidget(grp_defaults)

        # Botón Guardar
        btn_save = QPushButton("Guardar Cambios de Configuración")
        btn_save.setFixedHeight(40)
        btn_save.setStyleSheet("font-weight: bold; background-color: #2e6648; color: white;")
        btn_save.clicked.connect(self.save_settings)
        main_layout.addWidget(btn_save)

        main_layout.addStretch()

    def _browse_starnet(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar ejecutable de StarNet++", 
            "C:\\Program Files", "Ejecutables (*.exe);;Todos (*.*)"
        )
        if f:
            self.txt_starnet.setText(f)

    def save_settings(self):
        self.cfg["starnet_exe"] = self.txt_starnet.text().strip()
        self.cfg["default_kappa"] = self.spin_kappa.value()
        self.cfg["preview_max_dim"] = self.spin_proxy.value()
        save_config(self.cfg)
        QMessageBox.information(self, "Ajustes", "Configuración guardada correctamente en config.json.")