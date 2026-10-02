# gui/main_window.py
from PySide6.QtWidgets import QMainWindow, QTabWidget
from PySide6.QtGui import QIcon

from gui.tab_stacker import StackerTab
from gui.tab_developer import DeveloperTab
from gui.tab_settings import SettingsTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Apilator - Astrofotografía de Paisaje (v0.5.2)")
        self.resize(1360, 880)

        self._setup_ui()

    def _setup_ui(self):
        self.tab_widget = QTabWidget()

        # Instanciar las 3 pestañas principales
        self.tab_stacker = StackerTab()
        self.tab_developer = DeveloperTab()
        self.tab_settings = SettingsTab()

        # Añadir al contenedor
        self.tab_widget.addTab(self.tab_stacker, "1. Apilador (Stacker)")
        self.tab_widget.addTab(self.tab_developer, "2. Revelador / Editor")
        self.tab_widget.addTab(self.tab_settings, "3. Configuración / Ajustes")

        self.setCentralWidget(self.tab_widget)

        # Conectar el apilado terminado con el revelador automático
        self.tab_stacker.stacking_finished.connect(self._on_stacking_finished)
        
        #Resetear revelador al pulsar Nueva Sesión en el apilador
        self.tab_stacker.new_session_requested.connect(self.tab_developer.clear_session)

    def _on_stacking_finished(self, output_path: str):
        """Al terminar de apilar, transfiere imagen y máscara al revelador y cambia de pestaña."""
        mask = getattr(self.tab_stacker, "computed_mask", None)
        self.tab_developer.load_image_direct(output_path, mask=mask)
        self.tab_widget.setCurrentIndex(1)