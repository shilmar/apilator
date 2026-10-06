# gui/main_window.py
"""
gui/main_window.py - Ventana principal contenedora de la aplicación Apilator.
Coordina las pestañas principales (Apilador, Revelador y Ajustes) y sus señales.
"""
from typing import Optional
from PySide6.QtWidgets import QMainWindow, QTabWidget

from gui.tab_stacker import StackerTab
from gui.tab_developer import DeveloperTab
from gui.tab_settings import SettingsTab


class MainWindow(QMainWindow):
    """Ventana principal de la interfaz gráfica de Apilator."""

    BASE_TITLE = "Apilator - Astrofotografía de Paisaje (v0.5.11)"

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.BASE_TITLE)
        self.resize(1360, 880)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Inicializa las pestañas de navegación y enlaza las señales entre módulos."""
        self.tab_widget = QTabWidget()

        # Instanciar pestañas del flujo de trabajo
        self.tab_stacker = StackerTab()
        self.tab_developer = DeveloperTab()
        self.tab_settings = SettingsTab()

        self.tab_widget.addTab(self.tab_stacker, "1. Apilador (Stacker)")
        self.tab_widget.addTab(self.tab_developer, "2. Revelador / Editor")
        self.tab_widget.addTab(self.tab_settings, "3. Configuración / Ajustes")

        self.setCentralWidget(self.tab_widget)

        # 1. Transferencia automática tras finalizar apilado
        self.tab_stacker.stacking_finished.connect(self._on_stacking_finished)

        # 2. Reset del entorno completo al solicitar nueva sesión
        self.tab_stacker.new_session_requested.connect(self.tab_developer.clear_session)

        # 3. Actualización dinámica del título según la sesión activa
        self.tab_stacker.session_title_changed.connect(self._update_window_title)

    def _update_window_title(self, session_name: str) -> None:
        """Actualiza el texto de la barra de título con el archivo de proyecto activo."""
        if session_name:
            self.setWindowTitle(f"{self.BASE_TITLE} — [{session_name}]")
        else:
            self.setWindowTitle(self.BASE_TITLE)

    def _on_stacking_finished(self, output_path: str, mask: Optional[object] = None) -> None:
        """
        Transfiere el archivo 32-bit generado y su máscara calculada al Revelador,
        conmutando automáticamente a la pestaña 2.
        """
        final_mask = mask if mask is not None else getattr(self.tab_stacker, "computed_mask", None)
        self.tab_developer.load_image_direct(output_path, mask=final_mask)
        self.tab_widget.setCurrentIndex(1)