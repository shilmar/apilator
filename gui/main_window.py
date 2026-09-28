# gui/main_window.py
from PySide6.QtWidgets import QMainWindow, QTabWidget
from gui.tab_stacker import StackerTab
from gui.tab_developer import DeveloperTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Apilador & Revelador Astro - Vía Láctea")
        self.resize(1520, 940)

        self._setup_ui()

    def _setup_ui(self):
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabBar::tab { height: 36px; font-weight: bold; font-size: 13px; padding: 0 20px; }"
        )

        self.tab_stacker = StackerTab(self)
        self.tab_developer = DeveloperTab(self)

        self.tabs.addTab(self.tab_stacker, "1. Apilador Dual")
        self.tabs.addTab(self.tab_developer, "2. Revelador / Editor")

        # Conectar final de apilado con transferencia al revelador
        self.tab_stacker.stacking_finished.connect(self.on_stacking_completed)
        self.tab_stacker.session_title_changed.connect(self.update_window_title)

        self.setCentralWidget(self.tabs)

    def update_window_title(self, session_name: str):
        self.setWindowTitle(f"Apilador & Revelador Astro - [{session_name}]")

    def on_stacking_completed(self, output_path: str, computed_mask):
        # Transferir resultado al Revelador y activar la Pestaña 2
        self.tab_developer.load_image_direct(output_path, mask=computed_mask)
        self.tabs.setCurrentIndex(1)