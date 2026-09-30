"""
Widget interactivo de curvas tonales con histograma de fondo.
Permite añadir, arrastrar y eliminar puntos de control.
Genera una LUT (Look-Up Table) de 256 niveles evaluada en tiempo real.
"""

import numpy as np
import cv2
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath


class CurveWidget(QWidget):
    curveChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 180)
        self.setFixedHeight(190)

        # Puntos de control normalizados [(x, y)], con x, y en [0.0, 1.0]
        # Por defecto: recta identidad (0,0) -> (1,1)
        self.points = [(0.0, 0.0), (1.0, 1.0)]
        self.selected_idx = None
        self.dragging = False

        # LUT precalculada (256 elementos float32 en [0.0, 1.0])
        self.lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)

        # Histograma normalizado (256 bins)
        self.hist_data = None

    def reset_curve(self):
        """Devuelve la curva a la identidad lineal."""
        self.points = [(0.0, 0.0), (1.0, 1.0)]
        self.selected_idx = None
        self._update_lut()
        self.update()
        self.curveChanged.emit()

    def set_histogram_from_image(self, img_rgb: np.ndarray):
        """Calcula el histograma de luminancia con escala logarítmica para fondo de cielo."""
        if img_rgb is None:
            self.hist_data = None
            self.update()
            return

        # Submuestreo rápido para rendimiento interactivo
        sample = img_rgb[::4, ::4]
        lum = 0.2126 * sample[..., 0] + 0.7152 * sample[..., 1] + 0.0722 * sample[..., 2]
        u8 = (np.clip(lum, 0.0, 1.0) * 255.0).astype(np.uint8)

        hist = cv2.calcHist([u8], [0], None, [256], [0, 256]).ravel()
        # Escala logarítmica para visibilizar estructuras tenues y nebulosas
        hist_log = np.log1p(hist)
        max_v = float(np.max(hist_log))
        if max_v > 1e-5:
            self.hist_data = (hist_log / max_v).astype(np.float32)
        else:
            self.hist_data = None
        self.update()

    def _update_lut(self):
        """Calcula la LUT interpolada con monotonicidad y suavizado cúbico."""
        pts = sorted(self.points, key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # Asegurar anclaje en extremos si no están exactamente en 0 o 1
        if xs[0] > 0.0:
            xs.insert(0, 0.0)
            ys.insert(0, ys[0])
        if xs[-1] < 1.0:
            xs.append(1.0)
            ys.append(ys[-1])

        grid_x = np.linspace(0.0, 1.0, 256, dtype=np.float32)

        if len(xs) == 2:
            self.lut = np.interp(grid_x, xs, ys).astype(np.float32)
        else:
            try:
                # Interpolación PCHIP (Monotonic Piecewise Cubic) para evitar sobreoscilaciones
                from scipy.interpolate import PchipInterpolator
                pchip = PchipInterpolator(xs, ys)
                self.lut = np.clip(pchip(grid_x), 0.0, 1.0).astype(np.float32)
            except Exception:
                self.lut = np.interp(grid_x, xs, ys).astype(np.float32)

        self.lut = np.clip(self.lut, 0.0, 1.0)

    def get_lut(self) -> np.ndarray:
        return self.lut

    def is_identity(self) -> bool:
        """Verifica si la curva no produce alteraciones."""
        if len(self.points) == 2 and self.points[0] == (0.0, 0.0) and self.points[1] == (1.0, 1.0):
            return True
        return False

    # --- Renderizado ---
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        margin = 12
        inner_w = w - 2 * margin
        inner_h = h - 2 * margin

        # Fondo del recuadro
        painter.fillRect(margin, margin, inner_w, inner_h, QColor("#1e1e24"))

        # Cuadrícula sutil (guías de cuartos)
        grid_pen = QPen(QColor("#2f343f"), 1, Qt.DashLine)
        painter.setPen(grid_pen)
        for i in range(1, 4):
            x = margin + inner_w * (i / 4.0)
            y = margin + inner_h * (i / 4.0)
            painter.drawLine(int(x), margin, int(x), margin + inner_h)
            painter.drawLine(margin, int(y), margin + inner_w, int(y))

        # Diagonal de referencia identidad
        painter.setPen(QPen(QColor("#3a3f4b"), 1, Qt.DotLine))
        painter.drawLine(margin, margin + inner_h, margin + inner_w, margin)

        # Histograma de fondo
        if self.hist_data is not None:
            hist_brush = QBrush(QColor(100, 140, 180, 50))
            painter.setBrush(hist_brush)
            painter.setPen(Qt.NoPen)
            hist_path = QPainterPath()
            hist_path.moveTo(margin, margin + inner_h)
            for i, val in enumerate(self.hist_data):
                px = margin + (i / 255.0) * inner_w
                py = margin + inner_h - (val * inner_h * 0.90)
                hist_path.lineTo(px, py)
            hist_path.lineTo(margin + inner_w, margin + inner_h)
            hist_path.closeSubpath()
            painter.drawPath(hist_path)

        # Curva interpolada
        curve_pen = QPen(QColor("#88c0d0"), 2)
        painter.setPen(curve_pen)
        painter.setBrush(Qt.NoBrush)
        curve_path = QPainterPath()
        for i, val in enumerate(self.lut):
            px = margin + (i / 255.0) * inner_w
            py = margin + inner_h - (val * inner_h)
            if i == 0:
                curve_path.moveTo(px, py)
            else:
                curve_path.lineTo(px, py)
        painter.drawPath(curve_path)

        # Puntos de control
        for i, (px, py) in enumerate(self.points):
            cx = margin + px * inner_w
            cy = margin + inner_h - py * inner_h
            if i == self.selected_idx:
                painter.setBrush(QBrush(QColor("#ebcb8b")))
                painter.setPen(QPen(QColor("#ffffff"), 2))
                painter.drawEllipse(QPointF(cx, cy), 5, 5)
            else:
                painter.setBrush(QBrush(QColor("#eceff4")))
                painter.setPen(QPen(QColor("#2e3440"), 1))
                painter.drawEllipse(QPointF(cx, cy), 4, 4)

        # Marco exterior
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor("#4c566a"), 1.5))
        painter.drawRect(margin, margin, inner_w, inner_h)

    # --- Eventos de Ratón ---
    def _to_norm(self, pos):
        margin = 12
        inner_w = self.width() - 2 * margin
        inner_h = self.height() - 2 * margin
        nx = np.clip((pos.x() - margin) / inner_w, 0.0, 1.0)
        ny = np.clip(1.0 - (pos.y() - margin) / inner_h, 0.0, 1.0)
        return float(nx), float(ny)

    def mousePressEvent(self, event):
        margin = 12
        inner_w = self.width() - 2 * margin
        inner_h = self.height() - 2 * margin
        mx, my = self._to_norm(event.pos())

        # Buscar si pinchó cerca de un punto existente
        clicked_idx = None
        for i, (px, py) in enumerate(self.points):
            cx = margin + px * inner_w
            cy = margin + inner_h - py * inner_h
            dist = np.hypot(event.pos().x() - cx, event.pos().y() - cy)
            if dist <= 8.0:
                clicked_idx = i
                break

        if event.button() == Qt.RightButton:
            # Eliminar punto (excepto los bordes extremos si se desea)
            if clicked_idx is not None and len(self.points) > 2:
                # No eliminar si es el anclaje inicial de la izquierda o derecha si tienen x=0 / x=1
                del self.points[clicked_idx]
                self.selected_idx = None
                self._update_lut()
                self.update()
                self.curveChanged.emit()
            return

        if event.button() == Qt.LeftButton:
            if clicked_idx is not None:
                self.selected_idx = clicked_idx
            else:
                # Insertar nuevo punto
                new_pt = (mx, my)
                self.points.append(new_pt)
                self.points.sort(key=lambda p: p[0])
                self.selected_idx = self.points.index(new_pt)
            self.dragging = True
            self._update_lut()
            self.update()
            self.curveChanged.emit()

    def mouseMoveEvent(self, event):
        if self.dragging and self.selected_idx is not None:
            mx, my = self._to_norm(event.pos())
            # Restringir movimiento en X si es el primer o último punto
            if self.selected_idx == 0 and self.points[self.selected_idx][0] == 0.0:
                mx = 0.0
            elif self.selected_idx == len(self.points) - 1 and self.points[self.selected_idx][0] == 1.0:
                mx = 1.0
            
            self.points[self.selected_idx] = (mx, my)
            self._update_lut()
            self.update()
            self.curveChanged.emit()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            # Reordenar por X al soltar para mantener consistencia
            if self.selected_idx is not None:
                cur = self.points[self.selected_idx]
                self.points.sort(key=lambda p: p[0])
                self.selected_idx = self.points.index(cur)