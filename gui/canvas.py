# gui/canvas.py
import numpy as np
from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, QPoint, QRectF, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor


class MaskCanvas(QWidget):
    brush_size_changed = Signal(int)

    def __init__(self, parent=None, enable_masking=True):
        super().__init__(parent)
        self.enable_masking = enable_masking
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
        if self.enable_masking:
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
        if self.enable_masking:
            self.update_brush_cursor()

    def set_mask_visible(self, visible: bool):
        self.show_mask_overlay = visible
        self.update()

    def wheelEvent(self, event):
        if not self.enable_masking:
            return
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
            # En lugar de power(0.5) simple, normalizamos por el percentil 99 para garantizar visibilidad básica
            p_hi = np.percentile(self.orig_rgb, 99.5) if np.max(self.orig_rgb) > 0 else 1.0
            norm = np.clip(self.orig_rgb / max(p_hi, 1e-4), 0.0, 1.0)
            display = (np.power(norm, 0.4) * 255.0).astype(np.uint8)
        else:
            display = (np.clip(display_stretched, 0.0, 1.0) * 255.0).astype(np.uint8)

        display_contiguous = np.ascontiguousarray(display)
        qimg = QImage(display_contiguous.data, w, h, 3 * w, QImage.Format_RGB888).copy()

        self.base_pixmap = QPixmap.fromImage(qimg)
        if self.enable_masking:
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
        if not self.enable_masking:
            return
        if event.button() == Qt.LeftButton and self.base_pixmap and not self.base_pixmap.isNull():
            self.drawing = True
            ipt = self.widget_to_image_coords(event.position().toPoint())
            if ipt:
                self.last_img_pt = ipt
                self.paint_on_scribble(ipt, ipt)

    def mouseMoveEvent(self, event):
        if not self.enable_masking:
            return
        if self.drawing and self.base_pixmap and not self.base_pixmap.isNull():
            ipt = self.widget_to_image_coords(event.position().toPoint())
            if ipt and self.last_img_pt:
                self.paint_on_scribble(self.last_img_pt, ipt)
                self.last_img_pt = ipt

    def mouseReleaseEvent(self, event):
        if not self.enable_masking:
            return
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
            painter.drawText(self.rect(), Qt.AlignCenter, "Carga una toma o imagen para previsualizar aquí.")
            return

        rect, _ = self.get_render_rect()
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(rect, self.base_pixmap, QRectF(self.base_pixmap.rect()))

        if self.enable_masking and self.show_mask_overlay:
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