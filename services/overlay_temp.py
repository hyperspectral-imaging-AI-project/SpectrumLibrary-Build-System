# services/overlay_temp.py
from __future__ import annotations
import numpy as np
from PyQt5 import QtGui

class TempOverlayService:
    """레이어트리에 등록하지 않는 MapView 전용 임시 오버레이."""
    def __init__(self, map_view):
        self.map_view = map_view
        self._item = None

    def show_mask(self, mask: np.ndarray, rgba=(255, 200, 0, 120), z=990000):
        try:
            h, w = mask.shape
            r, g, b, a = rgba
            buf = np.zeros((h, w, 4), dtype=np.uint8)
            buf[mask, 0] = r; buf[mask, 1] = g; buf[mask, 2] = b; buf[mask, 3] = a

            self.clear()
            scene = self.map_view.scene() if callable(getattr(self.map_view, "scene", None)) else None
            if scene is None:
                return
            qimg = QtGui.QImage(buf.data, w, h, 4*w, QtGui.QImage.Format_RGBA8888).copy()
            pix  = QtGui.QPixmap.fromImage(qimg)
            item = scene.addPixmap(pix)
            item.setZValue(float(z))
            item.setOpacity(1.0)
            item.setOffset(0, 0)
            self._item = item
        except Exception:
            # 로깅은 상위에서
            pass

    def clear(self):
        try:
            if self._item is not None and self._item.scene() is not None:
                self._item.scene().removeItem(self._item)
        finally:
            self._item = None
