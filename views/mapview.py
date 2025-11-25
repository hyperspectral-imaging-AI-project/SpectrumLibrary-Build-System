# views/mapview.py
from __future__ import annotations

import numpy as np
import sip  # ★ 추가
import logging
from typing import Optional, Dict, Tuple, List

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QRect, QRectF, QPoint, QPointF, pyqtSignal, QSize
from PyQt5.QtGui import (
    QImage, QPixmap, QPen, QColor, QBrush, QPainter, QPainterPath, QPolygonF
)
from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPathItem
)
from PyQt5.QtWidgets import QRubberBand, QGraphicsRectItem
from controllers.pixel_click import ClickMode


class MapView(QGraphicsView):
    """
    - set_rgb_image(np.uint8 HxWx3) 또는 set_image(QImage)로 표시
    - ROI: set_roi_mode(None|'rect'|'free'), roiRectFinished/roiFreeFinished 신호
    - 픽셀 클릭: pixelClicked(row,col), pixelPicked(x,y), pixelPickedIndexed(x,y,idx)
    - 레이어: add_temporal_layer()/set_layer_visible()/reorder_layers()/remove_temporal_layer()
    - 휠 줌, 우/중버튼 팬
    """
    # ----- Signals -----
    pixelClicked        = pyqtSignal(int, int)            # (row, col)
    pixelPicked         = pyqtSignal(int, int)            # (x, y)
    pixelPickedIndexed  = pyqtSignal(int, int, int)       # (x, y, idx)
    roiChanged          = pyqtSignal()
    rectSelected        = pyqtSignal(int, int, int, int)  # x0,y0,x1,y1
    roiRectFinished     = pyqtSignal(QRect, object)       # (rect, mask)
    roiFreeFinished     = pyqtSignal(object, object)      # (QPolygonF(img coords), mask)
    inspectPixelPicked  = pyqtSignal(int, int)            # (y, x)
    diffusionSeedPicked = pyqtSignal(int, int)            # (y, x)
    labelPixelPicked    = pyqtSignal(int, int)            # (y, x)

    def __init__(self, parent=None):
        super().__init__(parent)

        # 렌더링/업데이트
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

        sc = QGraphicsScene(self)
        self.setScene(sc)
        self._img_item = QGraphicsPixmapItem()
        self._img_item.setAcceptedMouseButtons(Qt.NoButton)
        self._img_item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
        self._img_item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
        self.scene().addItem(self._img_item)

        # 레이어 스택 (아래→위)
        self._layer_order: List[str] = ["RGB"]
        self._layer_items: Dict[str, QGraphicsPixmapItem] = {}

        # 이미지 크기
        self.img_w = 0
        self.img_h = 0

        # 줌/팬
        self._zoom_steps = 0
        self._zoom_min = -10
        self._zoom_max = 10
        self._zoom_step = 1.25
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._panning = False
        self._pan_last = QPoint()

        # ROI 상태
        self._roi_mode = None
        self._rubber: Optional[QRubberBand] = None
        self._roi_origin: Optional[QPoint] = None
        self._poly_points: List[QPoint] = []
        self._poly_path_item: Optional[QGraphicsPathItem] = None  # ★ 폴리곤/자유형 실시간 프리뷰

        self._free_path: Optional[QPainterPath] = None
        self._free_path_item: Optional[QGraphicsPathItem] = None
        self._roi_overlay_items: List[QGraphicsPixmapItem] = []

        # 픽셀 픽킹/제한
        self._pick_counter = 1
        self._picking_enabled = True
        self._allowed_rect_img: Optional[QRect] = None
        self._allowed_mask: Optional[np.ndarray] = None     # ★ 클릭 허용(ROI) 마스크
        self.click_mode = ClickMode.NONE
        self._roi_active = False

        # ★ Diffusion 모드 동안 인덱스 지속 표기를 위한 상태
        self._idx_item: Optional[QtWidgets.QGraphicsSimpleTextItem] = None
        self._cross_item: Optional[QtWidgets.QGraphicsPathItem] = None    # ★ 십자 마커
        self._idx_persistent: bool = False
        
        # Marker
        self._seed_markers: list[tuple[QtWidgets.QGraphicsSimpleTextItem, QtWidgets.QGraphicsPathItem]] = []
        self._keep_seed_markers: bool = False  # DiffusionDialog가 켜져 있는 동안 True
        self._seed_palette = {}
        
        # ★ 추가: 컨텍스트별 라벨 마커 저장 {ctx: [(h_item, v_item, text_item), ...]}
        self._marker_layers: dict[str, list[tuple[QtWidgets.QGraphicsLineItem,
                                                QtWidgets.QGraphicsLineItem,
                                                QtWidgets.QGraphicsSimpleTextItem]]] = {}

    # =========================================================
    # 이미지 표시
    # =========================================================
    def set_rgb_image(self, rgb: np.ndarray, do_fit: bool = True) -> None:
        """
        np.ndarray (H,W,3) → QImage(RGB888)로 표시
        - clip 사용 안 함: 입력 범위를 신뢰
        - float: 0..1 가정 → *255 후 uint8 캐스팅 (no clip)
        - int: dtype이 8bit 초과여도 스케일링 없이 그대로 uint8 캐스팅 (mod 256 가능성 주의)
        """
        import numpy as np
        from PyQt5.QtGui import QImage, QPixmap

        assert rgb.ndim == 3 and rgb.shape[2] == 3, "RGB는 (H,W,3)이어야 합니다."

        arr = rgb

        # 필요 시 BGR → RGB 전환을 쓰려면 주석 해제
        # if getattr(self, "_source_is_bgr", False):
        #     arr = arr[..., ::-1]

        # dtype별 변환 (clip 없음)
        if np.issubdtype(arr.dtype, np.floating):
            a = (arr.astype(np.float32) * 255.0)  # no clip
            arr8 = a.astype(np.uint8)            # overflow 시 잘림/랩핑 주의
        else:
            # 정수형은 그대로 uint8 캐스팅 (예: uint16 → uint8 시 하위 8비트만 남음)
            arr8 = arr.astype(np.uint8)

        # 메모리 연속 보장
        if not arr8.flags["C_CONTIGUOUS"]:
            arr8 = np.ascontiguousarray(arr8)

        h, w, _ = arr8.shape

        # QImage stride 대응 복사 (clip 없음)
        qimg = QImage(w, h, QImage.Format_RGB888)
        actual_bpl = qimg.bytesPerLine()
        expected_bpl = w * 3

        ptr = qimg.bits()
        ptr.setsize(h * actual_bpl)

        if actual_bpl == expected_bpl:
            # 한 번에 복사
            ptr[:] = arr8.ravel().tobytes()[:h * actual_bpl]
        else:
            # 행 단위 복사 (패딩 고려)
            row_bytes = expected_bpl
            buf = arr8.reshape(h, w * 3)
            for y in range(h):
                off = y * actual_bpl
                ptr[off:off + row_bytes] = buf[y].tobytes()

        # 표시
        self.set_image(qimg, do_fit=do_fit)



    def set_image(self, qimg: QImage, do_fit: bool = False) -> None:
        if qimg.isNull():
            return
        self.img_w, self.img_h = qimg.width(), qimg.height()
        self._img_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene().setSceneRect(0, 0, self.img_w, self.img_h)
        if do_fit:
            self.fit_view()
            
    def _color_for_cid(self, cid:int):
        return self._seed_palette.get(int(cid), QtGui.QColor(0,0,255)) 
    
    def set_seed_palette(self, palette: dict):
        self._seed_palette = {int(k): QtGui.QColor(v) for k, v in palette.items()}

    def fit_view(self) -> None:
        if self.img_w and self.img_h:
            self._zoom_steps = 0
            self.setTransform(QtGui.QTransform())
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def reset_zoom(self) -> None:
        self._zoom_steps = 0
        self.setTransform(QtGui.QTransform())

    # =========================================================
    # 레이어 (RGB 위에 올리는 임시/오버레이)
    # =========================================================
    def add_temporal_layer(self, name: str, image: np.ndarray, opacity: float = 0.6, force_unique: bool = True) -> str:
        # 오버레이 크기 검증: 원본 이미지 크기와 일치해야 함
        if hasattr(self, 'img_h') and hasattr(self, 'img_w') and self.img_h and self.img_w:
            expected_h, expected_w = int(self.img_h), int(self.img_w)
            if image.ndim == 2:
                actual_h, actual_w = image.shape
            elif image.ndim == 3:
                actual_h, actual_w = image.shape[:2]
            else:
                raise ValueError(f"오버레이 이미지는 2D 또는 3D여야 합니다: {image.shape}")
            
            if actual_h != expected_h or actual_w != expected_w:
                logging.warning(f"[MapView] 오버레이 크기 불일치: {name} - 예상 ({expected_h}, {expected_w}), 실제 ({actual_h}, {actual_w})")
                # 크기 불일치 시 원본 크기로 리샘플링
                from scipy.ndimage import zoom
                zoom_h = expected_h / actual_h
                zoom_w = expected_w / actual_w
                if image.ndim == 2:
                    image = zoom(image, (zoom_h, zoom_w), order=1)
                else:
                    image = zoom(image, (zoom_h, zoom_w, 1), order=1)
                logging.info(f"[MapView] 오버레이 리샘플링 완료: {name}")
        
        qimg = self._np_to_qimage(image).copy()
        pm = QPixmap.fromImage(qimg)

        # 동일 이름 교체
        if not force_unique and name in self._layer_items:
            it = self._layer_items[name]
            it.setPixmap(pm)
            it.setOpacity(float(opacity))
            it.setVisible(True)
            # 위치를 (0,0)으로 명시적으로 설정 (원본 이미지와 정렬)
            it.setPos(0, 0)
            return name

        # 유니크 이름 부여
        final = name
        if force_unique and final in self._layer_items:
            i = 1
            while f"{name}#{i}" in self._layer_items:
                i += 1
            final = f"{name}#{i}"

        it = QGraphicsPixmapItem(pm)
        it.setOpacity(float(opacity))
        it.setVisible(True)
        # 위치를 (0,0)으로 명시적으로 설정 (원본 이미지와 정렬)
        it.setPos(0, 0)
        # 오버레이가 마우스를 가로채지 않도록
        it.setAcceptedMouseButtons(Qt.NoButton)
        it.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
        it.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
        self.scene().addItem(it)

        self._layer_items[final] = it
        if final not in self._layer_order:
            self._layer_order.append(final)
        self._apply_z()
        return final

    def remove_temporal_layer(self, name: str) -> None:
        it = self._layer_items.pop(name, None)
        if it:
            self.scene().removeItem(it)
        try:
            self._layer_order.remove(name)
        except ValueError:
            pass
        self._apply_z()

    def remove_rgb_image(self) -> None:
        """RGB 베이스 이미지를 제거하고 빈 이미지로 설정."""
        if hasattr(self, "_img_item") and self._img_item:
            # 빈 QPixmap으로 설정하여 이미지 제거
            from PyQt5.QtGui import QPixmap
            self._img_item.setPixmap(QPixmap())
            self._img_item.setVisible(False)
            # 이미지 크기 초기화
            self.img_w = 0
            self.img_h = 0
            # 레이어 순서에서 RGB 제거 (있는 경우)
            try:
                self._layer_order.remove("RGB")
            except ValueError:
                pass

    def set_layer_visible(self, name: str, on: bool) -> None:
        if name == "RGB":
            self._img_item.setVisible(bool(on))
            return
        it = self._layer_items.get(name)
        if it:
            it.setVisible(bool(on))

    def layer_visible(self, name: str) -> bool:
        if name == "RGB":
            return self._img_item.isVisible()
        it = self._layer_items.get(name)
        return bool(it.isVisible()) if it else False

    def layer_names(self) -> List[str]:
        return list(self._layer_order)

    def reorder_layers(self, new_order_top_to_bottom: List[str]) -> None:
        """
        new_order_top_to_bottom: 트리에 보이는 순서(위→아래)
        내부 Z적용은 아래→위가 필요하므로 여기서 뒤집어 저장한다.
        """
        have = {"RGB"} | set(self._layer_items.keys())
        # 1) 존재하는 항목만 보존(Top→Bottom)
        ordered_tb = [nm for nm in new_order_top_to_bottom if nm in have]
        # 2) 누락 항목은 말미에 보존
        for nm in (["RGB"] + list(self._layer_items.keys())):
            if nm not in ordered_tb:
                ordered_tb.append(nm)
        # 3) 내부 저장은 Bottom→Top
        self._layer_order = list(reversed(ordered_tb))
        self._apply_z()

    def _apply_z(self) -> None:
        for z, nm in enumerate(self._layer_order):
            if nm == "RGB":
                self._img_item.setZValue(z)
            else:
                it = self._layer_items.get(nm)
                if it:
                    it.setZValue(z)

    # =========================================================
    # ROI
    # =========================================================
    def set_roi_mode(self, mode):
        print(f"[MapView] set_roi_mode called: mode={mode}")
        self._roi_mode = mode
        # 프리뷰 정리 ...
        self._roi_active = bool(mode in ('rect', 'free', 'poly'))
        print(f"[MapView] ROI active: {self._roi_active}")
        self._apply_cursor()

    def clear_roi(self) -> None:
        self.roi_rect = None
        self._selecting = False
        for it in self._roi_overlay_items:
            try:
                self.scene().removeItem(it)
            except Exception:
                pass
        self._roi_overlay_items.clear()
        self.roiChanged.emit()

    def get_roi_mask(self, rows: int, cols: int) -> Optional[np.ndarray]:
        if not hasattr(self, "roi_rect") or not self.roi_rect:
            return None
        r = self.roi_rect
        x0, y0, x1, y1 = r.left(), r.top(), r.right() + 1, r.bottom() + 1
        x0 = max(0, min(cols, x0)); x1 = max(0, min(cols, x1))
        y0 = max(0, min(rows, y0)); y1 = max(0, min(rows, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        m = np.zeros((rows, cols), dtype=bool)
        m[y0:y1, x0:x1] = True
        return m

    def _image_size(self) -> tuple[int, int]:
        """현재 캔버스의 픽셀 크기(H,W). 이미지 한 장을 (0,0) 기준으로 올렸다는 가정."""
        if hasattr(self, "_img_item") and self._img_item:
            br = self._img_item.pixmap().rect()
            return br.height(), br.width()
        # fallback: scene rect
        r = self.sceneRect()
        return int(r.height()), int(r.width())

    def _rect_to_mask(self, rect: QRect) -> np.ndarray:
        H, W = self._image_size()
        r = rect.intersected(QRect(0, 0, W, H))
        if r.isEmpty():
            return np.zeros((H, W), dtype=bool)
        img = QImage(W, H, QImage.Format_Alpha8)
        img.fill(0)
        p = QPainter(img); p.fillRect(r, QColor(255, 255, 255)); p.end()

        ptr = img.bits()
        ptr.setsize(img.byteCount())
        bpl = img.bytesPerLine()
        arr = np.frombuffer(ptr, dtype=np.uint8, count=bpl * H).reshape(H, bpl)
        return (arr[:, :W] > 0)

    def _poly_to_mask(self, poly: QPolygonF) -> np.ndarray:
        H, W = self._image_size()
        img = QImage(W, H, QImage.Format_Alpha8)
        img.fill(0)
        p = QPainter(img)
        p.setBrush(QColor(255, 255, 255)); p.setPen(Qt.NoPen)
        p.drawPolygon(poly); p.end()

        ptr = img.bits()
        ptr.setsize(img.byteCount())
        bpl = img.bytesPerLine()
        arr = np.frombuffer(ptr, dtype=np.uint8, count=bpl * H).reshape(H, bpl)
        return (arr[:, :W] > 0)

    # =========================================================
    # 픽셀 픽킹/제한
    # =========================================================
    def enable_picking(self, enabled: bool) -> None:
        self._picking_enabled = bool(enabled)
        self._apply_cursor()

    def set_allowed_rect_img(self, rect: Optional[QRect]) -> None:
        """이미지 좌표계에서 클릭 허용 영역 제한 (None 이면 제한 없음)"""
        self._allowed_rect_img = QRect(rect) if rect else None

    def set_allowed_mask(self, mask: Optional[np.ndarray]) -> None:
        """
        이미지 좌표계 bool (H,W) 마스크로 클릭 허용 영역 제한.
        None이면 제한 해제.
        """
        if mask is None:
            self._allowed_mask = None
            return
        a = np.asarray(mask)
        if a.ndim != 2:
            raise ValueError("allowed_mask는 2D(bool) 마스크여야 합니다.")
        self._allowed_mask = a.astype(bool)

    # =========================================================
    # Events
    # =========================================================
    def wheelEvent(self, ev: QtGui.QWheelEvent):
        delta = ev.angleDelta().y()
        if delta > 0:
            if self._zoom_steps < self._zoom_max:
                self._zoom_steps += 1
                self.scale(self._zoom_step, self._zoom_step)
        elif delta < 0:
            if self._zoom_steps > self._zoom_min:
                self._zoom_steps -= 1
                self.scale(1.0 / self._zoom_step, 1.0 / self._zoom_step)
        else:
            super().wheelEvent(ev)

    # ----- 인덱스(선형) 지속 표기 유틸 -----
    def _ensure_idx_item(self) -> QtWidgets.QGraphicsSimpleTextItem:
        """상시 인덱스 표시에 사용할 텍스트 아이템을 보장"""
        if self._idx_item is not None:
            return self._idx_item
        ti = QtWidgets.QGraphicsSimpleTextItem("")
        f = QtGui.QFont(); f.setPointSize(10); f.setBold(True)
        ti.setFont(f)
        ti.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0)); pen.setWidth(3)
        ti.setPen(pen)
        ti.setZValue(1e6)
        sc = self.scene() if callable(getattr(self, "scene", None)) else None
        if sc:
            sc.addItem(ti)
        self._idx_item = ti
        return ti
    
    def _ensure_cross_item(self):
        if self._cross_item: return self._cross_item
        it = QtWidgets.QGraphicsPathItem()
        it.setZValue(1e6)
        pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
        pen.setWidth(2)
        it.setPen(pen)
        it.setBrush(QBrush(Qt.NoBrush))      # ★ BrushStyle → QBrush로 감싸기
        it.setAcceptedMouseButtons(Qt.NoButton)
        self.scene().addItem(it)
        self._cross_item = it
        return it

    def _update_persistent_marker(self, y: int, x: int):
        """선형 인덱스 + 작은 십자(+) 마커 동시 업데이트"""
        H, W = int(self.img_h), int(self.img_w)
        if not (0 <= y < H and 0 <= x < W): return
        lin = y * W + x

        # 숫자
        ti = self._ensure_idx_item()
        ti.setText(str(lin))
        ti.setPos(float(x + 6), float(y + 6))

        # 십자(길이 6px)
        cross = self._ensure_cross_item()
        L = 6
        p = QtGui.QPainterPath(QtCore.QPointF(x, y - L))
        p.lineTo(x, y + L)
        p.moveTo(x - L, y)
        p.lineTo(x + L, y)
        cross.setPath(p)

    def _hide_persistent_marker(self):
        if self._idx_item and self._idx_item.scene():
            self._idx_item.scene().removeItem(self._idx_item)
        if self._cross_item and self._cross_item.scene():
            self._cross_item.scene().removeItem(self._cross_item)
        self._idx_item = None
        self._cross_item = None

    # views/mapview.py의 mousePressEvent 수정
    def mousePressEvent(self, e):
        print(self._roi_active, e.button(), self.click_mode)

        # ROI 모드 처리 (기존 코드 유지)
        if self._roi_mode == 'rect' and e.button() == Qt.LeftButton:
            if self._rubber is None:
                self._rubber = QRubberBand(QRubberBand.Rectangle, self)
            self._roi_origin = e.pos()
            self._rubber.setGeometry(QRect(self._roi_origin, QSize()))
            self._rubber.show()
            return
        elif self._roi_mode == 'free' and e.button() == Qt.LeftButton:
            self._poly_points.append(e.pos())
            if self._poly_path_item is None:
                self._poly_path_item = QGraphicsPathItem()
                self._poly_path_item.setZValue(9999)
                self._poly_path_item.setPen(QPen(Qt.white, 2, Qt.DashLine))
                self._poly_path_item.setBrush(QBrush(Qt.NoBrush))
                self.scene().addItem(self._poly_path_item)
            return
        elif self._roi_mode == 'poly' and e.button() == Qt.LeftButton:
            self._poly_points.append(e.pos())
            if self._poly_path_item is None:
                self._poly_path_item = QGraphicsPathItem()
                self._poly_path_item.setZValue(9999)
                self._poly_path_item.setPen(QPen(Qt.white, 2, Qt.DashLine))
                self._poly_path_item.setBrush(QBrush(Qt.NoBrush))
                self.scene().addItem(self._poly_path_item)
            self._update_poly_preview(current_mouse=e.pos())
            return

        # ----- 픽셀 클릭 처리 -----
        if not self._roi_active and e.button() == Qt.LeftButton:
            y, x = self._to_image_coord(e.pos())
            if 0 <= y < self.img_h and 0 <= x < self.img_w:

                # ★ 허용 Rect 제한 (있을 때만)
                if self._allowed_rect_img is not None:
                    if not self._allowed_rect_img.contains(QPoint(x, y)):
                        return

                # ★ 허용 마스크 제한 (확산 시드 모드일 때 ROI 마스크 강제)
                if self.click_mode == ClickMode.DIFFUSION_SEED and self._allowed_mask is not None:
                    try:
                        if not bool(self._allowed_mask[y, x]):
                            return
                    except Exception:
                        return

                # ★ Diffusion 모드에서는 'Index만' 지속 표시 업데이트
                if self.click_mode == ClickMode.DIFFUSION_SEED and self._idx_persistent:
                    self._update_persistent_marker(y, x)

                # 모드별 시그널 방출
                if self.click_mode in (ClickMode.INSPECT, ClickMode.ANALYSIS):
                    self.inspectPixelPicked.emit(y, x)
                    e.accept(); return
                elif self.click_mode == ClickMode.DIFFUSION_SEED:
                # 기존: self._update_persistent_marker(y, x)
                    if self._idx_persistent:
                        if self._keep_seed_markers:
                            self._add_persistent_marker(y, x)   # ← 누적 추가
                        else:
                            # 과거 단일-표시 동작 유지 (원하면 삭제)
                            self._hide_persistent_marker()
                            self._add_persistent_marker(y, x)
                    self.diffusionSeedPicked.emit(y, x)
                    e.accept(); return
                elif self.click_mode == ClickMode.LABEL:
                    self.labelPixelPicked.emit(y, x)
                    e.accept(); return
            return

        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._roi_mode == 'rect' and self._roi_origin:
            self._rubber.setGeometry(QRect(self._roi_origin, e.pos()).normalized())
            return
        elif self._roi_mode == 'free' and self._poly_points:
            # 자유형: 일정 거리 이상 움직이면 점 추가
            if (self._poly_points[-1] - e.pos()).manhattanLength() > 2:
                self._poly_points.append(e.pos())
            self._update_poly_preview()  # 프리뷰 갱신
            return
        elif self._roi_mode == 'poly' and self._poly_points:
            self._update_poly_preview(current_mouse=e.pos())  # 현재 마우스까지 선 미리보기
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._roi_mode == 'rect' and self._roi_origin and e.button() == Qt.LeftButton:
            view_rect = QRect(self._roi_origin, e.pos()).normalized()
            self._rubber.hide()
            self._roi_origin = None
            # View → Scene → Image 좌표 사각형
            rectf = QRectF(self.mapToScene(view_rect.topLeft()),
                           self.mapToScene(view_rect.bottomRight())).normalized()
            rect = QRect(int(rectf.left()), int(rectf.top()),
                         int(rectf.width()), int(rectf.height()))
            mask = self._rect_to_mask(rect)
            self.roiRectFinished.emit(rect, mask)
            self.set_roi_mode(None)
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        if self._roi_mode == 'free' and len(self._poly_points) >= 3:
            poly_scene = QPolygonF([self.mapToScene(p) for p in self._poly_points])
            poly_img = QPolygonF([QPointF(int(pt.x()), int(pt.y())) for pt in poly_scene])
            mask = self._poly_to_mask(poly_img)
            self.roiFreeFinished.emit(poly_img, mask)
            # 프리뷰 제거
            if self._poly_path_item is not None:
                self.scene().removeItem(self._poly_path_item)
                self._poly_path_item = None
            self.set_roi_mode(None)
            return
        elif self._roi_mode == 'poly' and len(self._poly_points) >= 3:
            # 다각형도 더블클릭으로 완료
            poly_scene = QPolygonF([self.mapToScene(p) for p in self._poly_points])
            poly_img = QPolygonF([QPointF(int(pt.x()), int(pt.y())) for pt in poly_scene])
            mask = self._poly_to_mask(poly_img)
            self.roiFreeFinished.emit(poly_img, mask)
            if self._poly_path_item is not None:
                self.scene().removeItem(self._poly_path_item)
                self._poly_path_item = None
            self.set_roi_mode(None)
            return
        super().mouseDoubleClickEvent(e)

    # =========================================================
    # 내부 유틸
    # =========================================================
    def _np_to_qimage(self, arr: np.ndarray) -> QImage:
        a = np.asarray(arr)
        if a.ndim == 2:
            h, w = a.shape
            if a.dtype != np.uint8:
                a = np.clip(a.astype(np.float32), 0, 255).astype(np.uint8)
            if not a.flags["C_CONTIGUOUS"]:
                a = np.ascontiguousarray(a)
            return QImage(a.data, w, h, w, QImage.Format_Grayscale8)
        if a.ndim == 3 and a.shape[2] in (3, 4):
            h, w, c = a.shape
            if a.dtype != np.uint8:
                a = np.clip(a.astype(np.float32), 0, 255).astype(np.uint8)
            if not a.flags["C_CONTIGUOUS"]:
                a = np.ascontiguousarray(a)
            fmt = QImage.Format_RGBA8888 if c == 4 else QImage.Format_RGB888
            return QImage(a.data, w, h, w * c, fmt)
        raise ValueError("이미지는 (H,W) 또는 (H,W,3|4) uint8 이어야 합니다.")

    def _polygon_to_mask(self, poly_img: QPolygonF) -> np.ndarray:
        H, W = self.img_h, self.img_w
        mask = np.zeros((H, W), dtype=np.uint8)
        if H == 0 or W == 0 or poly_img.isEmpty():
            return mask.astype(bool)
        qimg = QImage(W, H, QImage.Format_Grayscale8)
        qimg.fill(0)
        p = QPainter(qimg)
        p.setPen(Qt.NoPen)
        p.setBrush(Qt.white)
        p.drawPolygon(poly_img)
        p.end()
        ptr = qimg.bits(); ptr.setsize(qimg.byteCount())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(H, W)
        return (arr > 0)

    def _show_roi_overlay(self, mask: np.ndarray, color: QColor):
        if mask.ndim != 2:
            return
        H, W = mask.shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[..., 0] = color.red()
        rgba[..., 1] = color.green()
        rgba[..., 2] = color.blue()
        rgba[..., 3] = (mask.astype(np.uint8) * color.alpha())
        qimg = QImage(rgba.data, W, H, 4 * W, QImage.Format_RGBA8888).copy()
        pm = QPixmap.fromImage(qimg)
        it = QGraphicsPixmapItem(pm)
        it.setZValue(800)
        # ROI 오버레이도 마우스 비활성
        it.setAcceptedMouseButtons(Qt.NoButton)
        it.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
        it.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
        self.scene().addItem(it)
        self._roi_overlay_items.append(it)

    def _update_poly_preview(self, current_mouse: QPoint | None = None):
        """_poly_points(+마우스 위치)로 QPainterPath를 만들어 프리뷰 표시"""
        if self._poly_path_item is None or not self._poly_points:
            return
        pts = self._poly_points.copy()
        if current_mouse is not None:
            pts.append(current_mouse)
        path = QPainterPath(self.mapToScene(pts[0]))
        for p in pts[1:]:
            path.lineTo(self.mapToScene(p))
        self._poly_path_item.setPath(path)

    # === 클릭 모드/ROI 활성화 제어 ===
    def set_click_mode(self, mode: ClickMode):
        self.click_mode = mode
        self._idx_persistent = (mode == ClickMode.DIFFUSION_SEED)
        if not self._idx_persistent:
            self._hide_persistent_marker()   # ★ 십자+숫자 동시 제거
        self._apply_cursor()

    def set_roi_active(self, active: bool):
        """ROI 드로잉 중엔 픽셀 픽업을 차단하기 위해 컨트롤러에서 True/False로 제어."""
        self._roi_active = bool(active)

    # views/mapview.py
    def _to_image_coord(self, pos) -> tuple[int, int]:
        sp = self.mapToScene(pos)
        # ε(−1e-12) 제거: 반올림 기준을 정확히 0.5에 둔다
        y = int(np.floor(sp.y()))
        x = int(np.floor(sp.x()))
        if self.img_h and self.img_w:
            y = max(0, min(self.img_h - 1, y))
            x = max(0, min(self.img_w - 1, x))
        return y, x

    def _apply_cursor(self):
        """
        ROI/ClickMode 상태에 맞춰 커서를 일관 적용.
        - viewport()에만 커서 지정
        - 남아있을 수 있는 전역 overrideCursor는 깨끗이 복구
        """
        try:
            app = QtWidgets.QApplication.instance()
            while app and QtGui.QGuiApplication.overrideCursor():
                QtGui.QGuiApplication.restoreOverrideCursor()
        except Exception:
            pass

        print(f"[MapView] _apply_cursor: roi_mode={self._roi_mode}, click_mode={self.click_mode}")

        if self._roi_mode in ('rect', 'free', 'poly'):
            self.viewport().setCursor(Qt.CrossCursor)
            return

        if self.click_mode in (ClickMode.LABEL, ClickMode.DIFFUSION_SEED, ClickMode.INSPECT, ClickMode.ANALYSIS):
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.viewport().unsetCursor()
            self.unsetCursor()
            self.viewport().setCursor(Qt.ArrowCursor)

    def enterEvent(self, e):
        self._apply_cursor()
        return super().enterEvent(e)

    def leaveEvent(self, e):
        self.viewport().unsetCursor()
        return super().leaveEvent(e)

    def resizeEvent(self, e):
        ret = super().resizeEvent(e)
        self._apply_cursor()
        return ret

    def enable_seed_marker_persistence(self, keep: bool):
        """DiffusionDialog 열림/닫힘에 따라 마커 유지 여부 토글"""
        self._keep_seed_markers = bool(keep)
        if not self._keep_seed_markers:
            self.clear_seed_markers()

    def clear_seed_markers(self):
        """모든 시드 마커(텍스트+십자) 제거"""
        for ti, cross in self._seed_markers:
            if ti.scene(): ti.scene().removeItem(ti)
            if cross.scene(): cross.scene().removeItem(cross)
        self._seed_markers.clear()
        
    def _add_persistent_marker(self, y: int, x: int, cid: Optional[int] = None):
        """현재 클릭 지점에 텍스트+십자 마커를 '추가' (누적)"""
        H, W = int(self.img_h), int(self.img_w)
        if not (0 <= y < H and 0 <= x < W): 
            return
        lin = y * W + x

        # 텍스트
        ti = QtWidgets.QGraphicsSimpleTextItem(str(lin))
        f = QtGui.QFont(); f.setPointSize(10); f.setBold(True)
        ti.setFont(f)
        ti.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0)); pen.setWidth(3)
        ti.setPen(pen)
        ti.setZValue(1e6)
        ti.setPos(float(x + 6), float(y + 6))
        self.scene().addItem(ti)

        # 십자 (cid 색상 적용)
        cross = QGraphicsPathItem()
        cross.setZValue(1e6)
        # cid가 있으면 팔레트에서 색상 가져오기, 없으면 파랑
        color = self._seed_palette.get(int(cid)) if cid is not None else QtGui.QColor(0, 0, 255)
        if color is None:
            color = QtGui.QColor(0, 0, 255)
        cp = QtGui.QPen(color)
        cp.setWidth(2)
        cross.setPen(cp)
        cross.setBrush(QBrush(Qt.NoBrush))
        cross.setAcceptedMouseButtons(Qt.NoButton)
        L = 6
        p = QtGui.QPainterPath(QtCore.QPointF(x, y - L))
        p.lineTo(x, y + L)
        p.moveTo(x - L, y)
        p.lineTo(x + L, y)
        cross.setPath(p)
        self.scene().addItem(cross)

        self._seed_markers.append((ti, cross))


    def rebuild_seed_markers(self, seeds: list[dict]):
        """[{y,x,cid}, ...] 기준으로 마커 전체 재구성"""
        self.clear_seed_markers()
        for s in seeds:
            y, x = int(s["y"]), int(s["x"])
            cid = int(s.get("cid", -1)) if "cid" in s else None
            # 이미 구현한 '추가' 유틸이 있으면 사용하세요. 없으면 아래 즉석 구현:
            # (텍스트)
            ti = QtWidgets.QGraphicsSimpleTextItem(str(y * int(self.img_w) + x))
            f = QtGui.QFont(); f.setPointSize(10); f.setBold(True)
            ti.setFont(f)
            ti.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
            pen = QtGui.QPen(QtGui.QColor(0, 0, 0)); pen.setWidth(3)
            ti.setPen(pen); ti.setZValue(1e6)
            ti.setPos(float(x + 6), float(y + 6))
            self.scene().addItem(ti)
            # (십자) cid 색상 적용
            cross = QGraphicsPathItem()
            cross.setZValue(1e6)
            color = self._seed_palette.get(int(cid)) if cid is not None and cid >= 0 else QtGui.QColor(0, 0, 255)
            if color is None:
                color = QtGui.QColor(0, 0, 255)
            cp = QtGui.QPen(color)
            cp.setWidth(2)
            cross.setPen(cp)
            cross.setBrush(QBrush(Qt.NoBrush))
            cross.setAcceptedMouseButtons(Qt.NoButton)
            L = 6
            p = QtGui.QPainterPath(QtCore.QPointF(x, y - L))
            p.lineTo(x, y + L); p.moveTo(x - L, y); p.lineTo(x + L, y)
            cross.setPath(p)
            self.scene().addItem(cross)

            self._seed_markers.append((ti, cross))
            
    # === 컨텍스트별 픽셀 마커 (중심 빨강 + 주변 검정 + 작은 번호) ===
    def add_ctx_marker(self, ctx: str, y: int, x: int, number: int,
                       color: Optional[QtGui.QColor] = None, size: int = 1):
        """
        ctx   : 마커 그룹 이름 (예: 'ANALYSIS_CLASS_ALL')
        (y,x) : 이미지 좌표
        number: 표시할 번호
        color : 중심 픽셀 색 (기본 빨간색)
        size  : 중심에서 몇 픽셀 반경까지 검은 테두리를 만들지 (기본 1 → 3x3)
        """
        sc = self.scene()
        if sc is None:
            return

        H, W = int(self.img_h), int(self.img_w)
        if not (0 <= y < H and 0 <= x < W):
            return

        px, py = float(x), float(y)

        # ----- 중심 빨간색 + 주변 검정 픽셀 패치 생성 -----
        radius = max(1, int(size))           # 1이면 3x3, 2면 5x5
        dim = 2 * radius + 1                 # 패치 한 변 길이
        img = QImage(dim, dim, QImage.Format_ARGB32)
        img.fill(Qt.transparent)

        center_color = color or QtGui.QColor(255, 0, 0)  # 중심 빨간색
        border_color = QtGui.QColor(0, 0, 0)             # 주변 검정

        for j in range(dim):
            for i in range(dim):
                if i == radius and j == radius:
                    # 중심 픽셀 → 빨간색
                    img.setPixelColor(i, j, center_color)
                else:
                    # 주변 픽셀 → 검정
                    img.setPixelColor(i, j, border_color)

        pm = QPixmap.fromImage(img)
        patch_item = QtWidgets.QGraphicsPixmapItem(pm)
        patch_item.setZValue(1e6 + 1)
        # 중심이 (x,y)에 오도록 좌상단을 (x-radius, y-radius)에 배치
        patch_item.setPos(px - radius, py - radius)
        patch_item.setAcceptedMouseButtons(Qt.NoButton)
        patch_item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
        patch_item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
        sc.addItem(patch_item)

        # ----- 번호: 기존보다 더 작게 (예: 6pt) -----
        txt = QtWidgets.QGraphicsSimpleTextItem(str(int(number)))
        f = txt.font()
        f.setPointSizeF(1.0)  # ★ 기존 8.0 → 6.0 로 더 줄임
        f.setBold(True)        # 🔹 볼드 적용
        txt.setFont(f)
        txt.setBrush(QtGui.QBrush(Qt.white))
        # 필요하면 외곽선도 추가 가능 (가독성용)
        # pen = QtGui.QPen(QtGui.QColor(0, 0, 0)); pen.setWidthF(1.0)
        # txt.setPen(pen)

        # 중심 픽셀 오른쪽 위쪽에 살짝 붙이기
        txt.setPos(px + radius + 1, py + radius - 5)
        txt.setZValue(1e6 + 2)
        txt.setAcceptedMouseButtons(Qt.NoButton)
        sc.addItem(txt)

        # 기존 구조 유지: (첫 번째, 두 번째, 텍스트) 튜플 저장
        # 두 번째는 더 이상 안 쓰므로 None 넣어도 clear_ctx_markers와 호환됨
        self._marker_layers.setdefault(ctx, []).append((patch_item, None, txt))

    def clear_ctx_markers(self, ctx: str):
        layer = self._marker_layers.get(ctx, [])
        sc = self.scene()
        for (h, v, t) in layer:
            if sc and h: sc.removeItem(h)
            if sc and v: sc.removeItem(v)
            if sc and t: sc.removeItem(t)
        self._marker_layers[ctx] = []

    def clear_all_ctx_markers(self):
        for k in list(self._marker_layers.keys()):
            self.clear_ctx_markers(k)

    def set_mode(self, mode):                  # 컨트롤러 호환용 래퍼
        self.set_roi_mode(mode)

    def set_accept_events(self, yes: bool):    # 컨트롤러 호환용 래퍼
        self.set_roi_active(bool(yes))