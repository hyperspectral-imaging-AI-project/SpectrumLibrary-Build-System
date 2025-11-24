# controllers/roi_controller.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any, List, Tuple
import numpy as np
from PyQt5.QtCore import QObject, QRect, pyqtSignal
from PyQt5.QtGui import QPolygonF

# ─────────────────────────────────────────────────────────────
# ROI 데이터 모델
# ─────────────────────────────────────────────────────────────
@dataclass
class ROIRecord:
    id: int
    name: str
    kind: str                     # 'rect' | 'poly'
    mask: np.ndarray              # (H,W) bool
    rect: Optional[QRect] = None
    poly: Optional[QPolygonF] = None
    color: Tuple[int,int,int,int] = (64,160,255,100)  # RGBA for overlay
    visible: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

# ─────────────────────────────────────────────────────────────
# ROI 컨트롤러
#  - MapView의 ROI 시그널을 받아 Repository로 저장
#  - register_layer(...) 콜백으로 LayersDock/MapView에 마스크를 레이어로 등록
#  - analyze/crop 등 후속 오퍼레이션의 허브
# ─────────────────────────────────────────────────────────────
class ROIController(QObject):
    # 외부로 알림(예: 로그/통계 UI 갱신)
    roiAdded       = pyqtSignal(object)     # 신규 ROI 등록 시
    roiRemoved     = pyqtSignal(int)           # ROI 삭제 시 (id)
    roiCleared     = pyqtSignal()              # 전체 삭제 시
    roiModeChanged = pyqtSignal(object)        # None | 'rect' | 'poly'
    _accept_events: bool = True

    def __init__(
        self,
        map_view,                                           # views.mapview.MapView
        layer_register: Callable[..., None],                # register_layer(name, data, type='mask', meta=..., as_base_rgb=False)
        img_shape_fn: Callable[[], Tuple[int,int]],         # (H,W) 반환
        parent=None
    ):
        super().__init__(parent)
        self.view = map_view
        self._register_layer = layer_register
        self._img_shape_fn = img_shape_fn

        # ID/색상 카운터
        self._next_id = 1
        self._color_cycle = [
            (64,160,255,100),   # 파랑
            (255,64,64,90),     # 빨강
            (60,180,75,110),    # 초록
            (255,225,25,110),   # 노랑
            (145,30,180,110),   # 보라
            (245,130,48,110),   # 주황
        ]
        self._color_idx = 0
        self.register_on_finish: bool = False

        # 저장소
        self._rois: Dict[int, ROIRecord] = {}    # id -> ROIRecord

        # MapView ROI 시그널 연결
        self.view.roiRectFinished.connect(self._on_rect_finished)
        self.view.roiFreeFinished.connect(self._on_poly_finished)

        self.accept_events = True
    # ─────────────────────────────────────────────────────────
    # 모드 제어
    # ─────────────────────────────────────────────────────────
    def set_mode(self, mode: Optional[str]) -> None:
        """mode: None | 'rect' | 'poly'"""
        mv_mode = None if mode is None else ('rect' if mode == 'rect' else 'free')
        self.view.set_roi_mode(mv_mode)
        self.roiModeChanged.emit(mode)

    def cancel(self) -> None:
        """현재 드로잉 취소(모드 해제 + 임시 경로 클리어)"""
        self.set_mode(None)

    def _on_rect_finished(self, rect: QRect, mask: np.ndarray) -> None:
        if not self._accept_events:
            return
        rid   = self._alloc_id()
        color = self._next_color()
        fixed_name = "작업 영역"  # ★ 트리에 Top-level로 한 줄만 보이게 'ROI'로 고정
        rec = ROIRecord(
            id=rid, name=fixed_name, kind='rect',
            mask=mask.astype(bool), rect=rect, poly=None, color=color
        )
        self._add_roi(rec)

    def _on_poly_finished(self, poly: QPolygonF, mask: np.ndarray) -> None:
        if not self._accept_events:
            return
        rid   = self._alloc_id()
        color = self._next_color()
        fixed_name = "작업 영역"  # ★ 다각형도 동일하게 Top-level 'ROI'
        rec = ROIRecord(
            id=rid, name=fixed_name, kind='poly',
            mask=mask.astype(bool), rect=None, poly=poly, color=color  # ★ poly 필드 저장
        )
        self._add_roi(rec)
        
    def set_accept_events(self, on: bool):
        self._accept_events = bool(on)
        if not on and hasattr(self.view, 'set_roi_mode'):
            self.view.set_roi_mode(None)
            
    def _add_roi(self, rec: ROIRecord) -> None:
        self._rois[rec.id] = rec

        if self.register_on_finish:
            # “작업 영역”은 앱 전반에서 mask.roi로 다룸
            self._register_layer(
                rec.name, rec.mask,
                type="mask.roi",
                visible=True,
                meta={"color": rec.color, "parent": None}
            )

        self.set_mode(None)
        self.roiAdded.emit(rec)

    def remove(self, roi_id: int) -> None:
        """내부 상태만 제거(레이어/MapView 삭제는 MainWindow 쪽 삭제 핸들러를 재사용 권장)"""
        if roi_id in self._rois:
            self._rois.pop(roi_id)
            self.roiRemoved.emit(roi_id)

    def clear(self) -> None:
        self._rois.clear()
        self.roiCleared.emit()

    # ─────────────────────────────────────────────────────────
    # 조회/가져오기
    # ─────────────────────────────────────────────────────────
    def list(self) -> List[ROIRecord]:
        return list(self._rois.values())

    def get(self, roi_id: int) -> ROIRecord:
        return self._rois[roi_id]

    # ─────────────────────────────────────────────────────────
    # 분석/크롭/유틸(향후 확장 훅)
    # ─────────────────────────────────────────────────────────
    def crop_array(self, arr: np.ndarray, roi_id: int, *, by_mask=True) -> np.ndarray:
        """
        arr: (H,W) or (H,W,C)
        by_mask=True → 마스크 영역만 추출(외부 0 또는 NaN)
        by_mask=False → ROI 바운딩박스 슬라이스만
        """
        m = self._rois[roi_id].mask
        if arr.ndim == 2:
            data = arr
        else:
            data = arr.copy()
        if by_mask:
            if arr.ndim == 2:
                out = np.where(m, data, 0)
            else:
                mask3 = m[..., None]
                out = np.where(mask3, data, 0)
            return out
        else:
            ys, xs = np.where(m)
            y0, y1 = ys.min(), ys.max()+1
            x0, x1 = xs.min(), xs.max()+1
            return arr[y0:y1, x0:x1] if arr.ndim == 2 else arr[y0:y1, x0:x1, :]

    def analyze(self, roi_id: int, analyzer: Callable[[np.ndarray], Any]) -> Any:
        """
        analyzer: mask(bool HxW) -> 결과
        예: 평균 스펙트럼/히스토그램 계산 등
        """
        return analyzer(self._rois[roi_id].mask)

    # (예비) 불리언 연산: 합집합/교집합/차집합 → 신규 ROI 반환
    def boolean(self, roi_ids: List[int], op: str, *, name: Optional[str] = None) -> ROIRecord:
        """
        op: 'union'|'intersection'|'difference'
        """
        masks = [self._rois[i].mask for i in roi_ids]
        base = masks[0].copy()
        if op == 'union':
            for m in masks[1:]: base |= m
        elif op == 'intersection':
            for m in masks[1:]: base &= m
        elif op == 'difference':
            for m in masks[1:]: base &= ~m
        # poly/rect 재구성은 생략(향후 marching squares 등); 우선 mask-only ROI
        rid = self._alloc_id()
        nm  = name or f"roi.{op}#{rid}"
        color = self._next_color()
        rec = ROIRecord(id=rid, name=nm, kind='poly', mask=base, color=color, meta={"op": op, "sources": roi_ids})
        self._add_roi(rec)
        return rec

    # ─────────────────────────────────────────────────────────
    # 내부 유틸
    # ─────────────────────────────────────────────────────────
    def _alloc_id(self) -> int:
        rid = self._next_id
        self._next_id += 1
        return rid

    def _next_color(self) -> Tuple[int,int,int,int]:
        c = self._color_cycle[self._color_idx % len(self._color_cycle)]
        self._color_idx += 1
        return c
    
    def set_register_on_finish(self, flag: bool) -> None:
        self.register_on_finish = bool(flag)