# controllers/class_roi_controller.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any, List, Tuple
import numpy as np
from PyQt5.QtCore import QObject, QRect, pyqtSignal
from PyQt5.QtGui import QPolygonF

# ─────────────────────────────────────────────────────────────
# 클래스 ROI 데이터 모델
# ─────────────────────────────────────────────────────────────
@dataclass
class ClassROIRecord:
    id: int
    class_id: int
    name: str
    kind: str                     # 'rect' | 'poly'
    mask: np.ndarray              # (H,W) bool
    rect: Optional[QRect] = None
    poly: Optional[QPolygonF] = None
    color: Tuple[int,int,int,int] = (64,160,255,100)  # RGBA for overlay
    
    meta: Dict[str, Any] = field(default_factory=dict)

# ─────────────────────────────────────────────────────────────
# 클래스 ROI 컨트롤러
#  - 작업 영역 내에서만 ROI 지정 가능
#  - 선택된 클래스에 대해 ROI 영역을 클래스로 지정
# ─────────────────────────────────────────────────────────────
class ClassROIController(QObject):
    # 외부로 알림
    classROIAdded = pyqtSignal(ClassROIRecord)     # 신규 클래스 ROI 등록 시
    classROIRemoved = pyqtSignal(int)              # 클래스 ROI 삭제 시 (id)
    classROICleared = pyqtSignal()                 # 전체 삭제 시
    classROIModeChanged = pyqtSignal(object)       # None | 'rect' | 'poly'

    def __init__(
        self,
        map_view,                                           # views.mapview.MapView
        layer_register: Callable[..., None],                # register_layer(name, data, type='mask', meta=..., as_base_rgb=False)
        img_shape_fn: Callable[[], Tuple[int,int]],         # (H,W) 반환
        get_work_area_mask: Callable[[], Optional[np.ndarray]],  # 작업 영역 마스크 반환
        temp_overlay,
        parent=None
    ):
        super().__init__(parent)
        self.view = map_view
        self._register_layer = layer_register
        self._img_shape_fn = img_shape_fn
        self._get_work_area_mask = get_work_area_mask

        # ID/색상 카운터
        self._next_id = 1
        self._color_cycle = [
            (0, 0, 255, 120),      # 파랑
            (0, 255, 0, 120),      # 초록
            (255, 0, 0, 120),      # 빨강
            (255, 255, 0, 120),    # 노랑
            (255, 0, 255, 120),    # 마젠타
            (0, 255, 255, 120),    # 시안
        ]
        self._color_idx = 0

        # 저장소
        self._class_rois: Dict[int, ClassROIRecord] = {}    # id -> ClassROIRecord
        self._current_class_id: Optional[int] = None        # 현재 선택된 클래스 ID
        self._is_active: bool = False                       # 활성화 상태

        # MapView ROI 시그널 연결
        self._accept_events = True
        self.view.roiRectFinished.connect(self._on_rect_finished)
        self.view.roiFreeFinished.connect(self._on_poly_finished)

    # ─────────────────────────────────────────────────────────
    # 모드 제어
    # ─────────────────────────────────────────────────────────
    def set_mode(self, mode: Optional[str], class_id: Optional[int] = None) -> None:
        if mode is None:
            self._is_active = False
            self._current_class_id = None
            self.view.set_roi_mode(None)
            self.classROIModeChanged.emit(None)
            return
        self._is_active = True
        # ★ None도 허용 (placeholder)
        self._current_class_id = class_id  # None 그대로 저장
        mv_mode = 'rect' if mode == 'rect' else 'free'
        self.view.set_roi_mode(mv_mode)
        self.classROIModeChanged.emit(mode)
        
    def set_accept_events(self, on: bool):
        self._accept_events = bool(on)
        if not on and hasattr(self.view, 'set_roi_mode'):
            self.view.set_roi_mode(None)
            
    def set_active(self, active: bool, class_id: Optional[int] = None):
        """활성화/비활성화 설정"""
        if active and class_id is not None:
            self._is_active = True
            self._current_class_id = class_id
        else:
            self._is_active = False
            self._current_class_id = None
            if hasattr(self.view, 'set_roi_mode'):
                self.view.set_roi_mode(None)

    def get_active_state(self) -> Tuple[bool, Optional[int]]:
        """현재 활성화 상태와 선택된 클래스 ID 반환"""
        return self._is_active, self._current_class_id

    def cancel(self) -> None:
        """현재 드로잉 취소(모드 해제 + 임시 경로 클리어)"""
        self.set_mode(None)

    # ─────────────────────────────────────────────────────────
    # ROI 처리
    # ─────────────────────────────────────────────────────────
    def _on_rect_finished(self, rect, mask):
        if not self._accept_events or not self._is_active:
            return
        if not np.any(mask):
            return
        # ★ None 허용: 그대로 전달
        self._create_class_roi(
            class_id=self._current_class_id,  # None 가능
            kind='rect', mask=mask, rect=rect, poly=None
        )

    def _on_poly_finished(self, poly_img, mask):
        if not self._accept_events or not self._is_active:
            return
        if not np.any(mask):
            return
        self._create_class_roi(
            class_id=self._current_class_id,  # None 가능
            kind='poly', mask=mask, rect=None, poly=poly_img
        )

    def _create_class_roi(self, class_id, kind, mask, rect, poly):
        roi_id = self._next_id
        self._next_id += 1

        # 색상 선택(placeholder 색상 유지 가능)
        color = self._color_cycle[self._color_idx % len(self._color_cycle)]
        self._color_idx += 1

        # ★ 이름: None이면 Unlabeled 표기
        label = "Unlabeled" if class_id is None else f"Class_{int(class_id)}"
        name  = f"{label}_ROI_{roi_id}"

        roi_record = ClassROIRecord(
            id=roi_id,
            class_id=(None if class_id is None else int(class_id)),
            name=name,
            kind=kind,
            mask=mask,
            rect=rect,
            poly=poly,
            color=color,
            meta={"created_at": "now", "class_id": class_id}
        )
        self._class_rois[roi_id] = roi_record

        # ★★ 레이어 등록은 하지 않음(임시 표출만; MainWindow가 show_mask 호출함)
        # self._register_layer(... )  ← 금지

        self.classROIAdded.emit(roi_record)
        self.set_mode(None)

    # ─────────────────────────────────────────────────────────
    # 조회/관리 API
    # ─────────────────────────────────────────────────────────
    def get_class_rois(self) -> Dict[int, ClassROIRecord]:
        """모든 클래스 ROI 반환"""
        return dict(self._class_rois)
    
    def get_class_rois_by_class(self, class_id: int) -> List[ClassROIRecord]:
        """특정 클래스의 모든 ROI 반환"""
        return [roi for roi in self._class_rois.values() if roi.class_id == class_id]
    
    def remove_class_roi(self, roi_id: int) -> bool:
        """클래스 ROI 제거"""
        if roi_id not in self._class_rois:
            return False
            
        roi_record = self._class_rois[roi_id]
        del self._class_rois[roi_id]
        
        # 레이어에서도 제거 (layer_manager에 위임)
        # TODO: layer_manager의 remove 메서드 호출
        # 수정 필요: 레이어 매니저를 통해 레이어 제거 로직 구현
        
        self.classROIRemoved.emit(roi_id)
        return True
    
    def clear_class_rois(self) -> None:
        """모든 클래스 ROI 제거"""
        self._class_rois.clear()
        self.classROICleared.emit()
    
    def get_class_roi_mask_for_class(self, class_id: int) -> Optional[np.ndarray]:
        """특정 클래스의 모든 ROI를 합친 마스크 반환"""
        class_rois = self.get_class_rois_by_class(class_id)
        if not class_rois:
            return None
            
        # 첫 번째 ROI의 크기를 기준으로 합친 마스크 생성
        first_roi = class_rois[0]
        combined_mask = np.zeros_like(first_roi.mask, dtype=bool)
        
        for roi in class_rois:
            if roi.mask.shape == combined_mask.shape:
                combined_mask |= roi.mask
                
        return combined_mask if np.any(combined_mask) else None
