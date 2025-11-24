from __future__ import annotations
from typing import Optional, Tuple, Callable
import numpy as np
from PyQt5 import QtCore, QtWidgets
import logging

class AnalysisSelectionController(QtCore.QObject):
    """
    분석 전용 사각형/픽셀 선택 컨트롤러 (상태만 관리)
    - 내부에 '단일 bool 맵(_mask)'만 관리: 픽셀/사각형 모두 여기에 누적
    - op: "union"(추가) / "subtract"(제거). subtract는 allowed 무시(어디서든 제거)
    - allowed: 작업영역 제한. union일 때만 적용
    - 컨트롤러는 '그리지 않는다' → 오버레이는 MainWindow가 단일 레이어로만 표시
    - 외부 API:
        set_shape(h,w), set_op(op), set_allowed_mask(mask),
        add_pixel(y,x), start_rect_selection(), cancel(),
        get_selection(), clear_selection()
    - 신호:
        rectFinished(QRect, np.ndarray[bool])       : 사각형 드로잉 1회 완료 시 원시 사각형 마스크 알림
        analysisMaskChanged(np.ndarray[bool], QRect): 내부 단일 맵이 바뀔 때(메인에서 그리기/상세뷰 갱신)
    """

    rectFinished = QtCore.pyqtSignal(QtCore.QRect, object)
    analysisMaskChanged = QtCore.pyqtSignal(object, QtCore.QRect)

    def __init__(
        self,
        map_view,
        roi_controller,
        overlay_service=None,                           # ← 더 이상 사용하지 않음(호환 파라미터)
        set_click_mode: Optional[Callable[[str], None]] = None,  # "NONE"|"INSPECT"
        get_op: Optional[Callable[[], str]] = None,   # ★ 추가: 최신 op 조회자
        parent=None
    ):
        super().__init__(parent)
        self.view = map_view
        self.roi = roi_controller
        self._set_click = set_click_mode

        # 내부 단일 맵 상태
        self._active: bool = False
        self._mask: Optional[np.ndarray] = None     # (H,W) bool
        self._allowed: Optional[np.ndarray] = None  # (H,W) bool | None
        self._op: str = "union"
        self._rect: Optional[QtCore.QRect] = None
        self._get_op = get_op or (lambda: self._op)  # ★ 추가
        self._rect_op_locked: Optional[str] = None   # ★ 추가: 사각형 1회용 op 잠금

        # MapView: 사각형 ROI 완료 신호
        try:
            self.view.roiRectFinished.connect(self._on_roi_rect_finished)
        except Exception:
            logging.exception("[AnalysisSel] map_view.roiRectFinished connect failed")

    # ===================== 외부 API =====================

    def set_shape(self, h: int, w: int, *, emit: bool = True) -> None:
        if h <= 0 or w <= 0: return
        if self._mask is None or self._mask.shape != (h, w):
            self._mask = np.zeros((h, w), dtype=bool)
            self._rect = None
            if emit:
                self._emit_mask_changed()

    def set_allowed_mask(self, allowed: Optional[np.ndarray]) -> None:
        """작업영역 제한. union일 때만 적용, subtract에는 미적용."""
        try:
            self._allowed = None if allowed is None else np.asarray(allowed, dtype=bool)
            if (self._mask is not None) and (self._allowed is not None):
                if self._allowed.shape != self._mask.shape:
                    logging.warning("[AnalysisSel] allowed shape %s != mask %s (무시)",
                                    getattr(self._allowed, 'shape', None), self._mask.shape)
                    self._allowed = None
        except Exception:
            logging.exception("[AnalysisSel] set_allowed_mask failed")

    def set_op(self, op: Optional[str]) -> None:
        """연산모드 설정: 'union' 또는 'subtract'(기타/None → 'union')."""
        self._op = op if op in ("union", "subtract") else "union"

    def add_pixel(self, y: int, x: int) -> None:
        """픽셀 클릭 누적(단일 맵 갱신)."""
        try:
            if self._mask is None:
                return
            h, w = self._mask.shape
            y = int(y); x = int(x)
            if not (0 <= y < h and 0 <= x < w):
                return

            if self._op == "subtract":
                if self._mask[y, x]:
                    self._mask[y, x] = False
                    self._rect = self._compute_bbox_from_mask(self._mask)
                    self._emit_mask_changed()
                return

            # union: allowed 제한
            if (self._allowed is not None) and (not bool(self._allowed[y, x])):
                return

            if not self._mask[y, x]:
                self._mask[y, x] = True
                self._rect = self._compute_bbox_from_mask(self._mask)
                self._emit_mask_changed()
        except Exception:
            logging.exception("[AnalysisSel] add_pixel failed")

    def start_rect_selection(self) -> None:
        if self._active:
            return
        if callable(self._set_click):
            self._set_click("NONE")
        self._active = True

        # ★ 이번 드로잉에서 사용할 op 잠금
        try:
            cur = self._get_op()
            self._rect_op_locked = cur if cur in ("union","subtract") else self._op
            self._op = self._rect_op_locked  # 내부 기본값도 맞춰둠(안전)
        except Exception:
            self._rect_op_locked = self._op

        if hasattr(self.roi, "set_accept_events"):
            self.roi.set_accept_events(True)
        if hasattr(self.roi, "register_on_finish"):
            self.roi.register_on_finish = False
        if hasattr(self.roi, "set_mode"):
            QtCore.QTimer.singleShot(0, lambda: self.roi.set_mode('rect'))


    def cancel(self) -> None:
        """현재 분석 드로잉 취소. 내부 맵은 유지(메인에서 그대로 그려짐)."""
        if not self._active:
            return
        self._active = False
        try:
            self.roi.set_mode(None)
        except Exception:
            pass
        if callable(self._set_click):
            self._set_click("INSPECT")

    def clear_selection(self) -> None:
        """단일 맵 자체 초기화."""
        try:
            if self._mask is not None and self._mask.any():
                self._mask[:] = False
            self._rect = None
            self._emit_mask_changed()
        except Exception:
            logging.exception("[AnalysisSel] clear_selection failed")

    def get_selection(self) -> Tuple[Optional[np.ndarray], Optional[QtCore.QRect]]:
        """현재 단일 맵 상태(마스크/사각형 bbox) 반환."""
        return self._mask, self._rect

    def show_details(
        self,
        parent_widget: QtWidgets.QWidget,
        classmap: Optional[np.ndarray],
        topk: Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = None,
        id_to_name: Optional[dict] = None
    ) -> None:
        """선택영역 통계 다이얼로그(단일 맵 기준)."""
        try:
            m, r = self.get_selection()

            if m is None or not isinstance(m, np.ndarray) or not np.any(m):
                QtWidgets.QMessageBox.information(parent_widget, "안내", "먼저 분석 영역을 지정하세요.")
                return
            if classmap is None or not isinstance(classmap, np.ndarray) or classmap.ndim != 2:
                QtWidgets.QMessageBox.critical(parent_widget, "오류", "분류맵이 없습니다. 먼저 분류를 수행하세요.")
                return
            if m.shape != classmap.shape:
                QtWidgets.QMessageBox.critical(parent_widget, "오류", f"마스크/분류맵 크기 불일치: {m.shape} vs {classmap.shape}")
                return

            masked_classes = classmap[m]
            vals, cnts = np.unique(masked_classes, return_counts=True)

            lines = ["[분석 선택영역 통계]"]
            if isinstance(r, QtCore.QRect):
                lines.append(f"- Rect: x={r.x()}, y={r.y()}, w={r.width()}, h={r.height()}")
            lines.append(f"- 픽셀수: {int(m.sum())}")
            lines.append("- 클래스 분포:")

            for v, c in zip(vals.astype(int), cnts.astype(int)):
                nm = str(int(v))
                if id_to_name is None:
                    nm = str(int(v))
                elif isinstance(id_to_name, dict):
                    nm = id_to_name.get(int(v), str(int(v)))
                elif isinstance(id_to_name, list):
                    try:
                        found_name = None
                        for item in id_to_name:
                            if isinstance(item, dict) and int(item.get('mtrl_cd', -9999)) == int(v):
                                found_name = item.get('mtrl_nm', str(int(v)))
                                break
                        nm = found_name if found_name else str(int(v))
                    except Exception:
                        nm = str(int(v))
                lines.append(f"  · {int(v)} ({nm}): {int(c)}")

            QtWidgets.QMessageBox.information(parent_widget, "분석 상세", "\n".join(lines))

        except Exception:
            logging.exception("[AnalysisSelectionController] show_details failed")
            QtWidgets.QMessageBox.critical(parent_widget, "오류", "분석 상세 생성 중 오류가 발생했습니다.")

    # ===================== 내부(슬롯/유틸) =====================

    @QtCore.pyqtSlot(QtCore.QRect, object)
    def _on_roi_rect_finished(self, rect: QtCore.QRect, mask_obj) -> None:
        if not self._active:
            # ★ 드물게 비활성 상태라도 바로 반영하고 싶다면 다음 한 줄을 유지
            # self.apply_rect(rect, mask_obj); return
            return

        m = np.asarray(mask_obj, dtype=bool)
        if self._mask is None or self._mask.shape != m.shape:
            self._mask = np.zeros_like(m, dtype=bool)

        # ★ 이번 드래그의 유효 op: (최신 provider) > (잠금) > (내부)
        # 사각형 그리기 중 op가 변경될 수 있으므로 최신 op를 우선 사용
        try:
            latest_op = self._get_op() if self._get_op else None
            effective_op = latest_op or self._rect_op_locked or self._op
        except Exception:
            effective_op = self._op

        if effective_op == "subtract":
            new_mask = np.logical_and(self._mask, np.logical_not(m))
        else:
            tgt = np.logical_and(m, self._allowed) if (self._allowed is not None) else m
            new_mask = np.logical_or(self._mask, tgt)

        self._mask[:] = new_mask
        self._rect = self._compute_bbox_from_mask(self._mask)
        self._emit_mask_changed()

        # ★ 잠금 해제
        self._rect_op_locked = None

        if hasattr(self.roi, "set_mode"):
            self.roi.set_mode(None)
        self._active = False
        try:
            self.rectFinished.emit(rect, m)
        except Exception:
            pass

    def _compute_bbox_from_mask(self, m: np.ndarray) -> Optional[QtCore.QRect]:
        try:
            if m is None or not np.any(m):
                return None
            ys, xs = np.where(m)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            return QtCore.QRect(x0, y0, x1 - x0, y1 - y0)
        except Exception:
            logging.exception("[AnalysisSel] bbox compute failed")
        return None

    def _emit_mask_changed(self) -> None:
        """내부 단일 맵 변경 신호만 방출(그리기는 MainWindow 담당)."""
        try:
            logging.info(f"[AnalysisSel] _emit_mask_changed called: mask={self._mask is not None}, rect={self._rect}")
            if self._mask is None:
                logging.warning("[AnalysisSel] _mask is None, skipping emit")
                return
            # ★ 오버레이를 직접 그리지 않는다
            # rect가 None이면 빈 QRect를 전달 (시그널 타입이 QRect이므로 None 불가)
            rect = self._rect if self._rect is not None else QtCore.QRect()
            logging.info(f"[AnalysisSel] emitting analysisMaskChanged: mask sum={self._mask.sum()}, rect={rect}")
            self.analysisMaskChanged.emit(self._mask, rect)
            logging.info("[AnalysisSel] analysisMaskChanged signal emitted successfully")
        except Exception:
            logging.exception("[AnalysisSel] emit_mask_changed failed")

    # 클래스 내부에 추가
    def apply_rect(self, rect: QtCore.QRect, mask_obj, *, op: Optional[str] = None, respect_allowed: bool = True) -> None:
        """사각형 결과를 즉시 내부 맵에 적용(_active 여부와 무관)."""
        try:
            m = np.asarray(mask_obj, dtype=bool)
        except Exception:
            logging.exception("[AnalysisSel] apply_rect: mask cast failed")
            return

        if self._mask is None or self._mask.shape != m.shape:
            self._mask = np.zeros_like(m, dtype=bool)

        op2 = (op if op in ("union","subtract") else self._op)
        if op2 == "subtract":
            new_mask = np.logical_and(self._mask, np.logical_not(m))
        else:
            tgt = np.logical_and(m, self._allowed) if (respect_allowed and self._allowed is not None) else m
            new_mask = np.logical_or(self._mask, tgt)

        self._mask[:] = new_mask
        self._rect = self._compute_bbox_from_mask(self._mask)
        self._emit_mask_changed()

    # 클래스 내부에 추가
    def add_pixels(self, coords: list[tuple[int, int]]) -> None:
        """여러 픽셀을 한 번에 누적."""
        try:
            if self._mask is None or not coords:
                return
            h, w = self._mask.shape
            sub = (self._op == "subtract")
            changed = False
            for y, x in coords:
                y = int(y); x = int(x)
                if not (0 <= y < h and 0 <= x < w): 
                    continue
                if sub:
                    if self._mask[y, x]:
                        self._mask[y, x] = False
                        changed = True
                else:
                    if (self._allowed is not None) and (not bool(self._allowed[y, x])):
                        continue
                    if not self._mask[y, x]:
                        self._mask[y, x] = True
                        changed = True
            if changed:
                self._rect = self._compute_bbox_from_mask(self._mask)
                self._emit_mask_changed()
        except Exception:
            logging.exception("[AnalysisSel] add_pixels failed")
