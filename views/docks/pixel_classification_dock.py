from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from PyQt5 import uic, QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, Qt, QSize
from PyQt5.QtGui import QColor, QPixmap, QIcon
from data.db import search_material_filtering_list
from services.resampling_cache import load_classes_from_info
from core.autoclass import UNKNOWN, MULTIPLE

class PixelClassificationDock(QtWidgets.QDockWidget):
    # === 외부로 내보내는 시그널 ===
    requestApplyThresholds = pyqtSignal(float, float)  # 유지
    # 추가
    requestResetOpacity = pyqtSignal()
    requestResetParams = pyqtSignal()
    requestFocusClass = pyqtSignal(int)                # class id (선택 해제: -9999)

    # 작업/클래스/분석 관련
    requestWorkRectROIStart = pyqtSignal()             # 작업영역 사각형 (필요시 유지)
    # requestViewSelectionDetails = pyqtSignal()         # 선택영역 상세보기
    requestClassRectROIStart = pyqtSignal()            # 클래스 ROI(사각형)
    requestVizClassChanged = pyqtSignal(int)
    requestApplyThresholdsForClass = pyqtSignal(int, float, float)
    # === 분석(SSOT 이벤트만) ===
    requestAnalysisRectROIStart = pyqtSignal()         # 분석 사각형 시작
    requestAnalysisPixelStart   = pyqtSignal()         # 분석 픽셀 시작
    requestAnalysisStop         = pyqtSignal()         # 분석 모드 종료
    requestAnalysisOpChanged    = pyqtSignal(str)      # "union" | "subtract"

    # 상태 브로드캐스트(선택적): MainWindow가 참고용으로 수신 가능
    analysisStateChanged        = pyqtSignal(str, bool, str)  # (mode="rect|pixel", active, op)

    # 기타
    requestViewAnalysisDetails  = pyqtSignal()
    requestClassDetail          = pyqtSignal(int)
    requestClearAnalysisRegion = pyqtSignal()   # 분석 영역(마스크/ROI) 초기화 요청

    def __init__(self, parent=None, ui_dir: Optional[Path] = None, ui_filename: str = "pixel_classification_dock.ui"):
        super().__init__("Pixel Classification", parent)
        self.setObjectName("dockPixelClassification")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # --- UI 로드 ---
        app_dir = Path(getattr(parent, "app_dir", Path(__file__).resolve().parents[2]))
        ui_dir = Path(ui_dir) if ui_dir else (app_dir / "ui")
        self._root = uic.loadUi(str(ui_dir / ui_filename))
        self.setWidget(self._root)

        # --- 상태 ---
        self._class_palette_ref: Optional[Dict[int, QColor]] = None
        self._use_row_background: bool = True
        self._selected_class_id: Optional[int] = None
        self._roi_mode_active: Optional[str] = None

        self._last_applied_strict: Optional[float] = None
        self._last_applied_base: Optional[float] = None

        # 분석 상태(도크 내부 캐시)
        self._analysis_op: Optional[str] = None        # "union" | "subtract" | None
        self._last_analysis_op: Optional[str] = None   # 마지막 op
        self._last_analysis_mode: Optional[str] = None # "pixel" | "rect" | None

        # --- 위젯 핸들 수집 ---
        self._class_meta: Dict[int, Dict[str, str]] = {}
        self._sp_tau    = self._find(QtWidgets.QDoubleSpinBox, "tauSpinBox")
        self._sp_delta  = self._find(QtWidgets.QDoubleSpinBox, "deltaSpinBox")
        self._sp_strict = self._find(QtWidgets.QDoubleSpinBox, "strictSpinBox")
        self._sp_base   = self._find(QtWidgets.QDoubleSpinBox, "baseSpinBox")

        self._btn_apply = self._find(QtWidgets.QPushButton, "btnApplyThresholds") or self._find(QtWidgets.QPushButton, "applyButton")
        self._btn_init  = self._find(QtWidgets.QPushButton, "btnResetOpacity")    or self._find(QtWidgets.QPushButton, "initializeButton")

        self._table            = self._find(QtWidgets.QTableWidget, "tableViewer") or self._find(QtWidgets.QTableWidget, "viewerTableWidget")
        self._btn_select_all   = self._find(QtWidgets.QPushButton, "btnViewerSelectAll")
        self._btn_clear_sel    = self._find(QtWidgets.QPushButton, "btnViewerClearSelection")
        self._btn_class_reset  = self._find(QtWidgets.QPushButton, "btnClassReset")

        self._val_great  = self._find(QtWidgets.QLabel, "valGreat")
        self._val_mid    = self._find(QtWidgets.QLabel, "valMid")
        self._val_little = self._find(QtWidgets.QLabel, "valLittle")

        # ROI/분석 버튼 (지정 영역 분석 기능 제거됨 - 버튼들은 찾지 않음)
        # self._btn_pick = self._find(QtWidgets.QPushButton, "btnPickPixel")
        # self._btn_rect = self._find(QtWidgets.QPushButton, "btnRectROI") or self._find(QtWidgets.QToolButton, "rectToolButton")
        # self._btn_free = self._find(QtWidgets.QPushButton, "btnFreeROI")  or self._find(QtWidgets.QToolButton, "freehandToolButton")
        # if self._btn_free:
        #     self._btn_free.setEnabled(False)  # 자유형 제외

        # "+" / "-" → 연산자 (제거됨)
        # self._btn_union = self._find(QtWidgets.QPushButton, "btnZoomIn")
        # self._btn_sub   = self._find(QtWidgets.QPushButton, "btnZoomOut")

        # '선택한 영역 분석' 버튼 (유지)
        self._btn_view_details = self._find(QtWidgets.QPushButton, "btnViewAnalysisDetails") or self._find(QtWidgets.QPushButton, "viewDetailsButton")
        
        # '지정 영역 분석' 그룹의 '초기화' 버튼 (제거됨)
        # self._btn_reset_analysis = self._find(QtWidgets.QPushButton, "pushButton")

        # --- 기본값/표시 초기화 ---
        if self._sp_tau and self._sp_tau.value() <= 0:
            self._sp_tau.setDecimals(3);   self._sp_tau.setSingleStep(0.01); self._sp_tau.setValue(0.05)
        if self._sp_delta and self._sp_delta.value() <= 0:
            self._sp_delta.setDecimals(3); self._sp_delta.setSingleStep(0.01); self._sp_delta.setValue(0.03)
        if self._sp_strict and self._sp_strict.value() <= 0:
            self._sp_strict.setDecimals(3); self._sp_strict.setSingleStep(0.01); self._sp_strict.setValue(0.02)
        if self._sp_base and self._sp_base.value() <= 0:
            self._sp_base.setDecimals(3);   self._sp_base.setSingleStep(0.01);   self._sp_base.setValue(0.05)

        if self._table:
            self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self._table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            self._table.itemSelectionChanged.connect(self._on_viewer_row_selected)

        if self._btn_class_reset:
            self._btn_class_reset.clicked.connect(self._on_click_class_reset)

        self._apply_value_cards_legend()

        # --- 시그널 연결 ---
        if self._btn_apply:
            self._btn_apply.clicked.connect(self._on_click_apply)
        if self._btn_init:
            self._btn_init.clicked.connect(self._on_click_reset_params)

        # 연산자 버튼 연결 제거 (지정 영역 분석 기능 제거됨)
        # if self._btn_union:
        #     self._btn_union.setCheckable(True)
        #     self._btn_union.toggled.connect(self._on_union_toggled)
        # if self._btn_sub:
        #     self._btn_sub.setCheckable(True)
        #     self._btn_sub.toggled.connect(self._on_subtract_toggled)

        # 선택 모드 버튼 연결 제거 (지정 영역 분석 기능 제거됨)
        # if self._btn_pick:
        #     self._btn_pick.setCheckable(True)
        #     self._btn_pick.toggled.connect(self._on_pick_toggled)
        # if self._btn_rect:
        #     self._btn_rect.setCheckable(True)
        #     self._btn_rect.toggled.connect(self._on_rect_toggled)

        # 분석 영역 초기화 버튼 제거 (지정 영역 분석 기능 제거됨)
        # if self._btn_reset_analysis:
        #     self._btn_reset_analysis.clicked.connect(self._on_click_reset_analysis_region)

        # (선택) 폴리곤 버튼은 비활성 (제거됨)
        # self._btn_poly = self._find(QtWidgets.QToolButton, "polygonToolButton")
        # if self._btn_poly:
        #     self._btn_poly.setEnabled(False)

        self.btnReset = self.findChild(QtWidgets.QPushButton, "btnReset") \
                        or self.findChild(QtWidgets.QPushButton, "btnInit") \
                        or self.findChild(QtWidgets.QPushButton, "btnResetParams")

        if self.btnReset:
            self.btnReset.clicked.connect(self._on_click_reset)
            
        if self._btn_view_details:
            self._btn_view_details.clicked.connect(self._on_click_view_analysis)
            
    # ---------- 유틸 ----------
    def _find(self, cls, name: str):
        try:
            return self._root.findChild(cls, name)
        except Exception:
            return None

    # ---------- 외부 API ----------
    def set_class_palette_ref(self, palette_ref: Dict[int, QColor]):
        self._class_palette_ref = palette_ref

    def set_class_palette(self, palette: Dict[int, Any]):
        normalized: Dict[int, QColor] = {}
        for k, v in palette.items():
            cid = int(k)
            if isinstance(v, QColor):
                normalized[cid] = v
            elif isinstance(v, (tuple, list)) and len(v) in (3, 4):
                normalized[cid] = QColor(*v)
            elif isinstance(v, str):
                normalized[cid] = QColor(v)
        self._class_palette_ref = normalized

    # ---------- Viewer ----------
    def sync_viewer(self, counts: Dict[int, int], id_to_name: Optional[List[Dict[str, Any]]] = None):
        if not self._table:
            return
        tv = self._table
        sorted_rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        total = max(int(sum(counts.values())), 1)
        tv.setRowCount(len(sorted_rows))
        self._class_meta = {}

        for r, (cid, cnt) in enumerate(sorted_rows):
            cid_i = int(cid)
            qc = self._lookup_color_strict(cid_i)

            color_item = QtWidgets.QTableWidgetItem()
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
            if qc is not None:
                color_item.setIcon(self._make_color_icon(qc))
                if self._use_row_background:
                    color_item.setBackground(qc)
            tv.setItem(r, 0, color_item)

            name_meta, desc_meta = self._lookup_meta_for(cid_i, id_to_name)
            # UNKNOWN(-1)과 MULTIPLE(-2)는 특별한 라벨 표시 (LayerManager와 동일)
            if cid_i == UNKNOWN:
                cls_text = "미분류"
                self._class_meta[cid_i] = {"name": "미분류", "desc": desc_meta or ""}
            elif cid_i == MULTIPLE:
                cls_text = "중복 클래스"
                self._class_meta[cid_i] = {"name": "중복 클래스", "desc": desc_meta or ""}
            elif name_meta:
                # name_meta만 표시 (class_id 제거)
                cls_text = str(name_meta)
                self._class_meta[cid_i] = {"name": name_meta, "desc": desc_meta or ""}
            else:
                cls_text = str(cid_i)
                self._class_meta[cid_i] = {"name": str(cid_i), "desc": desc_meta or ""}
            cls_item = QtWidgets.QTableWidgetItem(cls_text)
            cls_item.setFlags(cls_item.flags() & ~Qt.ItemIsEditable)
            tv.setItem(r, 1, cls_item)

            cnt_item = QtWidgets.QTableWidgetItem(str(int(cnt)))
            cnt_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            cnt_item.setFlags(cnt_item.flags() & ~Qt.ItemIsEditable)
            tv.setItem(r, 2, cnt_item)

            btn = QtWidgets.QPushButton("상세보기")
            btn.setProperty("cid", cid_i)
            btn.clicked.connect(lambda _=None, c=cid_i: self._on_click_class_detail(c))
            tv.setCellWidget(r, 3, btn)

        tv.resizeColumnsToContents()
        tv.horizontalHeader().setStretchLastSection(True)

    def _lookup_meta_for(self, cid: int, id_to_name: Optional[Any]) -> Tuple[Optional[str], Optional[str]]:
        name = None; desc = None
        try:
            if not id_to_name:
                return None, None
            if isinstance(id_to_name, dict):
                rec = id_to_name.get(cid)
                if isinstance(rec, dict):
                    name = rec.get("mtrl_nm") or rec.get("name") or rec.get("title")
                    desc = rec.get("desc") or rec.get("description")
                elif rec is not None:
                    name = str(rec)
            else:
                for rec in id_to_name:
                    if not isinstance(rec, dict): continue
                    key = rec.get("mtrl_cd"); 
                    if key is None: continue
                    try:
                        if int(key) != int(cid): continue
                    except (TypeError, ValueError):
                        continue
                    name = rec.get("mtrl_nm") or rec.get("name") or rec.get("title")
                    desc = rec.get("desc") or rec.get("description")
                    break
        except Exception:
            pass
        return name, desc

    def sync_viewer_with_palette(self, counts: Dict[int, int], palette: Dict[int, Any],
                                 id_to_name: Optional[List[Dict[str, Any]]] = None):
        self.set_class_palette(palette)
        self.sync_viewer(counts, id_to_name)

    def apply_dialog_params(self, params: Dict[str, Any]):
        metric = params.get("metric", "SAD").upper()
        tau    = float(params.get("tau", 0.05))
        delta  = float(params.get("delta", 0.03))

        if self._sp_tau:
            self._sp_tau.setValue(tau)
            self._sp_tau.setReadOnly(True)
            self._sp_tau.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
            self._sp_tau.setFocusPolicy(Qt.NoFocus)

        if self._sp_delta:
            self._sp_delta.setValue(delta)
            self._sp_delta.setReadOnly(True)
            self._sp_delta.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
            self._sp_delta.setFocusPolicy(Qt.NoFocus)

    # ---------- Getter ----------
    def get_metric(self) -> str:
        if hasattr(self._root, "radioSID") and self._root.radioSID.isChecked(): return "SID"
        if hasattr(self._root, "radioSCC") and self._root.radioSCC.isChecked(): return "SCC"
        return "SAD"

    def get_thresholds(self) -> Tuple[float, float]:
        tau   = float(self._sp_tau.value()) if self._sp_tau else 0.05
        delta = float(self._sp_delta.value()) if self._sp_delta else 0.03
        return tau, delta

    def get_levels(self) -> Tuple[float, float]:
        strict = float(self._sp_strict.value()) if self._sp_strict else 0.02
        base   = float(self._sp_base.value()) if self._sp_base else 0.05
        return strict, base

    def get_params(self) -> dict:
        tau, delta = self.get_thresholds()
        return {"metric": self.get_metric(), "tau": tau, "delta": delta}

    def get_selected_class_id(self) -> Optional[int]:
        return self._selected_class_id

    def set_value_cards(self, great: Optional[int] = None, mid: Optional[int] = None, little: Optional[int] = None):
        if self._val_great:  self._val_great.setText(str(great) if great is not None else "-")
        if self._val_mid:    self._val_mid.setText(str(mid) if mid is not None else "-")
        if self._val_little: self._val_little.setText(str(little) if little is not None else "-")

    # ---------- 버튼 핸들러 (간소/일원화) ----------
    def _on_click_apply(self):
        strict, base = self.get_levels()
        # 기존 신호 유지
        self.requestApplyThresholds.emit(strict, base)
        # 선택된 클래스와 함께 적용 → 해당 클래스만 시각화
        cid = self._selected_class_id if self._selected_class_id is not None else -9999
        self.requestApplyThresholdsForClass.emit(int(cid), float(strict), float(base))
        self._last_applied_strict = float(strict)
        self._last_applied_base   = float(base)
        self._set_viewer_enabled(False)

    def _on_click_reset_params(self):
        self.requestResetOpacity.emit()
        self.requestResetParams.emit()

        if self._sp_strict and (self._last_applied_strict is not None):
            self._sp_strict.blockSignals(True); self._sp_strict.setValue(float(self._last_applied_strict)); self._sp_strict.blockSignals(False)
        if self._sp_base and (self._last_applied_base is not None):
            self._sp_base.blockSignals(True); self._sp_base.setValue(float(self._last_applied_base)); self._sp_base.blockSignals(False)

        self.set_value_cards(None, None, None)
        self._set_viewer_enabled(True)

    def _on_viewer_row_selected(self):
        tv = self._table
        if not tv:
            return
        rows = tv.selectionModel().selectedRows()
        if not rows:
            self._selected_class_id = None
            self.requestVizClassChanged.emit(-9999)
            return
        r = rows[0].row()
        raw = tv.item(r, 1).text() if tv.item(r, 1) else ""
        cid = None
        try:
            import re
            m = re.match(r"\s*(\-?\d+)", raw)
            if m: cid = int(m.group(1))
        except Exception:
            cid = None
        if cid is not None:
            self._selected_class_id = cid
            self.requestVizClassChanged.emit(cid)
        else:
            self._selected_class_id = None
            self.requestVizClassChanged.emit(-9999)

    def _on_click_class_reset(self):
        tv = self._table
        if tv:
            tv.clearSelection()
            for r in range(tv.rowCount()):
                it = tv.item(r, 0)
                if it is not None:
                    it.setCheckState(Qt.Unchecked)
        self._selected_class_id = None
        self.requestVizClassChanged.emit(-9999)  # 포커스 대신 시각화 범위만 초기화
        self.requestResetOpacity.emit()

    def _on_click_class_detail(self, cid: int):
        self._selected_class_id = int(cid)
        self.requestVizClassChanged.emit(int(cid))  # 포커스 금지
        self._show_class_detail_dialog(int(cid))

    def _show_class_detail_dialog(self, cid: int):
        """클래스 상세 정보 다이얼로그 표시"""
        try:
            main_window = self.parent()
            user_type = getattr(main_window, "user_type", "server") if main_window else "server"
            result = None

            if user_type == "personal":
                try:
                    primary = None
                    cfg = getattr(main_window, "cfg", {}) if main_window else {}
                    if main_window and hasattr(main_window, "_cache_primary_path"):
                        primary = main_window._cache_primary_path(cfg)
                    if not primary and main_window and hasattr(main_window, "_extract_src_path"):
                        primary = main_window._extract_src_path(cfg)

                    if primary:
                        classes = load_classes_from_info(primary)
                        for cid_opt, name_opt, desc in classes:
                            if int(cid_opt) == int(cid):
                                result = {
                                    "mtrl_cd": int(cid_opt),
                                    "mtrl_nm": str(name_opt),
                                    "desc": str(desc),
                                }
                                break
                except Exception:
                    import logging
                    logging.exception("[ClassDetail] personal detail load failed")

                # personal .info에 없으면 viewer 캐시를 사용
                if result is None:
                    meta = self._class_meta.get(int(cid)) if hasattr(self, "_class_meta") else None
                    if meta:
                        result = {
                            "mtrl_cd": int(cid),
                            "mtrl_nm": meta.get("name"),
                            "desc": meta.get("desc", ""),
                        }

            if result is None:
                api_url = os.getenv('material_filtering_url')
                api_result = search_material_filtering_list(api_url, [cid]) if api_url else []
                if api_result:
                    result = api_result[0]

            info_text = f"클래스 ID: {cid}\n"
            if result and result.get("mtrl_nm"):
                info_text += f"이름: {result.get('mtrl_nm')}\n"
            if result and result.get("desc"):
                info_text += f"설명: {result.get('desc')}\n"

            from PyQt5 import QtWidgets
            QtWidgets.QMessageBox.information(
                self,
                "클래스 상세 정보",
                info_text if info_text.strip() else f"클래스 ID: {cid}\n(추가 정보 없음)"
            )
        except Exception:
            import logging
            logging.exception("[ClassDetail] dialog show failed")

    def _on_click_reset_analysis_region(self):
        """'지정 영역 분석' 그룹의 '초기화' 버튼 클릭 시: 분석 영역 초기화 요청"""
        self.requestClearAnalysisRegion.emit()

    # ---------- 분석 UX: 상호배타/이벤트만 송출 ----------
    # 지정 영역 분석 기능 제거됨 - 메서드들은 유지하되 비활성화
    def _on_union_toggled(self, checked: bool):
        """지정 영역 분석 기능 제거로 인해 비활성"""
        pass

    def _on_subtract_toggled(self, checked: bool):
        """지정 영역 분석 기능 제거로 인해 비활성"""
        pass

    def _on_pick_toggled(self, checked: bool):
        """지정 영역 분석 기능 제거로 인해 비활성"""
        pass

    def _on_rect_toggled(self, checked: bool):
        """지정 영역 분석 기능 제거로 인해 비활성"""
        pass

    # ---------- 내부 유틸 ----------
    def _make_color_icon(self, qc: QColor, size: int = 14) -> QIcon:
        pm = QPixmap(size, size); pm.fill(qc); return QIcon(pm)

    def _lookup_color_strict(self, cid: int) -> Optional[QColor]:
        if self._class_palette_ref is None: return None
        return self._class_palette_ref.get(int(cid))

    def _apply_value_cards_legend(self):
        try:
            cap_g = self._find(QtWidgets.QLabel, "capGreat")
            cap_m = self._find(QtWidgets.QLabel, "capMid")
            cap_l = self._find(QtWidgets.QLabel, "capLittle")
            text_g = cap_g.text() if cap_g else None
            text_m = cap_m.text() if cap_m else None
            text_l = cap_l.text() if cap_l else None
            def _apply(lbl, base, color_hex):
                if not lbl or not base: return
                lbl.setTextFormat(Qt.RichText)
                lbl.setText(f'<span style="color:{color_hex};font-weight:700;">■</span>&nbsp;{base}')
            _apply(cap_g, text_g, "#FFD54F")
            _apply(cap_m, text_m, "#66BB6A")
            _apply(cap_l, text_l, "#2196F3")
        except Exception:
            import logging; logging.exception("[Dock] apply value-cards legend failed")

    def _set_viewer_enabled(self, enabled: bool) -> None:
        if self._table: self._table.setEnabled(bool(enabled))
        for btn in (getattr(self, "_btn_select_all", None),
                    getattr(self, "_btn_clear_sel", None),
                    getattr(self, "_btn_class_reset", None)):
            if btn: btn.setEnabled(bool(enabled))

    @staticmethod
    def _set_button_checked(button, checked):
        if button is None or button.isChecked() == checked: return
        was = button.blockSignals(True)
        try: button.setChecked(checked)
        finally: button.blockSignals(was)

    def _toggle_analysis_op(self, op: Optional[str]):
        """내부 표시만: 상호배타 적용 + 캐시 업데이트 + 외부로는 requestAnalysisOpChanged는 여기서 직접 emit하지 않음. (지정 영역 분석 기능 제거로 인해 비활성)"""
        # 지정 영역 분석 기능 제거됨
        # op = op if op in ("union", "subtract") else None
        # self._set_button_checked(self._btn_union, op == "union")
        # self._set_button_checked(self._btn_sub,   op == "subtract")
        # self._analysis_op = op
        # if op in ("union","subtract"):
        #     self._last_analysis_op = op
        pass

    def _on_click_view_analysis(self):
        """ 분석 다이얼로그 표시 + 기존 requestViewAnalysisDetails 신호도 발행.
        가능한 경우 부모(MainWindow)에서 classmap/hsi 데이터를 수집해 전달."""
        # 분석 선택영역 상세보기 시그널 발행 (AnalysisSelectionDialog를 열기 위한 시그널)
        self.requestViewAnalysisDetails.emit()
        # 2) 선택영역 상세보기 시그널 발행
        # self.requestViewSelectionDetails.emit()
