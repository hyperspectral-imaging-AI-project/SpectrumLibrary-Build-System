# views/dialogs/classmap_labeling_dialog.py
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Callable

import numpy as np
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QColor, QPixmap, QPainter, QPen


# ---------- 타입/상수 ----------
TableRow = Dict[str, int]

class Msg:
    NOT_FOUND_COORD = "선택된 픽셀 좌표를 찾을 수 없습니다. 상단 그래프에서 스펙트럼을 선택하거나, 후보/비교 표의 행을 선택하세요."
    PICK_SELECT_SPEC = "상단 그래프에서 스펙트럼(빨간선)을 먼저 선택해 주세요."

class ColCmp:
    CHECK = 0; NO = 1; LOC = 2; CLASS = 3; METRIC = 4; VALUE = 5


class ClassmapLabelingDialog(QtWidgets.QDialog):
    """
    분류맵 기반 라벨링 다이얼로그
    - 좌 : 유사도 히스토그램(histWidget)
    - 우상 : 선택 스펙트럼 번들(spectrumWidget) (선택 1개 빨강)
    - 중간 버튼 : [선택 클래스 분석]
    - 좌하 : 비교 스펙트럼(cmpSpectrumWidget) (라이브러리/라벨링 평균 + 선택1)
    - 우하 : 비교 유사도 표(tableCompare) / [추가] 버튼으로 적재
    """
    labeling_requested = pyqtSignal(list)       # [{"y":int,"x":int,"cid":int}, ...]
    candidates_to_user_labeling = pyqtSignal(list)
    spectrum_pixel_selected = pyqtSignal(int, int)  # (y, x) : 상단 스펙트럼 클릭 시 MapView 반영 용도

    # ---------------------------------------------------------------------
    # 생성/초기화
    # ---------------------------------------------------------------------
    def __init__(self, parent=None, ui_dir: Optional[Path] = None):
        super().__init__(parent)
        self.setWindowTitle("분류맵 기반 라벨링")

        app_dir = Path(getattr(parent, "app_dir", Path(__file__).resolve().parents[3]))
        ui_dir = Path(ui_dir) if ui_dir else (app_dir / "ui")
        self._root = uic.loadUi(str(ui_dir / "classmap_labeling_dialog.ui"), baseinstance=self)

        # 내부 상태
        self._picked_pixels: List[Tuple[int, int, int]] = []  # (y, x, cid)
        self._classmap_options: List[str] = []
        self._class_options: List[Tuple[int, str]] = []       # [(cid, name)]
        self._cid_to_qcolor: Dict[int, QColor] = {}
        self._current_classmap_name: Optional[str] = None
        self._default_cid: Optional[int] = None  # user_labeling_dialog에서 선택한 기본 CID

        # ---- 글로벌 캐시(메인에서 주입) ----
        self._glob_cube: Optional[np.ndarray] = None       # (H,W,C)
        self._glob_classmap: Optional[np.ndarray] = None   # (H,W)
        self._glob_topk_cids: Optional[np.ndarray] = None  # (H,W,K)
        self._glob_topk_vals: Optional[np.ndarray] = None  # (H,W,K)
        self._glob_id2name: Dict[int, str] = {}
        self._glob_wavelengths: Optional[np.ndarray] = None

        # 데이터 콜백(메인에서 주입)
        self._get_classmap_data_callback: Optional[Callable[[str], np.ndarray]] = None
        self._get_similarity_callback: Optional[Callable[[str, int], np.ndarray]] = None
        self._get_spectrum_samples_callback: Optional[Callable[[str, int, int], List[np.ndarray]]] = None
        self._get_class_pixels_callback: Optional[Callable[[str, int], Tuple[np.ndarray, np.ndarray]]] = None  # ★ 복원
        self._get_class_library_callback: Optional[Callable[[int], List[np.ndarray]]] = None
        self._get_class_labeling_callback: Optional[Callable[[int], List[np.ndarray]]] = None
        self._get_bin_spectra_callback: Optional[Callable[[str, int, float, float, int], List[Tuple[np.ndarray, Tuple[int,int]]]]] = None
        self._get_map_is_distance: Optional[Callable[[str], bool]] = None
        self._on_spectrum_selected_cb: Optional[Callable[[int, int], None]] = None

        # 스펙트럼 선택 상태
        self._hist_selected_bin: Optional[int] = None
        self._spec_selected_index: Optional[int] = None
        self._spec_screen_polylines: List[List[Tuple[int,int]]] = []
        self._spec_data_cache: List[np.ndarray] = []
        self._spec_coords_cache: List[Optional[Tuple[int,int]]] = []
        self._last_selected_pixel_spec: Optional[np.ndarray] = None
        self._last_selected_yx: Optional[Tuple[int,int]] = None

        # 렌더 상태
        self._last_hist = {"vec": None, "edges": None, "counts": None, "bins": 0}
        self._hist_plot_rect: Optional[Tuple[int,int,int,int]] = None  # (left, top, right, bottom)

        # 클릭 이벤트 필터
        self.histWidget.installEventFilter(self)
        self.spectrumWidget.installEventFilter(self)

        # 버튼 연결
        self.btnAnalyzeSelectedClass.clicked.connect(self._on_analyze_selected_class)
        self.btnAddSelectedToCandidates.clicked.connect(self._on_add_selected_pixels_to_candidates)
        try:
            self.btnRegisterCandidates.clicked.disconnect()
        except Exception:
            pass
        self.btnRegisterCandidates.clicked.connect(self._on_register_checked_to_user_labeling)
        self.btnCancel.clicked.connect(self.reject)

        # 콤보/슬라이더 연결
        self.cmbClassmap.currentTextChanged.connect(self._on_classmap_changed)
        self.cmbClassFilter.currentIndexChanged.connect(self._on_class_changed)
        self.sldBinStep.valueChanged.connect(self._refresh_histogram_only)
        self.spnMaxSamples.valueChanged.connect(self._refresh_spectrum_only)

        # 표 초기화
        self._init_tables()

    # ---------------------------------------------------------------------
    # 외부 주입 API
    # ---------------------------------------------------------------------
    def set_on_spectrum_selected(self, fn: Callable[[int, int], None]):
        self._on_spectrum_selected_cb = fn

    def _norm_id2name(self, src) -> Dict[int, str]:
        try:
            if not src: return {}
            if isinstance(src, dict):
                return {int(k): str(v) for k, v in src.items()}
            if isinstance(src, (list, tuple)):
                out: Dict[int, str] = {}
                for it in src:
                    if isinstance(it, dict) and "mtrl_cd" in it and "mtrl_nm" in it:
                        out[int(it["mtrl_cd"])] = str(it["mtrl_nm"]); continue
                    if isinstance(it, (list, tuple)) and len(it) == 2:
                        cid, nm = it; out[int(cid)] = str(nm)
                return out
        except Exception:
            logging.exception("[set_global_caches] id_to_name normalize failed")
        return {}

    def set_global_caches(self,
                          cube: Optional[np.ndarray],
                          classmap: Optional[np.ndarray],
                          topk_cids: Optional[np.ndarray],
                          topk_vals: Optional[np.ndarray],
                          id_to_name=None):
        self._glob_cube = cube
        self._glob_classmap = classmap
        self._glob_topk_cids = topk_cids
        self._glob_topk_vals = topk_vals
        self._glob_id2name = self._norm_id2name(id_to_name)

    def set_classmap_options(self, items: List[str]):
        self._classmap_options = list(items)
        self.cmbClassmap.clear()
        for name in items:
            self.cmbClassmap.addItem(name)
        if items:
            self.cmbClassmap.setCurrentIndex(0)
            QtWidgets.QApplication.processEvents()
            self._refresh_histogram_only()

    def set_classmap_data_callback(self, cb: Callable[[str], np.ndarray]): self._get_classmap_data_callback = cb
    def set_similarity_data_callback(self, cb: Callable[[str, int], np.ndarray]): self._get_similarity_callback = cb
    def set_spectrum_samples_callback(self, cb: Callable[[str, int, int], List[np.ndarray]]): self._get_spectrum_samples_callback = cb
    def set_class_library_callback(self, cb: Callable[[int], List[np.ndarray]]): self._get_class_library_callback = cb
    def set_class_pixels_callback(self, cb: Callable[[str, int], Tuple[np.ndarray, np.ndarray]]):
        """하위호환: (map_name, cid) -> (ys, xs) 콜백 저장만 함."""
        self._get_class_pixels_callback = cb
    def set_class_labeling_callback(self, cb: Callable[[int], List[np.ndarray]]): self._get_class_labeling_callback = cb
    def set_bin_spectra_callback(self, cb: Callable[[str, int, float, float, int], List[Tuple[np.ndarray, Tuple[int,int]]]]):
        self._get_bin_spectra_callback = cb
    def set_map_semantics_callback(self, fn: Callable[[str], bool]): self._get_map_is_distance = fn
    def set_selected_pixel_callback(self, _cb: Callable[[], Optional[np.ndarray]]):
        # 호환용: 현재 내부 선택 캐시(_last_selected_pixel_spec)를 사용하므로 저장만 하거나 무시
        self._selected_pixel_external_cb = _cb
    def set_wavelengths(self, wavelengths: Optional[np.ndarray]):
        try:
            w = np.asarray(wavelengths, dtype=float).ravel()
            self._glob_wavelengths = w if w.size > 1 else None
        except Exception:
            self._glob_wavelengths = None

    def set_class_options(self, items: List[Tuple[int, str]]):
        # ★ 전체 클래스 목록 저장 (name lookup용)
        self._class_options = list(items) if items else []
        
        # 콤보박스는 classmap 변경 시 자동 갱신되므로, 여기서는 저장만 함
        # (초기 classmap이 설정된 후 _on_classmap_changed에서 _rebuild_class_options_from_map 호출됨)
        # 단, 초기값 설정을 위해 현재 classmap이 있으면 바로 재구성
        if self._current_classmap_name:
            self._rebuild_class_options_from_map(self._current_classmap_name)
        else:
            # classmap이 없으면 전체 옵션 표시
            self.cmbClassFilter.blockSignals(True)
            self.cmbClassFilter.clear()
            for cid, name in items:
                self.cmbClassFilter.addItem(name, int(cid))  # 클래스 이름만 표시, 내부 데이터는 CID
            self.cmbClassFilter.blockSignals(False)
            if items:
                # ★ 기본 CID가 있으면 해당 인덱스로 설정, 없으면 첫 번째
                if self._default_cid is not None:
                    idx = self.cmbClassFilter.findData(self._default_cid)
                    if idx >= 0:
                        self.cmbClassFilter.setCurrentIndex(idx)
                    else:
                        self.cmbClassFilter.setCurrentIndex(0)
                else:
                    self.cmbClassFilter.setCurrentIndex(0)
                QtWidgets.QApplication.processEvents()
                self._refresh_histogram_only()
    
    def set_default_cid(self, cid: Optional[int]) -> None:
        """user_labeling_dialog에서 선택한 기본 CID 설정."""
        self._default_cid = int(cid) if cid is not None else None

    def set_palette(self, cid_to_qcolor: Dict[int, QColor]): self._cid_to_qcolor = dict(cid_to_qcolor)

    # ---------------------------------------------------------------------
    # 내부 UI
    # ---------------------------------------------------------------------
    def _init_tables(self):
        # 후보 테이블
        table = getattr(self, "tablePickedPixels", None)
        if table is not None:
            table.setColumnCount(4)
            table.setHorizontalHeaderLabels(["#", "Locate", "Select Class", "option"])
            table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            table.setAlternatingRowColors(True)
            if table.horizontalHeader():
                table.horizontalHeader().setStretchLastSection(False)
                table.horizontalHeader().setDefaultSectionSize(110)
            if table.verticalHeader():
                table.verticalHeader().setVisible(False)
            table.setColumnWidth(0, 40); table.setColumnWidth(1, 120)
            table.setColumnWidth(2, 150); table.setColumnWidth(3, 80)

        # 비교 유사도 표
        cmp_tbl = getattr(self, "tableCompare", None)
        if cmp_tbl is not None:
            cmp_tbl.setColumnCount(6)
            cmp_tbl.setHorizontalHeaderLabels(["", "#", "Locate (X, Y)", "Class", "metric", "value"])
            cmp_tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            cmp_tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            cmp_tbl.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            if cmp_tbl.verticalHeader():
                cmp_tbl.verticalHeader().setVisible(False)

    # ---------------------------------------------------------------------
    # 이벤트 핸들러
    # ---------------------------------------------------------------------
    def _on_classmap_changed(self, name: str):
        self._current_classmap_name = name if name else None
        self._hist_selected_bin = None
        self._spec_selected_index = None
        self._last_selected_pixel_spec = None
        
        # ★ classmap 변경 시 해당 맵에 존재하는 클래스만 필터 콤보에 표시
        self._rebuild_class_options_from_map(self._current_classmap_name)
        
        self._refresh_class_count_table()
        self._refresh_histogram_only()
        self._refresh_spectrum_only()
        self.refresh_compare_spectrum()
        
    def _on_class_changed(self, _index: int):
        self._hist_selected_bin = None
        self._spec_selected_index = None
        self._last_selected_pixel_spec = None
        self._refresh_histogram_only()
        self._refresh_spectrum_only()
        self._refresh_class_count_table()
        self.refresh_compare_spectrum()  # ★ 추가


    def _on_analyze_selected_class(self):
        """좌하 비교창 갱신(라이브러리/라벨링/선택1)."""
        cid = self._current_cid()
        if cid is None:
            self._clear_widget(self.cmpSpectrumWidget); return

        # 선택(빨간선) 확인
        if self._last_selected_pixel_spec is None:
            QtWidgets.QMessageBox.information(self, "안내", Msg.PICK_SELECT_SPEC); return

        # 이하 동일 (라이브러리/라벨링 불러와 비교창 그리기)
        lib_specs, label_specs = [], []
        if callable(self._get_class_library_callback):
            try: lib_specs = self._get_class_library_callback(int(cid)) or []
            except Exception: logging.exception("[Analyze] class library fetch failed")
        if callable(self._get_class_labeling_callback):
            try: label_specs = self._get_class_labeling_callback(int(cid)) or []
            except Exception: logging.exception("[Analyze] class labeling fetch failed")

        selected_spec = self._last_selected_pixel_spec
        self._draw_compare_into(self.cmpSpectrumWidget, lib_specs, label_specs, selected_spec)

        # 라이브러리/라벨링
        lib_specs, label_specs = [], []
        if callable(self._get_class_library_callback):
            try: lib_specs = self._get_class_library_callback(int(cid)) or []
            except Exception: logging.exception("[Analyze] class library fetch failed")
        if callable(self._get_class_labeling_callback):
            try: label_specs = self._get_class_labeling_callback(int(cid)) or []
            except Exception: logging.exception("[Analyze] class labeling fetch failed")

        selected_spec = self._last_selected_pixel_spec
        self._draw_compare_into(self.cmpSpectrumWidget, lib_specs, label_specs, selected_spec)

        # 캡션
        if hasattr(self, "lblCmpCaption"):
            try:
                n_lib = sum(1 for s in lib_specs if s is not None and np.asarray(s).size > 1)
                n_lab = sum(1 for s in label_specs if s is not None and np.asarray(s).size > 1)
                picked = "Y" if (selected_spec is not None and np.asarray(selected_spec).size > 1) else "N"
                self.lblCmpCaption.setText(
                    f"비교 (Class {cid}): 라이브러리 {n_lib}개, 라벨링 {n_lab}개, 선택1={picked} "
                    "(파랑=Lib 평균, 초록=Label 평균, 빨강=선택1)"
                )
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # “추가” 버튼
    # ---------------------------------------------------------------------
    def _on_add_selected_pixels_to_candidates(self):
        tcmp = getattr(self, "tableCompare", None)
        if tcmp is None:
            QtWidgets.QMessageBox.information(self, "안내", "tableCompare가 없습니다."); return

        # 좌표 소스: ①상단선택 → ②후보테이블 → ③비교표 선택행
        yx = self._pick_yx()
        if yx is None:
            QtWidgets.QMessageBox.information(self, "안내", Msg.NOT_FOUND_COORD)
            return
        y, x = map(int, yx)

        # 스펙 검증
        spec = self._get_spectrum_at_xy(y, x)
        if spec is None or spec.size <= 1:
            QtWidgets.QMessageBox.warning(self, "오류", f"({y}, {x}) 위치의 스펙트럼을 가져오지 못했습니다.")
            return

        # 클래스 & 메트릭
        cid = self._current_cid()
        cid_name = ""
        if cid is not None:
            cid_name = next((n for c, n in self._class_options if int(c) == int(cid)), "")
        class_text = cid_name if cid_name else "-"
        metric = str(self._current_classmap_name) if self._current_classmap_name else "TopK"

        # 유사도 값(있으면)
        value = self._sim_at_xy_for_cid(y, x, cid)

        # 표에 한 행 추가
        r = tcmp.rowCount(); tcmp.insertRow(r)
        chk = QtWidgets.QCheckBox(); chk.setChecked(True); chk.setTristate(False)
        chk.setStyleSheet("margin-left:8px;")
        tcmp.setCellWidget(r, ColCmp.CHECK, chk)

        def _set(c, s, center=True):
            it = QtWidgets.QTableWidgetItem(s)
            if center: it.setTextAlignment(Qt.AlignCenter)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            tcmp.setItem(r, c, it)

        _set(ColCmp.NO, str(r + 1))
        _set(ColCmp.LOC, f"{x}, {y}")  # 표시용은 X, Y
        _set(ColCmp.CLASS, class_text)
        _set(ColCmp.METRIC, metric)
        _set(ColCmp.VALUE, "-" if value is None else f"{value:.6f}")

        # 내부 데이터(UserRole)
        tcmp.item(r, ColCmp.LOC).setData(Qt.UserRole, (y, x))
        tcmp.item(r, ColCmp.CLASS).setData(Qt.UserRole, int(cid) if cid is not None else -1)

        try:
            hdr = tcmp.horizontalHeader()
            if hdr:
                hdr.setStretchLastSection(True)
                hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # 후보 테이블
    # ---------------------------------------------------------------------
    def add_pixel_label(self, y: int, x: int, class_id: Optional[int] = None):
        if class_id is None:
            # ★ 기본 CID가 있으면 우선 사용, 없으면 첫 번째 클래스
            if self._default_cid is not None:
                class_id = self._default_cid
            else:
                class_id = self._class_options[0][0] if self._class_options else -1
        self._picked_pixels.append((int(y), int(x), int(class_id)))
        self._append_row(len(self._picked_pixels) - 1)
        self._last_selected_yx = (int(y), int(x))
        spec = self._get_spectrum_at_xy(int(y), int(x))
        if spec is not None and spec.size > 1:
            self._last_selected_pixel_spec = np.asarray(spec).ravel().copy()

    def _append_row(self, idx: int):
        table = self.tablePickedPixels
        table.insertRow(idx)
        y, x, cid = self._picked_pixels[idx]

        it_no = QtWidgets.QTableWidgetItem(str(idx + 1))
        it_no.setTextAlignment(Qt.AlignCenter)
        it_no.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        table.setItem(idx, 0, it_no)

        it_loc = QtWidgets.QTableWidgetItem(f"{y}, {x}")
        it_loc.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        table.setItem(idx, 1, it_loc)

        combo = QtWidgets.QComboBox(table)
        for cid_opt, name in self._class_options:
            combo.addItem(name, int(cid_opt))  # 클래스 이름만 표시, 내부 데이터는 CID
        if self._class_options:
            # ★ 기본값 결정: 기본 CID가 있고 목록에 있으면 우선 사용, 없으면 현재 CID
            existed_cids = {int(cid_opt) for cid_opt, _ in self._class_options}
            default_cid = self._default_cid if self._default_cid is not None and self._default_cid in existed_cids else cid
            
            # 기본 CID 선택
            for i in range(combo.count()):
                if int(combo.itemData(i)) == int(default_cid):
                    combo.setCurrentIndex(i); break
        combo.currentIndexChanged.connect(lambda _=None, row=idx, cb=combo: self._on_combo_changed(row, cb))
        table.setCellWidget(idx, 2, combo)

        btn = QtWidgets.QPushButton("✖")
        btn.setToolTip("삭제")
        btn.clicked.connect(lambda _=None, row=idx: self._remove_row(row))
        table.setCellWidget(idx, 3, btn)

    def _on_combo_changed(self, row: int, combo: QtWidgets.QComboBox):
        if 0 <= row < len(self._picked_pixels):
            y, x, _ = self._picked_pixels[row]
            cid = int(combo.currentData())
            self._picked_pixels[row] = (y, x, cid)

    def _remove_row(self, row: int):
        if not (0 <= row < len(self._picked_pixels)): return
        self._picked_pixels.pop(row)
        table = self.tablePickedPixels
        table.removeRow(row)
        # 인덱스/시그널 재정렬
        for i in range(table.rowCount()):
            if table.item(i, 0): table.item(i, 0).setText(str(i + 1))
            w_del = table.cellWidget(i, 3)
            if isinstance(w_del, QtWidgets.QPushButton):
                try: w_del.clicked.disconnect()
                except Exception: pass
                w_del.clicked.connect(lambda _=None, r=i: self._remove_row(r))
            w_cb = table.cellWidget(i, 2)
            if isinstance(w_cb, QtWidgets.QComboBox):
                try: w_cb.currentIndexChanged.disconnect()
                except Exception: pass
                w_cb.currentIndexChanged.connect(lambda _=None, r=i, cb=w_cb: self._on_combo_changed(r, cb))

    def get_pixel_labels(self) -> List[Dict[str, int]]:
        return [{"y": int(y), "x": int(x), "cid": int(cid)} for (y, x, cid) in self._picked_pixels]

    def _on_register(self):
        if not self._picked_pixels:
            QtWidgets.QMessageBox.warning(self, "경고", "라벨링할 픽셀이 없습니다."); return
        self.labeling_requested.emit(self.get_pixel_labels())
        self.accept()

    # ---------------------------------------------------------------------
    # 그래프 갱신
    # ---------------------------------------------------------------------
    def _refresh_histogram_only(self):
        map_name = self._current_classmap_name
        cid = self._current_cid()
        if (map_name is None) or (cid is None) or (self._get_similarity_callback is None):
            self._clear_widget(self.histWidget); return
        try:
            vec = self._get_similarity_callback(map_name, int(cid))
            vec = np.asarray(vec).ravel()
            vec = vec[np.isfinite(vec)]
            if vec.size == 0:
                self._clear_widget(self.histWidget); return
        except Exception:
            logging.exception("[ClassmapLabeling] similarity callback failed")
            self._clear_widget(self.histWidget); return

        bins = int(max(5, min(200, self.sldBinStep.value() if hasattr(self, "sldBinStep") else 20)))
        self._draw_histogram_into(self.histWidget, vec, bins=bins)

    def _refresh_histogram_only(self):
        """
        히스토그램 전용 갱신.
        우선순위:
        1) self._get_similarity_callback(map_name, cid) → 1D vec
        2) self._get_topk_for_map(map_name) + classmap 필터 → vec (Top-1 값)
        """
        map_name = self._current_classmap_name
        cid = self._current_cid()
        if (map_name is None) or (cid is None):
            self._clear_widget(self.histWidget)
            return

        vec = None

        # (1) 1순위: 외부 similarity 콜백
        if callable(self._get_similarity_callback):
            try:
                v = self._get_similarity_callback(map_name, int(cid))
                v = np.asarray(v).ravel()
                if v.size > 0:
                    vec = v[np.isfinite(v)]
            except Exception:
                logging.exception("[Histogram] similarity callback failed")

        # (2) 2순위: Top-K 캐시 + 클래스맵에서 cid만 추출
        if vec is None or vec.size == 0:
            try:
                cm = self._get_current_classmap()                          # (H,W)
                cids, vals = None, None
                if callable(getattr(self, "_get_topk_for_map", None)):
                    cids, vals = self._get_topk_for_map(map_name)          # (H,W,K)
                if (cm is not None) and isinstance(vals, np.ndarray) and vals.ndim == 3 and vals.shape[2] >= 1:
                    v1 = np.asarray(vals[..., 0], dtype=float)             # Top-1 값
                    m = (cm == int(cid)) & np.isfinite(v1)
                    vec2 = v1[m].ravel()
                    if vec2.size > 0:
                        vec = vec2
            except Exception:
                logging.exception("[Histogram] fallback(topk+cm) failed")

        # (3) 최종 드로잉
        if vec is None or vec.size == 0:
            self._clear_widget(self.histWidget)
            return

        bins = int(max(5, min(200, self.sldBinStep.value() if hasattr(self, "sldBinStep") else 20)))
        self._draw_histogram_into(self.histWidget, vec, bins=bins)


    def _current_cid(self) -> Optional[int]:
        if self.cmbClassFilter.count() == 0: return None
        data = self.cmbClassFilter.currentData()
        try: return int(data)
        except Exception: return None

    # ---------------------------------------------------------------------
    # 드로잉 유틸
    # ---------------------------------------------------------------------
    def _clear_widget(self, widget: QtWidgets.QWidget):
        canvas_w, canvas_h = max(10, widget.width()), max(10, widget.height())
        pix = QPixmap(canvas_w, canvas_h); pix.fill(Qt.white)
        p = QPainter(pix); p.fillRect(0, 0, canvas_w, canvas_h, Qt.white); p.end()
        self._ensure_overlay_label(widget).setPixmap(pix)

    def _draw_histogram_into(self, widget: QtWidgets.QWidget, vec: np.ndarray, bins: int = 20):
        canvas_w, canvas_h = max(60, widget.width()), max(60, widget.height())
        pix = QPixmap(canvas_w, canvas_h); pix.fill(Qt.white)
        p = QPainter(pix)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            counts, edges = (None, None)
            if vec.size > 0:
                counts, edges = np.histogram(vec, bins=bins)
            mx = max(1, int(counts.max())) if counts is not None else 1

            # 눈금 텍스트
            y_ticks = ["0", self._fmt_num(mx/2), self._fmt_num(mx)]
            if edges is not None:
                xmin, xmax = float(edges[0]), float(edges[-1])
                x_ticks = [self._fmt_num(xmin), self._fmt_num((xmin+xmax)/2), self._fmt_num(xmax)]
            else:
                x_ticks = ["0.0","0.5","1.0"]

            left_pad, top_pad, right_pad, bottom_pad, fm = self._measure_axes_pads(
                p, "Similarity", "Count", x_ticks, y_ticks
            )
            left, right, top, bottom = left_pad, canvas_w - right_pad, top_pad, canvas_h - bottom_pad

            # 클릭용 플롯 박스 저장
            self._hist_plot_rect = (int(left), int(top), int(right), int(bottom))

            # 프레임
            p.setPen(QPen(Qt.black)); p.drawRect(left, top, right - left, bottom - top)

            # 막대
            if counts is not None:
                bw = max(1, int((right - left) / bins))
                for i, cnt in enumerate(counts):
                    bh = int((cnt / mx) * (bottom - top))
                    x0 = left + i * bw; y0 = bottom - bh
                    color = Qt.red if (self._hist_selected_bin is not None and i == int(self._hist_selected_bin)) else Qt.gray
                    p.fillRect(x0, y0, max(1, bw - 1), bh, color)

            # 그리드
            p.setPen(QPen(QColor(225,225,225), 1, Qt.DotLine))
            for frac in (0.25, 0.5, 0.75):
                yline = int(bottom - frac*(bottom-top))
                p.drawLine(left, yline, right, yline)

            # 눈금/라벨 ─ Y
            p.setPen(QPen(Qt.black))
            max_yw = max(fm.horizontalAdvance(t) for t in y_ticks)
            for v, t in zip([0, mx/2, mx], y_ticks):
                ypix = int(bottom - (v/mx)*(bottom-top)) if mx>0 else bottom
                p.drawLine(left-4, ypix, left, ypix)
                p.drawText(left-6-max_yw, ypix - fm.height()//2, max_yw, fm.height(),
                           Qt.AlignRight|Qt.AlignVCenter, t)

            # 눈금/라벨 ─ X
            if edges is not None:
                xv = [xmin, 0.5*(xmin+xmax), xmax]
                for xval, txt in zip(xv, x_ticks):
                    rng = max(xmax - xmin, 1e-12)
                    t = (xval - xmin) / rng
                    xp = left + int(round(t * (right - left)))
                    xp = max(left, min(right, xp))  # 클램프
                    p.drawLine(xp, bottom, xp, bottom+4)
                    tw = fm.horizontalAdvance(txt)
                    p.drawText(xp - tw//2, bottom + 4 + fm.ascent(), txt)

            # 축 제목
            p.drawText((left+right)//2 - fm.horizontalAdvance("Similarity")//2, canvas_h - 6, "Similarity")
            p.save()
            p.translate(left - (max_yw + 10 + fm.height()//2), (top+bottom)//2)
            p.rotate(-90); p.drawText(-fm.horizontalAdvance("Count")//2, fm.ascent()//2, "Count"); p.restore()

        finally:
            p.end()
        self._ensure_overlay_label(widget).setPixmap(pix)
        self._last_hist.update({"vec": vec, "edges": edges, "counts": counts, "bins": int(bins)})

    def _draw_spectrum_bundle_into(
        self,
        widget: QtWidgets.QWidget,
        spectra: List[np.ndarray],
        coords: Optional[List[Optional[Tuple[int, int]]]] = None,
    ) -> None:
        """스펙트럼 번들을 그리고, spectra[i] ↔ coords[i]를 1:1로 캐시한다."""
        canvas_w = max(40, int(widget.width()))
        canvas_h = max(40, int(widget.height()))
        pixmap   = QPixmap(canvas_w, canvas_h); pixmap.fill(Qt.white)
        painter  = QPainter(pixmap)

        # 데이터/좌표 캐시 정규화
        self._spec_screen_polylines = []
        self._spec_data_cache = [
            np.asarray(s, dtype=float).ravel()
            for s in (spectra or [])
            if s is not None and np.asarray(s).size > 1
        ]
        if isinstance(coords, list) and len(coords) == len(self._spec_data_cache):
            self._spec_coords_cache = [
                (int(c[0]), int(c[1])) if (c and len(c) == 2) else None
                for c in coords
            ]
        else:
            self._spec_coords_cache = [None] * len(self._spec_data_cache)

        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            if not self._spec_data_cache:
                self._ensure_overlay_label(widget).setPixmap(pixmap)
                return

            series_len = min(len(s) for s in self._spec_data_cache)
            series = [s[:series_len] for s in self._spec_data_cache]
            mean_s = np.nanmean(series, axis=0)

            y_min = float(np.nanmin([*series, mean_s])); y_max = float(np.nanmax([*series, mean_s]))
            if not (np.isfinite(y_min) and np.isfinite(y_max)) or y_min >= y_max:
                y_min, y_max = 0.0, 1.0
            y_rng = (y_max - y_min)

            wave = getattr(self, "_glob_wavelengths", None)
            use_wave = isinstance(wave, np.ndarray) and wave.size >= series_len
            if use_wave:
                wave = wave[:series_len].astype(float)
                w_min, w_max = float(np.nanmin(wave)), float(np.nanmax(wave))
                w_rng = max(w_max - w_min, 1e-12)

            y_ticks = [self._fmt_num(y_min), self._fmt_num(0.5*(y_min + y_max)), self._fmt_num(y_max)]
            if use_wave:
                mid_w = 0.5 * (w_min + w_max)
                x_ticks = [f"{w_min:.0f}", f"{mid_w:.0f}", f"{w_max:.0f}"] if (w_max - w_min) >= 10 \
                          else [f"{w_min:.2f}", f"{mid_w:.2f}", f"{w_max:.2f}"]
                x_label = "Wavelength (nm)"
            else:
                x_ticks = ["0", str(int(0.5*(series_len - 1))), str(series_len - 1)]
                x_label = "Band Index"

            left_pad, top_pad, right_pad, bottom_pad, fm = self._measure_axes_pads(
                painter, x_label, "Reflectance/Intensity", x_ticks, y_ticks
            )
            plot_left, plot_right = left_pad, canvas_w - right_pad
            plot_top, plot_bottom = top_pad, canvas_h - bottom_pad

            def to_xy(i: int, val: float) -> Tuple[int, int]:
                if use_wave:
                    t = (wave[i] - w_min) / w_rng if series_len > 1 else 0.0
                else:
                    t = i / (series_len - 1) if series_len > 1 else 0.0
                x_pos = plot_left + t * (plot_right - plot_left)
                y_pos = plot_bottom - ((val - y_min) / y_rng) * (plot_bottom - plot_top)
                return int(x_pos), int(y_pos)

            # 그리드 + 프레임
            painter.setPen(QPen(QColor(225, 225, 225), 1, Qt.DotLine))
            for frac in (0.25, 0.5, 0.75):
                painter.drawLine(int(plot_left + frac*(plot_right-plot_left)), plot_top,
                                 int(plot_left + frac*(plot_right-plot_left)), plot_bottom)
                painter.drawLine(plot_left, int(plot_bottom - frac*(plot_bottom-plot_top)),
                                 plot_right, int(plot_bottom - frac*(plot_bottom-plot_top)))
            painter.setPen(QPen(Qt.black))
            painter.drawRect(plot_left, plot_top, plot_right - plot_left, plot_bottom - plot_top)

            # 번들
            self._spec_screen_polylines = [[to_xy(i, s[i]) for i in range(series_len)] for s in series]
            for idx, poly in enumerate(self._spec_screen_polylines):
                pen = QPen(Qt.red, 2) if (self._spec_selected_index is not None and idx == int(self._spec_selected_index)) \
                      else QPen(Qt.gray, 1)
                painter.setPen(pen)
                for i in range(series_len - 1):
                    x1, y1 = poly[i]; x2, y2 = poly[i+1]
                    painter.drawLine(x1, y1, x2, y2)

            # 평균(보라)
            painter.setPen(QPen(QColor(200, 0, 200), 2))
            for i in range(series_len - 1):
                x1, y1 = to_xy(i, mean_s[i]); x2, y2 = to_xy(i+1, mean_s[i+1])
                painter.drawLine(x1, y1, x2, y2)

            # 축 눈금/라벨
            painter.setPen(QPen(Qt.black))
            if use_wave:
                for xv, text in zip([w_min, 0.5*(w_min+w_max), w_max], x_ticks):
                    t = (xv - w_min) / w_rng
                    xp = int(plot_left + t*(plot_right-plot_left))
                    painter.drawLine(xp, plot_bottom, xp, plot_bottom+4)
                    painter.drawText(xp - fm.horizontalAdvance(text)//2, plot_bottom + 4 + fm.ascent(), text)
            else:
                for bi, text in zip([0, int(0.5*(series_len-1)), series_len-1], x_ticks):
                    xp, _ = to_xy(bi, y_min)
                    painter.drawLine(xp, plot_bottom, xp, plot_bottom+4)
                    painter.drawText(xp - fm.horizontalAdvance(text)//2, plot_bottom + 4 + fm.ascent(), text)

            max_y_text_w = max(fm.horizontalAdvance(t) for t in y_ticks)
            for val, text in zip([y_min, 0.5*(y_min+y_max), y_max], y_ticks):
                _, yp = to_xy(0, val)
                painter.drawLine(plot_left-4, yp, plot_left, yp)
                painter.drawText(plot_left - 6 - max_y_text_w, yp - fm.height()//2,
                                 max_y_text_w, fm.height(),
                                 Qt.AlignRight | Qt.AlignVCenter, text)

            painter.drawText((plot_left+plot_right)//2 - fm.horizontalAdvance(x_label)//2,
                             canvas_h - 6, x_label)
            painter.save()
            y_label = "Reflectance/Intensity"
            painter.translate(plot_left - (max_y_text_w + 10 + fm.height()//2), (plot_top + plot_bottom)//2)
            painter.rotate(-90)
            painter.drawText(-fm.horizontalAdvance(y_label)//2, fm.ascent()//2, y_label)
            painter.restore()

        finally:
            painter.end()

        self._ensure_overlay_label(widget).setPixmap(pixmap)

    def _ensure_overlay_label(self, host: QtWidgets.QWidget) -> QtWidgets.QLabel:
        lbl = getattr(host, "_overlay_label", None)
        if lbl is None:
            lbl = QtWidgets.QLabel(host); lbl.setAlignment(Qt.AlignCenter)
            host._overlay_label = lbl
            if host.layout() is None:
                lay = QtWidgets.QVBoxLayout(host); lay.setContentsMargins(0, 0, 0, 0); lay.addWidget(lbl)
            else:
                host.layout().addWidget(lbl)
            lbl.installEventFilter(self)
        else:
            lbl.installEventFilter(self)
        return lbl

    # ---------------------------------------------------------------------
    # 유틸(글로벌 캐시 사용)
    # ---------------------------------------------------------------------
    def _cube_ok(self) -> bool:
        return isinstance(self._glob_cube, np.ndarray) and self._glob_cube.ndim == 3

    def _cache_shape_ok(self) -> bool:
        return (self._cube_ok() and isinstance(self._glob_classmap, np.ndarray) and self._glob_classmap.ndim == 2)

    def _get_spectrum_at_xy(self, y: int, x: int) -> Optional[np.ndarray]:
        if not self._cube_ok():
            logging.debug("[_get_spectrum_at_xy] cube missing or invalid: %s", type(self._glob_cube))
            return None
        H, W, _ = self._glob_cube.shape
        y, x = int(y), int(x)
        if not (0 <= y < H and 0 <= x < W):
            logging.debug("[_get_spectrum_at_xy] coords out of range: y=%d (H=%d), x=%d (W=%d)", y, H, x, W)
            return None
        try:
            spec = np.asarray(self._glob_cube[y, x, :], dtype=float)
            return spec if spec.size > 1 else None
        except Exception:
            logging.exception("[_get_spectrum_at_xy] exception at (%d, %d)", y, x)
            return None

    def _get_top1_similarity_at_xy(self, y: int, x: int) -> Optional[float]:
        if not (isinstance(self._glob_topk_vals, np.ndarray) and self._glob_topk_vals.ndim == 3):
            return None
        y, x = int(y), int(x)
        Hc, Wc, _ = self._glob_topk_vals.shape
        if not (0 <= y < Hc and 0 <= x < Wc): return None
        row = np.asarray(self._glob_topk_vals[y, x, :]).ravel()
        if row.size == 0 or not np.isfinite(row[0]): return None
        return float(row[0])  # Top-K[0]

    # ---------------------------------------------------------------------
    # eventFilter (히스토그램/스펙트럼 클릭)
    # ---------------------------------------------------------------------
    def eventFilter(self, obj, ev):
        if obj is self.histWidget and ev.type() == ev.MouseButtonPress and ev.button() == Qt.LeftButton:
            self._on_hist_clicked(ev.pos().x(), ev.pos().y()); return True

        if ev.type() == ev.MouseButtonPress and ev.button() == Qt.LeftButton:
            is_src = (obj is self.spectrumWidget)
            ov = getattr(self.spectrumWidget, "_overlay_label", None)
            is_overlay = (ov is not None and obj is ov)
            if is_src or is_overlay:
                pt = ev.pos() if is_src else obj.mapTo(self.spectrumWidget, ev.pos())
                self._on_spectrum_clicked(int(pt.x()), int(pt.y())); return True
        return super().eventFilter(obj, ev)

    # ---------------------------------------------------------------------
    # 히스토그램 클릭 → 우측 번들 갱신
    # ---------------------------------------------------------------------
    def _on_hist_clicked(self, x: int, y: int):
        map_name = self._current_classmap_name
        cid = self._current_cid()
        if (map_name is None) or (cid is None):
            return

        vec = self._last_hist.get("vec")
        edges = self._last_hist.get("edges")
        bins = int(self._last_hist.get("bins") or 0)
        if vec is None or edges is None or bins <= 0 or np.asarray(vec).size == 0:
            return

        rect = self._hist_plot_rect
        if not rect: return
        left, top, right, bottom = rect
        if not (left <= x <= right and top <= y <= bottom): return

        bw_px = max(1, int((right - left) / bins))
        bin_ix = int(min(bins - 1, max(0, (x - left) // bw_px)))

        # bin 변경 시 초기화
        prev_bin = self._hist_selected_bin
        if (prev_bin is None) or (int(prev_bin) != int(bin_ix)):
            self._reset_selection_and_clear_graphs()

        # 하이라이트 후 재그림
        self._hist_selected_bin = bin_ix
        self._draw_histogram_into(self.histWidget, vec, bins=bins)

        lo, hi = float(edges[bin_ix]), float(edges[bin_ix + 1])
        # ★ 마지막 bin은 상한 포함(<=) 처리
        is_last_bin = (bin_ix == bins - 1)
        eps = np.finfo(float).eps * max(1.0, abs(hi))
        hi_sel = hi + (eps if is_last_bin else 0.0)

        specs_for_right: List[np.ndarray] = []
        coords_for_right: List[Tuple[int, int]] = []
        max_n = int(self.spnMaxSamples.value() if hasattr(self, "spnMaxSamples") else 50)

        # (A) bin 콜백
        out = []
        if callable(self._get_bin_spectra_callback):
            out = self._get_bin_spectra_callback(map_name, int(cid), lo, hi_sel, max_n) or []

        for item in out:
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple)):
                spec_i, (yy, xx) = item
                arr = np.asarray(spec_i).ravel()
                if arr.size > 1:
                    specs_for_right.append(arr)
                    coords_for_right.append((int(yy), int(xx)))

        # (B) 좌표를 못 받았으면 역추출 (classmap + similarity + cube)
        if not coords_for_right and self._cache_shape_ok() and callable(self._get_similarity_callback):
            try:
                sim_full = np.asarray(self._get_similarity_callback(map_name, int(cid)))
                cm = self._get_current_classmap()
                if cm is None: raise RuntimeError("classmap is None")
                if sim_full.ndim == 1:
                    H, W, _ = self._glob_cube.shape
                    if sim_full.size != H * W: raise RuntimeError("similarity size mismatch")
                    sim_full = sim_full.reshape(H, W)
                if cm.shape != sim_full.shape: raise RuntimeError("classmap/similarity shape mismatch")

                mask = (cm == int(cid)) & np.isfinite(sim_full) & (sim_full >= lo) & (sim_full < hi)
                ys, xs = np.where(mask)
                if ys.size > 0:
                    take = min(max_n, ys.size)
                    vals_in_bin = sim_full[ys, xs]
                    order = np.argsort(vals_in_bin)[:take] if self._is_distance_map() else np.argsort(-vals_in_bin)[:take]
                    ys, xs = ys[order], xs[order]
                    for yy, xx in zip(ys, xs):
                        spec_i = self._get_spectrum_at_xy(int(yy), int(xx))
                        if spec_i is None or np.asarray(spec_i).size <= 1: continue
                        specs_for_right.append(np.asarray(spec_i).ravel())
                        coords_for_right.append((int(yy), int(xx)))
            except Exception:
                logging.exception("[Histogram] reconstruct (y,x) failed")

        # 번들 + 좌표 1:1 전달
        self._draw_spectrum_bundle_into(self.spectrumWidget, specs_for_right, coords_for_right)

        # 대표 자동선택 금지: 사용자가 곡선을 클릭해야 좌표가 설정됨
        self._last_selected_pixel_spec = None
        self._last_selected_yx = None

        # (선택) 캡션 힌트
        if hasattr(self, "lblSpecCaption"):
            try:
                self.lblSpecCaption.setText(f"선택 픽셀 스펙트럼  [{lo:.4f} ~ {hi:.4f}] (N={len(specs_for_right)}) — 곡선을 클릭해 선택하세요")
            except Exception:
                pass

        # 캡션
        if hasattr(self, "lblSpecCaption"):
            try:
                self.lblSpecCaption.setText(f"선택 픽셀 스펙트럼  [{lo:.4f} ~ {hi:.4f}] (N={len(specs_for_right)})")
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # 상단 스펙트럼 클릭 → 선택/좌표 통지
    # ---------------------------------------------------------------------
    def _on_spectrum_clicked(self, cx: int, cy: int):
        if not self._spec_screen_polylines or not self._spec_data_cache:
            return

        # 드로잉 시 사용한 박스 근사(여백 포함). 과도한 가드 방지: 폴리라인 자체로 거리판정.
        def seg_dist(px, py, ax, ay, bx, by):
            vx, vy = bx - ax, by - ay
            wx, wy = px - ax, py - ay
            vv = vx*vx + vy*vy
            if vv <= 1e-9:
                dx, dy = px - ax, py - ay
                return dx*dx + dy*dy
            t = max(0.0, min(1.0, (wx*vx + wy*vy) / vv))
            qx, qy = ax + t*vx, ay + t*vy
            dx, dy = px - qx, py - qy
            return dx*dx + dy*dy

        best_idx, best_d = None, float("inf")
        for idx, poly in enumerate(self._spec_screen_polylines):
            dmin = float("inf")
            for i in range(len(poly) - 1):
                ax, ay = poly[i]; bx, by = poly[i + 1]
                d = seg_dist(cx, cy, ax, ay, bx, by)
                if d < dmin: dmin = d
            if dmin < best_d:
                best_d = dmin; best_idx = idx

        if best_idx is None: return

        self._spec_selected_index = int(best_idx)
        try:
            self._last_selected_pixel_spec = np.asarray(self._spec_data_cache[self._spec_selected_index]).copy()
        except Exception:
            self._last_selected_pixel_spec = None

        sel_xy = None
        if isinstance(self._spec_coords_cache, list) and 0 <= self._spec_selected_index < len(self._spec_coords_cache):
            sel_xy = self._spec_coords_cache[self._spec_selected_index]
        if isinstance(sel_xy, (tuple, list)) and len(sel_xy) == 2:
            y_sel, x_sel = int(sel_xy[0]), int(sel_xy[1])
            self._last_selected_yx = (y_sel, x_sel)
            # MapView 반영 통지
            try: self.spectrum_pixel_selected.emit(y_sel, x_sel)
            except Exception: pass
            cb = getattr(self, "_on_spectrum_selected_cb", None)
            if callable(cb):
                try: cb(y_sel, x_sel)
                except Exception: pass
        else:
            self._last_selected_yx = None

        # 선택 반영 재그림
        self._draw_spectrum_bundle_into(self.spectrumWidget, self._spec_data_cache, self._spec_coords_cache)

        # 좌표 캡션
        if hasattr(self, "lblSpecCaption") and self._last_selected_yx:
            try:
                yy, xx = self._last_selected_yx
                self.lblSpecCaption.setText(f"선택 픽셀 스펙트럼  (X, Y)=({xx}, {yy})")
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # 비교창 (라이브러리/라벨링/선택1)
    # ---------------------------------------------------------------------
    def refresh_compare_spectrum(self):
        cid = self._current_cid()
        if cid is None:
            self._draw_spectrum_bundle_into(self.cmpSpectrumWidget, [], None); return
        lib_specs = []
        if callable(self._get_class_library_callback):
            try: lib_specs = self._get_class_library_callback(int(cid)) or []
            except Exception: logging.exception("[Compare] class library fetch failed")
        label_specs = []
        if callable(self._get_class_labeling_callback):
            try: label_specs = self._get_class_labeling_callback(int(cid)) or []
            except Exception: logging.exception("[Compare] class labeling fetch failed")
        sel_spec = self._last_selected_pixel_spec
        self._draw_compare_into(self.cmpSpectrumWidget, lib_specs, label_specs, sel_spec)

    def _draw_compare_into(self, widget, lib_specs, label_specs, selected):
        canvas_w, canvas_h = max(40, widget.width()), max(40, widget.height())
        pix = QPixmap(canvas_w, canvas_h); pix.fill(Qt.white)
        p = QPainter(pix)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            def _arrs(lst):
                return [np.asarray(s).ravel() for s in (lst or []) if s is not None and np.asarray(s).size > 1]
            lib_arr, lab_arr, sel_arr = _arrs(lib_specs), _arrs(label_specs), _arrs([selected])
            if not (lib_arr or lab_arr or sel_arr):
                self._ensure_overlay_label(widget).setPixmap(pix); return
            L = min(len(s) for s in (lib_arr + lab_arr + sel_arr))
            if L <= 1: self._ensure_overlay_label(widget).setPixmap(pix); return
            all_arr = [s[:L] for s in (lib_arr + lab_arr + sel_arr)]
            ymin = float(np.nanmin(all_arr)); ymax = float(np.nanmax(all_arr))
            if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin >= ymax:
                ymin, ymax = 0.0, 1.0
            yrng = ymax - ymin

            wv = getattr(self, "_glob_wavelengths", None)
            use_wv = isinstance(wv, np.ndarray) and wv.size >= L
            if use_wv:
                wv = wv[:L].astype(float)
                wmin, wmax = float(np.nanmin(wv)), float(np.nanmax(wv))
                wrng = max(wmax - wmin, 1e-12)

            y_ticks = [self._fmt_num(ymin), self._fmt_num(0.5*(ymin + ymax)), self._fmt_num(ymax)]
            if use_wv:
                midw = 0.5*(wmin + wmax)
                x_ticks = [f"{wmin:.0f}", f"{midw:.0f}", f"{wmax:.0f}"] if (wmax - wmin) >= 10 \
                          else [f"{wmin:.2f}", f"{midw:.2f}", f"{wmax:.2f}"]
                x_label = "Wavelength (nm)"
            else:
                x_ticks = ["0", str(int(0.5*(L-1))), str(L-1)]; x_label = "Band Index"

            left_pad, top_pad, right_pad, bottom_pad, fm = self._measure_axes_pads(
                p, x_label, "Reflectance/Intensity", x_ticks, y_ticks
            )
            left, right, top, bottom = left_pad, canvas_w - right_pad, top_pad, canvas_h - bottom_pad

            def toXY(i: int, v: float):
                if use_wv:
                    t = (wv[i] - wmin) / wrng if L > 1 else 0.0
                else:
                    t = i / (L - 1) if L > 1 else 0.0
                x = left + t * (right - left)
                y = bottom - ((v - ymin) / yrng) * (bottom - top)
                return int(x), int(y)

            p.setPen(QPen(QColor(225, 225, 225), 1, Qt.DotLine))
            for frac in (0.25, 0.5, 0.75):
                xline = int(left + frac * (right - left))
                yline = int(bottom - frac * (bottom - top))
                p.drawLine(xline, top, xline, bottom); p.drawLine(left, yline, right, yline)
            p.setPen(QPen(Qt.black)); p.drawRect(left, top, right - left, bottom - top)

            for s in lib_arr:
                s = s[:L]; p.setPen(QPen(Qt.gray, 1))
                for i in range(L - 1):
                    xA, yA = toXY(i, s[i]); xB, yB = toXY(i + 1, s[i + 1]); p.drawLine(xA, yA, xB, yB)
            if lib_arr:
                M = np.nanmean([s[:L] for s in lib_arr], axis=0); p.setPen(QPen(Qt.blue, 2))
                for i in range(L - 1):
                    xA, yA = toXY(i, M[i]); xB, yB = toXY(i + 1, M[i + 1]); p.drawLine(xA, yA, xB, yB)

            for s in lab_arr:
                s = s[:L]; p.setPen(QPen(QColor(80, 200, 120), 1))
                for i in range(L - 1):
                    xA, yA = toXY(i, s[i]); xB, yB = toXY(i + 1, s[i + 1]); p.drawLine(xA, yA, xB, yB)
            if lab_arr:
                M = np.nanmean([s[:L] for s in lab_arr], axis=0); p.setPen(QPen(QColor(0, 160, 0), 2))
                for i in range(L - 1):
                    xA, yA = toXY(i, M[i]); xB, yB = toXY(i + 1, M[i + 1]); p.drawLine(xA, yA, xB, yB)

            if sel_arr:
                s = sel_arr[0][:L]; p.setPen(QPen(Qt.red, 3))
                for i in range(L - 1):
                    xA, yA = toXY(i, s[i]); xB, yB = toXY(i + 1, s[i + 1]); p.drawLine(xA, yA, xB, yB)

            p.setPen(QPen(Qt.black))
            if use_wv:
                for xv, txt in zip([wmin, 0.5*(wmin + wmax), wmax], x_ticks):
                    t = (xv - wmin) / wrng
                    xp = int(left + t * (right - left))
                    p.drawLine(xp, bottom, xp, bottom + 4)
                    p.drawText(xp - fm.horizontalAdvance(txt)//2, bottom + 4 + fm.ascent(), txt)
            else:
                for bi, txt in zip([0, int(0.5*(L - 1)), L - 1], x_ticks):
                    x, _ = toXY(bi, ymin)
                    p.drawLine(x, bottom, x, bottom + 4)
                    p.drawText(x - fm.horizontalAdvance(txt)//2, bottom + 4 + fm.ascent(), txt)

            max_yw = max(fm.horizontalAdvance(t) for t in y_ticks)
            for val, txt in zip([ymin, 0.5*(ymin + ymax), ymax], y_ticks):
                _, y = toXY(0, val)
                p.drawLine(left - 4, y, left, y)
                p.drawText(left - 6 - max_yw, y - fm.height()//2, max_yw, fm.height(),
                           Qt.AlignRight | Qt.AlignVCenter, txt)

            p.drawText((left + right)//2 - fm.horizontalAdvance(x_label)//2, canvas_h - 6, x_label)
            p.save()
            ylab = "Reflectance/Intensity"
            p.translate(left - (max_yw + 10 + fm.height()//2), (top + bottom)//2)
            p.rotate(-90); p.drawText(-fm.horizontalAdvance(ylab)//2, fm.ascent()//2, ylab); p.restore()

        finally:
            p.end()
        self._ensure_overlay_label(widget).setPixmap(pix)

    # ---------------------------------------------------------------------
    # 클래스 분포 테이블
    # ---------------------------------------------------------------------
    def _refresh_class_count_table(self):
        table = getattr(self, "tableClassCount", None)
        if table is None and not hasattr(self, "tableCompare"): return
        cm = self._get_current_classmap()
        if cm is None:
            if table is not None: table.setRowCount(0)
            return
        self._update_class_count_table(cm)

    def _update_class_count_table(self, classmap_data: np.ndarray):
        table = getattr(self, "tableClassCount", None)
        if table is None: return
        vals, cnts = np.unique(classmap_data, return_counts=True)
        mask = vals >= 0
        vals, cnts = vals[mask], cnts[mask]
        order = np.argsort(vals); vals, cnts = vals[order], cnts[order]
        table.setRowCount(len(vals))
        for i, (cid, c) in enumerate(zip(vals, cnts)):
            name = next((n for cid_opt, n in self._class_options if int(cid_opt) == int(cid)), str(int(cid)))
            it_class = QtWidgets.QTableWidgetItem(name)  # 클래스 이름만 표시
            it_class.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it_cnt = QtWidgets.QTableWidgetItem(str(int(c))); it_cnt.setTextAlignment(Qt.AlignCenter)
            it_cnt.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            table.setItem(i, 0, it_class); table.setItem(i, 1, it_cnt)
        table.resizeColumnsToContents()

    def _get_current_classmap(self) -> Optional[np.ndarray]:
        if self._current_classmap_name is None or self._get_classmap_data_callback is None: return None
        try:
            cm = self._get_classmap_data_callback(self._current_classmap_name)
            if cm is None: return None
            cm = np.asarray(cm)
            if cm.ndim == 0: return None
            return cm
        except Exception:
            logging.exception("[ClassmapLabeling] get_current_classmap failed")
            return None

    def _fmt_num(self, v: float) -> str:
        try:
            av = abs(float(v))
            if av >= 1000: return f"{v:.0f}"
            if av >= 1:    return f"{v:.2f}"
            return f"{v:.4f}"
        except Exception:
            return str(v)

    def _measure_axes_pads(self, painter: QtWidgets.QWidget, x_label: str, y_label: str,
                           x_tick_texts, y_tick_texts):
        fm = painter.fontMetrics()
        h = fm.height()
        max_yw = max([fm.horizontalAdvance(str(t)) for t in (y_tick_texts or ["0"])])
        left   = 10 + max_yw + 6 + h + 6
        bottom = h + 4 + h + 6
        top, right = 8, 10
        return int(left), int(top), int(right), int(bottom), fm

    # ---------------------------------------------------------------------
    # 비교표 체크 수집/후보 등록
    # ---------------------------------------------------------------------
    def _collect_checked_compare_rows(self) -> List[Dict[str, int]]:
        rows = []
        tbl = getattr(self, "tableCompare", None)
        if not isinstance(tbl, QtWidgets.QTableWidget):
            return rows
        for r in range(tbl.rowCount()):
            w = tbl.cellWidget(r, 0)  # 체크박스
            if not (isinstance(w, QtWidgets.QCheckBox) and w.isChecked()):
                continue
            # (y,x)는 UserRole로 저장되어 있어야 합니다.
            yx = None; cid = -1
            loc_item = tbl.item(r, 2)
            cls_item = tbl.item(r, 3)
            if loc_item is not None:
                data = loc_item.data(Qt.UserRole)
                if isinstance(data, (tuple, list)) and len(data) == 2:
                    yx = (int(data[0]), int(data[1]))
            if yx is None and loc_item and (loc_item.text() or "").strip():
                xx, yy = [int(s.strip()) for s in loc_item.text().split(",")]
                yx = (yy, xx)
            if cls_item is not None:
                d = cls_item.data(Qt.UserRole)
                if d is not None:
                    cid = int(d)
            if yx is not None:
                rows.append({"y": yx[0], "x": yx[1], "cid": cid})
        return rows
    
    def _on_register_checked_to_user_labeling(self):
        rows = self._collect_checked_compare_rows()
        if not rows:
            QtWidgets.QMessageBox.information(self, "안내", "체크된 행이 없습니다.")
            return
        self.candidates_to_user_labeling.emit(rows)
        QTimer.singleShot(0, self.accept)

    # ---------------------------------------------------------------------
    # 도우미들
    # ---------------------------------------------------------------------
    def _pick_yx(self) -> Optional[Tuple[int,int]]:
        return self._last_selected_yx or self._loc_from_picked() or self._loc_from_compare()

    def _loc_from_picked(self) -> Optional[Tuple[int,int]]:
        t = getattr(self, "tablePickedPixels", None)
        if not isinstance(t, QtWidgets.QTableWidget) or t.rowCount() == 0: return None
        row = t.currentRow() if t.currentRow() >= 0 else 0
        it = t.item(row, 1)
        if not it: return None
        try:
            y, x = [int(s.strip()) for s in (it.text() or "").replace("(", "").replace(")", "").split(",")]
            return y, x
        except Exception:
            return None

    def _loc_from_compare(self) -> Optional[Tuple[int,int]]:
        t = getattr(self, "tableCompare", None)
        if not isinstance(t, QtWidgets.QTableWidget) or t.rowCount() == 0: return None
        row = t.currentRow() if t.currentRow() >= 0 else 0
        it = t.item(row, ColCmp.LOC)
        if not it: return None
        data = it.data(Qt.UserRole)
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return int(data[0]), int(data[1])
        try:
            xx, yy = [int(s.strip()) for s in (it.text() or "").split(",")]
            return yy, xx
        except Exception:
            return None

    def _sim_at_xy_for_cid(self, y: int, x: int, cid: Optional[int]) -> Optional[float]:
        """
        (y,x)의 해당 class cid 유사도(Top-K 중 매칭되는 값)를 돌려준다.
        우선순위:
        1) self._glob_topk_cids/_vals 가 있으면 사용
        2) 없으면 self._get_topk_for_map(self._current_classmap_name) 호출
        """
        if cid is None:
            return None

        cids = vals = None

        # 1) 글로벌 배열 우선
        if isinstance(self._glob_topk_cids, np.ndarray) and isinstance(self._glob_topk_vals, np.ndarray):
            cids, vals = self._glob_topk_cids, self._glob_topk_vals

        # 2) 콜백 폴백 (classification/diffusion 등 맵별 Top-K)
        if (cids is None or vals is None) and callable(getattr(self, "_get_topk_for_map", None)):
            try:
                cids, vals = self._get_topk_for_map(self._current_classmap_name)
            except Exception:
                cids = vals = None

        # 유효성 검사
        if not (isinstance(cids, np.ndarray) and isinstance(vals, np.ndarray) and
                cids.ndim == 3 and vals.ndim == 3 and cids.shape[:2] == vals.shape[:2]):
            return None

        Hc, Wc, K = cids.shape
        if not (0 <= y < Hc and 0 <= x < Wc):
            return None

        row_c = np.asarray(cids[y, x, :], dtype=int).ravel()
        row_v = np.asarray(vals[y, x, :], dtype=float).ravel()

        hit = np.where(row_c == int(cid))[0]
        if hit.size == 0:
            return None  # K=1인데 다른 클래스거나 ROI 밖(inf/NaN)이면 없음

        v = float(row_v[int(hit[0])])
        return v if np.isfinite(v) else None


    def _reset_selection_and_clear_graphs(self):
        """상단/좌하 그래프 및 선택 상태 초기화."""
        self._spec_selected_index = None
        self._last_selected_pixel_spec = None
        self._last_selected_yx = None
        self._spec_coords_cache = []
        self._clear_widget(self.spectrumWidget)
        self._clear_widget(self.cmpSpectrumWidget)
        if hasattr(self, "lblSpecCaption"):
            try: self.lblSpecCaption.setText("선택 픽셀 스펙트럼")
            except Exception: pass
        if hasattr(self, "lblCmpCaption"):
            try: self.lblCmpCaption.setText("비교 스펙트럼")
            except Exception: pass

    def _is_distance_map(self) -> bool:
        name = getattr(self, "_current_classmap_name", None)
        if not name: return True
        if callable(self._get_map_is_distance):
            return bool(self._get_map_is_distance(name))
        return True  # 기본값: 거리형

    def _rebuild_class_options_from_map(self, map_name: Optional[str]) -> None:
        """
        선택한 classmap에 실제로 존재하는 cid(>=0)만 cmbClassFilter에 채운다.
        """
        if (map_name is None) or not hasattr(self, "cmbClassFilter"):
            return

        cm = self._get_current_classmap()
        if cm is None or cm.ndim != 2:
            # 맵이 없으면 전체 옵션 유지 (혹은 비우고 반환해도 됨)
            return

        present = [int(v) for v in np.unique(cm).tolist() if int(v) >= 0]  # UNKNOWN(<0) 제외
        # id->name 맵 (기보유 옵션 + 주입된 _glob_id2name 둘 다 사용)
        name_map: Dict[int, str] = {}
        try:
            for cid, nm in (self._class_options or []):
                name_map[int(cid)] = str(nm)
            for k, v in (self._glob_id2name or {}).items():
                name_map.setdefault(int(k), str(v))
        except Exception:
            pass

        # 현재 선택 cid 백업
        prev_cid = self._current_cid()

        # 콤보 갱신
        self.cmbClassFilter.blockSignals(True)
        self.cmbClassFilter.clear()
        for cid in sorted(present):
            nm = name_map.get(cid, str(cid))
            self.cmbClassFilter.addItem(nm, cid)  # 클래스 이름만 표시, 내부 데이터는 CID
        self.cmbClassFilter.blockSignals(False)

        # 이전 선택 유지 시도 → 없으면 첫 항목
        if self.cmbClassFilter.count() > 0:
            if (prev_cid is not None) and (prev_cid in present):
                # prev_cid가 존재하면 그 인덱스로 이동
                for i in range(self.cmbClassFilter.count()):
                    if int(self.cmbClassFilter.itemData(i)) == int(prev_cid):
                        self.cmbClassFilter.setCurrentIndex(i)
                        break
            else:
                self.cmbClassFilter.setCurrentIndex(0)

        # 선택 반영 후 히스토그램/스펙 갱신
        QtWidgets.QApplication.processEvents()
        self._refresh_histogram_only()
        self._refresh_spectrum_only()

        
    def _refresh_spectrum_only(self):
        """
        우상 스펙트럼 번들만 갱신한다.
        - set_spectrum_samples_callback 이 (spec,(y,x)) 또는 spec 단독을 줄 수 있음.
        - 좌표는 전달된 경우에만 1:1 매핑하여 사용(추정 금지).
        """
        map_name = self._current_classmap_name
        cid = self._current_cid()

        # 가드: 선택된 맵/클래스 없거나 콜백 미설정
        if (map_name is None) or (cid is None) or (self._get_spectrum_samples_callback is None):
            self._clear_widget(self.spectrumWidget)
            self._spec_coords_cache = []
            return

        # 개수 결정
        try:
            n = int(self.spnMaxSamples.value() if hasattr(self, "spnMaxSamples") else 50)
        except Exception:
            n = 50

        # 콜백 호출
        try:
            items = self._get_spectrum_samples_callback(map_name, int(cid), n) or []
        except Exception:
            logging.exception("[ClassmapLabeling] spectrum samples callback failed")
            items = []

        # 입력 정규화
        spec_list: List[np.ndarray] = []
        coord_list: List[Optional[Tuple[int, int]]] = []

        for it in items:
            try:
                # (spec, (y,x)) 케이스
                if isinstance(it, (list, tuple)) and len(it) == 2 and isinstance(it[1], (list, tuple)):
                    spec, yx = it
                    arr = np.asarray(spec, dtype=float).ravel()
                    if arr.size > 1:
                        spec_list.append(arr)
                        try:
                            y, x = int(yx[0]), int(yx[1])
                            coord_list.append((y, x))
                        except Exception:
                            coord_list.append(None)
                    continue

                # spec 단독 케이스
                arr = np.asarray(it, dtype=float).ravel()
                if arr.size > 1:
                    spec_list.append(arr)
                    coord_list.append(None)

            except Exception:
                logging.exception("[ClassmapLabeling] normalize spectrum item failed")
                continue

        # 렌더: 좌표 추정 금지 → 콜백이 준 coord_list만 사용
        self._draw_spectrum_bundle_into(self.spectrumWidget, spec_list, coord_list)

        # 내부 선택 상태 초기화(사용자가 곡선을 클릭해서 선택하도록)
        self._spec_selected_index = None
        self._last_selected_pixel_spec = None
        self._last_selected_yx = None

        # 좌표 캐시 동기화(명시 유지)
        self._spec_coords_cache = coord_list
