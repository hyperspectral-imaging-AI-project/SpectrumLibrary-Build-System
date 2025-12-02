# views/dialogs/endmember_detail_dialog.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import logging
import os

from PyQt5 import uic, QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
import numpy as np

# LabelingCandidateReviewDialog 에서 사용 중인 유틸 가져오기
from views.dialogs.labeling_candidate_review import (
    SAM, SID, SCC,                 # get_metric 기반 SAM/SID/SCC distance
    _build_class_lib,
    _build_labeling_lib,
    _merge_lib_dicts,
    _as_dict_lib,
    VLMResultWindow,
    VLMStreamWorker,
)

from core.vlm_generation import load_prompt  # openai_inference 는 VLMStreamWorker 내부에서 사용


try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("[EndmemberDetailDialog] matplotlib not available, plot disabled")


class SpectrumWidget(FigureCanvasQTAgg):
    """matplotlib 기반 스펙트럼 그래프 위젯 (엔드멤버 + 후보 스펙트럼 여러 개 표시 지원)"""

    def __init__(self, parent=None):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for SpectrumWidget")
        self._figure = Figure(figsize=(6, 4))
        super().__init__(self._figure)
        self.setParent(parent)
        self._ax = self._figure.add_subplot(111)
        self._spectrum: Optional[np.ndarray] = None          # 기준 엔드멤버
        self._wavelengths: Optional[np.ndarray] = None
        self._title: str = "Spectrum"

        # 추가: 겹쳐 그릴 후보 스펙트럼들
        self._overlays: List[np.ndarray] = []
        self._overlay_labels: List[str] = []

    def set_spectrum(
        self,
        spectrum: np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
        title: str = "Spectrum",
    ):
        """기준 스펙트럼(엔드멤버) 설정"""
        self._spectrum = np.asarray(spectrum).ravel()
        self._wavelengths = wavelengths
        self._title = title
        self._update_plot()

    def set_overlays(self, spectra: List[np.ndarray], labels: Optional[List[str]] = None):
        """
        후보 스펙트럼 리스트 설정 (엔드멤버 위에 겹쳐 그림)
        spectra: [(C,), ...]
        """
        self._overlays = []
        for s in spectra or []:
            if s is None:
                continue
            arr = np.asarray(s).ravel()
            if arr.size > 0:
                self._overlays.append(arr)
        self._overlay_labels = labels or []
        self._update_plot()

    def _update_plot(self):
        """그래프 업데이트"""
        self._ax.clear()

        # X축 준비
        if self._spectrum is None or len(self._spectrum) == 0:
            self._ax.text(
                0.5,
                0.5,
                "스펙트럼 데이터 없음",
                ha="center",
                va="center",
                transform=self._ax.transAxes,
                fontsize=12,
            )
            self._ax.set_xlabel("Wavelength (nm)")
            self._ax.set_ylabel("Reflectance")
            self.draw()
            return

        if self._wavelengths is not None and len(self._wavelengths) == len(self._spectrum):
            x_data = self._wavelengths
            x_label = "파장 (nm)"
        else:
            x_data = np.arange(len(self._spectrum), dtype=float)
            x_label = "밴드 인덱스"

        # 1) 기준 엔드멤버 (두꺼운 검은색)
        self._ax.plot(
            x_data,
            self._spectrum,
            color="black",
            linewidth=2.0,
            alpha=0.95,
            label="Endmember",
        )

        # 2) 선택된 후보 스펙트럼들 (색상 순환)
        color_cycle = [
            (0.9, 0.4, 0.2),
            (0.2, 0.7, 0.2),
            (0.8, 0.7, 0.1),
            (0.2, 0.6, 0.9),
            (0.7, 0.3, 0.7),
        ]
        for i, spec in enumerate(self._overlays):
            if spec.shape[0] != self._spectrum.shape[0]:
                continue
            c = color_cycle[i % len(color_cycle)]
            label = self._overlay_labels[i] if i < len(self._overlay_labels) else f"cand{i+1}"
            self._ax.plot(x_data, spec, color=c, linewidth=1.5, alpha=0.8, label=label)

        self._ax.set_xlabel(x_label)
        self._ax.set_ylabel("반사율")
        self._ax.set_title(self._title)
        self._ax.grid(True, alpha=0.3, linestyle="--")
        if self._overlays:
            self._ax.legend(loc="best", fontsize=8)
        self._figure.tight_layout()
        self.draw()


class EndmemberDetailDialog(QtWidgets.QDialog):
    """
    엔드멤버 상세 분석 다이얼로그 (UI: endmember_detail_dialog.ui)

    기능:
    - MainWindow 의 splib_raw/splib_cr/label_raw/label_cr 를 기준으로
      SAM/SID/SCC 유사도 Top-18 (raw/cr 각각 3개 × 3 metric = 18) 계산
    - # 컬럼의 체크박스로 선택된 후보 스펙트럼을 오른쪽 그래프에 겹쳐서 표시
    - VLM 버튼: Top-18 결과를 요약해서 VLM 스트리밍 분석 호출
    """

    def __init__(self, parent=None, ui_dir: Optional[Path] = None):
        super().__init__(parent)
        self.setWindowTitle("엔드멤버 상세분석")

        app_dir = Path(getattr(parent, "app_dir", Path(__file__).resolve().parents[3]))
        ui_dir = Path(ui_dir) if ui_dir else (app_dir / "ui")
        ui_path = ui_dir / "endmember_detail_dialog.ui"

        if not ui_path.exists():
            raise FileNotFoundError(f"UI file not found: {ui_path}")

        uic.loadUi(str(ui_path), self)

        # 위젯 핸들 찾기
        self.tableResults: QtWidgets.QTableWidget = self.findChild(QtWidgets.QTableWidget, "tableResults")
        self.framePlot: QtWidgets.QFrame = self.findChild(QtWidgets.QFrame, "framePlot")
        self.pushButtonVLM: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "pushButton")
        self.pushButtonClose: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "pushButtonClose")

        # 내부 상태
        self._mw = parent
        self._cube: Optional[np.ndarray] = getattr(parent, "cfg", {}).get("data") if parent is not None else None
        self._wavelengths_global: Optional[np.ndarray] = None
        if parent is not None:
            cfg = getattr(parent, "cfg", {}) or {}
            self._wavelengths_global = cfg.get("wavelength") or cfg.get("wavelength_list")

        # MainWindow 에서 라이브러리/라벨링 dict 가져오기
        self._spec_lib_raw = getattr(parent, "splib_raw", None) if parent is not None else None
        self._spec_lib_cr = getattr(parent, "splib_cr", None) if parent is not None else None
        self._label_raw = getattr(parent, "label_raw", None) if parent is not None else None
        self._label_cr = getattr(parent, "label_cr", None) if parent is not None else None

        # id → 재료명 메타
        raw_name_map = getattr(parent, "_last_id_to_name", {}) if parent is not None else {}
        self._id2name = {int(k): str(v) for k, v in raw_name_map.items()} if isinstance(raw_name_map, dict) else {}
        self._mtrl_meta: Dict[int, Dict[str, Any]] = getattr(parent, "_mtrl_meta", {}) if parent is not None else {}

        self._endmember_index: Optional[int] = None
        self._endmember_spectrum: Optional[np.ndarray] = None
        self._wavelengths: Optional[np.ndarray] = None

        # 라이브러리 캐시 (VLM/그래프에서 재사용)
        self._class_lib_raw: Dict[int, np.ndarray] = {}
        self._class_lib_cr: Dict[int, np.ndarray] = {}
        self._label_lib_raw: Dict[int, np.ndarray] = {}
        self._label_lib_cr: Dict[int, np.ndarray] = {}

        # Top-18 결과 캐시 (VLM/선택 그래프에서 재사용)
        self._top_rows_cache: List[Tuple[int, str, float, str, str]] = []
        self._suppress_item_changed: bool = False

        # 테이블 초기화
        if self.tableResults:
            self.tableResults.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.tableResults.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            self.tableResults.setColumnCount(6)
            self.tableResults.setHorizontalHeaderLabels(
                ["#", "Class", "Metric", "value", "Material Name", "Description"]
            )
            # 체크박스 변경 → 그래프 업데이트 연결
            self.tableResults.itemChanged.connect(self._on_item_changed)
            self.tableResults.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.tableResults.verticalHeader().setVisible(False)
            self.tableResults.horizontalHeader().setStretchLastSection(True)

        # 스펙트럼 플롯 위젯 설치
        self.spectrumWidget: Optional[SpectrumWidget] = None
        if self.framePlot and MATPLOTLIB_AVAILABLE:
            try:
                self.spectrumWidget = SpectrumWidget(self)
                layout = self.framePlot.layout()
                if layout is None:
                    layout = QtWidgets.QVBoxLayout(self.framePlot)
                    layout.setContentsMargins(4, 4, 4, 4)
                else:
                    # 기존 placeholder 라벨 제거
                    for i in reversed(range(layout.count())):
                        item = layout.itemAt(i)
                        if item.widget():
                            item.widget().setParent(None)
                layout.addWidget(self.spectrumWidget)
                self.spectrumWidget.setMinimumSize(280, 230)
            except Exception as e:
                logging.exception(f"[EndmemberDetailDialog] Failed to install spectrum widget: {e}")
                self.spectrumWidget = None

        # 버튼 연결
        if self.pushButtonClose:
            self.pushButtonClose.clicked.connect(self.accept)

        if self.pushButtonVLM:
            self.pushButtonVLM.clicked.connect(self._on_vlm_clicked)

    # ------------------------------------------------------------------
    # 외부에서 호출: 엔드멤버 스펙트럼 설정 + Top-18 계산/표시
    # ------------------------------------------------------------------
    def set_endmember_data(
        self,
        index: int,
        spectrum: np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
    ):
        """
        엔드멤버 데이터 설정 + SAM/SID/SCC Top-18 계산 및 테이블 채우기.
        """
        self._endmember_index = index
        self._endmember_spectrum = np.asarray(spectrum).ravel()
        # 파장이 명시되면 사용, 없으면 MainWindow 의 파장 사용
        if wavelengths is not None:
            self._wavelengths = np.asarray(wavelengths).ravel()
        else:
            self._wavelengths = (
                np.asarray(self._wavelengths_global).ravel()
                if self._wavelengths_global is not None
                else None
            )

        # 1) Top-18 결과 계산
        results = self._compute_top18(self._endmember_spectrum)
        self._top_rows_cache = [
            (r["class"], r["metric"], r["value"], r["material_name"], r["description"]) for r in results
        ]

        # 2) 테이블 채우기 (# 컬럼은 체크박스)
        if self.tableResults:
            self._suppress_item_changed = True
            try:
                self.tableResults.setRowCount(len(results))
                for i, row in enumerate(results, start=1):
                    # # 컬럼: 체크박스 + 랭크 숫자
                    item_rank = QtWidgets.QTableWidgetItem(str(i))
                    item_rank.setFlags(
                        Qt.ItemIsEnabled
                        | Qt.ItemIsUserCheckable
                        | Qt.ItemIsSelectable
                    )
                    item_rank.setCheckState(Qt.Unchecked)
                    self.tableResults.setItem(i - 1, 0, item_rank)

                    # Class
                    item_cls = QtWidgets.QTableWidgetItem(str(row["class"]))
                    self.tableResults.setItem(i - 1, 1, item_cls)

                    # Metric
                    item_metric = QtWidgets.QTableWidgetItem(str(row["metric"]))
                    self.tableResults.setItem(i - 1, 2, item_metric)

                    # value
                    item_val = QtWidgets.QTableWidgetItem(f"{row['value']:.6f}")
                    item_val.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.tableResults.setItem(i - 1, 3, item_val)

                    # Material Name
                    item_name = QtWidgets.QTableWidgetItem(str(row["material_name"]))
                    self.tableResults.setItem(i - 1, 4, item_name)

                    # Description
                    item_desc = QtWidgets.QTableWidgetItem(str(row["description"]))
                    self.tableResults.setItem(i - 1, 5, item_desc)
                self.tableResults.resizeColumnsToContents()
            finally:
                self._suppress_item_changed = False

        # 3) 기본 그래프: 엔드멤버만 표시
        if self.spectrumWidget:
            title = f"Endmember {index + 1} Spectrum"
            self.spectrumWidget.set_spectrum(self._endmember_spectrum, self._wavelengths, title=title)
            self.spectrumWidget.set_overlays([], [])

    # ------------------------------------------------------------------
    # Top-18 계산 (LabelingCandidateReviewDialog._fill_top_table 로직 기반)
    # ------------------------------------------------------------------
    def _compute_top18(self, target: np.ndarray, top_each: int = 3) -> List[Dict[str, Any]]:
        """
        요구사항: raw에서 SAM/SID/SCC 각 3개, cr에서 SAM/SID/SCC 각 3개 ⇒ 총 18개
        - Metric 컬럼에 'SAM (raw)' 형태로 표기
        - 값은 '작을수록 유사' 기준(거리형) 오름차순
        """
        rows: List[Dict[str, Any]] = []

        if target is None or target.size == 0:
            return rows

        if self._cube is not None:
            expect_c = self._cube.shape[2]
        else:
            expect_c = target.shape[0]

        # 1) 라이브러리(raw/cr) + 라벨링(raw/cr) 빌드/병합
        class_lib_raw = _build_class_lib(self._spec_lib_raw, None, expect_c)
        class_lib_cr = _build_class_lib(None, self._spec_lib_cr, expect_c)
        label_lib_raw = _build_labeling_lib(self._label_raw, None, expect_c)
        label_lib_cr = _build_labeling_lib(None, self._label_cr, expect_c)

        # 라벨링 포함해서 raw/cr 별로 병합
        class_lib_raw = _merge_lib_dicts(class_lib_raw, label_lib_raw, expect_c=expect_c)
        class_lib_cr = _merge_lib_dicts(class_lib_cr, label_lib_cr, expect_c=expect_c)

        # 이후 그래프/VLM에서도 재사용할 수 있도록 보관
        self._class_lib_raw = class_lib_raw
        self._class_lib_cr = class_lib_cr
        self._label_lib_raw = label_lib_raw
        self._label_lib_cr = label_lib_cr

        def _min_dists_for_lib(L: Dict[int, np.ndarray]) -> Dict[int, Tuple[float, float, float]]:
            """cid -> (sam_min, sid_min, scc_min)"""
            out: Dict[int, Tuple[float, float, float]] = {}
            L = _as_dict_lib(L)
            if not isinstance(L, dict) or len(L) == 0:
                return out

            tvec = target.astype(float, copy=False)

            for cid, refs in L.items():
                A = np.asarray(refs, dtype=float)
                if A.ndim == 1:
                    A = A[None, :]
                if A.ndim != 2 or A.shape[1] != tvec.shape[0]:
                    continue

                # SAM/SID/SCC 거리 (core.metrics 기반)
                sam_vals = SAM(A, tvec)
                sid_vals = SID(A, tvec)
                scc_vals = SCC(A, tvec)

                out[int(cid)] = (
                    float(np.min(sam_vals)),
                    float(np.min(sid_vals)),
                    float(np.min(scc_vals)),
                )
            return out

        d_raw = _min_dists_for_lib(class_lib_raw)
        d_cr = _min_dists_for_lib(class_lib_cr)

        # 2) 재료 메타 보강
        meta_map = self._mtrl_meta or {}
        try:
            all_cids = set(d_raw.keys()) | set(d_cr.keys())
            missing = [cid for cid in all_cids if cid not in meta_map]
            if missing:
                from data.db import search_material_filtering_list

                base_url = os.getenv("material_filtering_url")
                if base_url and missing:
                    recs = search_material_filtering_list(base_url=base_url, mtrl_ids=list(missing))
                    for rec in (recs or []):
                        try:
                            cid2 = int(rec.get("mtrl_cd"))
                        except Exception:
                            continue
                        meta_map[cid2] = {
                            "name": rec.get("mtrl_nm", str(cid2)),
                            "desc": rec.get("desc", rec.get("dsc", "")),
                        }
                    if self._mw is not None:
                        self._mw._mtrl_meta = meta_map
        except Exception:
            logging.exception("[EndmemberDetail] meta fetch failed")

        self._mtrl_meta = meta_map

        def _meta(cid_: int) -> Tuple[str, str]:
            m = meta_map.get(cid_) or {}
            name = self._id2name.get(cid_, m.get("name", str(cid_)))
            return name, m.get("desc", "")

        # 3) raw/cr × SAM/SID/SCC 별 top-3 선별
        def _pick_top(source: str, dmap: Dict[int, Tuple[float, float, float]]):
            metric_names = ["SAM", "SID", "SCC"]
            for mi, mname in enumerate(metric_names):
                cand = [(cid, vals[mi]) for cid, vals in dmap.items()]
                cand.sort(key=lambda x: x[1])
                for cid, val in cand[:top_each]:
                    name, desc = _meta(cid)
                    rows.append(
                        {
                            "class": int(cid),
                            "metric": f"{mname} ({source})",
                            "value": float(val),
                            "material_name": name,
                            "description": desc,
                        }
                    )

        _pick_top("raw", d_raw)
        _pick_top("cr", d_cr)

        return rows

    # ------------------------------------------------------------------
    # 테이블 체크박스 → 그래프 업데이트
    # ------------------------------------------------------------------
    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem):
        if self._suppress_item_changed:
            return
        if item.column() != 0:
            return
        # 체크 상태 변하면 그래프 갱신
        self._update_plot_from_selection()

    def _update_plot_from_selection(self):
        if self.spectrumWidget is None or self._endmember_spectrum is None:
            return

        # 1) 기준 엔드멤버 스펙트럼 세팅
        base_title = (
            f"Endmember {self._endmember_index + 1} Spectrum"
            if self._endmember_index is not None
            else "Endmember Spectrum"
        )
        self.spectrumWidget.set_spectrum(
            self._endmember_spectrum,
            self._wavelengths,
            title=base_title,
        )

        if not self.tableResults:
            return

        selected_specs: List[np.ndarray] = []
        labels: List[str] = []

        row_count = self.tableResults.rowCount()
        expect_c = self._endmember_spectrum.shape[0]

        # raw / cr 라이브러리 미리 병합 (클래스 + 라벨링)
        lib_raw_all = _merge_lib_dicts(
            self._class_lib_raw,
            self._label_lib_raw,
            expect_c=expect_c,
        )
        lib_cr_all = _merge_lib_dicts(
            self._class_lib_cr,
            self._label_lib_cr,
            expect_c=expect_c,
        )

        for r in range(row_count):
            item_rank = self.tableResults.item(r, 0)
            if item_rank is None or item_rank.checkState() != Qt.Checked:
                continue

            item_cid = self.tableResults.item(r, 1)
            item_metric = self.tableResults.item(r, 2)
            if item_cid is None:
                continue

            try:
                cid = int(item_cid.text())
            except Exception:
                continue

            metric_src = item_metric.text() if item_metric is not None else ""
            metric_src_lower = metric_src.lower()

            # metric에 따라 raw / cr 중 어떤 라이브러리를 쓸지 결정
            if "cr" in metric_src_lower:
                lib = lib_cr_all   # CR 기반 (continuum removal)
                src_tag = "cr"
            else:
                lib = lib_raw_all  # RAW 기반
                src_tag = "raw"

            refs = lib.get(cid)
            if refs is None:
                continue

            refs = np.asarray(refs, dtype=np.float32)
            if refs.ndim == 1:
                refs = refs[None, :]
            if refs.ndim != 2 or refs.shape[1] != expect_c:
                continue

            # 간단히 평균 스펙트럼을 대표로 사용
            rep = refs.mean(axis=0)
            selected_specs.append(rep)

            # CID → 물질명으로 변환
            mat_name = self._name_for_cid(cid)
            # 범례에 raw/cr 정보까지 같이 보여주고 싶으면:
            labels.append(f"{mat_name} ({src_tag})")
            # 만약 raw/cr 안 보이게 하고 싶으면 그냥:
            # labels.append(mat_name)

        self.spectrumWidget.set_overlays(selected_specs, labels)

    # ------------------------------------------------------------------
    # VLM 분석 (LabelingCandidateReviewDialog._on_detail_analysis 기반 간소 버전)
    # ------------------------------------------------------------------
    def _on_vlm_clicked(self):
        """VLM 분석 버튼 클릭 핸들러"""
        if not self._top_rows_cache:
            QtWidgets.QMessageBox.information(self, "안내", "분석 데이터가 없습니다. 먼저 엔드멤버를 설정하세요.")
            return

        if self._endmember_spectrum is None:
            QtWidgets.QMessageBox.information(self, "안내", "엔드멤버 스펙트럼 정보가 없습니다.")
            return

        # ---------- 1) SAM / SID / SCC Top-3 설명 문자열 ----------
        def _top_by_metric(metric_tag: str, top_k: int = 3):
            rows = [
                (cid, metric_src, val, name, desc)
                for (cid, metric_src, val, name, desc) in self._top_rows_cache
                if metric_tag in metric_src
            ]
            rows.sort(key=lambda r: r[2])
            return rows[:top_k]

        def _desc_lines_for_metric(metric_tag: str):
            rows = _top_by_metric(metric_tag, top_k=3)
            descs: List[str] = []
            for cid, metric_src, val, name, desc in rows:
                mtrl_nm = name or str(cid)
                mtrl_desc = desc or ""
                safe_nm = str(mtrl_nm).replace('"', '\\"')
                safe_desc = str(mtrl_desc).replace('"', '\\"').replace("\n", " ")
                json_str = f'{{"mtrl_nm":"{safe_nm}","mtrl_description":"{safe_desc}"}}'
                descs.append(json_str)
            while len(descs) < 3:
                descs.append('{"mtrl_nm":"(no candidates)","mtrl_description":""}')
            return descs

        sam_top1, sam_top2, sam_top3 = _desc_lines_for_metric("SAM")
        sid_top1, sid_top2, sid_top3 = _desc_lines_for_metric("SID")
        scc_top1, scc_top2, scc_top3 = _desc_lines_for_metric("SCC")

        # ---------- 2) CSV: 엔드멤버 + 각 metric별 top-3 대표 스펙트럼 ----------
        csv_text = ""
        try:
            # 파장
            n_band = self._endmember_spectrum.shape[0]
            wave = None
            if self._wavelengths is not None and self._wavelengths.shape[0] == n_band:
                wave = np.asarray(self._wavelengths, dtype=float)
            else:
                wave = np.arange(n_band, dtype=float)

            target_vec = self._endmember_spectrum.astype(np.float32, copy=False)
            expect_c = n_band

            class_lib_raw = _merge_lib_dicts(
                _build_class_lib(self._spec_lib_raw, None, expect_c),
                _build_labeling_lib(self._label_raw, None, expect_c),
                expect_c=expect_c,
            )
            class_lib_cr = _merge_lib_dicts(
                _build_class_lib(None, self._spec_lib_cr, expect_c),
                _build_labeling_lib(None, self._label_cr, expect_c),
                expect_c=expect_c,
            )

            metric_fn_map = {"SAM": SAM, "SID": SID, "SCC": SCC}

            def _best_ref_for_row(row_info):
                cid, metric_src, _, _, _ = row_info
                cid = int(cid)
                try:
                    if "SAM" in metric_src:
                        metric_name = "SAM"
                    elif "SID" in metric_src:
                        metric_name = "SID"
                    elif "SCC" in metric_src:
                        metric_name = "SCC"
                    else:
                        return None
                    metric_fn = metric_fn_map.get(metric_name)
                    if metric_fn is None:
                        return None
                    source_tag = "raw"
                    if "cr" in metric_src.lower():
                        source_tag = "cr"
                    lib = class_lib_raw if source_tag == "raw" else class_lib_cr
                    refs = lib.get(cid, None)
                    if refs is None:
                        return None
                    refs = np.asarray(refs, dtype=np.float32)
                    if refs.ndim == 1:
                        refs = refs[None, :]
                    if refs.ndim != 2 or refs.shape[1] != target_vec.shape[0]:
                        return None
                    try:
                        dists = metric_fn(refs, target_vec)
                    except Exception:
                        dists = np.asarray([metric_fn(refs[i, :], target_vec) for i in range(refs.shape[0])])
                    if dists.size == 0:
                        return None
                    best_idx = int(np.argmin(dists))
                    return refs[best_idx, :].astype(np.float32, copy=False)
                except Exception:
                    logging.exception("[Endmember VLM] _best_ref_for_row failed")
                    return None

            def _best_specs_for_metric(metric_tag: str):
                rows = _top_by_metric(metric_tag, top_k=3)
                specs = []
                for row_info in rows:
                    specs.append(_best_ref_for_row(row_info))
                while len(specs) < 3:
                    specs.append(None)
                return specs

            sam_specs = _best_specs_for_metric("SAM")
            sid_specs = _best_specs_for_metric("SID")
            scc_specs = _best_specs_for_metric("SCC")

            spectra_rows: List[Tuple[str, Optional[np.ndarray]]] = []
            spectra_rows.append(("endmember", target_vec))
            for i, spec in enumerate(sam_specs, start=1):
                spectra_rows.append((f"SAM_top{i}", spec))
            for i, spec in enumerate(sid_specs, start=1):
                spectra_rows.append((f"SID_top{i}", spec))
            for i, spec in enumerate(scc_specs, start=1):
                spectra_rows.append((f"SCC_top{i}", spec))

            header_cols = ["spectrum_id"] + [f"{float(l):.6f}" for l in wave]
            lines = [",".join(header_cols)]
            for name, spec in spectra_rows:
                row_vals = [name]
                if spec is None or spec.shape[0] != n_band:
                    row_vals += ["" for _ in range(n_band)]
                else:
                    row_vals += [f"{float(v):.6f}" for v in spec]
                lines.append(",".join(row_vals))
            csv_text = "\n".join(lines)
        except Exception:
            logging.exception("[Endmember VLM] CSV 생성 실패")
            csv_text = ""

        # ---------- 3) 프롬프트 템플릿 로드 + 치환 ----------
        try:
            # TODO: 경로는 프로젝트 구조에 맞게 수정
            text_prompt = load_prompt(
                prompt_path=r"C:\Users\pde15\Desktop\hsi_crop_preprcoessing\pyqt_code\third_pixel_classification_tool\prompt\vlm_prompt.md"
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "VLM", f"프롬프트 템플릿 로드 실패: {e}")
            return

        def _safe_replace(s: str, key: str, val: str) -> str:
            tag = "{" + key + "}"
            return s.replace(tag, val if val else "(no candidates)")

        prompt_text = text_prompt
        prompt_text = _safe_replace(prompt_text, "csv", csv_text)
        prompt_text = _safe_replace(prompt_text, "sam_top1", sam_top1)
        prompt_text = _safe_replace(prompt_text, "sam_top2", sam_top2)
        prompt_text = _safe_replace(prompt_text, "sam_top3", sam_top3)
        prompt_text = _safe_replace(prompt_text, "sid_top1", sid_top1)
        prompt_text = _safe_replace(prompt_text, "sid_top2", sid_top2)
        prompt_text = _safe_replace(prompt_text, "sid_top3", sid_top3)
        prompt_text = _safe_replace(prompt_text, "scc_top1", scc_top1)
        prompt_text = _safe_replace(prompt_text, "scc_top2", scc_top2)
        prompt_text = _safe_replace(prompt_text, "scc_top3", scc_top3)

        QtWidgets.QApplication.clipboard().setText(prompt_text)

        # 여기서는 엔드멤버 자체라 별도의 RGB 패치 이미지는 없다고 보고, img_url은 빈 문자열로 둠
        img_url = ""

        # ---------- 4) VLM 스트리밍 창 + 워커 실행 ----------
        self._vlm_text_win = VLMResultWindow("VLM 스트리밍 결과", parent=self)
        self._vlm_text_win.set_text("[VLM 분석 시작]\n결과가 순차적으로 표시됩니다.\n\n")
        self._vlm_text_win.show()
        self._vlm_text_win.raise_()
        self._vlm_text_win.activateWindow()

        self._vlm_thread = QtCore.QThread(self)
        self._vlm_worker = VLMStreamWorker(prompt_text, img_url)
        self._vlm_worker.moveToThread(self._vlm_thread)

        self._vlm_thread.started.connect(self._vlm_worker.run)

        def _on_chunk(chunk: str):
            if hasattr(self._vlm_text_win, "append_text"):
                self._vlm_text_win.append_text(chunk)
            else:
                cur = self._vlm_text_win._edit.toPlainText()
                self._vlm_text_win.set_text(cur + chunk)

        def _on_finished():
            self._vlm_text_win.append_text("\n\n[VLM 분석 완료]")
            self._vlm_text_win.setWindowTitle("VLM 스트리밍 결과 (완료)")
            self._vlm_thread.quit()
            self._vlm_thread.wait()

        def _on_failed(msg: str):
            self._vlm_text_win.append_text(f"\n\n{msg}")
            self._vlm_thread.quit()
            self._vlm_thread.wait()
            QtWidgets.QMessageBox.warning(self, "VLM", msg)

        self._vlm_worker.chunk_received.connect(_on_chunk)
        self._vlm_worker.finished.connect(_on_finished)
        self._vlm_worker.failed.connect(_on_failed)
        self._vlm_thread.finished.connect(self._vlm_worker.deleteLater)

        self._vlm_thread.start()

    def _name_for_cid(self, cid: int) -> str:
        """CID → 물질명 문자열로 변환 (없으면 CID 그대로)"""
        try:
            cid_i = int(cid)
        except Exception:
            return str(cid)

        # 1) MainWindow에서 넘어온 _last_id_to_name 기반
        name = self._id2name.get(cid_i)
        if name:
            return str(name)

        # 2) _mtrl_meta에 저장된 name 사용
        meta = (self._mtrl_meta or {}).get(cid_i) or {}
        name2 = meta.get("name") or meta.get("mtrl_nm")
        if name2:
            return str(name2)

        # 3) 전부 없으면 CID 문자열로
        return str(cid_i)
