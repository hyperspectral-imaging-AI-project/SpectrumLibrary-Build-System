# views/dialogs/labeling_candidate_review.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
import weakref
from typing import List, Dict, Tuple, Union
import numpy as np
from itertools import chain
from openai import OpenAI
from services.spec_library import build_library_from_spec_libs
from matplotlib import font_manager, rcParams
from dotenv import load_dotenv
from core.vlm_generation import load_prompt, openai_inference

# 한글 폰트 후보(윈도우/맥/리눅스 공통)
KOREAN_FONT_CANDIDATES = [
    "Malgun Gothic", "맑은 고딕",          # Windows
    "Apple SD Gothic Neo", "AppleGothic", # macOS
    "NanumGothic", "NanumGothicOTF", "Noto Sans CJK KR"  # Linux/common
]

load_dotenv()

class VLMResultWindow(QtWidgets.QDialog):
    def __init__(self, title: str = "VLM 결과", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 600)

        self._edit = QtWidgets.QTextEdit(self)
        self._edit.setReadOnly(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._edit)

    def set_text(self, text: str):
        self._edit.setPlainText(text)
        self._edit.moveCursor(QtGui.QTextCursor.End)

    def append_text(self, text: str):
        self._edit.moveCursor(QtGui.QTextCursor.End)
        self._edit.insertPlainText(text)
        self._edit.moveCursor(QtGui.QTextCursor.End)

        
class VLMStreamWorker(QtCore.QObject):
    chunk_received = pyqtSignal(str)   # 스트리밍 chunk
    finished = pyqtSignal()            # 전체 완료
    failed = pyqtSignal(str)           # 에러 메시지

    def __init__(self, prompt_text: str, img_url: str, parent=None):
        super().__init__(parent)
        self._prompt_text = prompt_text
        self._img_url = img_url

    @QtCore.pyqtSlot()
    def run(self):
        try:
            # ⚠️ 전제: openai_inference(..., streaming=True) 가
            # chunk(str)을 yield하는 제너레이터 형태
            for chunk in openai_inference(self._prompt_text, self._img_url, streaming=True):
                if isinstance(chunk, str) and chunk:
                    self.chunk_received.emit(chunk)
            self.finished.emit()
        except Exception as e:
            self.failed.emit(f"[VLM 스트리밍 실패] {e}")


def ensure_korean_font():
    """
    matplotlib에 한글 글꼴을 등록/지정한다.
    1) 시스템에 설치된 폰트 중 후보를 우선 사용
    2) 없다면 로컬 번들 폰트(NotoSansCJKkr-Regular.otf/NanumGothic.ttf 등)를 추가 로드
    """
    # 1) 시스템에 있는지 검사
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in KOREAN_FONT_CANDIDATES:
        if name in installed:
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False  # 음수기호 깨짐 방지
            return name

    # 2) 프로젝트에 번들한 폰트가 있으면 수동 등록 (예: ui/fonts/NotoSansCJKkr-Regular.otf)
    for rel in ["ui/fonts/NotoSansCJKkr-Regular.otf",
                "ui/fonts/NanumGothic.ttf",
                "assets/fonts/NotoSansCJKkr-Regular.otf"]:
        p = Path(__file__).resolve().parent.parent.parent / rel
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            fam = font_manager.FontProperties(fname=str(p)).get_name()
            rcParams["font.family"] = fam
            rcParams["axes.unicode_minus"] = False
            return fam

    # 3) 끝까지 실패하면 경고만 남기고 DejaVu 유지
    logging.warning("한글 폰트를 찾지 못했습니다. 시스템에 '맑은 고딕' 또는 'NanumGothic/Noto Sans CJK KR'를 설치하세요.")
    rcParams["axes.unicode_minus"] = False
    return None

# ===== metrics: core.metrics가 있으면 사용, 없으면 간이구현 =====
# def _import_metrics():
#     try:
#         from core.metrics import sam as _sam, sid as _sid, scc_distance as _scc
#         return _sam, _sid, _scc
#     except Exception:
#         pass

    # # fallback
    # def _sam(a: np.ndarray, b: np.ndarray) -> float:
    #     a = a.astype(np.float64, copy=False); b = b.astype(np.float64, copy=False)
    #     na = np.linalg.norm(a); nb = np.linalg.norm(b)
    #     if na == 0 or nb == 0:
    #         return np.pi / 2.0
    #     v = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    #     return float(np.arccos(v))

    # def _sid(a: np.ndarray, b: np.ndarray) -> float:
    #     a = a.astype(np.float64, copy=False); b = b.astype(np.float64, copy=False)
    #     a = np.clip(a, 1e-12, None); b = np.clip(b, 1e-12, None)
    #     a = a / a.sum(); b = b / b.sum()
    #     return float((a * np.log(a / b)).sum() + (b * np.log(b / a)).sum())

    # def _scc(a: np.ndarray, b: np.ndarray) -> float:
    #     a = a.astype(np.float64, copy=False); b = b.astype(np.float64, copy=False)
    #     a = a - a.mean(); b = b - b.mean()
    #     da = np.linalg.norm(a); db = np.linalg.norm(b)
    #     if da == 0 or db == 0:
    #         return 1.0
    #     corr = np.dot(a, b) / (da * db)
    #     return float(1.0 - corr)  # distance
    # return _sam, _sid, _scc

# SAM, SID, SCC = _import_metrics()
from core.metrics import get_metric
SAM = get_metric("sad")
SID = get_metric("sid")
SCC = get_metric("scc")
def _as_dict_lib(lib_like):
    """(lib, id_to_name) / [lib1, lib2] / dict → 항상 dict로 표준화"""
    if isinstance(lib_like, tuple) and lib_like and isinstance(lib_like[0], dict):
        return lib_like[0]
    if isinstance(lib_like, (list, tuple)) and lib_like and all(isinstance(x, dict) for x in lib_like):
        merged = {}
        for d in lib_like:
            merged.update(d)
        return merged
    return lib_like if isinstance(lib_like, dict) else {}

def _build_class_lib(parent_lib_raw, parent_lib_cr, expect_c: int) -> Dict[int, np.ndarray]:
    """
    MainWindow의 splib_raw/splib_cr은 이미 Dict[int, np.ndarray] 형식으로 저장되어 있음.
    리스트 형식인 경우에만 build_library_from_spec_libs를 호출.
    """
    result: Dict[int, np.ndarray] = {}
    
    # parent_lib_raw 처리
    if parent_lib_raw is not None:
        if isinstance(parent_lib_raw, dict):
            # 이미 Dict[int, np.ndarray] 형식
            for cid, arr in parent_lib_raw.items():
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.ndim == 2 and arr.shape[1] == expect_c:
                    if int(cid) in result:
                        result[int(cid)] = np.vstack([result[int(cid)], arr])
                    else:
                        result[int(cid)] = arr
        else:
            # 리스트 형식인 경우 build_library_from_spec_libs 사용
            lib_dict = _as_dict_lib(build_library_from_spec_libs([{"spec_lib": parent_lib_raw}], expect_c))
            for cid, arr in lib_dict.items():
                if int(cid) in result:
                    result[int(cid)] = np.vstack([result[int(cid)], arr])
                else:
                    result[int(cid)] = arr
    
    # parent_lib_cr 처리
    if parent_lib_cr is not None:
        if isinstance(parent_lib_cr, dict):
            # 이미 Dict[int, np.ndarray] 형식
            for cid, arr in parent_lib_cr.items():
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.ndim == 2 and arr.shape[1] == expect_c:
                    if int(cid) in result:
                        result[int(cid)] = np.vstack([result[int(cid)], arr])
                    else:
                        result[int(cid)] = arr
        else:
            # 리스트 형식인 경우 build_library_from_spec_libs 사용
            lib_dict = _as_dict_lib(build_library_from_spec_libs([{"spec_lib": parent_lib_cr}], expect_c))
            for cid, arr in lib_dict.items():
                if int(cid) in result:
                    result[int(cid)] = np.vstack([result[int(cid)], arr])
                else:
                    result[int(cid)] = arr
    
    return result

def _build_labeling_lib(label_raw, label_cr, expect_c: int) -> Dict[int, np.ndarray]:
    """
    MainWindow의 label_raw/label_cr도 이미 Dict[int, np.ndarray] 형식으로 저장되어 있음.
    """
    result: Dict[int, np.ndarray] = {}
    
    # label_raw 처리
    if label_raw is not None:
        if isinstance(label_raw, dict):
            # 이미 Dict[int, np.ndarray] 형식
            for cid, arr in label_raw.items():
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.ndim == 2 and arr.shape[1] == expect_c:
                    if int(cid) in result:
                        result[int(cid)] = np.vstack([result[int(cid)], arr])
                    else:
                        result[int(cid)] = arr
        else:
            # 리스트 형식인 경우 build_library_from_spec_libs 사용
            lib_dict = _as_dict_lib(build_library_from_spec_libs([{"spec_lib": label_raw}], expect_c))
            for cid, arr in lib_dict.items():
                if int(cid) in result:
                    result[int(cid)] = np.vstack([result[int(cid)], arr])
                else:
                    result[int(cid)] = arr
    
    # label_cr 처리
    if label_cr is not None:
        if isinstance(label_cr, dict):
            # 이미 Dict[int, np.ndarray] 형식
            for cid, arr in label_cr.items():
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.ndim == 2 and arr.shape[1] == expect_c:
                    if int(cid) in result:
                        result[int(cid)] = np.vstack([result[int(cid)], arr])
                    else:
                        result[int(cid)] = arr
        else:
            # 리스트 형식인 경우 build_library_from_spec_libs 사용
            lib_dict = _as_dict_lib(build_library_from_spec_libs([{"spec_lib": label_cr}], expect_c))
            for cid, arr in lib_dict.items():
                if int(cid) in result:
                    result[int(cid)] = np.vstack([result[int(cid)], arr])
                else:
                    result[int(cid)] = arr
    
    return result


def _merge_lib_dicts(*libs: Dict[int, np.ndarray], expect_c: Optional[int] = None) -> Dict[int, np.ndarray]:
    """여러 라이브러리 dict를 병합하여 CID별 스펙트럼을 수평 스택."""
    merged: Dict[int, List[np.ndarray]] = {}
    for lib in libs:
        lib_dict = _as_dict_lib(lib)
        if not isinstance(lib_dict, dict):
            continue
        for cid, arr in lib_dict.items():
            if arr is None:
                continue
            a = np.asarray(arr, dtype=float)
            if a.ndim == 1:
                a = a[None, :]
            if a.ndim != 2 or a.size == 0:
                continue
            if expect_c is not None and a.shape[1] != expect_c:
                continue
            merged.setdefault(int(cid), []).append(a)

    out: Dict[int, np.ndarray] = {}
    for cid, blocks in merged.items():
        try:
            out[cid] = np.vstack(blocks).astype(np.float32, copy=False)
        except ValueError:
            # shape mismatch 시 expect_c가 있으면 거기에 맞는 것만 사용
            filtered = [b for b in blocks if expect_c is None or b.shape[1] == expect_c]
            if not filtered:
                continue
            out[cid] = np.vstack(filtered).astype(np.float32, copy=False)
    return out

def get_openai_client() -> Optional[OpenAI]:
    """
    전역적으로 OpenAI 클라이언트 하나만 생성해서 재사용.
    OPENAI_API_KEY 가 없으면 None 반환.
    """
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.warning("OPENAI_API_KEY가 설정되어 있지 않습니다. ChatGPT 연동 불가.")
        return None
    
    _openai_client = OpenAI(api_key=api_key)
    return _openai_client

class LabelingCandidateReviewDialog(QtWidgets.QDialog):
    """
    .ui: LabelingCandidateReviewDialog.ui
    입력: parent=MainWindow (rgb_image, cfg['data'], splib_raw/splib_cr, _last_id_to_name 등 접근)
    기능:
      - 선택 픽셀 스펙트럼 vs 라이브러리(raw/cr) 전부 비교
      - SAM/SID/SCC 모두 계산하여 상위 Top-N 테이블 채움
      - RGB 50x50 패치/스펙트럼(빨간 선) 표시
    """
    
    class_changed = pyqtSignal(int, int)

    
    def __init__(self, parent: QtWidgets.QWidget, ui_dir: Optional[Path] = None):
        super().__init__(parent)
        app_dir = Path(getattr(parent, "app_dir", Path(__file__).resolve().parents[2]))
        ui_dir = Path(ui_dir) if ui_dir else (app_dir / "ui")
        self._root = uic.loadUi(str(ui_dir / "validation_labeling_candidate.ui"), self)

        # 위젯 핸들
        self.tableTopK: QtWidgets.QTableWidget = self.findChild(QtWidgets.QTableWidget, "tableTopK")
        self.lblPreview: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, "viewPreview")
        self.spectrumHost: QtWidgets.QWidget = self.findChild(QtWidgets.QWidget, "spectrumWidget")
        self.btnCancel: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btnCancel")
        self.btnChange: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btnChangeClass")
        self.btnDetail: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "pushButton")
        self.btnCancel.clicked.connect(self.reject)
        self.btnChange.clicked.connect(self._class_change)
        self.btnDetail.clicked.connect(self._on_detail_analysis)
        # matplotlib 임베드
        self._init_matplotlib()

        # 모델 캐시(부모에서 끌어옴)
        self._mw = parent
        self._cube: np.ndarray = getattr(parent, "cfg", {}).get("data")
        self._rgb: Optional[np.ndarray] = getattr(parent, "rgb_image", None)

        # meta
        raw_name_map = getattr(parent, "_last_id_to_name", {}) or {}
        self._id2name = {int(k): str(v) for k, v in raw_name_map.items()} if isinstance(raw_name_map, dict) else {}
        # ▼ 라이브러리/라벨링 원본 보관 (MainWindow에서 이미 만들어 둔 것 사용)
        self._spec_lib_raw = getattr(parent, "splib_raw", None)
        self._spec_lib_cr  = getattr(parent, "splib_cr",  None)
        self._label_raw    = getattr(parent, "label_raw", None)
        self._label_cr     = getattr(parent, "label_cr",  None)
        
        self._top_rows_cache: List[Tuple[int, str, float, str, str]] = []  # (cid, metric, value, name, desc)

        # 현재 타겟 보관
        self._target_spec: Optional[np.ndarray] = None
        self._target_xy: Optional[Tuple[int, int]] = None

        # 테이블 초기 구성
        self._setup_table()
        # ★ 안전 초기화 (중요)
        self._origin_row: Optional[int] = None
        self._origin_dlg: Optional[weakref.ReferenceType] = None
    # ---------- 외부에서 호출 ----------
    def run_with_target(self, y: int, x: int, *,
                        origin_row: Optional[int] = None,
                        origin_dialog: Optional[QtWidgets.QDialog] = None) -> None:
        """선택 픽셀 (y,x)로 대화상자를 채우고 띄운다."""
        # ★ 원본 행/다이얼로그 보관
        self._origin_row = int(origin_row) if isinstance(origin_row, int) else None
        self._origin_dlg = weakref.ref(origin_dialog) if origin_dialog is not None else None

        if self._cube is None:
            QtWidgets.QMessageBox.warning(self, "경고", "HSI 데이터가 없습니다.")
            return
        H, W, C = self._cube.shape
        if not (0 <= y < H and 0 <= x < W):
            QtWidgets.QMessageBox.warning(self, "경고", "좌표가 이미지 범위를 벗어났습니다.")
            return

        self._target_xy = (int(y), int(x))
        self._target_spec = self._cube[y, x, :].astype(float, copy=False)

        self._draw_patch(y, x)
        self._plot_target(self._target_spec)

        # # 라이브러리 빌드(raw + cr)
        # spec_lib_raw = getattr(self._mw, "splib_raw", None)
        # spec_lib_cr  = getattr(self._mw, "splib_cr",  None)
        # spec_dicts = []
        # if spec_lib_raw is not None:
        #     spec_dicts.append({"spec_lib": spec_lib_raw})
        # if spec_lib_cr is not None:
        #     spec_dicts.append({"spec_lib": spec_lib_cr})
            
        # lib = _as_dict_lib(build_library_from_spec_libs(spec_dicts, expect_c=self._cube.shape[2]))
        # if not isinstance(lib, dict) or len(lib) == 0:
        #     QtWidgets.QMessageBox.information(self, "안내", "스펙트럼 라이브러리가 비어있습니다.")
        #     return
        
        # 변경 후: 분류에 실제 사용한 '통합 라이브러리' 우선 사용 + 라벨링 데이터 포함
        mw_lib = getattr(self._mw, "lib", {})
        if isinstance(mw_lib, dict) and mw_lib:
            lib = mw_lib
        else:
            expect_c = self._cube.shape[2]
            packs = []
            for key in ("splib_raw", "splib_cr", "label_raw", "label_cr"):
                data = getattr(self._mw, key, None)
                if data:
                    packs.append({"spec_lib": data})
            lib = _as_dict_lib(build_library_from_spec_libs(packs, expect_c=expect_c))

        # 계산 & 표 채우기
        self._fill_top_table(self._target_spec, lib)
        self._select_first_row()

        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.show()

    # ---------- 내부 구현 ----------
    def _setup_table(self):
        t = self.tableTopK
        t.setColumnCount(6)
        t.setHorizontalHeaderLabels(["#", "Class", "Metric", "value", "Material Name", "Description"])
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        t.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setStretchLastSection(True)
        t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        # ★ 한글 폰트 적용
        db = QtGui.QFontDatabase()
        for fam in KOREAN_FONT_CANDIDATES:
            if fam in db.families():
                f = QtGui.QFont(fam, 9)
                t.setFont(f)
                t.horizontalHeader().setFont(f)
                break
        t.itemSelectionChanged.connect(self._on_row_changed)


    def _init_matplotlib(self):
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            # ★ 한글 폰트 보장
            used = ensure_korean_font()

            self._fig = Figure(figsize=(3.0, 2.2), dpi=100)
            self._ax = self._aac = None  # 안전 초기화
            self._ax = self._fig.add_subplot(111)
            self._ax.set_xlabel("Wavelength")
            self._ax.set_ylabel("Reflectance")

            self._canvas = FigureCanvas(self._fig)
            # 레이아웃 추가
            lay = self.spectrumHost.layout()
            if lay is None:
                lay = QtWidgets.QVBoxLayout()
                lay.setContentsMargins(0, 0, 0, 0)
                lay.setSpacing(0)
                self.spectrumHost.setLayout(lay)
            # 기존 자식 위젯 정리(겹치는 플레이스홀더가 있을 가능성 제거)
            for ch in list(self.spectrumHost.children()):
                if isinstance(ch, QtWidgets.QWidget) and ch is not self._canvas:
                    ch.setParent(None)
            lay.addWidget(self._canvas)
        except Exception:
            self._fig = None
            self._ax = None
            self._canvas = None


    def _plot_target(self,
                    target_spec: np.ndarray,
                    refs_class: Optional[np.ndarray] = None,
                    refs_label: Optional[np.ndarray] = None,
                    title: Optional[str] = None):
        """
        빨간(클래스 라이브러리), 파란(선택 스펙트럼), 초록(라벨링 데이터) 규칙으로 그림.
        """
        if self._ax is None:
            return
        self._ax.clear()

        # 초록: 라벨링 데이터 번들
        if isinstance(refs_label, np.ndarray) and refs_label.ndim == 2 and refs_label.size > 0:
            for i in range(refs_label.shape[0]):
                self._ax.plot(refs_label[i, :], linewidth=1.0, alpha=0.35, color="green", label=None)

        # 빨강: 스펙트럼 라이브러리 번들
        if isinstance(refs_class, np.ndarray) and refs_class.ndim == 2 and refs_class.size > 0:
            for i in range(refs_class.shape[0]):
                self._ax.plot(refs_class[i, :], linewidth=1.0, alpha=0.45, color="red", label=None)

        # 파랑: 타깃(선택) 스펙트럼
        self._ax.plot(target_spec, linewidth=2.0, color="blue", label="selected pixel")

        # 범례(간결하게 수동 항목)
        handles = []
        labels  = []
        if isinstance(refs_class, np.ndarray) and refs_class.size > 0:
            handles.append(QtGui.QPen(QtGui.QColor("red")));   labels.append("class lib")
        if isinstance(refs_label, np.ndarray) and refs_label.size > 0:
            handles.append(QtGui.QPen(QtGui.QColor("green"))); labels.append("labeling data")
        # matplotlib 범례 생성(가짜 라인으로 생성)
        if labels:
            import matplotlib.lines as mlines
            ml = []
            if ("class lib" in labels):
                ml.append(mlines.Line2D([], [], color='red',  linewidth=1.2, label='class lib'))
            if ("labeling data" in labels):
                ml.append(mlines.Line2D([], [], color='green',linewidth=1.2, label='labeling data'))
            ml.append(mlines.Line2D([], [], color='blue', linewidth=2.0, label='selected pixel'))
            self._ax.legend(handles=ml, loc="best")

        if title:
            self._ax.set_title(title)
        self._ax.set_xlabel("Wavelength"); self._ax.set_ylabel("Reflectance")
        self._canvas.draw_idle()


    def _draw_patch(self, y: int, x: int, size: int = 50):
        if self._rgb is None:
            return
        H, W, _ = self._rgb.shape
        r = size // 2
        y0, y1 = max(0, y - r), min(H, y + r)
        x0, x1 = max(0, x - r), min(W, x + r)
        patch = self._rgb[y0:y1, x0:x1, :].copy()
        # 빨간 점 마킹(중심)
        cy = min(r, patch.shape[0]-1); cx = min(r, patch.shape[1]-1)
        if patch.ndim == 3 and patch.shape[2] >= 3:
            rr = 3
            sy0, sy1 = max(0, cy-rr), min(patch.shape[0], cy+rr+1)
            sx0, sx1 = max(0, cx-rr), min(patch.shape[1], cx+rr+1)
            patch[sy0:sy1, sx0:sx1, 0] = 255; patch[sy0:sy1, sx0:sx1, 1:] = 0

        h, w, _ = patch.shape
        qimg = QtGui.QImage(patch.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        qimg = qimg.copy()
        self.lblPreview.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.lblPreview.width(), self.lblPreview.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def _fill_top_table(self, target: np.ndarray, lib_merged_unused: Dict[int, np.ndarray], top_each: int = 3):
        """
        요구사항: raw에서 SAM/SID/SCC 각 3개, cr에서 SAM/SID/SCC 각 3개 ⇒ 총 18개
        - Metric 칼럼에 'SAM (raw)' 형태로 표기
        - 값은 '작을수록 유사' 기준(거리형) 오름차순
        """
        # 0) 준비
        t = self.tableTopK
        t.clearContents()
        t.setRowCount(0)
        rows: list[tuple[int, str, float, str, str]] = []  # (cid, metric_with_src, value, name, desc)

        # 메타(재료명/설명) 확보
        meta_map = getattr(self._mw, "_mtrl_meta", {}) or {}
        try:
            missing = []
            # lib들을 만들기 전에 id 후보를 모르니, 만들고 나서 보강함
        except Exception:
            pass

        # 1) 클래스/라벨링 라이브러리(raw/cr) 각각 빌드 후 병합
        expect_c = self._cube.shape[2]
        class_lib_raw = _build_class_lib(self._spec_lib_raw, None, expect_c)
        class_lib_cr  = _build_class_lib(None, self._spec_lib_cr, expect_c)
        label_lib_raw = _build_labeling_lib(self._label_raw, None, expect_c)
        label_lib_cr  = _build_labeling_lib(None, self._label_cr, expect_c)

        class_lib_raw = _merge_lib_dicts(class_lib_raw, label_lib_raw, expect_c=expect_c)
        class_lib_cr  = _merge_lib_dicts(class_lib_cr, label_lib_cr, expect_c=expect_c)

        def _min_dists_for_lib(L: Dict[int, np.ndarray]) -> dict[int, tuple[float, float, float]]:
            """cid -> (sam_min, sid_min, scc_min)"""
            out: dict[int, tuple[float, float, float]] = {}
            L = _as_dict_lib(L)
            if not isinstance(L, dict) or len(L) == 0:
                return out
            # 빠른 벡터화 계산
            tvec = target.astype(float, copy=False)
            tnorm = np.linalg.norm(tvec)
            tclip = np.clip(tvec, 1e-12, None)
            tprob = tclip / tclip.sum()
            tc = tvec - tvec.mean()
            tcn = np.linalg.norm(tc) + 1e-12

            for cid, refs in L.items():  # refs: (P,C)
                A = refs.astype(float, copy=False)

                # # SAM
                # na = np.linalg.norm(A, axis=1) + 1e-12
                # v = np.clip((A @ tvec) / (na * (tnorm + 1e-12)), -1.0, 1.0)
                # sam_vals = np.arccos(v)
                sam_vals = SAM(A, tvec)

                # # SID
                # Ar = np.clip(A, 1e-12, None)
                # Ar = Ar / Ar.sum(axis=1, keepdims=True)
                # sid_vals = (Ar * np.log(Ar / tprob)).sum(axis=1) + (tprob * np.log(tprob / Ar)).sum(axis=1)
                sid_vals = SID(A, tvec)


                # # SCC(distance)
                # Ac = A - A.mean(axis=1, keepdims=True)
                # den = (np.linalg.norm(Ac, axis=1) * tcn) + 1e-12
                # corr = (Ac @ tc) / den
                # scc_vals = 1.0 - corr
                scc_vals = SCC(A, tvec)

                out[int(cid)] = (float(np.min(sam_vals)),
                                float(np.min(sid_vals)),
                                float(np.min(scc_vals)))
            return out

        d_raw = _min_dists_for_lib(class_lib_raw)
        d_cr  = _min_dists_for_lib(class_lib_cr)

        # 2) 메타 보강(이름/설명)
        try:
            all_cids = set(d_raw.keys()) | set(d_cr.keys())
            missing = [cid for cid in all_cids if cid not in meta_map]
            if missing:
                from data.db import search_material_filtering_list
                import os
                base_url = os.getenv('material_filtering_url')
                recs = search_material_filtering_list(base_url=base_url, mtrl_ids=list(missing))
                for rec in (recs or []):
                    cid2 = int(rec.get('mtrl_cd'))
                    meta_map[cid2] = {'name': rec.get('mtrl_nm', str(cid2)),
                                    'desc': rec.get('desc', '')}
                self._mw._mtrl_meta = meta_map
        except Exception:
            pass

        def _meta(cid_: int):
            m = meta_map.get(cid_) or {}
            name = self._id2name.get(cid_, m.get('name', str(cid_)))
            return name, m.get('desc', '')

        # 3) 각 소스(raw/cr) X 각 메트릭(SAM,SID,SCC)별로 top-3 선별
        def _pick_top(source: str, dmap: dict[int, tuple[float, float, float]]):
            # metric index: 0=SAM, 1=SID, 2=SCC
            metric_names = ["SAM", "SID", "SCC"]
            for mi, mname in enumerate(metric_names):
                # (cid, value)
                cand = [(cid, vals[mi]) for cid, vals in dmap.items()]
                cand.sort(key=lambda x: x[1])  # 오름차순
                for cid, val in cand[:top_each]:
                    name, desc = _meta(cid)
                    rows.append((cid, f"{mname} ({source})", float(val), name, desc))

        _pick_top("raw", d_raw)
        _pick_top("cr",  d_cr)

        # 4) 최종 표 반영(요청대로 총 18개; 부족하면 적은 만큼)
        t.setRowCount(len(rows))
        self._top_rows_cache = []
        for i, (cid, metric_src, val, name, desc) in enumerate(rows, start=1):
            t.setItem(i-1, 0, QtWidgets.QTableWidgetItem(str(i)))
            t.setItem(i-1, 1, QtWidgets.QTableWidgetItem(str(int(cid))))
            t.setItem(i-1, 2, QtWidgets.QTableWidgetItem(metric_src))
            itv = QtWidgets.QTableWidgetItem(f"{val:.6f}")
            itv.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            t.setItem(i-1, 3, itv)
            t.setItem(i-1, 4, QtWidgets.QTableWidgetItem(name))
            t.setItem(i-1, 5, QtWidgets.QTableWidgetItem(desc))
            # 캐시(선택 이벤트 대응)
            self._top_rows_cache.append((cid, metric_src, val, name, desc))


    def _select_first_row(self):
        if self.tableTopK.rowCount() > 0:
            self.tableTopK.selectRow(0)
            self._update_plot_from_row(0)

    def _on_row_changed(self):
        idxs = self.tableTopK.selectionModel().selectedRows()
        if not idxs:
            return
        row = int(idxs[0].row())
        # 선택은 미리보기/그래프만 갱신
        self._update_plot_from_row(row)

    def _update_plot_from_row(self, row: int):
        if self._target_spec is None:
            return
        cid = int(self.tableTopK.item(row, 1).text())

        # 클래스 라이브러리 번들(빨강)
        class_lib = _as_dict_lib(_build_class_lib(self._spec_lib_raw, self._spec_lib_cr, self._cube.shape[2]))
        refs_class = class_lib.get(cid)

        labeling_lib = _as_dict_lib(_build_labeling_lib(self._label_raw, self._label_cr, self._cube.shape[2]))
        refs_label = labeling_lib.get(cid)

        self._plot_target(self._target_spec,
                        refs_class=refs_class,
                        refs_label=refs_label,
                        title=f"Class : {cid}")
        
    def _class_change(self):
        sel = self.tableTopK.selectionModel().selectedRows()
        if not sel:
            QtWidgets.QMessageBox.information(self, "안내", "표에서 클래스를 먼저 선택하세요.")
            return
        row = int(sel[0].row())
        item_cid = self.tableTopK.item(row, 1)
        if item_cid is None:
            return
        new_cid = int(item_cid.text())

        origin_row = getattr(self, "_origin_row", None)
        if origin_row is None:
            QtWidgets.QMessageBox.warning(self, "경고", "원본 행 정보를 찾을 수 없습니다.")
            return

        # ✅ 이제 '클래스 변경' 버튼을 눌렀을 때만 반영
        self.class_changed.emit(int(origin_row), int(new_cid))
        self.accept()
    
    def _on_detail_analysis(self):
        """
        '상세 분석' 버튼 클릭 시
        - SAM/SID/SCC Top-3 정보로 설명 문자열 생성
        - 선택 픽셀 및 후보 스펙트럼들을 행(row), 파장을 열(column)로 하는 CSV 표 생성
        - 프롬프트 템플릿에 {csv}, {sam_top1} 등 치환 후 VLM 스트리밍 호출
        """
        if not self._top_rows_cache:
            QtWidgets.QMessageBox.information(self, "안내", "분석 데이터가 없습니다. 먼저 픽셀을 선택해주세요.")
            return

        if self._target_xy is None or self._target_spec is None:
            QtWidgets.QMessageBox.information(self, "안내", "타겟 픽셀 정보가 없습니다. 먼저 픽셀을 선택해주세요.")
            return

        # ---------- 1) SAM / SID / SCC Top-3 설명 문자열 만들기 ----------
        # self._top_rows_cache: [(cid, metric_src, val, name, desc), ...]
        def _top_by_metric(metric_tag: str, top_k: int = 3):
            rows = [
                (cid, metric_src, val, name, desc)
                for (cid, metric_src, val, name, desc) in self._top_rows_cache
                if metric_tag in metric_src  # "SAM", "SID", "SCC" 문자열로 필터
            ]
            rows.sort(key=lambda r: r[2])  # value(거리) 오름차순
            return rows[:top_k]

        def _desc_lines_for_metric(metric_tag: str):
            """
            metric_tag = "SAM" / "SID" / "SCC"
            return: [top1_json, top2_json, top3_json]
            각 원소는  {"mtrl_nm":"...", "mtrl_description":"..."} 형식의 문자열
            """
            rows = _top_by_metric(metric_tag, top_k=3)
            descs: list[str] = []

            for _, (cid, metric_src, val, name, desc) in enumerate(rows, start=1):
                mtrl_nm = name or str(cid)
                mtrl_desc = desc or ""

                # 따옴표/줄바꿈 간단 정리
                safe_nm = str(mtrl_nm).replace('"', '\\"')
                safe_desc = str(mtrl_desc).replace('"', '\\"').replace("\n", " ")

                json_str = f'{{"mtrl_nm":"{safe_nm}","mtrl_description":"{safe_desc}"}}'
                descs.append(json_str)

            # 후보가 부족한 경우 자리 채우기
            while len(descs) < 3:
                descs.append('{"mtrl_nm":"(no candidates)","mtrl_description":""}')

            return descs

        sam_top1, sam_top2, sam_top3 = _desc_lines_for_metric("SAM")
        sid_top1, sid_top2, sid_top3 = _desc_lines_for_metric("SID")
        scc_top1, scc_top2, scc_top3 = _desc_lines_for_metric("SCC")

        # ---------- 2) CSV 생성: 행=스펙트럼, 열=파장 ----------
        csv_text = ""
        try:
            # 2-1) wavelength (없으면 band index를 파장처럼 사용)
            wave = None
            try:
                cfg = getattr(self._mw, "cfg", {}) or {}
                wl_raw = cfg.get("wavelength") or cfg.get("wavelength_list")
                if wl_raw is not None:
                    wave_arr = np.asarray(wl_raw, dtype=float).ravel()
                    if wave_arr.shape[0] == self._target_spec.shape[0]:
                        wave = wave_arr
            except Exception:
                wave = None

            n_band = self._target_spec.shape[0]
            if wave is None:
                wave = np.arange(n_band, dtype=float)

            target_vec = self._target_spec.astype(np.float32, copy=False)
            expect_c = self._cube.shape[2]

            # 2-2) 라이브러리(스펙트럼 + 라벨링) 재구성
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

            metric_fn_map = {
                "SAM": SAM,
                "SID": SID,
                "SCC": SCC,
            }

            def _best_ref_for_row(row_info):
                """
                row_info: (cid, metric_src, val, name, desc)
                return: 1D np.ndarray or None
                """
                cid, metric_src, _, _, _ = row_info
                cid = int(cid)
                try:
                    # metric_src 예: "SAM (raw)", "SID (cr)" 등
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

                    # 거리 계산 (가능하면 벡터화)
                    try:
                        dists = metric_fn(refs, target_vec)
                    except Exception:
                        d_list = [metric_fn(refs[i, :], target_vec) for i in range(refs.shape[0])]
                        dists = np.asarray(d_list, dtype=float)

                    if dists.size == 0:
                        return None

                    best_idx = int(np.argmin(dists))
                    return refs[best_idx, :].astype(np.float32, copy=False)
                except Exception:
                    logging.exception("[VLM] _best_ref_for_row failed")
                    return None

            # 2-3) SAM/SID/SCC 각각에서 top-3 스펙트럼 추출 → 총 9행
            def _best_specs_for_metric(metric_tag: str):
                rows = _top_by_metric(metric_tag, top_k=3)
                specs = []
                for row_info in rows:
                    spec = _best_ref_for_row(row_info)
                    specs.append(spec)
                while len(specs) < 3:
                    specs.append(None)
                return specs  # [spec1, spec2, spec3]

            sam_specs = _best_specs_for_metric("SAM")  # 3개
            sid_specs = _best_specs_for_metric("SID")  # 3개
            scc_specs = _best_specs_for_metric("SCC")  # 3개
            
            spectra_rows: List[Tuple[str, Optional[np.ndarray]]] = []
            spectra_rows.append(("selected_pixel", target_vec))
            for i, spec in enumerate(sam_specs, start=1):
                spectra_rows.append((f"SAM_top{i}", spec))
            for i, spec in enumerate(sid_specs, start=1):
                spectra_rows.append((f"SID_top{i}", spec))
            for i, spec in enumerate(scc_specs, start=1):
                spectra_rows.append((f"SCC_top{i}", spec))

            # 2-4) CSV 문자열 생성
            # 헤더: spectrum_id, λ1, λ2, ...
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
            logging.exception("[VLM] CSV 생성 실패")
            csv_text = ""

        # ---------- 3) 프롬프트 템플릿 로드 + 치환 ----------
        try:
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
        # 표
        prompt_text = _safe_replace(prompt_text, "csv", csv_text)

        # SAM / SID / SCC description (템플릿의 키와 일치하도록 수정)
        prompt_text = _safe_replace(prompt_text, "sam_top1", sam_top1)
        prompt_text = _safe_replace(prompt_text, "sam_top2", sam_top2)
        prompt_text = _safe_replace(prompt_text, "sam_top3", sam_top3)

        prompt_text = _safe_replace(prompt_text, "sid_top1", sid_top1)
        prompt_text = _safe_replace(prompt_text, "sid_top2", sid_top2)
        prompt_text = _safe_replace(prompt_text, "sid_top3", sid_top3)

        prompt_text = _safe_replace(prompt_text, "scc_top1", scc_top1)
        prompt_text = _safe_replace(prompt_text, "scc_top2", scc_top2)
        prompt_text = _safe_replace(prompt_text, "scc_top3", scc_top3)

        # 디버깅용: 프롬프트를 클립보드에 복사
        QtWidgets.QApplication.clipboard().setText(prompt_text)
        
        with open('final_prompt.txt', 'w', encoding = 'utf8') as f:
            f.write(prompt_text)
        
        # ---------- 4) 선택 픽셀 주변 패치 → data URL ----------
        img_url = ""
        if self._rgb is not None:
            import base64, io
            from PIL import Image

            H, W, _ = self._rgb.shape
            y, x = self._target_xy
            y = int(y); x = int(x)

            win = 128
            r = win // 2
            y0, y1 = max(0, y - r), min(H, y + r)
            x0, x1 = max(0, x - r), min(W, x + r)

            patch = self._rgb[y0:y1, x0:x1, :].astype(np.uint8)
            patch = np.ascontiguousarray(patch)

            buf = io.BytesIO()
            Image.fromarray(patch).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            img_url = f"data:image/png;base64,{b64}"

        # ---------- 5) 결과 창 + 스트리밍 ----------
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
