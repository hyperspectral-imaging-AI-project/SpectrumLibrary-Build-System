# views/dialogs/analysis_selection_dialog.py
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
from PyQt5 import uic, QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush
from PyQt5.QtWidgets import (
    QDialog, QTableWidget, QTableWidgetItem, QPushButton,
    QComboBox, QHeaderView, QWidget
)

# matplotlib imports
try:
    import matplotlib
    matplotlib.use('Qt5Agg')  # PyQt5 백엔드 사용
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    MATPLOTLIB_AVAILABLE = True

    # 한글 폰트 설정 함수
    def _setup_korean_font():
        """matplotlib 한글 폰트 설정"""
        try:
            korean_fonts = [
                'Malgun Gothic', 'NanumGothic', 'NanumBarunGothic',
                'Gulim', 'Batang', 'Gungsuh', 'Dotum'
            ]
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            font_found = None
            for font in korean_fonts:
                if font in available_fonts:
                    font_found = font
                    break

            if font_found:
                matplotlib.rcParams['font.family'] = font_found
                matplotlib.rcParams['axes.unicode_minus'] = False
                logging.debug(f"[matplotlib] Korean font set to: {font_found}")
            else:
                logging.warning(
                    "[matplotlib] Korean font not found. Available fonts: %s",
                    set(available_fonts[:10])
                )
                matplotlib.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            logging.exception(f"[matplotlib] Failed to setup Korean font: {e}")

    _setup_korean_font()

except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not available, falling back to custom widget")


def _qcolor_to_matplotlib_color(qcolor: QColor) -> Tuple[float, float, float]:
    """QColor를 matplotlib RGB 튜플로 변환"""
    return (qcolor.red() / 255.0, qcolor.green() / 255.0, qcolor.blue() / 255.0)


if MATPLOTLIB_AVAILABLE:
    class SpectrumWidget(FigureCanvasQTAgg):
        """matplotlib 기반 스펙트럼 그래프 위젯"""

        def __init__(self, parent=None):
            self._figure = Figure(figsize=(6, 4))
            super().__init__(self._figure)
            self.setParent(parent)

            self._ax = self._figure.add_subplot(111)

            self._spectra: List[np.ndarray] = []
            self._wavelengths: Optional[np.ndarray] = None
            self._colors: List[QColor] = []
            self._selected_indices: List[int] = []
            self._label_provider = None
            self._metas: List[Any] = []
            self._lines = []
            self.on_pick = None
            self.mpl_connect("pick_event", self._on_pick)

            # 그래프 제목(TL/TR에서 사용)
            self._title: str = ""

            self._ax.text(
                0.5, 0.5, 'No spectrum data',
                ha='center', va='center',
                transform=self._ax.transAxes,
                fontsize=12
            )
            # self._ax.set_xlabel('Wavelength')
            self._ax.set_xlabel('Band')
            self._ax.set_ylabel('Intensity')
            self.draw()

        def set_spectra(
            self,
            spectra: List[np.ndarray],
            wavelengths: Optional[np.ndarray] = None,
            colors: Optional[List[QColor]] = None,
            metas: Optional[List[Any]] = None
        ):
            self._spectra = spectra or []
            self._wavelengths = wavelengths
            self._colors = (colors or []) if colors else []
            self._metas = metas or [None] * len(self._spectra)

            if self._colors and len(self._colors) < len(self._spectra):
                base = self._colors[:]
                i = 0
                while len(self._colors) < len(self._spectra):
                    self._colors.append(base[i % len(base)])
                    i += 1

            self._selected_indices = []
            self._update_plot()

        def set_selected_indices(self, indices: List[int]):
            self._selected_indices = indices
            self._update_plot()

        def set_title(self, title: str):
            """그래프 제목 설정(TL/TR에서 사용)."""
            self._title = str(title)
            try:
                self._ax.set_title(self._title)
            except Exception:
                pass
            self.draw()

        def _update_plot(self):
            self._ax.clear()
            self._lines = []

            spectra = self._spectra or []
            if len(spectra) == 0:
                self._ax.text(
                    0.5, 0.5, 'No spectrum data',
                    ha='center', va='center',
                    transform=self._ax.transAxes,
                    fontsize=12
                )
                # self._ax.set_xlabel('Wavelength')
                self._ax.set_xlabel('Band')
                self._ax.set_ylabel('Intensity')
                self._ax.grid(True, alpha=0.3, linestyle='--')

                # 제목 적용
                if getattr(self, "_title", ""):
                    self._ax.set_title(self._title)

                self._ax.relim()
                self._ax.autoscale_view()
                self._figure.tight_layout()
                self.draw()
                return

            if self._wavelengths is not None and len(self._wavelengths) > 0:
                x_w = np.asarray(self._wavelengths).ravel()
            else:
                x_w = None

            sel_set = set(self._selected_indices or [])

            fallback_colors = [
                (100/255, 150/255, 255/255),
                (255/255, 100/255, 100/255),
                (100/255, 255/255, 100/255),
                (255/255, 255/255, 100/255),
                (255/255, 100/255, 255/255),
            ]

            def _xy(idx):
                y = np.asarray(spectra[idx]).ravel().astype(float)
                y = np.nan_to_num(
                    y,
                    nan=np.nanmedian(y) if np.isfinite(np.nanmedian(y)) else 0.0,
                    posinf=np.nanmax(y[np.isfinite(y)]) if np.any(np.isfinite(y)) else 0.0,
                    neginf=np.nanmin(y[np.isfinite(y)]) if np.any(np.isfinite(y)) else 0.0,
                )
                if x_w is not None and len(x_w) == len(y):
                    x = x_w
                else:
                    x = np.arange(len(y), dtype=float)
                return x, y

            def _base_color(idx):
                if self._colors and idx < len(self._colors) and isinstance(self._colors[idx], QtGui.QColor):
                    return _qcolor_to_matplotlib_color(self._colors[idx])
                return fallback_colors[idx % len(fallback_colors)]

            def _plot_idx(idx, strong=False):
                x, y = _xy(idx)
                if strong:
                    color = (1.0, 0.0, 0.0)
                    lw, alpha, z = 1.0, 1.0, 10
                else:
                    bc = _base_color(idx)
                    color = tuple(min(1.0, c * 0.85) for c in bc)
                    lw, alpha, z = 1.4, 0.6, 1
                line, = self._ax.plot(
                    x, y,
                    color=color,
                    linewidth=lw,
                    alpha=alpha,
                    zorder=z,
                    picker=5
                )
                line._spec_index = idx
                self._lines.append(line)

            for i in range(len(spectra)):
                if i not in sel_set:
                    _plot_idx(i, strong=False)
            for i in range(len(spectra)):
                if i in sel_set:
                    _plot_idx(i, strong=True)

            # self._ax.set_xlabel('Wavelength')
            self._ax.set_ylabel('Intensity')
            self._ax.set_xlabel('Band')
            self._ax.grid(True, alpha=0.3, linestyle='--')

            # 제목 적용
            if getattr(self, "_title", ""):
                self._ax.set_title(self._title)

            self._ax.relim()
            self._ax.autoscale_view()
            try:
                self._ax.margins(x=0.02, y=0.05)
            except Exception:
                pass

            self._figure.tight_layout()
            self.draw()

        def _on_pick(self, event):
            try:
                line = getattr(event, "artist", None)
                if not line or line not in self._lines:
                    return
                idx = getattr(line, "_spec_index", None)
                if idx is None or idx < 0 or idx >= len(self._spectra):
                    return

                modifiers = QtWidgets.QApplication.keyboardModifiers()
                multi = bool(modifiers & (Qt.ControlModifier | Qt.ShiftModifier))

                if multi:
                    cur = set(self._selected_indices or [])
                    if idx in cur:
                        cur.remove(idx)
                    else:
                        cur.add(idx)
                    self._selected_indices = sorted(cur)
                else:
                    self._selected_indices = [idx]

                self._update_plot()

                if callable(self.on_pick):
                    meta = self._metas[idx] if self._metas and idx < len(self._metas) else None
                    self.on_pick(meta, event, multi)
            except Exception:
                logging.exception("[SpectrumWidget] _on_pick failed")

else:
    class SpectrumWidget(QWidget):
        """Fallback 위젯 (matplotlib 없음)"""

        def __init__(self, parent=None):
            super().__init__(parent)
            self._spectra: List[np.ndarray] = []
            self._wavelengths: Optional[np.ndarray] = None
            self._colors: List[QColor] = []
            self._selected_indices: List[int] = []
            self._label_provider = None
            self._metas: List[Any] = []
            self._title: str = ""

        def set_title(self, title: str):
            """그래프 제목 설정(TL/TR에서 사용)."""
            self._title = str(title)
            self.update()

        def set_spectra(
            self,
            spectra: List[np.ndarray],
            wavelengths: Optional[np.ndarray] = None,
            colors: Optional[List[QColor]] = None,
            metas: Optional[List[Any]] = None
        ):
            self._spectra = spectra or []
            self._wavelengths = wavelengths
            self._colors = (colors or [])
            self._metas = metas or [None] * len(self._spectra)
            if self._colors and len(self._colors) < len(self._spectra):
                base = self._colors[:]
                i = 0
                while len(self._colors) < len(self._spectra):
                    self._colors.append(base[i % len(base)])
                    i += 1
            self._selected_indices = []
            self.update()

        def set_selected_indices(self, indices: List[int]):
            self._selected_indices = indices
            self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor(32, 32, 32))

            if not self._spectra:
                painter.setPen(QPen(QColor(200, 200, 200)))
                text = "No spectrum data\n(matplotlib not available)"
                if self._title:
                    text = self._title + "\n" + text
                painter.drawText(
                    self.rect(),
                    Qt.AlignCenter,
                    text
                )
                return

class AnalysisSelectionDialog(QDialog):
    """
    지정한 영역 분석 다이얼로그
    - 좌: 클래스 정보 테이블
    - 우상: 선택한 클래스의 픽셀 정보 테이블
    - 우하: 스펙트럼 비교 그래프
    - TL: 선택 픽셀 스펙트럼 번들
    - TR: 사용된 reference 번들
    """

    pixels_class_set = pyqtSignal(list, int)  # ([(y,x),...], new_cid)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        ui_dir: Optional[Path] = None
    ):
        super().__init__(parent)

        app_dir = Path(getattr(parent, "app_dir", Path(__file__).resolve().parents[3]))
        ui_dir = Path(ui_dir) if ui_dir else (app_dir / "ui")
        ui_path = (ui_dir / "analysis_selection_dialog.ui").resolve()
        if not ui_path.exists():
            raise FileNotFoundError(f"UI 파일을 찾을 수 없습니다: {ui_path}")
        self._root = uic.loadUi(str(ui_path), baseinstance=self)

        # --- 위젯 핸들 ---
        self.tableClassInfo: QtWidgets.QTableWidget = self.findChild(QtWidgets.QTableWidget, "tableClassInfo")
        self.tablePixels: QtWidgets.QTableWidget = self.findChild(QtWidgets.QTableWidget, "tablePixels")
        self.twSelected: QtWidgets.QTreeWidget = self.findChild(QtWidgets.QTreeWidget, "tw_selected")

        self.cbIntermediateTarget: QtWidgets.QComboBox = self.findChild(QtWidgets.QComboBox, "cb_intermediate_target")
        self.cbClassRange: QtWidgets.QComboBox = self.findChild(QtWidgets.QComboBox, "cb_class_range")
        self.sldBins: QtWidgets.QSlider = self.findChild(QtWidgets.QSlider, "sld_bins")
        self.lblHist: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, "lbl_hist")
        self.gbHist: QtWidgets.QGroupBox = self.findChild(QtWidgets.QGroupBox, "gb_hist")

        self.btnSelect: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btnSelect")
        self.btnApply: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btn_apply")
        self.btnAnalyzeSelection: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btnAnalyzeSelection")
        self.btnCancel: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btn_cancel")
        self.cboSetClass: QtWidgets.QComboBox = self.findChild(QtWidgets.QComboBox, "comboBox")

        self.btnPlus: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btn_plus")
        self.btnMinus: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btn_minus")
        self.btnReset: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btn_reset")
        self.btnPixel: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btn_pixel")
        self.btnRect: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btn_rect")
        self.btnFree: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btn_free")

        self._plotTL: QtWidgets.QWidget = self.findChild(QtWidgets.QFrame, "plot_tl")
        self._plotTR: QtWidgets.QWidget = self.findChild(QtWidgets.QFrame, "plot_tr")
        self._plotBR: QtWidgets.QWidget = self.findChild(QtWidgets.QFrame, "plot_br")

        if not self._plotTL:
            logging.warning("[AnalysisDialog] plot_tl widget not found")
        if not self._plotTR:
            logging.warning("[AnalysisDialog] plot_tr widget not found")
        if not self._plotBR:
            logging.warning("[AnalysisDialog] plot_br widget not found")

        # --- TL / TR 스펙트럼 위젯 설치 ---
        self.spectrumWidgetTL: Optional[SpectrumWidget] = None
        self.spectrumWidgetTR: Optional[SpectrumWidget] = None
        self._focused_point: Optional[Tuple[int, int]] = None

        def _install_spectrum_widget(container: QtWidgets.QWidget) -> SpectrumWidget:
            if container is None:
                logging.warning("[AnalysisDialog] _install_spectrum_widget: container is None")
                return None
            w = SpectrumWidget(self)
            layout = container.layout()
            if layout is None:
                layout = QtWidgets.QVBoxLayout(container)
                layout.setContentsMargins(4, 4, 4, 4)
                layout.setSpacing(4)
            else:
                while layout.count() > 0:
                    item = layout.takeAt(0)
                    if item.widget():
                        item.widget().setParent(None)
            layout.addWidget(w)
            w.setMinimumSize(400, 250)
            logging.debug(f"[AnalysisDialog] installed spectrum widget to: {container.objectName()}")
            return w

        # ✅ TL/TR 그래프 제목 설정 (영어)
        # --- TL / TR 위젯 실제 생성 ---
        if self._plotTL and not self.spectrumWidgetTL:
            self.spectrumWidgetTL = _install_spectrum_widget(self._plotTL)
        if self._plotTR and not self.spectrumWidgetTR:
            self.spectrumWidgetTR = _install_spectrum_widget(self._plotTR)

        # ✅ TL/TR 그래프 제목 설정 (영어)
        if isinstance(self.spectrumWidgetTL, SpectrumWidget):
            # TL : 선택된 스펙트럼 그래프
            self.spectrumWidgetTL._title = "Selected Spectra"
        if isinstance(self.spectrumWidgetTR, SpectrumWidget):
            # TR : 선택된 스펙트럼의 레퍼런스 그래프
            self.spectrumWidgetTR._title = "Reference Spectra of Selected Pixels"

        # --- BR 스펙 비교용 ---
        self.spectrumWidgetBR: Optional[FigureCanvasQTAgg] = None
        if MATPLOTLIB_AVAILABLE and self._plotBR:
            parent_container = self._plotBR
            layout = parent_container.layout()
            if layout:
                while layout.count() > 0:
                    item = layout.takeAt(0)
                    if item.widget():
                        item.widget().setParent(None)

                self.spectrumWidgetBR = FigureCanvasQTAgg(Figure(figsize=(6, 4)))
                self.spectrumWidgetBR.setMinimumSize(400, 250)
                layout.addWidget(self.spectrumWidgetBR)
                logging.debug("[AnalysisDialog] installed BR spectrum widget")

        # --- 히스토그램 캔버스는 _ensure_hist_widget 에서 생성 ---
        self.histogramWidget: Optional[FigureCanvasQTAgg] = None

        # --- 내부 상태 ---
        self._classmap: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._hsi_data: Optional[np.ndarray] = None
        self._topk_cids: Optional[np.ndarray] = None
        self._topk_vals: Optional[np.ndarray] = None
        self._id_to_name: Optional[Dict[int, str]] = None
        self._wavelengths: Optional[np.ndarray] = None
        self._metric: str = "SAD"
        self._palette: Dict[int, QtGui.QColor] = {}
        self._match_provider = None

        self._selected_class_ids: List[int] = []
        self._pixel_data: List[Dict[str, Any]] = []
        self._filling_pixels_table: bool = False
        self._tl_mode: str = "tree"
        self._tl_hist_cache: dict = {}
        self._spectrum_library_provider = None
        self._suppress_tree_item_changed = False
        self._selected_ref_keys = set()
        self._hist_mode: str = "checked"

        self._current_op: str = "union"
        self._current_mode: Optional[str] = None
        self._focus_source = None
        self._current_hist_cid: Optional[int] = None
        self._ensure_hist_widget()
        self._hist_user_locked_until = 0.0
        self._hist_user_lock_secs = 2.0

        self._hist_cache = {
            "edges": None,
            "vals": None,
            "indices": None,
            "selected_bin": None,
            "selected_mask": None,
        }

        if self.histogramWidget is not None:
            self._hist_click_cid = self.histogramWidget.mpl_connect(
                "button_press_event", self._on_hist_click
            )
        else:
            self._hist_click_cid = None

        # --- 테이블 설정 ---
        if self.tableClassInfo:
            self.tableClassInfo.setColumnCount(4)
            self.tableClassInfo.setHorizontalHeaderLabels(["#", "Class", "Name", "Count"])
            self.tableClassInfo.horizontalHeader().setStretchLastSection(True)
            self.tableClassInfo.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)

        if self.tablePixels:
            self.tablePixels.setColumnCount(5)
            self.tablePixels.setHorizontalHeaderLabels(["#", "Locate", "Class", "metric", "value"])
            self.tablePixels.horizontalHeader().setStretchLastSection(True)
            self.tablePixels.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
            self.tablePixels.setSelectionMode(QtWidgets.QTableWidget.ExtendedSelection)

        if self.twSelected and self.twSelected.header():
            self.twSelected.header().setVisible(False)

        # --- 시그널 연결 ---
        if self.btnSelect:
            self.btnSelect.clicked.connect(self._on_select_class)
        if self.tableClassInfo:
            self.tableClassInfo.itemChanged.connect(self._on_class_check_changed)
            self.tableClassInfo.itemSelectionChanged.connect(self._on_class_selection_changed)
        if self.tablePixels:
            self.tablePixels.itemSelectionChanged.connect(self._on_pixel_selection_changed)
            self.tablePixels.itemChanged.connect(self._on_pixel_checkbox_changed)
        if self.twSelected:
            if self.twSelected.header():
                self.twSelected.header().setVisible(False)
            self.twSelected.itemChanged.connect(self._on_selected_tree_changed)
            
                # ▼▼▼ 여기부터 twSelected 관련 설정 ▼▼▼
        if self.twSelected and self.twSelected.header():
            self.twSelected.header().setVisible(False)

        # ★ Layer(선택된 픽셀 트리) 글꼴 크기 줄이기
        if self.twSelected:
            font = self.twSelected.font()
            orig_size = font.pointSize()

            # pointSize가 -1이면 OS 기본폰트라서 안전한 기본값 지정
            if orig_size > 0:
                font.setPointSize(max(7, orig_size - 1))  # 한 단계만 줄이되 최소 7pt
            else:
                font.setPointSize(8)

            self.twSelected.setFont(font)

            # 컬럼/폭도 적당히 설정(선택사항)
            self.twSelected.setColumnCount(1)
            self.twSelected.setColumnWidth(0, 230)

        if self.btnApply:
            self.btnApply.clicked.connect(self._on_apply_class)
        if self.btnAnalyzeSelection:
            self.btnAnalyzeSelection.clicked.connect(self._on_analyze_selection)
        if self.btnCancel:
            self.btnCancel.clicked.connect(self.reject)

        if self.btnPlus:
            self.btnPlus.setCheckable(True)
            self.btnPlus.clicked.connect(self._on_plus_clicked)
        if self.btnMinus:
            self.btnMinus.setCheckable(True)
            self.btnMinus.clicked.connect(self._on_minus_clicked)
        if self.btnReset:
            self.btnReset.clicked.connect(self._on_reset_clicked)
        if self.btnPixel:
            self.btnPixel.setCheckable(True)
            self.btnPixel.clicked.connect(self._on_pixel_mode_clicked)
        if self.btnRect:
            self.btnRect.setCheckable(True)
            self.btnRect.clicked.connect(self._on_rect_mode_clicked)
        if self.btnFree:
            self.btnFree.setCheckable(True)
            self.btnFree.clicked.connect(self._on_free_mode_clicked)

        if self.cbIntermediateTarget:
            self.cbIntermediateTarget.currentTextChanged.connect(self._on_intermediate_target_changed)

        # ▶ 클래스 범위 콤보: 단일 콤보만 사용
        # 디버깅: 모든 콤보박스 찾기
        all_combos = self.findChildren(QtWidgets.QComboBox)
        logging.info(
            "[AnalysisDialog] Found %d comboboxes: %s",
            len(all_combos),
            [cb.objectName() for cb in all_combos]
        )
        
        if self.cbClassRange:
            logging.info(
                "[AnalysisDialog] cbClassRange found: objectName=%s, parent=%s",
                self.cbClassRange.objectName(),
                self.cbClassRange.parent().objectName() if self.cbClassRange.parent() else None
            )
            
            # 기존 연결 전부 제거
            try:
                self.cbClassRange.currentIndexChanged.disconnect()
            except Exception:
                pass
            try:
                self.cbClassRange.activated.disconnect()
            except Exception:
                pass
            try:
                self.cbClassRange.currentTextChanged.disconnect()
            except Exception:
                pass

            self.cbClassRange.setEditable(False)
            self.cbClassRange.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
            
            # 시그널 연결
            self.cbClassRange.currentIndexChanged.connect(self._on_class_range_changed)
            logging.info("[AnalysisDialog] Connected currentIndexChanged to _on_class_range_changed")
            
            # 드롭다운에서 항목을 클릭하는 순간에도 바로 반영
            try:
                self.cbClassRange.view().pressed.connect(self._on_class_combo_pressed)
                logging.info("[AnalysisDialog] Connected view().pressed to _on_class_combo_pressed")
            except Exception:
                logging.exception("[AnalysisDialog] failed to hook cbClassRange.view().pressed")

        else:
            logging.error("[class combo] cbClassRange not found in UI")
            # 대안: objectName으로 직접 찾기
            for cb in all_combos:
                if cb.objectName() == "cb_class_range":
                    self.cbClassRange = cb
                    logging.warning("[AnalysisDialog] Found cb_class_range by alternative method")
                    # 시그널 연결
                    self.cbClassRange.currentIndexChanged.connect(self._on_class_range_changed)
                    break

        if self.sldBins:
            self.sldBins.setMinimum(5)
            self.sldBins.setMaximum(200)
            self.sldBins.setValue(50)
            self.sldBins.valueChanged.connect(self._on_bins_changed)

        self._op_button_group = QtWidgets.QButtonGroup(self)
        if self.btnPlus:
            self._op_button_group.addButton(self.btnPlus, 0)
        if self.btnMinus:
            self._op_button_group.addButton(self.btnMinus, 1)
        if self.btnPlus:
            self.btnPlus.setChecked(True)

        self._mode_button_group = QtWidgets.QButtonGroup(self)
        if self.btnPixel:
            self._mode_button_group.addButton(self.btnPixel, 0)
        if self.btnRect:
            self._mode_button_group.addButton(self.btnRect, 1)
        if self.btnFree:
            self._mode_button_group.addButton(self.btnFree, 2)

        if isinstance(self.spectrumWidgetTL, SpectrumWidget):
            self.spectrumWidgetTL.set_spectra([], None)
        if isinstance(self.spectrumWidgetTR, SpectrumWidget):
            self.spectrumWidgetTR.set_spectra([], None)

        logging.debug("[AnalysisDialog] initialized: TL/TR spectrum widgets installed, tree mode enabled")

    def set_data(
        self,
        mask: np.ndarray,
        classmap: np.ndarray,
        hsi_data: np.ndarray,
        topk_cids: Optional[np.ndarray] = None,
        topk_vals: Optional[np.ndarray] = None,
        id_to_name: Optional[Dict[int, str]] = None,
        wavelengths: Optional[np.ndarray] = None,
        metric: str = "SAD"
    ):
        """
        분석 데이터 설정
        
        Args:
            mask: 분석 영역 마스크 (H, W) bool
            classmap: 분류맵 (H, W) int
            hsi_data: HSI 데이터 (H, W, C) float
            topk_cids: Top-K 클래스 ID (H, W, K) int
            topk_vals: Top-K 값 (H, W, K) float
            id_to_name: 클래스 ID → 이름 매핑
            wavelengths: 파장 배열 (C,)
            metric: 메트릭 이름 ("SAD", "SID", "SCC")
        """
        try:
            self._mask = np.asarray(mask, dtype=bool)
            self._classmap = np.asarray(classmap, dtype=int)
            self._hsi_data = np.asarray(hsi_data)
            self._topk_cids = topk_cids
            self._topk_vals = topk_vals
            self._id_to_name = id_to_name or {}
            self._wavelengths = wavelengths
            self._metric = metric
            
            # 클래스 정보 테이블 업데이트
            self._update_class_table()
            # Histogram 콤보박스 갱신
            self._update_histogram_class_combo(apply=True)
            # 클래스 콤보박스 초기화
            self._update_class_combo()

            # ★ 처음엔 비워두고, 맵에서 영역을 실제로 그린 뒤 update_mask(...)가 채우게 함
            self._clear_right_panels()
            if self._mask is not None and np.any(self._mask):
                QtCore.QTimer.singleShot(100, self._update_pixel_table)
                # ★ SSOT 기반으로 TL/TR/Hist 강제 1회 동기화
                QtCore.QTimer.singleShot(120, self._refresh_all_from_checked)
            else:
                # 트리도 비워둠 (맵에서 영역을 그리고 나면 MainWindow가 dlg.update_mask(...)를 호출)
                if self.twSelected:
                    self.twSelected.clear()
                # 마스크가 비어있으면 픽셀 테이블도 비워둠
                if self.tablePixels is not None:
                    self.tablePixels.setRowCount(0)
                self._pixel_data = []
                logging.debug("[AnalysisDialog] set_data: empty mask, cleared panels")
            
        except Exception:
            logging.exception("[AnalysisDialog] set_data failed")
    
    def update_mask(self, mask: np.ndarray):
        """
        분석 영역 마스크를 갱신하고 다이얼로그의 모든 패널을 최신화한다.
        - classmap/hsi 등은 최초 set_data(...)에서 이미 세팅되어 있다고 가정
        - 마스크 변경 즉시: 좌측 클래스 테이블, 우측 픽셀표, 트리, TL/TR 그래프 갱신
        - TR(우측 상단): 선택 클래스 범위 내에서 '실제로 선택된 reference' 스펙만 표시
        """
        try:
            if mask is None:
                return

            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.ndim != 2:
                logging.warning(f"[AnalysisDialog] update_mask: invalid mask shape {mask_array.shape}")
                return

            if self._classmap is not None and mask_array.shape != self._classmap.shape[:2]:
                logging.warning(f"[AnalysisDialog] update_mask: mask shape {mask_array.shape} != classmap {self._classmap.shape[:2]}")
                return

            # 1) 마스크 저장(항상 복사본)
            self._mask = mask_array.copy()

            # 2) 마스크 내 존재 클래스 수집 → 선택 리스트 기본값 구성
            if self._classmap is not None and np.any(self._mask):
                masked = self._classmap[self._mask]
                self._selected_class_ids = [int(c) for c in np.unique(masked)]
            else:
                self._selected_class_ids = []

            # 3) 마스크가 비면 모든 패널 비움 후 종료
            if not np.any(mask_array):
                logging.info("[AnalysisDialog] update_mask: empty mask, clearing data")
                self._pixel_data = []
                if self.tablePixels is not None:
                    self.tablePixels.setRowCount(0)
                if self.twSelected is not None:
                    self.twSelected.clear()
                self._clear_right_panels()
                # 좌측 클래스 테이블도 비움
                if self.tableClassInfo is not None:
                    self.tableClassInfo.setRowCount(0)
                # TR도 비움
                if isinstance(self.spectrumWidgetTR, SpectrumWidget):
                    self.spectrumWidgetTR.set_spectra([], None)
                # 포커스 해제
                self._focused_point = None
                return

            # 4) 좌측 클래스 테이블/히스토그램 콤보 갱신
            self._update_class_table()
            self._update_histogram_class_combo(apply=True)

            # 5) 우측 픽셀 테이블 + 트리 + TL 갱신
            #    (작으면 즉시, 크면 비동기) — 여기서는 즉시 실행
            self._update_pixel_table()

            # 6) TR(우측 상단) 갱신 — '지정 영역 내에서 실제로 선택된 reference 스펙'만
            #    cbClassRange가 설정되어 있으면 그 클래스 기준, 아니면 비움
            keys = self._ref_keys_for_points(self._get_checked_points())
            self._update_tr_refs_from_checked(keys)
            
            # 7) 포커스 초기화(마스크 변경 시 TL 클릭 포커스는 무효)
            self._focused_point = None
            # ★ SSOT 동기화
            self._refresh_all_from_checked()
            logging.info(f"[AnalysisDialog] mask updated: shape={mask_array.shape}, sum={int(mask_array.sum())}")

        except Exception:
            logging.exception("[AnalysisDialog] update_mask failed")

    
    def _update_class_table(self):
        if self.tableClassInfo is None or self._classmap is None or self._mask is None:
            return
        try:
            masked_classes = self._classmap[self._mask]
            unique_classes, counts = np.unique(masked_classes, return_counts=True)
            sorted_idx = np.argsort(-counts)
            unique_classes = unique_classes[sorted_idx]
            counts = counts[sorted_idx]

            self.tableClassInfo.setRowCount(len(unique_classes))
            self._selected_class_ids = []

            for row, (cid, count) in enumerate(zip(unique_classes, counts)):
                # 체크박스(기본 체크)
                checkbox = QtWidgets.QTableWidgetItem()
                checkbox.setFlags(checkbox.flags() | Qt.ItemIsUserCheckable)
                checkbox.setCheckState(Qt.Checked)
                self.tableClassInfo.setItem(row, 0, checkbox)

                # Class ID
                cid_item = QtWidgets.QTableWidgetItem(str(int(cid)))
                cid_item.setFlags(cid_item.flags() & ~Qt.ItemIsEditable)
                self.tableClassInfo.setItem(row, 1, cid_item)

                # Name
                name = self._id_to_name.get(int(cid), str(int(cid))) if self._id_to_name else str(int(cid))
                name_item = QtWidgets.QTableWidgetItem(name)
                name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                self.tableClassInfo.setItem(row, 2, name_item)

                # Count
                count_item = QtWidgets.QTableWidgetItem(str(int(count)))
                count_item.setFlags(count_item.flags() & ~Qt.ItemIsEditable)
                count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tableClassInfo.setItem(row, 3, count_item)

                # 선택 목록에 추가
                self._selected_class_ids.append(int(cid))

            self.tableClassInfo.resizeColumnsToContents()

            # ✅ 루프 종료 후 1회만 호출
            self._update_histogram_class_combo(apply=True)

        except Exception:
            logging.exception("[AnalysisDialog] _update_class_table failed")

    def _update_histogram_class_combo(self, *, apply: bool = False):
        if self.cbClassRange is None or self._classmap is None or self._mask is None:
            return

        prev_cid = self._current_hist_cid
        try:
            self.cbClassRange.blockSignals(True)
            self.cbClassRange.clear()

            m = np.asarray(self._mask, dtype=bool)
            if m.size == 0 or not np.any(m):
                self.cbClassRange.addItem("No classes", -1)
                self._current_hist_cid = None
            else:
                masked_classes = np.asarray(self._classmap)[m]
                classes = sorted(int(c) for c in np.unique(masked_classes) if int(c) >= 0)
                if not classes:
                    self.cbClassRange.addItem("No classes", -1)
                    self._current_hist_cid = None
                else:
                    for cid in classes:
                        name = self._id_to_name.get(cid, str(cid)) if self._id_to_name else str(cid)
                        self.cbClassRange.addItem(f"Class {cid} ({name})", cid)

                    idx = self.cbClassRange.findData(prev_cid) if prev_cid in classes else -1
                    if idx < 0:
                        self._current_hist_cid = classes[0]
                        idx = self.cbClassRange.findData(self._current_hist_cid)
                    else:
                        self._current_hist_cid = prev_cid
                    if idx >= 0:
                        self.cbClassRange.setCurrentIndex(idx)
        except Exception:
            logging.exception("[AnalysisDialog] _update_histogram_class_combo failed")
        finally:
            try: self.cbClassRange.blockSignals(False)
            except Exception: pass

        # ✅ 필요 시 ‘한 번’ 즉시 그리기
        if apply:
            cid = self.cbClassRange.currentData()
            try: cid = int(cid) if cid is not None else None
            except Exception: cid = None
            self._apply_histogram_class_selection(cid, source="combo-populate")


        # ⛔ 기존의 즉시 갱신 제거
        # 기존: self._apply_histogram_class_selection(cid, source="combo-populate")

    def _update_class_combo(self):
        """클래스 설정 콤보박스 업데이트"""
        if self.cboSetClass is None:
            return
        
        self.cboSetClass.clear()
        
        if self._classmap is None:
            return
        
        # 모든 클래스 ID 수집
        unique_classes = np.unique(self._classmap)
        sorted_classes = sorted([int(c) for c in unique_classes])
        
        for cid in sorted_classes:
            name = self._id_to_name.get(cid, str(cid)) if self._id_to_name else str(cid)
            self.cboSetClass.addItem(f"{cid} ({name})", cid)
        
    def _on_class_check_changed(self, item: QTableWidgetItem):
        if item.column() != 0:
            return
        row = item.row()
        cid_item = self.tableClassInfo.item(row, 1)
        if cid_item is None:
            return
        try:
            cid = int(cid_item.text())
            is_checked = (item.checkState() == Qt.Checked)
            if is_checked:
                if cid not in self._selected_class_ids:
                    self._selected_class_ids.append(cid)
            else:
                if cid in self._selected_class_ids:
                    self._selected_class_ids.remove(cid)

            # ★ 삭제: 즉시 갱신
            # self._update_pixel_table()

        except Exception:
            logging.exception("[AnalysisDialog] _on_class_check_changed failed")

    
    def _on_class_selection_changed(self):
        """클래스 행 선택 변경 시"""
        # 체크박스는 itemChanged로 처리되므로 여기서는 추가 처리 없음
        pass
    
    def _on_select_class(self):
        picked: List[int] = []
        if self.tableClassInfo:
            for row in range(self.tableClassInfo.rowCount()):
                it = self.tableClassInfo.item(row, 0)
                cid_item = self.tableClassInfo.item(row, 1)
                if it is None or cid_item is None:
                    continue
                if it.checkState() == Qt.Checked:
                    try:
                        picked.append(int(cid_item.text()))
                    except Exception:
                        pass
        self._selected_class_ids = picked
        self._update_pixel_table()   # 오른쪽 표 + 스펙트럼 갱신

    def _update_pixel_table(self):
        """
        선택된 클래스(좌측 체크)를 기준으로 내부 _pixel_data를 채우고,
        '선택된 픽셀' 트리(Class → (y,x))를 재구성한다.
        - tablePixels 유무와 관계없이 동작
        - ★ 지정된 분석 영역(마스크) 내의 픽셀에 대한 정보만 처리
        """
        # ★ 테이블 유무와 상관없이 동작하도록 수정
        if self._classmap is None or self._mask is None:
            return

        try:
            # ★ 마스크가 비어있으면 빈 상태로 표시
            if not np.any(self._mask):
                logging.debug("[AnalysisDialog] _update_pixel_table: empty mask")
                self._pixel_data = []
                if self.tablePixels is not None:
                    self.tablePixels.setRowCount(0)
                if self.twSelected is not None:
                    self.twSelected.clear()
                self._clear_right_panels()
                return

            # --- 선택 클래스 비어 있으면 마스크 내 등장 클래스로 자동 채움
            if not self._selected_class_ids:
                masked = self._classmap[self._mask]
                if masked.size > 0:
                    self._selected_class_ids = [int(c) for c in np.unique(masked)]

            if not self._selected_class_ids:
                self._pixel_data = []
                self._clear_right_panels()
                return

            # ★ 지정된 분석 영역(마스크) 내의 픽셀 좌표만 수집
            ys, xs = np.where(self._mask)
            self._pixel_data = []
            pixel_rows = []  # (y, x, cid, metric_val)

            for y, x in zip(ys.tolist(), xs.tolist()):
                cid = int(self._classmap[y, x])
                if cid not in self._selected_class_ids:
                    continue

                metric_val = None
                if self._topk_cids is not None and self._topk_vals is not None:
                    try:
                        cids_row = self._topk_cids[y, x, ...]
                        vals_row = self._topk_vals[y, x, ...]
                        hit = np.where(cids_row == cid)[0]
                        if hit.size > 0:
                            metric_val = float(vals_row[int(hit[0])])
                    except Exception:
                        metric_val = None

                spec = None
                if self._hsi_data is not None:
                    try:
                        spec = np.asarray(self._hsi_data[y, x, :], dtype=np.float32)
                    except Exception:
                        spec = None

                label_id = self._get_label_id_for_pixel(y, x, cid)  # ← 없으면 나중에 구현

                self._pixel_data.append({
                    "index": len(self._pixel_data),
                    "y": int(y),
                    "x": int(x),
                    "cid": cid,
                    "label_id": label_id,        # ★ 추가
                    "sad": metric_val,           # ★ SAD 별도 필드 (필요하면)
                    "metric": self._metric,
                    "value": metric_val,
                    "spectrum": spec,
                })
                
                pixel_rows.append((int(y), int(x), cid, metric_val))

            # ------- (있으면) 우측 표 채우기 -------
            if self.tablePixels is not None:
                self._filling_pixels_table = True
                try:
                    self.tablePixels.setRowCount(len(pixel_rows))
                    for row, (yy, xx, cid, metric_val) in enumerate(pixel_rows):
                        it_chk = QtWidgets.QTableWidgetItem()
                        it_chk.setFlags((it_chk.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
                        it_chk.setCheckState(Qt.Checked)
                        self.tablePixels.setItem(row, 0, it_chk)

                        it_loc = QtWidgets.QTableWidgetItem(f"({xx},{yy})")
                        it_loc.setFlags(it_loc.flags() & ~Qt.ItemIsEditable)
                        self.tablePixels.setItem(row, 1, it_loc)

                        it_cid = QtWidgets.QTableWidgetItem(str(cid))
                        it_cid.setFlags(it_cid.flags() & ~Qt.ItemIsEditable)
                        self.tablePixels.setItem(row, 2, it_cid)

                        it_metric = QtWidgets.QTableWidgetItem(self._metric)
                        it_metric.setFlags(it_metric.flags() & ~Qt.ItemIsEditable)
                        self.tablePixels.setItem(row, 3, it_metric)

                        val_str = f"{metric_val:.4f}" if metric_val is not None else "-"
                        it_val = QtWidgets.QTableWidgetItem(val_str)
                        it_val.setFlags(it_val.flags() & ~Qt.ItemIsEditable)
                        it_val.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        self.tablePixels.setItem(row, 4, it_val)

                    self.tablePixels.resizeColumnsToContents()
                finally:
                    self._filling_pixels_table = False

            # ------- 트리 재구성 + 스펙트럼 갱신 -------
            self._rebuild_selected_tree()

        except Exception:
            logging.exception("[AnalysisDialog] _update_pixel_table failed")

    def _get_label_id_for_pixel(self, y: int, x: int, cid: int):
        """
        (y,x,cid) 픽셀이 사용한 reference 스펙트럼을 기준으로
        '라벨 ID'를 만들어서 돌려준다.

        - 같은 reference 스펙트럼을 쓰는 픽셀들은 같은 label_id 로 묶인다.
        - 실제 LabelStore의 label_id 가 아니라, 이 다이얼로그 내부에서만
          쓰는 'reference 그룹 번호'라고 보면 된다.
        """
        # reference 정보를 못 쓰면 라벨 구분 안 함
        if not callable(self._match_provider):
            return None
        if self._classmap is None:
            return None

        try:
            # 1) 이 픽셀에서 사용된 reference 스펙트럼 가져오기
            ref = self._match_provider(int(y), int(x), int(cid))
            if ref is None:
                return None

            v = np.asarray(ref, dtype=np.float32).reshape(-1)

            # 2) reference 스펙트럼을 key 로 변환 (TR 쪽에서 쓰는 방식과 동일)
            key = np.round(v, 6).tobytes()

            # 3) key -> label_id 매핑 테이블 (클래스 멤버에 캐시)
            if not hasattr(self, "_ref_label_index"):
                self._ref_label_index = {}      # key -> label_id
                self._ref_label_counter = 1     # 1,2,3,...

            label_id = self._ref_label_index.get(key)
            if label_id is None:
                label_id = self._ref_label_counter
                self._ref_label_index[key] = label_id
                self._ref_label_counter += 1

            return label_id
        except Exception:
            logging.exception("[AnalysisDialog] _get_label_id_for_pixel failed")
            return None



    def _on_pixel_selection_changed(self):
        if getattr(self, "_tl_mode", "tree") == "hist":
            return 
        self._refresh_spectra_from_tree()  # ← 테이블 체크 기반이 아니라 트리 체크 기반을 사용
    
    def _update_spectrum(self):
        if not isinstance(self.spectrumWidgetTL, SpectrumWidget):
            return
        spectra = []
        if self.tablePixels:
            for row in range(self.tablePixels.rowCount()):
                it_chk = self.tablePixels.item(row, 0)
                if it_chk and it_chk.checkState() == Qt.Checked:
                    if 0 <= row < len(self._pixel_data):
                        s = self._pixel_data[row].get("spectrum")
                        if s is not None: spectra.append(s)
        self.spectrumWidgetTL.set_spectra(spectra if spectra else [], self._wavelengths if spectra else None)

    
    def _on_apply_class(self):
        """
        적용: 체크된 픽셀을 콤보박스 Class로 변경하고 다이얼로그 종료.
        좌/우 테이블 갱신, 체크 상태 복원은 생략(즉시 닫힘 가정).
        """
        try:
            # 1) 체크된 좌표
            pixel_coords = self._get_checked_pixel_coords()
            if not pixel_coords:
                QtWidgets.QMessageBox.information(self, "안내", "변경할 픽셀(# 체크)을 선택하세요.")
                return

            # 2) 새 클래스
            data = self.cboSetClass.currentData() if self.cboSetClass else None
            if data is None:
                QtWidgets.QMessageBox.information(self, "안내", "설정할 클래스를 선택하세요.")
                return
            new_cid = int(data)

            # 3) classmap 직접 반영
            if self._classmap is None:
                QtWidgets.QMessageBox.critical(self, "오류", "classmap이 설정되지 않았습니다.")
                return

            ys = np.fromiter((y for (y, _) in pixel_coords), dtype=int, count=len(pixel_coords))
            xs = np.fromiter((x for (_, x) in pixel_coords), dtype=int, count=len(pixel_coords))
            self._classmap[ys, xs] = new_cid

            # 4) (선택) 외부 처리용 시그널: 저장/팔레트/리렌더 등
            self.pixels_class_set.emit(pixel_coords, new_cid)

            # 5) 안내 후 닫기
            QtWidgets.QMessageBox.information(
                self, "완료", f"{len(pixel_coords)}개 픽셀의 클래스를 {new_cid}로 변경했습니다."
            )
            self.accept()

        except Exception:
            logging.exception("[AnalysisDialog] _on_apply_class failed")
            QtWidgets.QMessageBox.critical(self, "오류", "클래스 설정 중 오류가 발생했습니다.")

    def set_palette(self, palette: Dict[int, Any]):
        """외부 팔레트 주입 (cid -> QColor/tuple/#hex)"""
        norm: Dict[int, QColor] = {}
        for k, v in (palette or {}).items():
            cid = int(k)
            if isinstance(v, QColor):
                norm[cid] = v
            elif isinstance(v, (tuple, list)) and len(v) in (3, 4):
                norm[cid] = QColor(*v)
            elif isinstance(v, str):
                norm[cid] = QColor(v)
        self._palette = norm
        
    def _check_all_classes_and_refresh(self):
        if not self.tableClassInfo:
            return
        self._selected_class_ids = []
        for row in range(self.tableClassInfo.rowCount()):
            it = self.tableClassInfo.item(row, 0)
            cid_item = self.tableClassInfo.item(row, 1)
            if it is None or cid_item is None:
                continue
            it.setCheckState(Qt.Checked)
            try:
                cid = int(cid_item.text())
                if cid not in self._selected_class_ids:
                    self._selected_class_ids.append(cid)
            except Exception:
                pass
        # 오른쪽 표/스펙트럼 최신화
        self._update_pixel_table()
        
    def _clear_right_panels(self):
        if self.tablePixels:
            self.tablePixels.setRowCount(0)
        if isinstance(self.spectrumWidgetTL, SpectrumWidget):
            self.spectrumWidgetTL.set_spectra([], None)
        if isinstance(self.spectrumWidgetTR, SpectrumWidget):
            self.spectrumWidgetTR.set_spectra([], None)
        
        # MapView 마커도 제거
        self._clear_mapview_red_markers()
            
    def _on_pixel_checkbox_changed(self, item: QtWidgets.QTableWidgetItem):
        # 테이블 채우는 중이면 무시
        if getattr(self, "_filling_pixels_table", False):
            return
        # 체크박스 열만 처리
        if item.column() != 0:
            return
        # 체크 변경 시 스펙트럼 재계산
        self._update_spectrum()

    def _get_checked_pixel_coords(self) -> List[Tuple[int, int]]:
        """우측 픽셀 테이블에서 체크된 행들의 (y,x) 좌표를 반환"""
        coords: List[Tuple[int, int]] = []
        if not self.tablePixels:
            return coords

        row_cnt = self.tablePixels.rowCount()
        for row in range(row_cnt):
            it_chk = self.tablePixels.item(row, 0)
            if it_chk is None:
                continue
            if it_chk.checkState() == Qt.Checked:
                # _pixel_data는 _update_pixel_table에서 행과 동일한 순서로 적재됨
                if 0 <= row < len(self._pixel_data):
                    coords.append((self._pixel_data[row]["y"], self._pixel_data[row]["x"]))
        return coords
      
    # ★ 지정 영역 분석 버튼 핸들러
    def _on_plus_clicked(self):
        """'+' 버튼 클릭: union 모드"""
        try:
            self._current_op = "union"
            if self.btnMinus:
                self.btnMinus.setChecked(False)
            self._update_analysis_op()
            # ★ 버튼 상태 변경 시 항상 클릭 모드 동기화 (픽셀/사각형 모드에 따라 다르게 동작)
            parent = self.parent()
            if parent and hasattr(parent, "_sync_click_mode_with_analysis_state"):
                QtCore.QTimer.singleShot(10, lambda: parent._sync_click_mode_with_analysis_state())
        except Exception:
            logging.exception("[AnalysisDialog] plus button failed")
    
    def _on_minus_clicked(self):
        """'-' 버튼 클릭: subtract 모드"""
        try:
            self._current_op = "subtract"
            if self.btnPlus:
                self.btnPlus.setChecked(False)
            self._update_analysis_op()
            # ★ 버튼 상태 변경 시 항상 클릭 모드 동기화 (픽셀/사각형 모드에 따라 다르게 동작)
            parent = self.parent()
            if parent and hasattr(parent, "_sync_click_mode_with_analysis_state"):
                QtCore.QTimer.singleShot(10, lambda: parent._sync_click_mode_with_analysis_state())
        except Exception:
            logging.exception("[AnalysisDialog] minus button failed")
    
    def _on_reset_clicked(self):
        """'초기화' 버튼 클릭: 분석 영역 초기화"""
        try:
            parent = self.parent()
            if parent and hasattr(parent, "_on_clear_analysis_region"):
                parent._on_clear_analysis_region()

            self._tl_mode = "tree"
            self._tl_hist_cache.clear()            # 모드 버튼도 해제
            if self.btnPixel:
                self.btnPixel.setChecked(False)
            if self.btnRect:
                self.btnRect.setChecked(False)
            if self.btnFree:
                self.btnFree.setChecked(False)
            self._current_mode = None
            self._stop_analysis_mode()
        except Exception:
            logging.exception("[AnalysisDialog] reset button failed")
    
    def _on_pixel_mode_clicked(self):
        """'픽셀' 버튼 클릭: 픽셀 선택 모드 시작"""
        try:
            if self.btnPixel and self.btnPixel.isChecked():
                self._current_mode = "pixel"
                if self.btnRect:
                    self.btnRect.setChecked(False)
                if self.btnFree:
                    self.btnFree.setChecked(False)
                self._start_analysis_pixel()
                # ★ 클릭 모드 동기화
                parent = self.parent()
                if parent and hasattr(parent, "_sync_click_mode_with_analysis_state"):
                    QtCore.QTimer.singleShot(10, lambda: parent._sync_click_mode_with_analysis_state())
            else:
                # ★ Dialog가 열려있으면 픽셀 버튼을 해제해도 분석 모드 유지 (다른 모드로 전환하지 않음)
                self._current_mode = None
                # "+" 또는 "-" 버튼이 활성화되어 있으면 분석 모드 유지
                if (self.btnPlus and self.btnPlus.isChecked()) or (self.btnMinus and self.btnMinus.isChecked()):
                    # op만 활성화되어 있으면 분석 모드 유지
                    parent = self.parent()
                    if parent and hasattr(parent, "_sync_click_mode_with_analysis_state"):
                        QtCore.QTimer.singleShot(10, lambda: parent._sync_click_mode_with_analysis_state())
                else:
                    self._stop_analysis_mode()
        except Exception:
            logging.exception("[AnalysisDialog] pixel mode button failed")
    
    def _on_rect_mode_clicked(self):
        """'사각형' 버튼 클릭: 사각형 선택 모드 시작"""
        try:
            if self.btnRect and self.btnRect.isChecked():
                self._current_mode = "rect"
                if self.btnPixel:
                    self.btnPixel.setChecked(False)
                if self.btnFree:
                    self.btnFree.setChecked(False)
                self._start_analysis_rect()
                # ★ 클릭 모드 동기화
                parent = self.parent()
                if parent and hasattr(parent, "_sync_click_mode_with_analysis_state"):
                    QtCore.QTimer.singleShot(10, lambda: parent._sync_click_mode_with_analysis_state())
            else:
                # ★ Dialog가 열려있으면 사각형 버튼을 해제해도 분석 모드 유지
                self._current_mode = None
                # "+" 또는 "-" 버튼이 활성화되어 있으면 분석 모드 유지
                if (self.btnPlus and self.btnPlus.isChecked()) or (self.btnMinus and self.btnMinus.isChecked()):
                    # op만 활성화되어 있으면 분석 모드 유지
                    parent = self.parent()
                    if parent and hasattr(parent, "_sync_click_mode_with_analysis_state"):
                        QtCore.QTimer.singleShot(10, lambda: parent._sync_click_mode_with_analysis_state())
                else:
                    self._stop_analysis_mode()
        except Exception:
            logging.exception("[AnalysisDialog] rect mode button failed")
    
    def _on_free_mode_clicked(self):
        """'자유형' 버튼 클릭: 자유형 선택 모드 시작"""
        try:
            if self.btnFree and self.btnFree.isChecked():
                self._current_mode = "free"
                if self.btnPixel:
                    self.btnPixel.setChecked(False)
                if self.btnRect:
                    self.btnRect.setChecked(False)
                # 자유형은 아직 구현되지 않았으므로 안내 메시지
                QtWidgets.QMessageBox.information(self, "안내", "자유형 모드는 아직 구현되지 않았습니다.")
                self.btnFree.setChecked(False)
                self._current_mode = None
            else:
                self._current_mode = None
                self._stop_analysis_mode()
        except Exception:
            logging.exception("[AnalysisDialog] free mode button failed")
    
    def _update_analysis_op(self):
        """MainWindow에 op 변경 요청"""
        try:
            parent = self.parent()
            if parent and hasattr(parent, "_set_analysis_op"):
                parent._set_analysis_op(self._current_op)
            elif parent and hasattr(parent, "analysisCtrl") and parent.analysisCtrl:
                parent.analysisCtrl.set_op(self._current_op)
        except Exception:
            logging.exception("[AnalysisDialog] update analysis op failed")
    
    def _start_analysis_rect(self):
        """MainWindow에 사각형 모드 시작 요청"""
        try:
            parent = self.parent()
            if parent and hasattr(parent, "_on_start_analysis_rect_from_dock"):
                parent._on_start_analysis_rect_from_dock()
            elif parent and hasattr(parent, "analysisCtrl") and parent.analysisCtrl:
                parent.analysisCtrl.start_rect_selection()
        except Exception:
            logging.exception("[AnalysisDialog] start analysis rect failed")
    
    def _start_analysis_pixel(self):
        """MainWindow에 픽셀 모드 시작 요청"""
        try:
            parent = self.parent()
            if parent and hasattr(parent, "_on_start_analysis_pixel"):
                parent._on_start_analysis_pixel()
            elif parent and hasattr(parent, "analysisCtrl") and parent.analysisCtrl:
                # 픽셀 모드는 click mode를 변경해야 함
                if hasattr(parent, "_sync_click_mode_with_analysis_state"):
                    parent._sync_click_mode_with_analysis_state()
        except Exception:
            logging.exception("[AnalysisDialog] start analysis pixel failed")
    
    def _stop_analysis_mode(self):
        """MainWindow에 분석 모드 종료 요청"""
        try:
            parent = self.parent()
            if parent and hasattr(parent, "_on_analysis_stop"):
                parent._on_analysis_stop()
            elif parent and hasattr(parent, "analysisCtrl") and parent.analysisCtrl:
                parent.analysisCtrl.cancel()
        except Exception:
            logging.exception("[AnalysisDialog] stop analysis mode failed")

    def _rebuild_selected_tree(self):
        if not (self.twSelected and self._pixel_data):
            if self.twSelected:
                self.twSelected.clear()
            return

        # 1) cid / label_id 기준으로 2단계 그룹핑
        by_cls_label: Dict[int, Dict[Any, list[int]]] = {}
        for i, rec in enumerate(self._pixel_data):
            y, x = int(rec["y"]), int(rec["x"])
            live_cid = int(self._classmap[y, x]) if self._classmap is not None else int(rec.get("cid", -1))
            rec["cid"] = live_cid

            label_id = rec.get("label_id", None)  # 없으면 None/기본값
            by_cls_label.setdefault(live_cid, {}).setdefault(label_id, []).append(i)

        self.twSelected.blockSignals(True)
        try:
            self.twSelected.clear()

            for cid in sorted(by_cls_label.keys()):
                name = self._id_to_name.get(cid, str(cid)) if self._id_to_name else str(cid)

                # ✅ 최상위(Class) 노드
                top_text = f"Class {cid} ({name})"
                top = QtWidgets.QTreeWidgetItem([top_text])
                top.setToolTip(0, top_text)  # ✅ 길면 툴팁으로 전체 표시
                top.setFlags(top.flags() | Qt.ItemIsUserCheckable)
                top.setCheckState(0, Qt.Checked)
                top.setData(0, Qt.UserRole + 1, Qt.Checked)

                if self._palette and (cid in self._palette):
                    top.setForeground(0, QtGui.QBrush(self._palette[cid]))

                # 2) Label 단위 서브 노드
                label_map = by_cls_label[cid]

                # ★ 1) label_id마다 평균 SAD를 미리 계산
                label_stats = []   # (label_id, idx_list, mean_sim)
                for label_id, idx_list in label_map.items():
                    vals = []
                    for idx in idx_list:
                        v = self._pixel_data[idx].get("sad", self._pixel_data[idx].get("value"))
                        if v is not None:
                            vals.append(float(v))
                    mean_sim = float(np.mean(vals)) if vals else None
                    label_stats.append((label_id, idx_list, mean_sim))

                # ★ 2) 평균 SAD 기준 오름차순 정렬 (None 은 맨 뒤)
                label_stats.sort(key=lambda t: (t[2] is None, t[2]))

                # ★ 3) 정렬된 순서로 Label 노드 생성
                for (label_id, idx_list, mean_sim) in label_stats:
                    if label_id is None:
                        label_prefix = "Label ?"
                    else:
                        label_prefix = f"Label {label_id}"

                    if mean_sim is not None:
                        label_text = f"{label_prefix} ({len(idx_list)}, {mean_sim:.2f})"
                    else:
                        label_text = f"{label_prefix} ({len(idx_list)}, -)"

                    label_item = QtWidgets.QTreeWidgetItem([label_text])
                    label_item.setToolTip(0, label_text)  # ✅ 툴팁
                    label_item.setFlags(label_item.flags() | Qt.ItemIsUserCheckable)
                    label_item.setCheckState(0, Qt.Checked)
                    label_item.setData(0, Qt.UserRole + 1, Qt.Checked)

                    if self._palette and (cid in self._palette):
                        c = QtGui.QColor(self._palette[cid]); c.setAlpha(220)
                        label_item.setForeground(0, QtGui.QBrush(c))

                    # 3) 픽셀 단위 leaf — "SAD : 0.076 (x,y)" 형식
                    def _pixel_sad(idx: int) -> float:
                        v = self._pixel_data[idx].get("sad", self._pixel_data[idx].get("value"))
                        if v is None:
                            return float("inf")
                        try:
                            return float(v)
                        except Exception:
                            return float("inf")

                    sorted_idx_list = sorted(idx_list, key=_pixel_sad)

                    # ★ 4) 정렬된 순서대로 픽셀 노드 생성
                    for idx in sorted_idx_list:
                        rec = self._pixel_data[idx]
                        y, x = int(rec["y"]), int(rec["x"])
                        v = rec.get("sad", rec.get("value"))

                        if v is not None:
                            txt = f"{float(v):.2f} ({x},{y})"   # 표시용은 (x, y)
                        else:
                            txt = f"({x},{y})"

                        ch = QtWidgets.QTreeWidgetItem([txt])
                        ch.setToolTip(0, txt)  # ✅ 툴팁
                        ch.setFlags(ch.flags() | Qt.ItemIsUserCheckable)
                        ch.setCheckState(0, Qt.Checked)
                        ch.setData(0, Qt.UserRole + 1, Qt.Checked)
                        ch.setData(0, Qt.UserRole, int(idx))

                        if self._palette and (cid in self._palette):
                            c_child = QtGui.QColor(self._palette[cid]); c_child.setAlpha(200)
                            ch.setForeground(0, QtGui.QBrush(c_child))

                        label_item.addChild(ch)

                    top.addChild(label_item)

                self.twSelected.addTopLevelItem(top)

            self.twSelected.expandAll()

        finally:
            self.twSelected.blockSignals(False)

        self._refresh_spectra_from_tree()

                
    def _on_selected_tree_changed(self, item: QtWidgets.QTreeWidgetItem, col: int):
        """
        좌측 트리(twSelected)의 체크박스/항목 상태가 바뀌었을 때 호출된다.

        동작 요약
        ----------
        1) 프로그램적으로 발생한 변경(self._suppress_tree_item_changed=True)은 무시.
        2) 체크박스가 아닌 변화는 무시.
        3) 이전 체크 상태(Qt.UserRole+1에 저장)와 현재 checkState를 비교해 '실제 변경'만 처리.
        4) 부모(클래스) 항목이면 모든 '라벨 + 픽셀' 자식 체크 상태를 동일하게 동기화.
        5) 라벨 항목이면 그 아래 픽셀 자식들의 체크 상태를 동일하게 동기화.
        6) 히스토그램 주도 모드였다면 트리 주도 모드로 전환(self._tl_mode="tree").
        7) 트리 주도 모드에서는 TL/TR 표시를 트리 체크 결과로 갱신.
        8) 언제나 최종적으로 '체크된 픽셀 집합'을 기준으로 TL/TR/Histogram 전체를 동기화.
        """
        # 1) 프로그램적 변경 무시
        if getattr(self, "_suppress_tree_item_changed", False):
            return

        try:
            # 2) 체크박스가 아닌 변화는 무시
            if not (item.flags() & Qt.ItemIsUserCheckable):
                return

            # 3) '실제 변경'만 처리 (직전 상태는 Qt.UserRole+1에 저장해 둠)
            prev = item.data(0, Qt.UserRole + 1)
            curr = item.checkState(0)
            if prev == curr:
                return
            item.setData(0, Qt.UserRole + 1, curr)

            # 4) 최상위(클래스) 항목이면 → 아래 라벨/픽셀 전부 동일 상태로 전파
            if item.parent() is None:
                st = curr
                self._suppress_tree_item_changed = True
                try:
                    for i in range(item.childCount()):
                        label_item = item.child(i)
                        if not label_item:
                            continue
                        # 라벨 노드 체크 동기화
                        label_item.setCheckState(0, st)
                        label_item.setData(0, Qt.UserRole + 1, st)

                        # 라벨 아래 픽셀 노드들까지 동기화
                        for j in range(label_item.childCount()):
                            ch = label_item.child(j)
                            if not ch:
                                continue
                            ch.setCheckState(0, st)
                            ch.setData(0, Qt.UserRole + 1, st)
                finally:
                    self._suppress_tree_item_changed = False

            # 5) 라벨 항목이면 → 그 아래 픽셀들만 동일 상태로 전파
            elif item.parent() is not None and item.childCount() > 0:
                st = curr
                self._suppress_tree_item_changed = True
                try:
                    for j in range(item.childCount()):
                        ch = item.child(j)
                        if not ch:
                            continue
                        ch.setCheckState(0, st)
                        ch.setData(0, Qt.UserRole + 1, st)
                finally:
                    self._suppress_tree_item_changed = False

            # 픽셀 leaf(자식 없음)는 전파 대상 아님 → 위에서 curr만 갱신하고 끝

        except Exception:
            logging.exception("[AnalysisDialog] tree itemChanged failed")

        # 6) 히스토그램 주도 모드였다면 트리 주도 모드로 전환
        if getattr(self, "_tl_mode", "tree") == "hist":
            self._tl_mode = "tree"

        # 7) 트리 주도 모드에서는 TL(TR 내부 로직 포함)을 트리 체크 결과로 갱신
        if self._tl_mode == "tree":
            self._refresh_spectra_from_tree()

        # 8) 언제나 최종적으로 '체크된 픽셀 집합' 기준으로 TL/TR/Histogram 동기화
        #    (요구사항: 왼쪽에 체크된 픽셀들 전체를 기준으로 모든 패널을 갱신)
        if hasattr(self, "_refresh_all_from_checked"):
            self._refresh_all_from_checked()

    def _refresh_spectra_from_tree(self):
        """
        TL(좌상단): 트리에서 체크된 '픽셀(child of label)'들의 원본 스펙 표시
        TR(우상단): 체크 집합 기준 reference 표시 (기존 로직 재사용)
        """
        if not self.twSelected:
            return

        # ----- TL: 체크된 child(픽셀) 수집 → 개별 스펙 표시 -----
        checked_indices: List[int] = []
        class_groups: Dict[int, List[np.ndarray]] = {}

        root_count = self.twSelected.topLevelItemCount()
        for r in range(root_count):
            top = self.twSelected.topLevelItem(r)
            if top is None:
                continue
            cid = self._cid_from_top_item_text(top.text(0))

            # ★ 1단계: Label 노드들
            for i in range(top.childCount()):
                label_item = top.child(i)
                if label_item is None:
                    continue

                # ★ 2단계: Pixel 노드들
                for j in range(label_item.childCount()):
                    ch = label_item.child(j)
                    if ch is None:
                        continue
                    if ch.checkState(0) != Qt.Checked:
                        continue

                    idx = ch.data(0, Qt.UserRole)
                    if isinstance(idx, int) and 0 <= idx < len(self._pixel_data):
                        checked_indices.append(idx)
                        spec = self._pixel_data[idx].get("spectrum")
                        if spec is not None:
                            class_groups.setdefault(cid, []).append(
                                np.asarray(spec, dtype=np.float32)
                            )

        # TL: 개별 스펙 라인들
        specs_tl = []
        metas_tl = []
        for i in checked_indices:
            spec = self._pixel_data[i].get("spectrum")
            if spec is not None:
                specs_tl.append(spec)
                y = self._pixel_data[i].get("y")
                x = self._pixel_data[i].get("x")
                cid_i = self._pixel_data[i].get("cid", -1)
                metas_tl.append(
                    {"y": int(y), "x": int(x), "cid": int(cid_i)}
                    if y is not None and x is not None
                    else None
                )

        if isinstance(self.spectrumWidgetTL, SpectrumWidget):
            colors = []
            if self._palette and specs_tl:
                for i in checked_indices:
                    cid_i = int(self._pixel_data[i]["cid"])
                    colors.append(self._palette.get(cid_i, QtGui.QColor(100, 150, 255)))
            self.spectrumWidgetTL.set_spectra(
                specs_tl, self._wavelengths, colors, metas=metas_tl
            )

        # ----- TR: 체크된 픽셀들이 사용한 reference 표시 -----
        if isinstance(self.spectrumWidgetTR, SpectrumWidget):
            self._update_tr_refs_from_checked(None)

        # 지도 오버레이 동기화
        self._update_mapview_red_markers()

    def _cid_from_top_item_text(self, text: str) -> int:
        # "Class 17 (Name)" → 17 추출
        try:
            # 앞 2단어가 "Class", "<cid>"
            parts = text.split()
            for p in parts:
                try:
                    return int(p)
                except Exception:
                    pass
        except Exception:
            pass
        return -1

    def _update_class_layer(self):
        """
        self._mask(True인 영역)와 self._classmap, self._hsi_data, (옵션)topk 값을 바탕으로
        self._pixel_data를 구성하고, '선택된 픽셀' 트리를 Class → (y,x) 구조로 만든다.
        """
        if self._mask is None or self._classmap is None:
            if self.twSelected: self.twSelected.clear()
            self._pixel_data = []
            return

        try:
            m = np.asarray(self._mask, dtype=bool)
            H, W = self._classmap.shape[:2]
            if m.shape != (H, W) or not np.any(m):
                if self.twSelected: self.twSelected.clear()
                self._pixel_data = []
                return

            ys, xs = np.where(m)
            pixel_data = []
            for y, x in zip(ys.tolist(), xs.tolist()):
                cid = int(self._classmap[y, x])
                metric_val = None
                if self._topk_cids is not None and self._topk_vals is not None:
                    try:
                        cids_row = self._topk_cids[y, x, ...]
                        vals_row = self._topk_vals[y, x, ...]
                        hit = np.where(cids_row == cid)[0]
                        if hit.size > 0:
                            metric_val = float(vals_row[int(hit[0])])
                    except Exception:
                        metric_val = None

                spec = None
                if self._hsi_data is not None:
                    try:
                        spec = np.asarray(self._hsi_data[y, x, :], dtype=np.float32)
                    except Exception:
                        spec = None

                pixel_data.append({
                    "y": int(y), "x": int(x), "cid": cid,
                    "metric": self._metric, "value": metric_val,
                    "spectrum": spec,
                })

            # SSOT에 저장
            self._pixel_data = pixel_data

            # 트리로 뿌리기
            self._rebuild_selected_tree_from_pixel_data()

        except Exception:
            logging.exception("[AnalysisDialog] _update_class_layer failed")
            if self.twSelected: self.twSelected.clear()
            self._pixel_data = []
            
    def set_label_provider(self, provider):
        """
        provider: 
        - callable(cid) -> List[np.ndarray]  (각 원소 shape=(C,))
        - 또는 dict {cid: List[np.ndarray]}
        """
        self._label_provider = provider

    def set_spectrum_library_provider(self, provider):
        """
        스펙트럼 라이브러리 provider 설정
        provider: 
        - callable(cid) -> List[np.ndarray]  (각 원소 shape=(C,))
        - 또는 dict {cid: List[np.ndarray]}
        """
        self._spectrum_library_provider = provider

    # === 히스토그램 관련 메서드 ===
    
    def _on_intermediate_target_changed(self, text: str):
        """대상 분류 맵 변경 시 (classification/probability)"""
        if text == "classification":
            # classification 모드일 때만 히스토그램 업데이트
            self._update_histogram()
        # probability는 아직 구현하지 않음
            
    def _on_class_range_changed(self, idx: int):
        """히스토그램 클래스 콤보가 바뀌었을 때"""
        logging.info(
            "[_on_class_range_changed] CALLED: idx=%s, cbClassRange=%s, objectName=%s",
            idx,
            "None" if self.cbClassRange is None else "Found",
            self.cbClassRange.objectName() if self.cbClassRange else "N/A"
        )
        
        if self.cbClassRange is None:
            logging.warning("[_on_class_range_changed] cbClassRange is None, returning")
            return

        if idx is None or idx < 0:
            # 선택 해제 → 히스토그램 초기화
            logging.info("[class_range] cleared (idx=%r)", idx)
            self._apply_histogram_class_selection(None, source="combo-change")
            return

        data = self.cbClassRange.itemData(idx)
        try:
            cid = int(data) if data is not None else None
        except Exception:
            cid = None
            logging.warning(
                "[class_range] Failed to parse itemData: idx=%s, data=%r",
                idx, data
            )

        # itemData가 None이면 텍스트에서 추출 시도
        if cid is None:
            text = self.cbClassRange.itemText(idx)
            logging.warning(
                "[class_range] itemData is None, trying to parse from text: %r",
                text
            )
            # "Class {cid} ({name})" 형식에서 추출
            import re
            match = re.search(r'Class\s+(\d+)', text)
            if match:
                try:
                    cid = int(match.group(1))
                    logging.info("[class_range] Extracted CID from text: %s", cid)
                except Exception:
                    pass

        logging.info(
            "[class_range] changed: idx=%s, text=%r, cid=%r",
            idx,
            self.cbClassRange.itemText(idx),
            cid,
        )

        if cid is None:
            logging.warning("[class_range] CID is None after parsing, histogram may not update")

        # ✅ 콤보 변경 시에도 항상 동일 경로 사용
        self._apply_histogram_class_selection(cid, source="combo-change")

        # 포커스/지도 마커 초기화
        self._focused_point = None
        self._selected_ref_keys = set()
        self._reset_tw_selected_colors()
        self._clear_mapview_red_markers()
        
        # 체크된 클래스들의 TL/TR 그래프만 업데이트 (체크 상태는 변경하지 않음)
        # _refresh_spectra_from_tree()는 현재 트리의 체크 상태를 읽어서 그래프만 업데이트
        if hasattr(self, '_refresh_spectra_from_tree'):
            self._refresh_spectra_from_tree()

    def _apply_histogram_class_selection(self, cid: Optional[int], *, source: str = "combo"):
        logging.info("[hist_apply] source=%s, cid=%r", source, cid)
        self._current_hist_cid = (None if (cid is not None and cid < 0) else cid)

        # 콤보가 유효하면 반드시 'combo' 모드 고정
        self._hist_mode = "combo" if self._current_hist_cid is not None else "checked"

        # 캐시 전체 초기화(간헐적 덮어쓰기 방지)
        if not hasattr(self, "_hist_cache") or not isinstance(self._hist_cache, dict):
            self._hist_cache = {}
        self._hist_cache.update({
            "edges": None, "vals": None, "indices": None,
            "selected_bin": None, "selected_mask": None
        })

        # 위젯 보장 후 즉시 재그림
        self._ensure_hist_widget()
        self._update_histogram()

        # TL/TR 갱신과 충돌하지 않도록 모드 고정
        self._tl_mode = "tree"  # TL는 트리 기준 그대로 두고, 히스토그램만 콤보 기준 유지

    def _on_bins_changed(self, value: int):
        """히스토그램 간격 변경 시"""
        self._update_histogram()
        
    def _update_histogram(self):
        # ★ 콤보박스를 진실 소스로 사용하되, 이미 _current_hist_cid가 설정되어 있으면 우선 사용
        if hasattr(self, "cbClassRange") and self.cbClassRange is not None:
            # _current_hist_cid가 없거나 유효하지 않을 때만 콤보박스에서 읽기
            if self._current_hist_cid is None or self._current_hist_cid < 0:
                data = self.cbClassRange.currentData()
                try:
                    cid_from_combo = int(data) if data is not None else None
                except Exception:
                    cid_from_combo = None
                if cid_from_combo is not None and cid_from_combo >= 0:
                    self._current_hist_cid = cid_from_combo
                    logging.info("[update_histogram] Read CID from combo: %s", cid_from_combo)
            else:
                # 이미 설정된 CID가 있으면 콤보박스와 동기화 확인 (디버깅용)
                combo_cid = None
                try:
                    data = self.cbClassRange.currentData()
                    combo_cid = int(data) if data is not None else None
                except Exception:
                    pass
                if combo_cid is not None and combo_cid != self._current_hist_cid:
                    logging.warning(
                        "[update_histogram] CID mismatch: combo=%s, current=%s, using current",
                        combo_cid, self._current_hist_cid
                    )

        logging.info(
            "[AnalysisDialog] update_histogram start: cid=%s, mask_set=%s, topk_set=%s, mode=%s",
            self._current_hist_cid,
            self._mask is not None,
            self._topk_vals is not None,
            getattr(self, "_hist_mode", "unknown"),
        )

        # matplotlib/위젯 체크
        if not MATPLOTLIB_AVAILABLE or self.histogramWidget is None:
            return

        fig = self.histogramWidget.figure
        fig.clear()
        ax = fig.add_subplot(111)

        # 필수 데이터 체크
        if self._classmap is None or self._mask is None:
            logging.warning(
                "[AnalysisDialog] update_histogram aborted: classmap_set=%s mask_set=%s",
                self._classmap is not None,
                self._mask is not None,
            )
            ax.text(0.5, 0.5, 'No classmap/mask', ha='center', va='center', transform=ax.transAxes, color='red')
            self.histogramWidget.draw()
            return

        if self._current_hist_cid is None or self._current_hist_cid < 0:
            logging.info("[AnalysisDialog] update_histogram aborted: current_hist_cid=%s", self._current_hist_cid)
            ax.text(0.5, 0.5, 'No class selected', ha='center', va='center', transform=ax.transAxes, color='red')
            self.histogramWidget.draw()
            return

        if self._topk_vals is None:
            logging.warning("[AnalysisDialog] update_histogram aborted: topk_vals is None")
            ax.text(0.5, 0.5, 'No top-k values (topk_vals=None)', ha='center', va='center', transform=ax.transAxes, color='red')
            self.histogramWidget.draw()
            return

        if self._topk_vals.ndim != 3 or self._topk_vals.shape[2] < 1:
            logging.warning(
                "[AnalysisDialog] update_histogram aborted: invalid topk_vals shape=%s",
                getattr(self._topk_vals, "shape", None),
            )
            ax.text(0.5, 0.5, f'Invalid topk_vals shape={self._topk_vals.shape}', ha='center', va='center', transform=ax.transAxes, color='red')
            self.histogramWidget.draw()
            return

        if (self._topk_vals.shape[0] != self._classmap.shape[0]) or (self._topk_vals.shape[1] != self._classmap.shape[1]):
            logging.warning(
                "[AnalysisDialog] update_histogram aborted: topk_shape=%s classmap_shape=%s",
                self._topk_vals.shape[:2],
                self._classmap.shape[:2],
            )
            ax.text(
                0.5, 0.5,
                f'shape mismatch: topk={self._topk_vals.shape[:2]} vs classmap={self._classmap.shape[:2]}',
                ha='center', va='center', transform=ax.transAxes, color='red'
            )
            self.histogramWidget.draw()
            return

        try:
            # 기준 클래스의 마스크 내부 픽셀 좌표
            m = np.asarray(self._mask, dtype=bool)
            sel = (self._classmap == int(self._current_hist_cid)) & m
            ys, xs = np.where(sel)

            if ys.size == 0:
                logging.info(
                    "[AnalysisDialog] update_histogram: class %s has zero pixels in mask",
                    self._current_hist_cid,
                )
                ax.text(0.5, 0.5, 'No pixels for selected class', ha='center', va='center', transform=ax.transAxes)
                self.histogramWidget.draw()
                if not hasattr(self, "_hist_cache"):
                    self._hist_cache = {}
                self._hist_cache.update({
                    "edges": None, "vals": None, "indices": None,
                    "selected_bins": set(), "selected_mask": None
                })
                return

            # 해당 좌표들의 top-1 값 수집
            vals, idx = [], []
            H, W = self._topk_vals.shape[:2]
            for y, x in zip(ys.tolist(), xs.tolist()):
                if 0 <= y < H and 0 <= x < W:
                    v = self._topk_vals[y, x, 0]
                    if np.isfinite(v):
                        vals.append(float(v))
                        idx.append((int(y), int(x)))

            if not vals:
                logging.info(
                    "[AnalysisDialog] update_histogram: class %s has no valid top-1 values",
                    self._current_hist_cid,
                )
                ax.text(0.5, 0.5, 'No valid Top-1 values', ha='center', va='center', transform=ax.transAxes)
                self.histogramWidget.draw()
                if not hasattr(self, "_hist_cache"):
                    self._hist_cache = {}
                self._hist_cache.update({
                    "edges": None, "vals": None, "indices": None,
                    "selected_bins": set(), "selected_mask": None
                })
                return

            # bin 개수: 슬라이더 우선
            bins = 50
            if self.sldBins:
                try:
                    bins = int(self.sldBins.value())
                except Exception:
                    pass

            # 히스토그램 그리기
            counts, edges, patches = ax.hist(vals, bins=bins, edgecolor='black', alpha=0.85)
            ax.set_xlabel(f"Similarity ({self._metric})")
            ax.set_ylabel("Count")
            cname = self._id_to_name.get(int(self._current_hist_cid), str(int(self._current_hist_cid))) if self._id_to_name else str(int(self._current_hist_cid))
            ax.set_title(f"Class {self._current_hist_cid} ({cname}) - Top-1 Histogram")
            ax.grid(True, alpha=0.3, linestyle="--")
            fig.tight_layout()
            self.histogramWidget.draw()

            # 캐시 갱신 (히스토그램 클릭용)
            if not hasattr(self, "_hist_cache"):
                self._hist_cache = {}
            self._hist_cache.update({
                "edges": np.asarray(edges, dtype=float),
                "vals": np.asarray(vals, dtype=float),
                "indices": np.asarray(idx, dtype=int),  # (N,2) (y,x)
                "selected_bins": set(),                 # 여러 bin 선택 지원
                "selected_mask": None
            })

            # 기존에 선택된 bin(들)이 있었다면 재강조 (필요 시)
            try:
                selected_bins = self._hist_cache.get("selected_bins", set()) or set()
                if selected_bins:
                    self._highlight_hist_bin(selected_bins)
            except Exception:
                pass

        except Exception:
            logging.exception("[AnalysisDialog] _update_histogram failed")
            ax.text(0.5, 0.5, 'Error updating histogram', ha='center', va='center', transform=ax.transAxes, color='red')
            self.histogramWidget.draw()

    def _on_hist_click(self, event):
        """
        히스토그램 bin 클릭 →
        1) 캐시가 없으면 현재 클래스 기준으로 히스토그램 1회 갱신
        2) 클릭된 bin 인덱스를 (단일 / 다중) 선택 집합에 반영
        3) 선택된 bin 전체에 속하는 좌표 집합 points 계산
        4) points 기준으로 TL/TR/Histogram/트리/지도 동기화
        """
        try:
            if self.histogramWidget is None or event.inaxes is None:
                return

            # 1) 캐시 보장
            cache = getattr(self, "_hist_cache", None)
            if not cache or cache.get("edges") is None:
                self._update_histogram()
                cache = getattr(self, "_hist_cache", None)
                if not cache:
                    return

            edges = cache.get("edges")
            vals  = cache.get("vals")
            idx2d = cache.get("indices")
            if edges is None or vals is None or idx2d is None:
                return

            # 2) 클릭된 bin 인덱스
            x = event.xdata
            if x is None:
                return
            j = np.searchsorted(edges, x, side="right") - 1
            if j < 0 or j >= len(edges) - 1:
                return

            # 2-1) 현재 modifier 확인 (Ctrl/Shift → 다중 선택)
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            multi = bool(modifiers & (Qt.ControlModifier | Qt.ShiftModifier))

            # 2-2) 기존 선택 집합
            selected_bins = cache.get("selected_bins", set()) or set()
            selected_bins = set(int(b) for b in selected_bins)

            # 2-3) 새 선택 집합 구성
            if multi:
                # Ctrl/Shift: 토글
                if j in selected_bins:
                    selected_bins.remove(j)
                else:
                    selected_bins.add(j)
            else:
                # 일반 클릭: 이 bin 하나만
                selected_bins = {j}

            # 선택이 하나도 없으면 전체 해제
            if not selected_bins:
                cache["selected_bins"] = set()
                cache["selected_mask"] = None
                self._highlight_hist_bin(None)
                self._clear_mapview_red_markers()
                return

            # 3) 선택된 bin 전체에 대한 mask 계산 (union)
            selected_mask = np.zeros_like(vals, dtype=bool)
            for b in selected_bins:
                left, right = float(edges[b]), float(edges[b + 1])
                if b == len(edges) - 2:
                    mask_b = (vals >= left) & (vals <= right)
                else:
                    mask_b = (vals >= left) & (vals < right)
                selected_mask |= mask_b

            points: list[tuple[int, int]] = []
            if np.any(selected_mask):
                sel_idx = idx2d[selected_mask]  # (M,2)
                for k in range(sel_idx.shape[0]):
                    points.append((int(sel_idx[k, 0]), int(sel_idx[k, 1])))

            if not points:
                cache["selected_bins"] = set()
                cache["selected_mask"] = None
                self._highlight_hist_bin(None)
                self._clear_mapview_red_markers()
                return

            # 4) 공통 동기화
            self._focus_source = 'bin'
            cache["selected_bins"] = selected_bins
            cache["selected_mask"] = selected_mask

            # TL/TR/Histogram/지도 전체를 points 기준으로 동기화
            self._apply_focus_from_points(points)

            # bin 강조
            self._highlight_hist_bin(selected_bins)

        except Exception:
            logging.exception("[AnalysisDialog] _on_hist_click failed")


    def _highlight_hist_bin(self, bins: Optional[Any]):
        """
        선택된 bin(들)을 강조.
        - bins: None 이면 모두 기본색
                int 이면 해당 bin만
                Iterable[int] 이면 여러 bin 강조
        """
        if self.histogramWidget is None:
            return
        try:
            fig = self.histogramWidget.figure
            if fig is None or not fig.axes:
                return
            ax = fig.axes[0]

            # 선택 집합 정규화
            if bins is None:
                selected = set()
            elif isinstance(bins, (list, tuple, set, np.ndarray)):
                selected = {int(b) for b in bins}
            else:
                selected = {int(bins)}

            # 팔레트 기본색
            color_base = (0.5, 0.5, 0.8)
            color_sel  = (1.0, 0.6, 0.2)  # 강조색(오렌지)

            patches = ax.patches
            if not patches:
                return

            for i, p in enumerate(patches):
                if i in selected:
                    p.set_facecolor(color_sel)
                    p.set_alpha(0.95)
                else:
                    p.set_facecolor(color_base)
                    p.set_alpha(0.8)

            # 캐시에도 저장
            if hasattr(self, "_hist_cache") and isinstance(self._hist_cache, dict):
                self._hist_cache["selected_bins"] = set(selected)

            self.histogramWidget.draw()
        except Exception:
            logging.exception("[AnalysisDialog] _highlight_hist_bin failed")


    def _on_tl_line_picked(self, meta: dict, event, multi: bool):
        """
        TL 라인 클릭:
        1) 클릭 픽셀 (y,x) 기준으로 BR, TR, Histogram, 트리/지도 동기화
        2) TL: 클릭/멀티선택 그대로 유지(빨간 라인)
        """
        try:
            if not isinstance(meta, dict):
                return
            y, x = int(meta.get("y", -1)), int(meta.get("x", -1))
            if y < 0 or x < 0 or self._classmap is None:
                return

            # TL 메타가 비어 있으면 먼저 채움(콜백도 다시 물림)
            if isinstance(self.spectrumWidgetTL, SpectrumWidget):
                metas = getattr(self.spectrumWidgetTL, "_metas", []) or []
                if not metas:
                    self._refresh_all_from_checked()
                    if getattr(self.spectrumWidgetTL, "on_pick", None) is not self._on_tl_line_picked:
                        self.spectrumWidgetTL.on_pick = self._on_tl_line_picked

            # 포커스 저장
            self._focused_point = (y, x)

            # 현재 TL에서 선택된 모든 픽셀 → points로 수집
            pts = []
            metas = getattr(self.spectrumWidgetTL, "_metas", []) or []
            sel   = getattr(self.spectrumWidgetTL, "_selected_indices", []) or []
            for i in sel:
                if 0 <= i < len(metas) and isinstance(metas[i], dict):
                    yy, xx = metas[i].get("y"), metas[i].get("x")
                    if yy is not None and xx is not None:
                        pts.append((int(yy), int(xx)))
            if not pts:
                pts = [(y, x)]

            # ✅ TL / TR / Histogram / BR / Layer / MapView를 한 번에 동기화
            self._focus_source = 'tl'
            self._apply_focus_from_points(pts)

        except Exception:
            logging.exception("[AnalysisDialog] _on_tl_line_picked failed")


    def _on_tr_line_picked(self, meta: dict, event, multi: bool):
        """
        클릭한 reference를 쓰는 '체크집합 내' 모든 픽셀 → points로 만들고 그걸 기준으로 동기화
        - TL: 해당 reference를 쓰는 픽셀들 스펙트럼 빨간색
        - Histogram: 이 픽셀들의 top1 값이 들어가는 bin 강조
        """
        try:
            if not isinstance(meta, dict):
                return
            ref = meta.get("ref_spec")
            if ref is None or not callable(self._match_provider) or self._classmap is None:
                return
            clicked_key = np.round(np.asarray(ref, dtype=np.float32).reshape(-1), 6).tobytes()

            # 1) 체크 집합에서 동일 reference를 쓰는 픽셀 모두 수집
            pts = []
            for (y, x) in self._get_checked_points():
                try:
                    cid = int(self._classmap[int(y), int(x)])
                    r = self._match_provider(int(y), int(x), cid)
                    if r is None:
                        continue
                    key = np.round(np.asarray(r, dtype=np.float32).reshape(-1), 6).tobytes()
                    if key == clicked_key:
                        pts.append((int(y), int(x)))
                except Exception:
                    continue

            if not pts:
                return

            # 2) 기준 클래스(최빈값)로 히스토그램 클래스 설정
            try:
                from collections import Counter
                cids = [int(self._classmap[y, x]) for (y, x) in pts]
                if cids:
                    major_cid, _ = Counter(cids).most_common(1)[0]
                    import time
                    if time.time() >= getattr(self, "_hist_user_locked_until", 0.0):
                        self._set_hist_class_combo_to(int(major_cid), update_hist=True)
            except Exception:
                pass

            # 3) TL/TR/Histogram/지도 동기화 (공통 헬퍼 사용)
            self._focus_source = 'tr'
            self._apply_focus_from_points(pts)

        except Exception:
            logging.exception("[AnalysisDialog] _on_tr_line_picked failed")

    def _reset_tw_selected_colors(self):
        """트리 항목 색을 팔레트(또는 기본값)로 되돌림"""
        if not self.twSelected:
            return

        root_cnt = self.twSelected.topLevelItemCount()
        for r in range(root_cnt):
            top = self.twSelected.topLevelItem(r)
            if not top:
                continue

            # 1) Class 노드 색 복원
            cid = self._cid_from_top_item_text(top.text(0))
            if self._palette and (cid in self._palette):
                top.setForeground(0, QtGui.QBrush(self._palette[cid]))
            else:
                top.setForeground(0, QtGui.QBrush(QtGui.QColor(Qt.black)))

            # 2) Label 노드 + Pixel 노드 색 복원
            for i in range(top.childCount()):
                label_item = top.child(i)
                if not label_item:
                    continue

                # Label 노드 색
                if self._palette and (cid in self._palette):
                    c_label = QtGui.QColor(self._palette[cid])
                    c_label.setAlpha(220)
                    label_item.setForeground(0, QtGui.QBrush(c_label))
                else:
                    label_item.setForeground(0, QtGui.QBrush(QtGui.QColor(Qt.darkGray)))

                # Pixel 노드 색
                for j in range(label_item.childCount()):
                    leaf = label_item.child(j)
                    if not leaf:
                        continue
                    if self._palette and (cid in self._palette):
                        c_child = QtGui.QColor(self._palette[cid])
                        c_child.setAlpha(200)
                        leaf.setForeground(0, QtGui.QBrush(c_child))
                    else:
                        leaf.setForeground(0, QtGui.QBrush(QtGui.QColor(Qt.darkGray)))

    def _mark_tree_selection(self, points: List[Tuple[int,int]]):
        """지정 좌표 리스트를 트리에서 빨간색으로 강조"""
        if not self.twSelected or not points:
            return

        # 먼저 모두 초기화
        self._reset_tw_selected_colors()

        want = set(points)  # {(y,x),...}
        root_cnt = self.twSelected.topLevelItemCount()
        for r in range(root_cnt):
            top = self.twSelected.topLevelItem(r)
            if not top:
                continue

            # Class → Label → Pixel 구조
            for i in range(top.childCount()):
                label_item = top.child(i)
                if not label_item:
                    continue

                for j in range(label_item.childCount()):
                    ch = label_item.child(j)
                    if not ch:
                        continue

                    txt = ch.text(0)  # "0.82 (x,y)" 또는 "(x,y)"
                    try:
                        start = txt.rfind('(')
                        end = txt.rfind(')')
                        if start == -1 or end == -1 or start >= end:
                            continue
                        xy = txt[start+1:end].split(',')
                        x, y = int(xy[0]), int(xy[1])   # ← 표시 순서 기준
                    except Exception:
                        continue

                    if (y, x) in want:  # 내부 기준은 (y, x)
                        ch.setForeground(0, QtGui.QBrush(QtGui.QColor(255, 0, 0)))
                        self.twSelected.scrollToItem(ch)

    def _update_br_spectrum(self, y: int, x: int):
        """우하단 그래프 업데이트: 해당 스펙트럼 + 스펙트럼 라이브러리 + 라벨링 데이터"""
        if not MATPLOTLIB_AVAILABLE or self.spectrumWidgetBR is None:
            return
        
        if self._classmap is None or self._hsi_data is None:
            return
        
        try:
            # 클래스 ID 가져오기
            cid = int(self._classmap[y, x])
            
            # 해당 픽셀의 스펙트럼 (빨간색)
            pixel_spectrum = None
            if 0 <= y < self._hsi_data.shape[0] and 0 <= x < self._hsi_data.shape[1]:
                pixel_spectrum = np.asarray(self._hsi_data[y, x, :], dtype=np.float32)
            
            # 스펙트럼 라이브러리 데이터 가져오기 (파란색)
            library_spectra = []
            if self._spectrum_library_provider:
                if callable(self._spectrum_library_provider):
                    library_spectra = self._spectrum_library_provider(cid) or []
                elif isinstance(self._spectrum_library_provider, dict):
                    library_spectra = self._spectrum_library_provider.get(cid, []) or []
            
            # 라벨링 데이터 가져오기 (초록색)
            labeling_spectra = []
            if self._label_provider:
                if callable(self._label_provider):
                    labeling_spectra = self._label_provider(cid) or []
                elif isinstance(self._label_provider, dict):
                    labeling_spectra = self._label_provider.get(cid, []) or []
            
            # 그래프 그리기
            self.spectrumWidgetBR.figure.clear()
            ax = self.spectrumWidgetBR.figure.add_subplot(111)
            
            # X축 데이터 준비
            if self._wavelengths is not None and len(self._wavelengths) > 0:
                x_data_base = self._wavelengths
            else:
                # wavelengths가 없으면 인덱스 사용
                max_len = 0
                if pixel_spectrum is not None:
                    max_len = max(max_len, len(pixel_spectrum))
                for spec in library_spectra:
                    max_len = max(max_len, len(spec))
                for spec in labeling_spectra:
                    max_len = max(max_len, len(spec))
                if max_len == 0:
                    max_len = 100
                x_data_base = np.arange(max_len, dtype=float)
            
            # 1. 해당 스펙트럼 (빨간색)
            if pixel_spectrum is not None:
                pixel_spectrum = pixel_spectrum.flatten()
                if len(x_data_base) == len(pixel_spectrum):
                    x_pixel = x_data_base
                else:
                    x_pixel = np.arange(len(pixel_spectrum), dtype=float)
                ax.plot(x_pixel, pixel_spectrum, color='red', linewidth=1.0, 
                       label='Selected Pixel', alpha=0.9)
            
            # 2. 스펙트럼 라이브러리 (파란색)
            for idx, lib_spec in enumerate(library_spectra):
                lib_spec = np.asarray(lib_spec).flatten()
                if len(x_data_base) == len(lib_spec):
                    x_lib = x_data_base
                else:
                    x_lib = np.arange(len(lib_spec), dtype=float)
                label = 'Spectrum Library' if idx == 0 else None
                ax.plot(x_lib, lib_spec, color='blue', linewidth=1.5, 
                       label=label, alpha=0.7, linestyle='--')
            
            # 3. 라벨링 데이터 (초록색)
            for idx, label_spec in enumerate(labeling_spectra):
                label_spec = np.asarray(label_spec).flatten()
                if len(x_data_base) == len(label_spec):
                    x_label = x_data_base
                else:
                    x_label = np.arange(len(label_spec), dtype=float)
                label = 'Labeling Data' if idx == 0 else None
                ax.plot(x_label, label_spec, color='green', linewidth=1.5, 
                       label=label, alpha=0.7, linestyle=':')
            
            # 클래스 이름 가져오기
            class_name = self._id_to_name.get(cid, str(cid)) if self._id_to_name else str(cid)
            
            # ax.set_xlabel('Wavelength')
            ax.set_xlabel('Band')
            ax.set_ylabel('Intensity')
            # BR : {mtrl_nm}의 라벨링 데이터와 선택 픽셀 그래프
            ax.set_title(f"{class_name}: Labeling Data and Selected Pixels", fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')

            ax.legend(loc='best', fontsize=9)
            
            self.spectrumWidgetBR.figure.tight_layout()
            self.spectrumWidgetBR.draw()
            
        except Exception:
            logging.exception("[AnalysisDialog] _update_br_spectrum failed")
            if self.spectrumWidgetBR:
                self.spectrumWidgetBR.figure.clear()
                ax = self.spectrumWidgetBR.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'Error updating spectrum', 
                       ha='center', va='center', transform=ax.transAxes, color='red')
                self.spectrumWidgetBR.draw()
    
    def _update_br_from_points(self, points: List[Tuple[int, int]]):
        """
        BR (우하단) 그래프 업데이트:
        - 초록색: 해당 클래스의 라벨링 데이터(_label_provider에서 가져온 스펙트럼들)
        - 빨간색: TL에서 선택된 픽셀 스펙트럼
        """
        if not MATPLOTLIB_AVAILABLE or self.spectrumWidgetBR is None:
            return
        
        if self._classmap is None or self._hsi_data is None:
            return
        
        if not points:
            return
        
        try:
            # points에서 클래스 ID 가져오기 (첫 번째 픽셀의 클래스 사용)
            y, x = points[0]
            if not (0 <= y < self._classmap.shape[0] and 0 <= x < self._classmap.shape[1]):
                return
            cid = int(self._classmap[y, x])

            # 1) 라벨링 데이터 가져오기 (초록색) — _label_provider 만 사용
            labeling_spectra: List[np.ndarray] = []
            if self._label_provider:
                try:
                    if callable(self._label_provider):
                        raw_list = self._label_provider(cid) or []
                    elif isinstance(self._label_provider, dict):
                        raw_list = self._label_provider.get(cid, []) or []
                    else:
                        raw_list = []

                    for spec in raw_list:
                        try:
                            arr = np.asarray(spec, dtype=np.float32).reshape(-1)
                            if arr.size > 0:
                                labeling_spectra.append(arr)
                        except Exception:
                            continue
                except Exception:
                    logging.exception("[AnalysisDialog] _update_br_from_points: _label_provider failed")

            # 2) TL에서 선택된 픽셀 스펙트럼 가져오기 (빨간색)
            selected_pixel_spectra: List[np.ndarray] = []
            if isinstance(self.spectrumWidgetTL, SpectrumWidget):
                tl_selected_indices = getattr(self.spectrumWidgetTL, "_selected_indices", []) or []
                tl_metas = getattr(self.spectrumWidgetTL, "_metas", []) or []
                tl_spectra = getattr(self.spectrumWidgetTL, "_spectra", []) or []

                for idx in tl_selected_indices:
                    if 0 <= idx < len(tl_metas) and 0 <= idx < len(tl_spectra):
                        meta = tl_metas[idx]
                        if isinstance(meta, dict):
                            spec = tl_spectra[idx]
                            if spec is not None:
                                selected_pixel_spectra.append(np.asarray(spec, dtype=np.float32))

            # 3) 그래프 그리기
            self.spectrumWidgetBR.figure.clear()
            ax = self.spectrumWidgetBR.figure.add_subplot(111)

            # X축 데이터 준비
            if self._wavelengths is not None and len(self._wavelengths) > 0:
                x_data_base = np.asarray(self._wavelengths, dtype=float).ravel()
            else:
                max_len = 0
                for spec in labeling_spectra:
                    max_len = max(max_len, spec.size)
                for spec in selected_pixel_spectra:
                    max_len = max(max_len, spec.size)
                if max_len == 0:
                    max_len = 100
                x_data_base = np.arange(max_len, dtype=float)

            # 3-1. 라벨링 데이터 (초록색, 점선)
            for idx, label_spec in enumerate(labeling_spectra):
                label_spec = np.asarray(label_spec, dtype=np.float32).ravel()
                if x_data_base.size == label_spec.size:
                    x_label = x_data_base
                else:
                    x_label = np.arange(label_spec.size, dtype=float)
                label = "Labeling Data (Reference)" if idx == 0 else None
                ax.plot(
                    x_label,
                    label_spec,
                    color="green",
                    linewidth=1.5,
                    alpha=0.7,
                    linestyle=":",
                    label=label,
                )

            # 3-2. TL에서 선택된 픽셀 스펙트럼 (빨간색, 실선)
            for idx, pixel_spec in enumerate(selected_pixel_spectra):
                pixel_spec = np.asarray(pixel_spec, dtype=np.float32).ravel()
                if x_data_base.size == pixel_spec.size:
                    x_pixel = x_data_base
                else:
                    x_pixel = np.arange(pixel_spec.size, dtype=float)
                label = "Selected Pixels (TL)" if idx == 0 else None
                ax.plot(
                    x_pixel,
                    pixel_spec,
                    color="red",
                    linewidth=1.0,
                    alpha=0.9,
                    label=label,
                )

            # 4) 축/제목/범례 설정
            class_name = (
                self._id_to_name.get(cid, str(cid))
                if self._id_to_name
                else str(cid)
            )

            # ax.set_xlabel("Wavelength")
            ax.set_xlabel('Band')
            ax.set_ylabel("Intensity")
            ax.set_title(f"{class_name}- Labeling Data & Selected Pixels", fontsize=8)
            ax.grid(True, alpha=0.3, linestyle="--")

            if labeling_spectra or selected_pixel_spectra:
                ax.legend(loc="best", fontsize=9)

            self.spectrumWidgetBR.figure.tight_layout()
            self.spectrumWidgetBR.draw()

        except Exception:
            logging.exception("[AnalysisDialog] _update_br_from_points failed")
            if self.spectrumWidgetBR:
                self.spectrumWidgetBR.figure.clear()
                ax = self.spectrumWidgetBR.figure.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    "Error updating BR graph",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="red",
                )
                self.spectrumWidgetBR.draw()

    def set_match_provider(self, provider):
        """
        provider:
        callable(y:int, x:int, cid:int) -> np.ndarray(shape=(C,)) or None
        분류 시 사용된(또는 그에 가장 가까운) 레퍼런스 스펙트럼을 돌려주는 콜백.
        """
        self._match_provider = provider
        
    def _update_tr_reference_only(self, y: int, x: int):
        """
        TR(우측 상단): 선택 픽셀(y,x)의 '사용된 reference' 1개만 표시
        """
        if not MATPLOTLIB_AVAILABLE or not isinstance(self.spectrumWidgetTR, SpectrumWidget):
            return
        if self._classmap is None:
            self.spectrumWidgetTR.set_spectra([], None); return
        cid = int(self._classmap[y, x])
        ref_spec = self._match_provider(int(y), int(x), int(cid)) if callable(self._match_provider) else None
        if ref_spec is None:
            self.spectrumWidgetTR.set_spectra([], None); return

        color = self._palette.get(cid, QtGui.QColor(150, 80, 220)) if self._palette else QtGui.QColor(150, 80, 220)
        self.spectrumWidgetTR.set_spectra(
            [np.asarray(ref_spec, dtype=np.float32).reshape(-1)],
            self._wavelengths,
            [color],
            metas=[{"kind":"used_reference","cid":cid,"y":y,"x":x}]
        )

    def _update_tr_region_used_refs(self):
        """
        TR(우상): 현 마스크 + 선택 클래스(cbClassRange) 픽셀들이 실제로 사용한 reference 스펙트럼 전부 표시.
        - self._selected_ref_keys: TL에서 '빨간(선택)'으로 표시된 라인들의 reference key 집합
        - _focused_point(단일 클릭 픽셀)의 reference도 계속 지원
        """
        if not MATPLOTLIB_AVAILABLE or not isinstance(self.spectrumWidgetTR, SpectrumWidget):
            return
        if self._classmap is None or self._mask is None:
            self.spectrumWidgetTR.set_spectra([], None); return

        # 선택 클래스
        selected_cid = None
        if self.cbClassRange and self.cbClassRange.currentData() is not None:
            try:
                selected_cid = int(self.cbClassRange.currentData())
            except Exception:
                selected_cid = None
        if selected_cid is None or selected_cid < 0:
            self.spectrumWidgetTR.set_spectra([], None); return

        # (옵션) 단일 포커스 픽셀의 reference key
        clicked_key = None
        if getattr(self, "_focused_point", None) and callable(self._match_provider):
            try:
                y_click, x_click = self._focused_point
                ref = self._match_provider(int(y_click), int(x_click), int(selected_cid))
                if ref is not None:
                    clicked_key = np.round(np.asarray(ref, dtype=np.float32).reshape(-1), 6).tobytes()
            except Exception:
                clicked_key = None

        # '빨간 기준' key 집합 구성: (히스토그램/TL 선택 key들) ∪ (단일 클릭 key)
        red_keys = set(getattr(self, "_selected_ref_keys", set()) or set())
        if clicked_key is not None:
            red_keys.add(clicked_key)

        # 마스크 내 선택 클래스 픽셀들에서 reference 수집(중복 제거)
        m = np.asarray(self._mask, dtype=bool)
        sel = (self._classmap == selected_cid) & m
        ys, xs = np.where(sel)
        if ys.size == 0:
            self.spectrumWidgetTR.set_spectra([], None); return

        specs, colors, metas, dedup = [], [], [], set()
        MAX_DRAW = 80
        pal_color = self._palette.get(selected_cid, QtGui.QColor(80, 200, 255)) if self._palette else QtGui.QColor(80, 200, 255)
        red_color = QtGui.QColor(255, 0, 0)

        if not callable(self._match_provider):
            self.spectrumWidgetTR.set_spectra([], None); return

        for y, x in zip(ys.tolist(), xs.tolist()):
            ref = self._match_provider(int(y), int(x), int(selected_cid))
            if ref is None:
                continue
            v = np.asarray(ref, dtype=np.float32).reshape(-1)
            key = np.round(v, 6).tobytes()
            if key in dedup:
                continue
            dedup.add(key)

            # 🔴 빨간 여부 판단: red_keys에 포함되면 빨간색
            is_red = key in red_keys
            specs.append(v)
            colors.append(red_color if is_red else pal_color)
            metas.append({"kind": "used_ref", "cid": selected_cid, "ref_spec": v})

            if len(specs) >= MAX_DRAW:
                break

        if specs:
            self.spectrumWidgetTR.set_spectra(specs, self._wavelengths, colors, metas=metas)
            if not getattr(self, "_tr_pick_connected", False):
                self.spectrumWidgetTR.on_pick = self._on_tr_line_picked
                self._tr_pick_connected = True
        else:
            self.spectrumWidgetTR.set_spectra([], None)

        # 지도 마커도 갱신
        self._update_mapview_red_markers()

    def _update_mapview_red_markers(self):
        """
        MapView 오버레이 동기화
        - TL 선택된 픽셀만 '빨간색' 1픽셀 채움 + 얇은 검정 외곽선 표시
        - TR(노란색) 표시는 하지 않음(기존 레이어도 제거)
        """
        try:
            parent = self.parent()
            if not parent:
                return
            map_view = getattr(parent, "_map_view", None)
            if not map_view or not hasattr(map_view, "add_temporal_layer") or not hasattr(map_view, "remove_temporal_layer"):
                return
            if self._classmap is None:
                return

            H, W = self._classmap.shape[:2]

            # 기존 오버레이 제거 (TR 노랑도 함께 제거)
            try: map_view.remove_temporal_layer("ANALYSIS_TL_SELECTED_OVERLAY")
            except Exception: pass
            try: map_view.remove_temporal_layer("ANALYSIS_TR_SELECTED_OVERLAY")
            except Exception: pass

            # ========= TL(빨강) 오버레이 =========
            # TL 그래프에서 선택된(체크된) 부분만 기준으로 빨간색 표시
            tl_points: List[Tuple[int, int]] = []
            if isinstance(self.spectrumWidgetTL, SpectrumWidget):
                selected_indices = getattr(self.spectrumWidgetTL, "_selected_indices", []) or []
                metas = getattr(self.spectrumWidgetTL, "_metas", []) or []
                for idx in selected_indices:
                    if 0 <= idx < len(metas) and isinstance(metas[idx], dict):
                        y = metas[idx].get("y"); x = metas[idx].get("x")
                        if y is None or x is None:
                            continue
                        tl_points.append((int(y), int(x)))

            tl_rgba = np.zeros((H, W, 4), dtype=np.uint8)
            if tl_points:
                # 채움(정확히 해당 픽셀만)
                for (y, x) in tl_points:
                    if 0 <= y < H and 0 <= x < W:
                        tl_rgba[y, x, :] = (255, 0, 0, 230)

                # 얇은 외곽선(인접 8방향 1픽셀 검정 링)
                border = np.zeros((H, W), dtype=bool)
                selmask = np.zeros((H, W), dtype=bool)
                yy, xx = zip(*tl_points)
                selmask[list(yy), list(xx)] = True
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        border |= np.roll(np.roll(selmask, dy, axis=0), dx, axis=1)
                border &= ~selmask
                tl_rgba[border, 0] = 0
                tl_rgba[border, 1] = 0
                tl_rgba[border, 2] = 0
                tl_rgba[border, 3] = 160

                map_view.add_temporal_layer("ANALYSIS_TL_SELECTED_OVERLAY", tl_rgba, opacity=1.0, force_unique=False)

            # === TR 깜빡임/캐시 정리 (더 이상 사용하지 않음) ===
            if hasattr(self, "_pix_blink_timer") and self._pix_blink_timer:
                try:
                    self._pix_blink_timer.stop()
                except Exception:
                    pass
            self._pix_blink_timer = None
            self._tr_overlay_rgba_cache = None
            self._tr_blink_mask = None

        except Exception:
            logging.exception("[AnalysisDialog] _update_mapview_red_markers failed")



    def _clear_mapview_red_markers(self):
        """
        TL/TR 픽셀 채색 오버레이 제거 (마커 사용 안 함)
        """
        try:
            parent = self.parent()
            if not parent:
                return
            map_view = getattr(parent, "_map_view", None)
            if map_view and hasattr(map_view, "remove_temporal_layer"):
                try: map_view.remove_temporal_layer("ANALYSIS_TL_SELECTED_OVERLAY")
                except Exception: pass
                try: map_view.remove_temporal_layer("ANALYSIS_TR_SELECTED_OVERLAY")
                except Exception: pass
        except Exception:
            logging.exception("[AnalysisDialog] _clear_mapview_red_markers failed")

    def _highlight_all_pixels_of_class(self, cid: int):
        """
        지정 클래스 cid에 대해, (mask == True) AND (classmap == cid) 인 모든 픽셀을
        MapView에 '빨간색' 마커로 표시한다.
        """
        try:
            parent = self.parent()
            if not parent or self._classmap is None or self._mask is None:
                return
            map_view = getattr(parent, "_map_view", None)
            if not map_view or not hasattr(map_view, "add_ctx_marker") or not hasattr(map_view, "clear_ctx_markers"):
                return

            # 기존 클래스-전체 하이라이트 마커 제거
            map_view.clear_ctx_markers("ANALYSIS_CLASS_ALL")

            m = np.asarray(self._mask, dtype=bool)
            sel = (self._classmap == int(cid)) & m
            ys, xs = np.where(sel)
            if ys.size == 0:
                return

            red = QtGui.QColor(255, 0, 0)
            # 성능 보호: 너무 많으면 샘플링(원하면 조정)
            MAX_MARK = 5000
            total = min(MAX_MARK, ys.size)
            for i in range(total):
                y, x = int(ys[i]), int(xs[i])
                map_view.add_ctx_marker("ANALYSIS_CLASS_ALL", y, x, number=None, color=red, size=1)
        except Exception:
            logging.exception("[AnalysisDialog] _highlight_all_pixels_of_class failed")
            
    def _get_checked_points(self) -> List[Tuple[int, int]]:
        """
        왼쪽 트리(twSelected)의 체크된 픽셀(child-of-label)을 우선 수집.
        트리가 비어 있으면 tablePixels의 체크된 행으로 대체.
        반환: [(y,x), ...]
        """
        pts: List[Tuple[int, int]] = []

        # 1) 트리 우선
        if self.twSelected:
            root_cnt = self.twSelected.topLevelItemCount()
            for r in range(root_cnt):
                top = self.twSelected.topLevelItem(r)
                if not top:
                    continue

                # Class → Label → Pixel 구조
                for i in range(top.childCount()):
                    label_item = top.child(i)
                    if not label_item:
                        continue

                    for j in range(label_item.childCount()):
                        ch = label_item.child(j)
                        if not ch:
                            continue
                        if ch.checkState(0) != Qt.Checked:
                            continue

                        try:
                            txt = ch.text(0)  # "0.82 (x,y)" 형태
                            start = txt.rfind('(')
                            end = txt.rfind(')')
                            if start != -1 and end != -1 and start < end:
                                xy = txt[start+1:end].split(',')
                                x, y = int(xy[0]), int(xy[1])   # ← 먼저 x, 그 다음 y
                                pts.append((y, x))              # ← 내부는 (y, x)로 돌려줌
                        except Exception:
                            continue

        # 2) 트리에서 아무것도 못 모았으면 테이블로 백업 (기존 로직 유지)
        if not pts and self.tablePixels:
            for row in range(self.tablePixels.rowCount()):
                it_chk = self.tablePixels.item(row, 0)
                if it_chk and it_chk.checkState() == Qt.Checked:
                    if 0 <= row < len(self._pixel_data):
                        rec = self._pixel_data[row]
                        pts.append((int(rec["y"]), int(rec["x"])))
        return pts


    def _refresh_all_from_checked(self):
        """
        왼쪽에서 체크된 픽셀들 전체를 기준으로:
        - TL: 개별 스펙트럼 모두 표시
        - TR: 사용된 reference 스펙트럼(중복 제거) 모두 표시
        - Histogram: 이미 그려진 히스토그램 위에서, 체크된 픽셀들의 bin만 강조
        """
        pts = self._get_checked_points()

        # --- TL ---
        if isinstance(self.spectrumWidgetTL, SpectrumWidget):
            specs, colors, metas = [], [], []
            for (y, x) in pts:
                s = None if self._hsi_data is None else self._hsi_data[y, x, :]
                if s is None:
                    continue
                specs.append(np.asarray(s, dtype=np.float32))
                cid = int(self._classmap[y, x]) if self._classmap is not None else -1
                c = self._palette.get(cid, QtGui.QColor(100, 150, 255)) if self._palette else QtGui.QColor(100, 150, 255)
                colors.append(c)
                metas.append({"y": int(y), "x": int(x), "cid": int(cid)})

            self.spectrumWidgetTL.set_spectra(specs, self._wavelengths, colors, metas=metas)
            self.spectrumWidgetTL.on_pick = self._on_tl_line_picked
            self._tl_pick_connected = True

        # --- TR ---
        if isinstance(self.spectrumWidgetTR, SpectrumWidget):
            specs_tr, colors_tr, metas_tr = [], [], []
            dedup = set()
            if callable(self._match_provider):
                for (y, x) in pts:
                    cid = int(self._classmap[y, x]) if self._classmap is not None else -1
                    ref = self._match_provider(int(y), int(x), int(cid))
                    if ref is None:
                        continue
                    v = np.asarray(ref, dtype=np.float32).reshape(-1)
                    key = np.round(v, 6).tobytes()
                    if key in dedup:
                        continue
                    dedup.add(key)
                    specs_tr.append(v)
                    col = self._palette.get(cid, QtGui.QColor(80, 200, 255)) if self._palette else QtGui.QColor(80, 200, 255)
                    colors_tr.append(col)
                    metas_tr.append({"kind": "used_ref", "cid": cid, "ref_spec": v})
            self.spectrumWidgetTR.set_spectra(specs_tr, self._wavelengths, colors_tr, metas=metas_tr)
            if not getattr(self, "_tr_pick_connected", False):
                self.spectrumWidgetTR.on_pick = self._on_tr_line_picked
                self._tr_pick_connected = True

        # --- Histogram ---
        # ✅ 더 이상 히스토그램을 다시 그리지 않고,
        #    현재 클래스 기준 히스토그램 위에서 체크된 픽셀들의 bin만 강조
        self._hist_highlight_points(pts)

        # 지도 마커 동기화
        self._update_mapview_red_markers()

    def _update_histogram_from_points(self, points: List[Tuple[int,int]]):
        """
        체크된 픽셀들의 top1 값을 대상으로 히스토그램을 그린다.
        클래스 콤보/마스크와 무관하게 '현재 체크 집합'만 반영.
        """
        if not MATPLOTLIB_AVAILABLE or self.histogramWidget is None:
            return

        fig = self.histogramWidget.figure
        fig.clear()
        ax = fig.add_subplot(111)

        if not points or self._topk_vals is None:
            ax.text(0.5, 0.5, 'No checked pixels', ha='center', va='center', transform=ax.transAxes)
            self.histogramWidget.draw()
            # 캐시 초기화
            if not hasattr(self, "_hist_cache"):
                self._hist_cache = {}
            self._hist_cache.update({
                "edges": None, "vals": None, "indices": None,
                "selected_bins": set(), "selected_mask": None
            })
            return

        vals, idx = [], []
        H, W = self._topk_vals.shape[:2]
        for (y, x) in points:
            if 0 <= y < H and 0 <= x < W:
                v = self._topk_vals[y, x, 0]
                if np.isfinite(v):
                    vals.append(float(v))
                    idx.append((int(y), int(x)))

        if not vals:
            ax.text(0.5, 0.5, 'No valid Top-1 values', ha='center', va='center', transform=ax.transAxes)
            self.histogramWidget.draw()
            if not hasattr(self, "_hist_cache"):
                self._hist_cache = {}
            self._hist_cache.update({
                "edges": None, "vals": None, "indices": None,
                "selected_bins": set(), "selected_mask": None
            })
            return

        # bin 수: 기존 슬라이더 존중
        bins = 50
        if self.sldBins:
            try:
                bins = int(self.sldBins.value())
            except Exception:
                pass

        counts, edges, patches = ax.hist(vals, bins=bins, edgecolor='black', alpha=0.85)
        ax.set_xlabel(f"Similarity ({self._metric})")
        ax.set_ylabel("Count")
        ax.set_title("Checked Pixels (Top-1) Histogram")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        self.histogramWidget.draw()

        # 캐시(클릭 대응)
        if not hasattr(self, "_hist_cache"):
            self._hist_cache = {}
        self._hist_cache.update({
            "edges": np.asarray(edges, dtype=float),
            "vals": np.asarray(vals, dtype=float),
            "indices": np.asarray(idx, dtype=int),  # (N,2)
            "selected_bins": set(),                 # 여러 bin 선택 지원
            "selected_mask": None
        })

        # 현재 모드는 히스토그램 주도
        self._tl_mode = "hist"
        self._hist_mode = "checked"

    def _update_tr_refs_from_checked(self, red_ref_keys: Optional[set] = None):
        """
        TR(오른쪽): 왼쪽에서 '체크된 픽셀 전체'가 실제 사용한 reference 스펙트럼을 모두 표시.
        - red_ref_keys: TL에서 '빨간(선택)'으로 강조된 라인들이 사용하는 reference key 집합
                        (여기에 포함된 것만 빨간색, 나머지는 기본 팔레트색)
        - TL에서 선택된 픽셀의 스펙트럼도 함께 표시 (빨간색으로 강조)
        """
        if not MATPLOTLIB_AVAILABLE or not isinstance(self.spectrumWidgetTR, SpectrumWidget):
            return
        if not callable(self._match_provider):
            self.spectrumWidgetTR.set_spectra([], None); return

        pts = self._get_checked_points()  # [(y,x), ...]
        if not pts:
            self.spectrumWidgetTR.set_spectra([], None); return

        red_ref_keys = red_ref_keys or set()
        specs, colors, metas, dedup = [], [], [], set()
        
        # 1) 체크된 픽셀들이 사용한 reference 스펙트럼 표시
        for (y, x) in pts:
            try:
                cid = int(self._classmap[y, x]) if self._classmap is not None else -1
                ref = self._match_provider(int(y), int(x), cid)
                if ref is None:
                    continue
                v = np.asarray(ref, dtype=np.float32).reshape(-1)
                key = np.round(v, 6).tobytes()
                if key in dedup:
                    continue
                dedup.add(key)

                # 강조 여부: red_ref_keys에 들어있으면 빨강, 아니면 팔레트 색
                is_red = key in red_ref_keys
                col = QtGui.QColor(255, 0, 0) if is_red else (
                    self._palette.get(cid, QtGui.QColor(80, 200, 255)) if self._palette else QtGui.QColor(80, 200, 255)
                )

                specs.append(v)
                colors.append(col)
                metas.append({"kind": "used_ref", "cid": cid, "ref_spec": v})
            except Exception:
                continue

        # TR 그래프에는 reference 스펙트럼만 표시 (TL 선택 픽셀 스펙트럼은 BR 그래프에 표시)
        self.spectrumWidgetTR.set_spectra(specs, self._wavelengths, colors, metas=metas)
        if not getattr(self, "_tr_pick_connected", False):
            self.spectrumWidgetTR.on_pick = self._on_tr_line_picked
            self._tr_pick_connected = True

        # 지도 오버레이(TL=빨강, TR=노랑) 갱신
        self._update_mapview_red_markers()


    def _update_tr_refs_only_selected(self, red_ref_keys: set):
        """
        TR(오른쪽): TL에서 '빨간(선택)'으로 강조된 라인들이 사용하는 reference만 표시.
        - red_ref_keys: 빨간 TL 라인들의 reference key 집합
        """
        if not MATPLOTLIB_AVAILABLE or not isinstance(self.spectrumWidgetTR, SpectrumWidget):
            return
        if not callable(self._match_provider):
            self.spectrumWidgetTR.set_spectra([], None); return

        pts = self._get_checked_points()  # 체크된 픽셀 전체
        if not pts or not red_ref_keys:
            self.spectrumWidgetTR.set_spectra([], None); return

        specs, colors, metas, dedup = [], [], [], set()
        for (y, x) in pts:
            try:
                cid = int(self._classmap[y, x]) if self._classmap is not None else -1
                ref = self._match_provider(int(y), int(x), cid)
                if ref is None:
                    continue
                v = np.asarray(ref, dtype=np.float32).reshape(-1)
                key = np.round(v, 6).tobytes()
                if key not in red_ref_keys or key in dedup:
                    continue
                dedup.add(key)

                # 표시 색(가시성 위해 빨강): 원하면 팔레트색으로 바꿔도 됩니다.
                col = QtGui.QColor(255, 0, 0)

                specs.append(v)
                colors.append(col)
                metas.append({"kind": "used_ref", "cid": cid, "ref_spec": v})
            except Exception:
                continue

        self.spectrumWidgetTR.set_spectra(specs, self._wavelengths, colors, metas=metas)
        if not getattr(self, "_tr_pick_connected", False):
            self.spectrumWidgetTR.on_pick = self._on_tr_line_picked
            self._tr_pick_connected = True

        # 지도 오버레이(빨강 TL, 노랑 TR) 갱신
        self._update_mapview_red_markers()

    def _ensure_blink_timer(self):
        if getattr(self, "_blink_timer", None):
            return
        self._blink_alpha_up = True
        self._blink_timer = QtCore.QTimer(self)
        self._blink_timer.setInterval(220)  # ms
        self._blink_timer.timeout.connect(self._tick_blink)
        self._blink_timer.start()

    def _tick_blink(self):
        """TR 노란 오버레이의 알파를 160~230 범위에서 펄스."""
        try:
            parent = self.parent()
            if not parent:
                return
            map_view = getattr(parent, "_map_view", None)
            if not map_view or not hasattr(map_view, "remove_temporal_layer") or not hasattr(map_view, "add_temporal_layer"):
                return

            rgba = getattr(self, "_tr_overlay_rgba_cache", None)
            if rgba is None:
                return

            a = rgba[..., 3].astype(np.int16)
            delta = 22 if self._blink_alpha_up else -22
            a = np.clip(a + delta, 160, 230)
            # 왕복 스위치
            if a.max() == 230 or a.min() == 160:
                self._blink_alpha_up = not self._blink_alpha_up
            rgba[..., 3] = a.astype(np.uint8)

            # 다시 올려서 교체
            try: map_view.remove_temporal_layer("ANALYSIS_TR_SELECTED_OVERLAY")
            except Exception: pass
            map_view.add_temporal_layer("ANALYSIS_TR_SELECTED_OVERLAY", rgba, opacity=1.0, force_unique=False)

            # 캐시 갱신
            self._tr_overlay_rgba_cache = rgba
        except Exception:
            pass

    def _ensure_pixel_blink_timer(self):
        """선택 픽셀만 깜빡이도록 하는 타이머 준비."""
        if getattr(self, "_pix_blink_timer", None):
            return
        self._pix_blink_up = True
        self._pix_blink_timer = QtCore.QTimer(self)
        self._pix_blink_timer.setInterval(220)  # 속도 조절
        self._pix_blink_timer.timeout.connect(self._tick_pixel_blink)
        self._pix_blink_timer.start()

    def _tick_pixel_blink(self):
        """
        TR(또는 TL) 오버레이에서 '점멸 대상 마스크(True)'인 픽셀의 알파만 토글.
        - self._tr_overlay_rgba_cache : (H,W,4) 현재 오버레이 복사본
        - self._tr_blink_mask         : (H,W) bool  ← 깜빡일 채움 픽셀만 True (외곽선/배경 False)
        """
        try:
            parent = self.parent()
            if not parent:
                return
            map_view = getattr(parent, "_map_view", None)
            if not map_view or not hasattr(map_view, "add_temporal_layer") or not hasattr(map_view, "remove_temporal_layer"):
                return

            rgba = getattr(self, "_tr_overlay_rgba_cache", None)
            mask = getattr(self, "_tr_blink_mask", None)
            if rgba is None or mask is None:
                return

            # 선택 픽셀만 알파 토글 (외곽선은 건드리지 않음)
            a = rgba[..., 3].astype(np.int16)
            delta = 22 if self._pix_blink_up else -22
            a[mask] = np.clip(a[mask] + delta, 160, 230)
            # 왕복 스위칭(상/하 포화 시 방향 반전)
            if a[mask].max() == 230 or a[mask].min() == 160:
                self._pix_blink_up = not self._pix_blink_up
            rgba[..., 3] = a.astype(np.uint8)

            # 레이어 교체 적용
            try: map_view.remove_temporal_layer("ANALYSIS_TR_SELECTED_OVERLAY")
            except Exception: pass
            map_view.add_temporal_layer("ANALYSIS_TR_SELECTED_OVERLAY", rgba, opacity=1.0, force_unique=False)

            # 캐시 갱신
            self._tr_overlay_rgba_cache = rgba
        except Exception:
            logging.exception("[AnalysisDialog] _tick_pixel_blink failed")

    def _select_tl_by_points(self, points: List[Tuple[int, int]]):
        """TL에서 메타(y,x)가 points에 포함되는 라인만 '선택(빨강)' 처리. 나머지는 그대로."""
        if not isinstance(self.spectrumWidgetTL, SpectrumWidget):
            return
        metas = getattr(self.spectrumWidgetTL, "_metas", []) or []
        want = set((int(y), int(x)) for (y, x) in (points or []))
        red_indices = []
        for i, m in enumerate(metas):
            if isinstance(m, dict):
                y, x = m.get("y"), m.get("x")
                if y is not None and x is not None and (int(y), int(x)) in want:
                    red_indices.append(i)
        self.spectrumWidgetTL.set_selected_indices(red_indices)
        # ★ TL 선택 변경 시 Layer 빨간색 표시 동기화
        self._update_mapview_red_markers()

    def _ref_keys_for_points(self, points: List[Tuple[int, int]]) -> set:
        """points에 해당하는 픽셀들이 사용하는 reference key 집합 계산."""
        keys = set()
        if not callable(self._match_provider) or self._classmap is None:
            return keys
        for (y, x) in points or []:
            try:
                cid = int(self._classmap[int(y), int(x)])
                ref = self._match_provider(int(y), int(x), cid)
                if ref is not None:
                    k = np.round(np.asarray(ref, dtype=np.float32).reshape(-1), 6).tobytes()
                    keys.add(k)
            except Exception:
                continue
        return keys

    def _hist_highlight_points(self, points: List[Tuple[int, int]]):
        """
        체크집합 기반 히스토그램이 없으면 먼저 그린 뒤,
        points의 top1 값이 들어가는 bin(들)을 강조.
        """
        if not MATPLOTLIB_AVAILABLE or self.histogramWidget is None or self._topk_vals is None:
            return
        cache = getattr(self, "_hist_cache", None)
        # 히스토그램이 아직 없으면 현재 모드에 맞춰 생성
        if not cache or cache.get("edges") is None:
            if getattr(self, "_hist_mode", "checked") == "combo":
                self._update_histogram()
            else:
                self._update_histogram_from_points(self._get_checked_points())
            cache = getattr(self, "_hist_cache", None)
            if not cache or cache.get("edges") is None:
                return

        edges = cache["edges"]
        fig = self.histogramWidget.figure
        if not fig.axes:
            return
        ax = fig.axes[0]

        H, W = self._topk_vals.shape[:2]
        bins_to_mark = set()
        for (y, x) in points or []:
            if 0 <= y < H and 0 <= x < W and np.isfinite(self._topk_vals[y, x, 0]):
                v = float(self._topk_vals[y, x, 0])
                j = np.searchsorted(edges, v, side="right") - 1
                j = max(0, min(len(edges) - 2, j))
                bins_to_mark.add(int(j))

        # 실제 강조는 공통 함수 사용
        self._highlight_hist_bin(bins_to_mark if bins_to_mark else None)

        # 캐시에도 표시
        cache["selected_bins"] = bins_to_mark


    def _apply_focus_from_points(self, points: List[Tuple[int, int]]):
        """
        단일 출처(TL/ TR/ BIN)에서 얻은 points[(y,x),...] 기준으로
        - TL: 해당 points만 '선택(빨강)'으로 강조 (전체 라인 유지)
        - TR: 체크집합 전체 reference 표시 + points의 reference만 빨강 강조
        - HIST: points의 top1 값이 포함된 bin(들) 강조
        """
        # 1) TL 전체 스펙 갱신 (트리 체크 기준)
        if hasattr(self, '_refresh_spectra_from_tree'):
            self._refresh_spectra_from_tree()
        
        # 2) TL에서 points에 해당하는 라인만 '선택(빨강)'
        self._select_tl_by_points(points)

        # 3) TR: 전체 reference + points 기준 빨강 강조
        red_ref_keys = self._ref_keys_for_points(points)
        self._update_tr_refs_from_checked(red_ref_keys)

        # 4) BR: 라벨링 데이터 + TL 선택 픽셀 스펙
        self._update_br_from_points(points)

        # 5) Histogram: points의 top1 값이 속한 bin 강조
        self._hist_highlight_points(points)

        # 6) 지도 오버레이 동기화
        self._update_mapview_red_markers()

        # 7) ★ Layer 트리에서 동일 픽셀 빨간색 표시
        self._mark_tree_selection(points)


    def _set_hist_class_combo_to(self, cid: int, update_hist: bool = True):
        if cid is None or cid < 0 or self.cbClassRange is None:
            return

        import time
        # 사용자 우선권이 살아있으면 프로그램적 강제 변경 무시
        if time.time() < getattr(self, "_hist_user_locked_until", 0.0):
            return

        self._current_hist_cid = int(cid)
        self.cbClassRange.blockSignals(True)
        try:
            idx = self.cbClassRange.findData(self._current_hist_cid)
            if idx < 0:
                name = self._id_to_name.get(self._current_hist_cid, str(self._current_hist_cid)) if self._id_to_name else str(self._current_hist_cid)
                self.cbClassRange.addItem(f"Class {self._current_hist_cid} ({name})", self._current_hist_cid)
                idx = self.cbClassRange.findData(self._current_hist_cid)
            if idx >= 0:
                self.cbClassRange.setCurrentIndex(idx)
        finally:
            self.cbClassRange.blockSignals(False)

        if update_hist:
            self._apply_histogram_class_selection(self._current_hist_cid, source="combo-set")


    def _ensure_hist_widget(self):
        # 이미 위젯이 있고 레이아웃에 추가되어 있으면 반환
        if self.histogramWidget is not None:
            # 위젯이 실제로 레이아웃에 있는지 확인
            if self.histogramWidget.parent() is not None:
                return
            else:
                # 위젯은 있지만 레이아웃에 없으면 None으로 설정하고 재생성
                logging.warning("[AnalysisDialog] histogramWidget exists but not in layout, recreating")
                self.histogramWidget = None
        
        # 1) 우선 lbl_hist 아래에 캔버스 생성 시도
        if self.lblHist:
            parent_container = self.lblHist.parent()
            if parent_container:
                lay = parent_container.layout() or QtWidgets.QVBoxLayout(parent_container)
                parent_container.setLayout(lay)
                try:
                    lay.removeWidget(self.lblHist)
                    self.lblHist.setParent(None)
                except Exception:
                    pass
                self.histogramWidget = FigureCanvasQTAgg(Figure(figsize=(6, 4)))
                self.histogramWidget.setMinimumSize(400, 220)
                lay.addWidget(self.histogramWidget)
                logging.debug("[AnalysisDialog] installed histogram widget in lbl_hist container")
                
        # 2) 그래도 없으면 기타 컨테이너로 시도
        if self.histogramWidget is None:
            for name in ("plot_hist", "gb_hist", "hist_container"):
                host = self.findChild(QtWidgets.QWidget, name)
                if host:
                    if host.layout() is None:
                        host.setLayout(QtWidgets.QVBoxLayout(host))
                    self.histogramWidget = FigureCanvasQTAgg(Figure(figsize=(6, 4)))
                    self.histogramWidget.setMinimumSize(400, 220)
                    host.layout().addWidget(self.histogramWidget)
                    logging.debug(f"[AnalysisDialog] installed histogram widget in {name} container")
                    break
                
        # 3) 클릭 이벤트 연결
        if self.histogramWidget is not None:
            try:
                # 기존 연결이 있으면 해제
                if hasattr(self, "_hist_click_cid") and self._hist_click_cid is not None:
                    try:
                        self.histogramWidget.mpl_disconnect(self._hist_click_cid)
                    except Exception:
                        pass
                self._hist_click_cid = self.histogramWidget.mpl_connect(
                    "button_press_event", self._on_hist_click
                )
            except Exception:
                logging.exception("[AnalysisDialog] failed to connect histogram click event")
                self._hist_click_cid = None

    def _on_class_combo_pressed(self, model_index: QtCore.QModelIndex):
        try:
            if model_index is None:
                return
            row = int(model_index.row())
            logging.info("[AnalysisDialog] class combo pressed row=%s", row)
            if self.cbClassRange is not None:
                self.cbClassRange.setCurrentIndex(row)
                # 바로 적용 트리거 (일부 스타일에서 currentIndexChanged가 지연될 수 있음)
                self._on_class_range_changed(row)
        except Exception:
            logging.exception("[AnalysisDialog] class combo pressed handler failed")
            
    def _apply_histogram_class_selection(self, cid: Optional[int], *, source: str = "combo"):
        logging.info("[hist_apply] source=%s, cid=%r", source, cid)
        self._current_hist_cid = (None if (cid is not None and cid < 0) else cid)

        # 콤보가 유효하면 반드시 'combo' 모드 고정
        self._hist_mode = "combo" if self._current_hist_cid is not None else "checked"

        # 캐시 전체 초기화(간헐적 덮어쓰기 방지)
        if not hasattr(self, "_hist_cache") or not isinstance(self._hist_cache, dict):
            self._hist_cache = {}
        self._hist_cache.update({
            "edges": None,
            "vals": None,
            "indices": None,
            "selected_bins": set(),   # ✅ 여러 bin 인덱스를 저장
            "selected_mask": None,
        })

        # 위젯 보장 후 즉시 재그림
        self._ensure_hist_widget()
        self._update_histogram()

        # TL/TR 갱신과 충돌하지 않도록 모드 고정
        self._tl_mode = "tree"
