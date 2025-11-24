# recommendation_wizard.py
from __future__ import annotations
import sys, random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, QObject

# matplotlib 임베드
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ---------------------------
# 상태 보관
# ---------------------------
@dataclass
class State:
    target_map_path: Optional[Path] = None
    algo: str = "clustering"  # or "similarity"

    selected_class_ids: List[int] = field(default_factory=list)
    inner_tau: float = 0.002
    outer_tau: float = 0.005

    # step3
    recommended: List[Dict[str, Any]] = field(default_factory=list)
    selected_candidates: Set[int] = field(default_factory=set)


# ---------------------------
# 백그라운드 작업 (데모용)
# ---------------------------
class _JobSignals(QObject):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)


class RecommendJob(QtCore.QRunnable):
    def __init__(self, state: State):
        super().__init__()
        self.state = state
        self.signals = _JobSignals()

    @staticmethod
    def _fake_spectrum(n=210):
        # 400~1000nm 구간 가정, 임의의 부드러운 스펙트럼 생성(데모)
        x = np.linspace(0, 1, n)
        y = 0.3 + 0.7 * np.sin(6 * np.pi * (x + random.uniform(-0.02, 0.02)))
        y += 0.1 * np.cos(18 * np.pi * (x + random.uniform(-0.02, 0.02)))
        y = (y - y.min()) / max(1e-6, (y.max() - y.min()))
        return y.astype(np.float32)

    def run(self):
        try:
            # ===== 여기에 실제 추천 로직을 넣으시면 됩니다 =====
            # self.state.algo, self.state.selected_class_ids,
            # self.state.inner_tau, self.state.outer_tau 등을 사용.
            # 예시: 상위 K개 후보 반환
            K = 12
            results = []
            base_class = self.state.selected_class_ids[0] if self.state.selected_class_ids else 1
            for i in range(K):
                x = random.randint(0, 2047)
                y = random.randint(0, 2047)
                metric = "SAD" if self.state.algo == "similarity" else "clusterDist"
                value = round(random.uniform(0.01, 0.25), 3)
                spec = self._fake_spectrum()
                results.append({
                    "idx": i + 1,
                    "x": x,
                    "y": y,
                    "class_id": base_class,
                    "class_name": f"Class {base_class}",
                    "metric": metric,
                    "value": value,
                    "spectrum": spec,  # (N,)
                })
            self.signals.finished.emit(results)
        except Exception as e:
            self.signals.error.emit(str(e))


# ---------------------------
# Matplotlib 캔버스
# ---------------------------
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

# ---------------------------
# 메인 다이얼로그
# ---------------------------
class RecommendationWizard(QtWidgets.QDialog):
    candidatesSelected = pyqtSignal(list)  # 3단계 '후보 등록' 클릭 시 외부로 결과 emit

    def __init__(self, parent=None, ui_path: Optional[Path] = None):
        super().__init__(parent)
        if ui_path is None:
            ui_path = Path("recommend_label_wizard.ui")
        self._root = uic.loadUi(str(ui_path), baseinstance=self)

        # 상태/스레드 풀
        self.state = State()
        self.pool = QtCore.QThreadPool.globalInstance()

        # 위젯 참조(가독성)
        self.stacked: QtWidgets.QStackedWidget = self.findChild(QtWidgets.QStackedWidget, "stackedMain")
        self.lblStep1: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, "lblStep1")
        self.lblStep2: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, "lblStep2")
        self.lblStep3: QtWidgets.QLabel = self.findChild(QtWidgets.QLabel, "lblStep3")

        self.btnPrev: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btnPrev")
        self.btnNext: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btnNext")
        self.btnCancel: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btnCancel")
        self.btnRegister: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btnRegister")

        # step1
        self.cboMap: QtWidgets.QComboBox = self.findChild(QtWidgets.QComboBox, "cboMap")
        self.rClustering: QtWidgets.QRadioButton = self.findChild(QtWidgets.QRadioButton, "rClustering")
        self.rSimilarity: QtWidgets.QRadioButton = self.findChild(QtWidgets.QRadioButton, "rSimilarity")
        self.txtAlgoDesc: QtWidgets.QPlainTextEdit = self.findChild(QtWidgets.QPlainTextEdit, "txtAlgoDesc")

        # step2
        self.tableClasses: QtWidgets.QTableWidget = self.findChild(QtWidgets.QTableWidget, "tableClasses")
        self.txtInner: QtWidgets.QLineEdit = self.findChild(QtWidgets.QLineEdit, "txtInner")
        self.txtOuter: QtWidgets.QLineEdit = self.findChild(QtWidgets.QLineEdit, "txtOuter")

        # step3
        self.tableRec: QtWidgets.QTableWidget = self.findChild(QtWidgets.QTableWidget, "tableRec")
        self.plotArea: QtWidgets.QWidget = self.findChild(QtWidgets.QWidget, "plotArea")

        # 그래프
        self.canvas = MplCanvas(self.plotArea)
        lay = QtWidgets.QVBoxLayout(self.plotArea)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

        # 이벤트 연결
        self.btnPrev.clicked.connect(self._on_prev)
        self.btnNext.clicked.connect(self._on_next)
        self.btnCancel.clicked.connect(self.reject)
        self.btnRegister.clicked.connect(self._on_register)

        self.rClustering.toggled.connect(self._update_algo_desc)
        self.rSimilarity.toggled.connect(self._update_algo_desc)
        self.cboMap.currentIndexChanged.connect(self._validate_step1)
        self.txtInner.textChanged.connect(self._validate_step2)
        self.txtOuter.textChanged.connect(self._validate_step2)
        self.tableClasses.itemChanged.connect(self._on_class_item_changed)

        # 초기화
        self._init_step1_demo()
        self._init_step2_demo()
        self._init_step3_table()
        self._goto(0)

    # ---------------------------
    # Step1
    # ---------------------------
    def _init_step1_demo(self):
        # 데모용 맵 채우기 (실제는 프로젝트 내 분류맵 목록으로 교체)
        self.cboMap.addItems(["Classification1.map", "Classification2.map"])
        self._update_algo_desc()
        self._validate_step1()

    def _update_algo_desc(self):
        if self.rClustering.isChecked():
            self.txtAlgoDesc.setPlainText("클러스터링 설명: 선택 클래스의 분포를 군집화하여 대표 스펙트럼 후보를 제안합니다.")
        else:
            self.txtAlgoDesc.setPlainText("유사도 설명: 기준 라벨 스펙트럼과 유사한 픽셀을 거리(metric) 기반으로 추천합니다.")
        self._validate_step1()

    def _validate_step1(self):
        ok = self.cboMap.currentIndex() >= 0 and (self.rClustering.isChecked() or self.rSimilarity.isChecked())
        self.btnNext.setEnabled(ok and self.stacked.currentIndex() == 0)

    def _collect_step1(self):
        text = self.cboMap.currentText()
        self.state.target_map_path = Path(text) if text else None
        self.state.algo = "clustering" if self.rClustering.isChecked() else "similarity"

    # ---------------------------
    # Step2
    # ---------------------------
    def _init_step2_demo(self):
        self.tableClasses.setRowCount(5)
        self.tableClasses.setColumnCount(5)
        headers = ["#", "■", "Class", "Name", "Count"]
        self.tableClasses.setHorizontalHeaderLabels(headers)
        self.tableClasses.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        # 데모 데이터
        classes = [
            (17, "Name 17", 121, QtGui.QColor("#00bcd4")),
            (25, "Name 25", 30,  QtGui.QColor("#8bc34a")),
            (27, "Name 27", 35,  QtGui.QColor("#ffc107")),
            (29, "Name 29", 256, QtGui.QColor("#ff9800")),
            (33, "Name 33", 3,   QtGui.QColor("#f44336")),
        ]
        for r, (cid, name, cnt, color) in enumerate(classes):
            # 체크박스
            it_chk = QtWidgets.QTableWidgetItem()
            it_chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            it_chk.setCheckState(Qt.Unchecked)
            it_chk.setData(Qt.UserRole, cid)  # class id 저장
            self.tableClasses.setItem(r, 0, it_chk)

            # 색칩
            it_color = QtWidgets.QTableWidgetItem(" ")
            it_color.setFlags(Qt.ItemIsEnabled)
            it_color.setBackground(color)
            self.tableClasses.setItem(r, 1, it_color)

            # Class
            it_c = QtWidgets.QTableWidgetItem(f"Class {cid}")
            it_c.setFlags(Qt.ItemIsEnabled)
            self.tableClasses.setItem(r, 2, it_c)

            # Name
            it_n = QtWidgets.QTableWidgetItem(name)
            it_n.setFlags(Qt.ItemIsEnabled)
            self.tableClasses.setItem(r, 3, it_n)

            # Count
            it_cnt = QtWidgets.QTableWidgetItem(str(cnt))
            it_cnt.setFlags(Qt.ItemIsEnabled)
            self.tableClasses.setItem(r, 4, it_cnt)

        # 숫자 검증기
        self.txtInner.setValidator(QtGui.QDoubleValidator(0.0, 1e6, 6, self))
        self.txtOuter.setValidator(QtGui.QDoubleValidator(0.0, 1e6, 6, self))
        self._validate_step2()

    def _on_class_item_changed(self, item: QtWidgets.QTableWidgetItem):
        if item.column() == 0:  # 체크 변경
            self._validate_step2()

    def _validate_step2(self):
        # 체크된 클래스가 하나라도 있어야 함 + 파라미터 유효
        checked = self._get_checked_class_ids()
        ok = len(checked) > 0 and self._safe_float(self.txtInner.text()) is not None and self._safe_float(self.txtOuter.text()) is not None
        if self.stacked.currentIndex() == 1:
            self.btnNext.setEnabled(ok)

    def _collect_step2(self):
        self.state.selected_class_ids = self._get_checked_class_ids()
        self.state.inner_tau = float(self.txtInner.text())
        self.state.outer_tau = float(self.txtOuter.text())

    def _get_checked_class_ids(self) -> List[int]:
        ids = []
        for r in range(self.tableClasses.rowCount()):
            it = self.tableClasses.item(r, 0)
            if it and it.checkState() == Qt.Checked:
                cid = it.data(Qt.UserRole)
                ids.append(int(cid))
        return ids

    @staticmethod
    def _safe_float(s: str):
        try:
            return float(s)
        except Exception:
            return None

    # ---------------------------
    # Step3
    # ---------------------------
    def _init_step3_table(self):
        self.tableRec.setRowCount(0)
        self.tableRec.setColumnCount(5)
        self.tableRec.setHorizontalHeaderLabels(["#", "Locate", "Class", "metric", "value"])
        self.tableRec.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tableRec.itemChanged.connect(self._on_rec_item_changed)
        self.tableRec.itemSelectionChanged.connect(self._on_rec_selection_changed)

    def _fill_step3(self, items: List[Dict[str, Any]]):
        self.tableRec.blockSignals(True)
        self.tableRec.setRowCount(len(items))
        for r, it in enumerate(items):
            # 체크박스
            cell0 = QtWidgets.QTableWidgetItem()
            cell0.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            cell0.setCheckState(Qt.Checked if r < 3 else Qt.Unchecked)
            self.tableRec.setItem(r, 0, cell0)

            # Locate
            cell1 = QtWidgets.QTableWidgetItem(f"{it['x']}, {it['y']}")
            cell1.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tableRec.setItem(r, 1, cell1)

            # Class
            cell2 = QtWidgets.QTableWidgetItem(it["class_name"])
            cell2.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tableRec.setItem(r, 2, cell2)

            # metric
            cell3 = QtWidgets.QTableWidgetItem(it["metric"])
            cell3.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tableRec.setItem(r, 3, cell3)

            # value
            cell4 = QtWidgets.QTableWidgetItem(f"{it['value']:.3f}")
            cell4.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tableRec.setItem(r, 4, cell4)
        self.tableRec.blockSignals(False)
        self._update_register_button()
        # 첫 행 선택하여 그래프 표시
        if self.tableRec.rowCount() > 0:
            self.tableRec.setCurrentCell(0, 1)
        self._update_plot()

    def _on_rec_item_changed(self, item: QtWidgets.QTableWidgetItem):
        if item.column() == 0:  # 체크박스 변경
            self._update_register_button()

    def _on_rec_selection_changed(self):
        self._update_plot()

    def _checked_rows_in_rec(self) -> List[int]:
        rows = []
        for r in range(self.tableRec.rowCount()):
            it = self.tableRec.item(r, 0)
            if it and it.checkState() == Qt.Checked:
                rows.append(r)
        return rows

    def _update_register_button(self):
        rows = self._checked_rows_in_rec()
        self.btnRegister.setVisible(True if self.stacked.currentIndex() == 2 else False)
        self.btnRegister.setEnabled(len(rows) > 0)

    def _update_plot(self):
        self.canvas.ax.clear()
        # 기준(라벨 평균) 스펙트럼 - 데모용: 매끄러운 기준 곡선
        base = np.linspace(0, 1, 210)
        base_spec = 0.4 + 0.6 * np.sin(4 * np.pi * base)
        base_spec = (base_spec - base_spec.min()) / (base_spec.max() - base_spec.min() + 1e-6)
        self.canvas.ax.plot(base, base_spec, linewidth=2.5, label="기준 라벨 평균")

        # 모든 후보(얇게)
        for it in self.state.recommended:
            y = it["spectrum"]
            self.canvas.ax.plot(base, y, linewidth=0.8, alpha=0.6)

        # 선택된 행 강조
        row = self.tableRec.currentRow()
        if 0 <= row < len(self.state.recommended):
            y = self.state.recommended[row]["spectrum"]
            self.canvas.ax.plot(base, y, linewidth=3.0, marker="o", markevery=20, label="선택 후보")

        self.canvas.ax.set_xlabel("Wavelength (norm)")
        self.canvas.ax.set_ylabel("Reflectance (norm)")
        self.canvas.ax.set_title(self._current_class_title())
        self.canvas.ax.legend(loc="best")
        self.canvas.draw_idle()

    def _current_class_title(self) -> str:
        if self.state.recommended:
            return self.state.recommended[0]["class_name"]
        if self.state.selected_class_ids:
            return f"Class {self.state.selected_class_ids[0]}"
        return "Class"

    # ---------------------------
    # 네비게이션
    # ---------------------------
    def _goto(self, idx: int):
        self.stacked.setCurrentIndex(idx)
        # 헤더 굵게/활성도 표현
        self.lblStep1.setText("<b>1단계</b>" if idx == 0 else "1단계")
        self.lblStep2.setEnabled(idx >= 1)
        self.lblStep2.setText("<b>2단계</b>" if idx == 1 else ("2단계" if idx != 0 else "2단계"))
        self.lblStep3.setEnabled(idx >= 2)
        self.lblStep3.setText("<b>3단계</b>" if idx == 2 else ("3단계" if idx != 0 else "3단계"))

        self.btnPrev.setEnabled(idx > 0)
        if idx == 0:
            self._validate_step1()
        elif idx == 1:
            self._validate_step2()
        elif idx == 2:
            self.btnNext.setEnabled(False)
            self._update_register_button()

    def _on_prev(self):
        i = self.stacked.currentIndex()
        if i > 0:
            self._goto(i - 1)

    def _on_next(self):
        i = self.stacked.currentIndex()
        if i == 0:
            self._collect_step1()
            self._goto(1)
        elif i == 1:
            self._collect_step2()
            self._run_recommend()
        elif i == 2:
            self.accept()

    # ---------------------------
    # 실행/등록
    # ---------------------------
    def _run_recommend(self):
        self.btnNext.setEnabled(False)
        job = RecommendJob(self.state)
        job.signals.finished.connect(self._on_recommend_done)
        job.signals.error.connect(self._on_recommend_error)
        self.pool.start(job)

    def _on_recommend_done(self, items: List[Dict[str, Any]]):
        self.state.recommended = items
        self._fill_step3(items)
        self._goto(2)

    def _on_recommend_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "오류", f"추천 계산 실패:\n{msg}")
        self.btnNext.setEnabled(True)

    def _on_register(self):
        rows = self._checked_rows_in_rec()
        payload = [self.state.recommended[r] for r in rows]
        self.candidatesSelected.emit(payload)
        QtWidgets.QMessageBox.information(self, "등록", f"{len(payload)}개 후보를 등록했습니다.")
        self.accept()


# ---------------------------
# 단독 실행 테스트
# ---------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui_file = Path("RecommendationWizard.ui")  # 같은 폴더에 .ui가 있다고 가정
    dlg = RecommendationWizard(ui_path=ui_file)
    dlg.show()
    sys.exit(app.exec_())
