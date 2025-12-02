# views/docks/unmixing_dock.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Any, Dict
import logging

from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
from views.dialogs.endmember_detail_dialog import EndmemberDetailDialog


class UnmixingDock(QtWidgets.QDockWidget):
    """
    분광 혼합 분석 Dock
    UI: unmixing_dock.ui

    - MainWindow → Dock:
        * set_results(endmembers, abundance_map, threshold)
        * set_endmember_count(count)
        * set_threshold(threshold)

    - Dock → MainWindow:
        * thresholdChanged(float)  # 0.0 ~ 1.0 범위의 임계값
    """

    # --- MainWindow로 보낼 신호: 임계값 변경 ---
    thresholdChanged = pyqtSignal(float)
    classMappingApplied = pyqtSignal(dict)  # {endmember_index: class_id}

    def __init__(self, parent=None, ui_dir: Optional[Path] = None):
        super().__init__("Unmixing Analysis", parent)
        self.setObjectName("dockUnmixing")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # 내부 상태
        self._endmembers: Optional[np.ndarray] = None    # (K, C) 또는 (K, ...)
        self._abundance_map: Optional[np.ndarray] = None # (H, W, K)
        self._current_threshold: Optional[float] = None  # 0~1
        self._default_threshold: Optional[float] = None  # 처음 설정된 임계값(Reset용)
        self._wavelengths: Optional[np.ndarray] = None  # 파장 배열 (C,)

        # --- UI 로드 ---
        app_dir = Path(getattr(parent, "app_dir", Path(__file__).resolve().parents[2]))
        ui_dir = Path(ui_dir) if ui_dir else (app_dir / "ui")
        ui_path = ui_dir / "unmixing_dock.ui"

        if not ui_path.exists():
            raise FileNotFoundError(f"UI file not found: {ui_path}")

        self._root = uic.loadUi(str(ui_path))
        self.setWidget(self._root)

        # --- 위젯 핸들 수집 ---
        self.labelEndmemberCount: QtWidgets.QLabel = self._find(QtWidgets.QLabel, "labelEndmemberCount")
        self.lineEditThreshold: QtWidgets.QLineEdit = self._find(QtWidgets.QLineEdit, "lineEditThreshold")
        self.pushButtonApplyThreshold: QtWidgets.QPushButton = self._find(QtWidgets.QPushButton, "pushButtonApplyThreshold")
        self.tableEndmember: QtWidgets.QTableWidget = self._find(QtWidgets.QTableWidget, "tableEndmember")
        self.pushButtonApply: QtWidgets.QPushButton = self._find(QtWidgets.QPushButton, "pushButtonApply")
        self.pushButtonReset: QtWidgets.QPushButton = self._find(QtWidgets.QPushButton, "pushButtonReset")

        # --- 테이블 초기화 ---
        if self.tableEndmember:
            self.tableEndmember.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.tableEndmember.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        # --- 버튼/이벤트 연결 ---
        if self.pushButtonApplyThreshold:
            self.pushButtonApplyThreshold.clicked.connect(self._on_apply_threshold_clicked)

        # 필요하다면 Apply 버튼도 같은 역할로 사용 가능
        if self.pushButtonApply:
            self.pushButtonApply.clicked.connect(self._on_apply_threshold_clicked)

        if self.pushButtonReset:
            self.pushButtonReset.clicked.connect(self._on_reset_clicked)
        self._class_options: List[tuple[int, str]] = []  # ← 추가

    def _find(self, cls, name: str):
        """위젯 찾기 헬퍼"""
        try:
            return self._root.findChild(cls, name)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # MainWindow → Dock 인터페이스
    # ------------------------------------------------------------------
    def set_endmember_count(self, count: int):
        """에드멤버 수 라벨 표시"""
        if self.labelEndmemberCount:
            self.labelEndmemberCount.setText(f"엔드멤버 수 : {count}")

    def set_threshold(self, threshold: float):
        """
        임계값 설정
        threshold: 0.0 ~ 1.0
        """
        self._current_threshold = float(threshold)
        # 최초 한 번 들어온 값을 기본값으로 저장 (Reset 용)
        if self._default_threshold is None:
            self._default_threshold = float(threshold)

        if self.lineEditThreshold:
            # UI에는 퍼센트로 표시 (예: 0.8 → "80%")
            self.lineEditThreshold.setText(f"{threshold * 100:.0f}%")

    def get_threshold(self) -> Optional[float]:
        """
        임계값 가져오기
        Returns:
            float: 0.0 ~ 1.0, 파싱 실패 시 None
        """
        if not self.lineEditThreshold:
            return None
        try:
            text = self.lineEditThreshold.text()
            if text is None:
                return None
            text = text.replace("%", "").strip()
            if text == "":
                return None
            value = float(text)
            # 1보다 크면 퍼센트로 보고 100으로 나눔
            return value / 100.0 if value > 1.0 else value
        except ValueError:
            return None

    def set_endmember_table(self, endmembers: List[dict]):
        """
        에드멤버 테이블 설정
        Args:
            endmembers: [{"index": int, "class_id": int|None, ...}, ...] 형태를 권장
        """
        if not self.tableEndmember:
            return

        self.tableEndmember.setRowCount(len(endmembers))
        for i, em in enumerate(endmembers):
            # ----- 1) # 컬럼 -----
            index_val = em.get("index", i + 1)
            index_item = QtWidgets.QTableWidgetItem(str(index_val))
            index_item.setFlags(index_item.flags() & ~Qt.ItemIsEditable)
            self.tableEndmember.setItem(i, 0, index_item)

            # ----- 2) Class 콤보박스 -----
            combo = QtWidgets.QComboBox(self.tableEndmember)
            combo.setEditable(False)

            # 옵션: 텍스트는 '물질명', data에는 cid
            for cid, name in self._class_options:
                combo.addItem(str(name), cid)

            # 기본 선택 CID 결정
            default_cid = em.get("class_id", None)

            # (A) row 정보에 class_id가 있다면 그걸 우선 사용
            if default_cid is not None:
                try:
                    default_cid = int(default_cid)
                except ValueError:
                    default_cid = None

            # (B) class_id가 비어 있으면, # 번호(=index)에 맞는 순번의 클래스를 자동으로 선택
            # 예: 1행(#=1) → _class_options[0]의 cid, 2행 → _class_options[1]의 cid ...
            if default_cid is None and self._class_options:
                # index_val은 1-based, 리스트는 0-based 이므로 -1
                idx_in_options = max(0, min(index_val - 1, len(self._class_options) - 1))
                default_cid = self._class_options[idx_in_options][0]

            # 실제 콤보박스 선택 반영
            if default_cid is not None:
                idx_match = combo.findData(default_cid)
                if idx_match >= 0:
                    combo.setCurrentIndex(idx_match)

            self.tableEndmember.setCellWidget(i, 1, combo)

            # ----- 3) Detail 버튼 -----
            detail_btn = QtWidgets.QPushButton("Detail")
            detail_btn.setProperty("index", i)
            detail_btn.clicked.connect(lambda checked, idx=i: self._on_detail_clicked(idx))
            self.tableEndmember.setCellWidget(i, 2, detail_btn)

        self.tableEndmember.resizeColumnsToContents()


    # MainWindow에서 직접 결과를 넘겨줄 때 사용할 통합 메서드
    def set_results(self, endmembers: Any, abundance_map: Any, threshold: float):
        """
        Unmixing 결과를 Dock에 전달 (MainWindow에서 호출)

        Args:
            endmembers: 엔드멤버 스펙트럼 (np.ndarray 또는 list)
            abundance_map: 풍부도 맵 (np.ndarray 또는 list)
            threshold: 현재 임계값 (0.0 ~ 1.0)
        """
        self._endmembers = np.asarray(endmembers)
        self._abundance_map = np.asarray(abundance_map)

        # 엔드멤버 개수 표시
        if self._endmembers.ndim >= 1:
            k = int(self._endmembers.shape[0])
        else:
            k = 0
        self.set_endmember_count(k)

        # 임계값 표시/저장
        self.set_threshold(threshold)

        # 엔드멤버 테이블 구성 (간단 버전: EM0, EM1, ...)
        rows = []
        for i in range(k):
            rows.append({
                "index": i + 1,
                "class_id": None,   # 아직 매핑 전이므로 None
            })
        self.set_endmember_table(rows)
   
    def set_wavelengths(self, wavelengths: np.ndarray):
        """파장 배열 설정"""
        self._wavelengths = np.asarray(wavelengths)

    # ------------------------------------------------------------------
    # 버튼 핸들러
    # ------------------------------------------------------------------
    def _on_apply_threshold_clicked(self):
        """
        '설정' 버튼(Apply): 임계값 + 엔드멤버→클래스 매핑을 MainWindow로 보냄
        """
        thr = self.get_threshold()
        if thr is None:
            QtWidgets.QMessageBox.warning(self, "경고", "임계값 형식이 올바르지 않습니다.")
            if self._current_threshold is not None:
                self.set_threshold(self._current_threshold)
            return

        thr = max(0.0, min(1.0, thr))
        self.set_threshold(thr)

        # 1) 임계값 변경 알림
        self.thresholdChanged.emit(thr)

        # 2) 엔드멤버 → 클래스 매핑 알림
        mapping = self._collect_class_mapping()
        if mapping:
            self.classMappingApplied.emit(mapping)

    def _on_reset_clicked(self):
        """
        'Reset' 버튼:
        - 최초 설정된 기본 임계값(_default_threshold)로 되돌리고
        - thresholdChanged(default) 신호 발행
        """
        if self._default_threshold is None:
            # 기본값이 없는 경우에는 아무것도 하지 않음
            return

        self.set_threshold(self._default_threshold)
        self.thresholdChanged.emit(self._default_threshold)
    
    def _on_detail_clicked(self, index: int):
        """
        Detail 버튼 클릭 핸들러
        
        Args:
            index: 엔드멤버 인덱스 (0-based)
        """
        try:
            # 엔드멤버 스펙트럼 가져오기
            if self._endmembers is None:
                QtWidgets.QMessageBox.warning(self, "경고", "엔드멤버 데이터가 없습니다.")
                return
            
            if index < 0 or index >= len(self._endmembers):
                QtWidgets.QMessageBox.warning(self, "경고", f"유효하지 않은 엔드멤버 인덱스: {index}")
                return
            
            endmember_spectrum = self._endmembers[index]
            
            # 다이얼로그 생성 및 표시
            app_dir = Path(getattr(self.parent(), "app_dir", Path(__file__).resolve().parents[2]))
            dlg = EndmemberDetailDialog(parent=self.parent(), ui_dir=app_dir / "ui")
            
            # 데이터 설정
            dlg.set_endmember_data(
                index=index,
                spectrum=endmember_spectrum,
                wavelengths=self._wavelengths,
            )
            
            # 모델리스로 표시
            dlg.setModal(False)
            dlg.show()
            
        except Exception as e:
            logging.exception(f"[UnmixingDock] Detail dialog failed for index {index}: {e}")
            QtWidgets.QMessageBox.critical(self, "오류", f"상세 다이얼로그 표시 중 오류가 발생했습니다: {e}")

    def set_class_options(self, options: List[tuple[int, str]]):
        """
        Class 콤보박스에 들어갈 옵션 설정.
        options: [(cid, material_name), ...]
        """
        self._class_options = [(int(cid), str(name)) for cid, name in options or []]

        # 이미 테이블이 채워진 상태라면, 현재 index/class_id 정보를 읽어서 다시 테이블 구성
        if self.tableEndmember and self.tableEndmember.rowCount() > 0:
            rows: List[Dict[str, Any]] = []
            row_count = self.tableEndmember.rowCount()
            for r in range(row_count):
                # # 컬럼에서 index 읽기
                idx_item = self.tableEndmember.item(r, 0)
                try:
                    index_val = int(idx_item.text()) if idx_item else (r + 1)
                except Exception:
                    index_val = r + 1

                # Class 콤보에서 현재 선택된 cid 읽기 (있으면 유지, 없으면 None)
                cur_cid = None
                combo = self.tableEndmember.cellWidget(r, 1)
                if isinstance(combo, QtWidgets.QComboBox):
                    cur_cid = combo.currentData()

                rows.append({
                    "index": index_val,
                    "class_id": cur_cid,
                })

            # 새 옵션(self._class_options)을 반영해서 콤보박스를 다시 채운다
            self.set_endmember_table(rows)
        
    def _collect_class_mapping(self) -> Dict[int, int]:
        """
        테이블에서 endmember index → class_id 매핑을 수집.
        row index(=endmember index) 기준으로 콤보박스의 cid를 읽는다.
        """
        mapping: Dict[int, int] = {}
        if not self.tableEndmember:
            return mapping

        row_count = self.tableEndmember.rowCount()
        for row in range(row_count):
            combo = self.tableEndmember.cellWidget(row, 1)
            if isinstance(combo, QtWidgets.QComboBox):
                cid = combo.currentData()
                if cid is not None:
                    mapping[row] = int(cid)  # row == endmember index
        return mapping
