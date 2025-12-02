# views/dialogs/unmixing_dialog.py
from __future__ import annotations
from pathlib import Path
from typing import Optional

from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import pyqtSignal


class UnmixingDialog(QtWidgets.QDialog):
    """
    분광 혼합 분석 맵 생성 다이얼로그 (UI: unmixing_dialog.ui)
    - 에드멤버 수(lineEditEndmember): 분광 혼합 분석에 사용할 에드멤버 개수
    - 임계값(lineEditThreshold): 분광 혼합 분석 임계값
    """
    unmixing_requested = pyqtSignal(dict)  # {endmember_count, threshold}

    def __init__(self, parent=None, ui_dir: Optional[Path] = None):
        super().__init__(parent)
        self.setWindowTitle("분광 혼합 분석 맵 생성")

        app_dir = Path(getattr(parent, "app_dir", Path(__file__).resolve().parents[3]))
        ui_dir = Path(ui_dir) if ui_dir else (app_dir / "ui")
        ui_path = ui_dir / "unmixing_dialog.ui"
        
        if not ui_path.exists():
            raise FileNotFoundError(f"UI file not found: {ui_path}")
        
        uic.loadUi(str(ui_path), self)

        # 위젯 핸들(디자이너 objectName 기준)
        self.lineEditEndmember: QtWidgets.QLineEdit = self.findChild(QtWidgets.QLineEdit, "lineEditEndmember")
        self.lineEditThreshold: QtWidgets.QLineEdit = self.findChild(QtWidgets.QLineEdit, "lineEditThreshold")
        self.pushButtonRun: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "pushButtonRun")

        # 버튼 연결
        if self.pushButtonRun:
            self.pushButtonRun.clicked.connect(self._on_run_clicked)

    def get_parameters(self) -> dict:
        """현재 설정된 파라미터를 반환"""
        try:
            endmember_count = int(self.lineEditEndmember.text() or "3")
        except ValueError:
            endmember_count = 3
        
        try:
            threshold = float(self.lineEditThreshold.text() or "0.8")
        except ValueError:
            threshold = 0.8
        
        return {
            "endmember_count": endmember_count,
            "threshold": threshold,
        }

    def _on_run_clicked(self):
        """분광 혼합 분석 실행 버튼 클릭 핸들러"""
        try:
            params = self.get_parameters()
            
            # 유효성 검사
            if params["endmember_count"] < 1:
                QtWidgets.QMessageBox.warning(self, "경고", "에드멤버 수는 1 이상이어야 합니다.")
                return
            
            if not (0.0 <= params["threshold"] <= 1.0):
                QtWidgets.QMessageBox.warning(self, "경고", "임계값은 0.0과 1.0 사이의 값이어야 합니다.")
                return
            
            # 신호 발생
            self.unmixing_requested.emit(params)
            self.accept()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", f"파라미터 처리 중 오류가 발생했습니다: {e}")

