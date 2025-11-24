# views/dialogs/workspace_dialog.py
from __future__ import annotations
from pathlib import Path
from typing import Optional

from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import pyqtSignal


class WorkspaceDialog(QtWidgets.QDialog):
    """
    작업 영역 선택 다이얼로그
    - 사각형, 다각형, 자유형 버튼으로 ROI 모드 선택
    """
    
    # 시그널 정의: ROI 모드 변경 요청 ('rect', 'poly', 'none')
    roi_mode_requested = pyqtSignal(str)
    # 완료 버튼 클릭 시 작업 영역 등록 요청
    register_workspace_requested = pyqtSignal()
    
    def __init__(self, parent=None, ui_dir: Optional[Path] = None):
        super().__init__(parent)
        
        # UI 로드
        app_dir = Path(getattr(parent, "app_dir", Path(__file__).resolve().parents[3]))
        ui_dir = Path(ui_dir) if ui_dir else (app_dir / "ui")
        ui_path = ui_dir / "workspace_dialog.ui"
        
        if not ui_path.exists():
            raise FileNotFoundError(f"UI file not found: {ui_path}")
        
        uic.loadUi(str(ui_path), self)
        
        # 버튼 그룹 설정 (Rect, Polygon, Freehand는 배타적)
        self._shape_group = QtWidgets.QButtonGroup(self)
        self._shape_group.addButton(self.btnRect, 0)
        self._shape_group.addButton(self.btnPolygon, 1)
        self._shape_group.addButton(self.btnFreehand, 2)
        
        # 버튼 연결
        self.btnRect.clicked.connect(lambda: self._on_shape_button_clicked('rect'))
        self.btnPolygon.clicked.connect(lambda: self._on_shape_button_clicked('poly'))
        self.btnFreehand.clicked.connect(lambda: self._on_shape_button_clicked('poly'))  # 자유형도 poly 모드
        
        # 완료/취소 버튼
        self.btnOk.clicked.connect(self._on_ok_clicked)
        self.btnCancel.clicked.connect(self._on_cancel_clicked)
        
        # 다이얼로그가 닫혀도 계속 작동하도록 모달리스로 설정하지 않음
        # (모달 다이얼로그로 열리되, ROI 작업은 메인 윈도우에서 계속)
    
    def _on_shape_button_clicked(self, mode: str):
        """도형 버튼 클릭 시 ROI 모드 요청"""
        self.roi_mode_requested.emit(mode)
    
    def _on_ok_clicked(self):
        """완료 버튼 클릭 시 작업 영역 등록 요청"""
        self.register_workspace_requested.emit()
        self.accept()
    
    def _on_cancel_clicked(self):
        """취소 버튼 클릭 시 Dialog 닫기"""
        self.reject()
    
    def reset_shape_selection(self):
        """도형 선택 해제"""
        self._shape_group.setExclusive(False)
        for btn in [self.btnRect, self.btnPolygon, self.btnFreehand]:
            btn.setChecked(False)
        self._shape_group.setExclusive(True)

