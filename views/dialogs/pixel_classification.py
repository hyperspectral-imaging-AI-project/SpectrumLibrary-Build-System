# views/dialogs/pixel_classification.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt


class PixelClassificationDialog(QtWidgets.QDialog):
    """
    분류 맵 생성 다이얼로그
    """
    
    # 시그널 정의
    classification_requested = pyqtSignal(dict)  # 분류 요청 시그널
    
    def __init__(self, parent=None, ui_dir: Optional[Path] = None):
        super().__init__(parent)
        self.setWindowTitle("분류 맵 생성")
        
        # UI 로드
        app_dir = Path(getattr(parent, "app_dir", Path(__file__).resolve().parents[3]))
        ui_dir = Path(ui_dir) if ui_dir else (app_dir / "ui")
        self._root = uic.loadUi(str(ui_dir / "classmap.ui"))
        
        # 레이아웃 설정
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._root)
        
        # 기본값 설정
        self._root.lineEdit_threshold.setText("0.05")
        self._root.lineEdit_margin.setText("0.03")
        
        # 버튼 연결
        self._root.pushButton_save.clicked.connect(self._on_save_map)
        self._root.pushButton_cancel.clicked.connect(self.reject)
        
        # 라디오 버튼 그룹 설정
        self._metric_group = QtWidgets.QButtonGroup(self)
        self._metric_group.addButton(self._root.radioButton_sad, 0)
        self._metric_group.addButton(self._root.radioButton_sid, 1)
        self._metric_group.addButton(self._root.radioButton_scc, 2)
        
    def get_metric(self) -> str:
        """선택된 유사도 지표 반환"""
        checked_id = self._metric_group.checkedId()
        if checked_id == 0:
            return "SAD"
        elif checked_id == 1:
            return "SID"
        elif checked_id == 2:
            return "SCC"
        else:
            return "SAD"  # 기본값
            
    def get_thresholds(self) -> tuple[float, float]:
        """임계값과 마진값 반환"""
        try:
            tau = float(self._root.lineEdit_threshold.text())
            delta = float(self._root.lineEdit_margin.text())
            return tau, delta
        except ValueError:
            return 0.05, 0.03  # 기본값
            
    def get_map_name(self) -> str:
        """맵 이름 반환"""
        name = self._root.lineEdit_map_name.text().strip()
        return name if name else "classification1"
        
    def get_parameters(self) -> dict:
        """분류 파라미터 반환"""
        try:
            metric = self.get_metric()
            tau, delta = self.get_thresholds()
            map_name = self.get_map_name()
            
            return {
                "metric": metric,
                "tau": tau,
                "delta": delta,
                "map_name": map_name
            }
        except Exception:
            return {
                "metric": "SAD",
                "tau": 0.05,
                "delta": 0.03,
                "map_name": "classification1"
            }
            
    def _on_save_map(self):
        """맵 저장 버튼 클릭"""
        params = self.get_parameters()
        
        # 파라미터 유효성 검사
        # if params["tau"] <= 0 or params["delta"] <= 0:
        #     QtWidgets.QMessageBox.warning(self, "경고", "임계값과 마진값은 0보다 커야 합니다.")
        #     return
            
        self.classification_requested.emit(params)
        self.accept()
        
    def set_parameters(self, params: dict):
        """파라미터 설정"""
        # 메트릭 설정
        metric = params.get("metric", "SAD")
        if metric == "SAD":
            self._root.radioButton_sad.setChecked(True)
        elif metric == "SID":
            self._root.radioButton_sid.setChecked(True)
        elif metric == "SCC":
            self._root.radioButton_scc.setChecked(True)
            
        # 임계값 설정
        self._root.lineEdit_threshold.setText(str(params.get("tau", 0.05)))
        self._root.lineEdit_margin.setText(str(params.get("delta", 0.03)))
        
        # 맵 이름 설정
        self._root.lineEdit_map_name.setText(params.get("map_name", "classification1"))