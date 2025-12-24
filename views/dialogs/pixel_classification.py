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
        
        # 임계값/마진값 변경 시 결정 규칙 텍스트 업데이트
        self._root.lineEdit_threshold.textChanged.connect(self._update_decision_rules)
        self._root.lineEdit_margin.textChanged.connect(self._update_decision_rules)
        
        # 버튼 연결
        self._root.pushButton_save.clicked.connect(self._on_save_map)
        self._root.pushButton_cancel.clicked.connect(self.reject)
        
        # 라디오 버튼 그룹 설정
        self._metric_group = QtWidgets.QButtonGroup(self)
        self._metric_group.addButton(self._root.radioButton_sad, 0)
        self._metric_group.addButton(self._root.radioButton_sid, 1)
        self._metric_group.addButton(self._root.radioButton_scc, 2)
        
        # 라디오 버튼 변경 시 결정 규칙 텍스트 업데이트
        self._root.radioButton_sad.toggled.connect(self._update_decision_rules)
        self._root.radioButton_sid.toggled.connect(self._update_decision_rules)
        self._root.radioButton_scc.toggled.connect(self._update_decision_rules)
        
        # 초기 결정 규칙 텍스트 설정
        self._update_decision_rules()
        
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
        
        # 결정 규칙 텍스트 업데이트
        self._update_decision_rules()
    
    def _update_decision_rules(self):
        """선택된 메트릭에 따라 결정 규칙 텍스트 업데이트"""
        metric = self.get_metric()
        
        # plainTextEdit_rules 위젯 찾기
        rules_widget = None
        try:
            rules_widget = self._root.findChild(QtWidgets.QPlainTextEdit, "plainTextEdit_rules")
        except Exception:
            pass
        
        if rules_widget is None:
            return
        
        # 현재 임계값과 마진값 가져오기
        try:
            tau = float(self._root.lineEdit_threshold.text())
            delta = float(self._root.lineEdit_margin.text())
        except ValueError:
            tau = 0.05
            delta = 0.03
        
        # 메트릭별 결정 규칙 텍스트 생성
        if metric == "SAD":
            rules_text = f"""Unknown : 각도 > τ ({tau:.3f})
  → 가장 유사한 클래스와의 각도가 임계값보다 크면 미분류

c_i : 각도 ≤ τ ({tau:.3f}) AND (각도차 ≥ δ ({delta:.3f}))
  → 각도가 임계값 이하이고, 1등과 2등의 각도 차이가 마진 이상이면 단일 클래스

multiple : 그 외
  → 위 조건에 해당하지 않으면 중복 클래스"""
        
        elif metric == "SID":
            rules_text = f"""Unknown : 거리 > τ ({tau:.3f})
  → 가장 유사한 클래스와의 정보 거리가 임계값보다 크면 미분류

c_i : 거리 ≤ τ ({tau:.3f}) AND (거리차 ≥ δ ({delta:.3f}))
  → 거리가 임계값 이하이고, 1등과 2등의 거리 차이가 마진 이상이면 단일 클래스

multiple : 그 외
  → 위 조건에 해당하지 않으면 중복 클래스"""
        
        else:  # SCC
            rules_text = f"""Unknown : 거리 > τ ({tau:.3f})
  → 가장 유사한 클래스와의 상관 거리가 임계값보다 크면 미분류

c_i : 거리 ≤ τ ({tau:.3f}) AND (거리차 ≥ δ ({delta:.3f}))
  → 거리가 임계값 이하이고, 1등과 2등의 거리 차이가 마진 이상이면 단일 클래스

multiple : 그 외
  → 위 조건에 해당하지 않으면 중복 클래스"""
        
        rules_widget.setPlainText(rules_text)