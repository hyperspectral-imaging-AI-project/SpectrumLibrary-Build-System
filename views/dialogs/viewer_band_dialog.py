# views/dialogs/viewer_band_dialog.py
from __future__ import annotations
from typing import Iterable, List, Dict, Any, Optional, Tuple

from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtCore import Qt, pyqtSignal
from pathlib import Path

class ViewerBandDialog(QtWidgets.QDialog):
    """
    '뷰어 대역 선택' 다이얼로그 래퍼
    - set_wavelengths(wl_list)로 파장 선택지 세팅
    - exec_() 후 get_result()로 선택 결과(dict) 반환
    """
    selectionChanged = pyqtSignal(dict)   # {"mode":"gray","gray":700.1} 또는 {"mode":"rgb","r":..,"g":..,"b":..}
    
    def __init__(self, parent=None, ui_dir: Optional[Path] = None, ui_filename: str = "viewer_band_dialog.ui"):
        super().__init__(parent)
        app_dir = Path(getattr(parent, "app_dir", Path(__file__).resolve().parents[2]))
        ui_dir = Path(ui_dir) if ui_dir else (app_dir / "ui")
        uic.loadUi(str(ui_dir / ui_filename), self)

        # 필수 위젯 핸들
        self.radGray: QtWidgets.QRadioButton = self.findChild(QtWidgets.QRadioButton, "radGray")
        self.radRGB:  QtWidgets.QRadioButton = self.findChild(QtWidgets.QRadioButton, "radRGB")
        self.stack:   QtWidgets.QStackedWidget = self.findChild(QtWidgets.QStackedWidget, "stack")

        self.cboGray: QtWidgets.QComboBox = self.findChild(QtWidgets.QComboBox, "cboGray")
        self.cboR:    QtWidgets.QComboBox = self.findChild(QtWidgets.QComboBox, "cboR")
        self.cboG:    QtWidgets.QComboBox = self.findChild(QtWidgets.QComboBox, "cboG")
        self.cboB:    QtWidgets.QComboBox = self.findChild(QtWidgets.QComboBox, "cboB")

        self.btnOk:     QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btnOk")
        self.btnCancel: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "btnCancel")
        
        # ✅ 변경 이벤트 연결 (디바운스 60ms)
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(60)
        self._debounce.timeout.connect(lambda: self.selectionChanged.emit(self.get_result()))

        if self.radGray:
            self.radGray.toggled.connect(self._on_any_changed)
        if self.radRGB:
            self.radRGB.toggled.connect(self._on_any_changed)
        for cbo in (self.cboGray, self.cboR, self.cboG, self.cboB):
            if cbo:
                cbo.currentTextChanged.connect(self._on_any_changed)
                cbo.currentIndexChanged.connect(self._on_any_changed)
        # 연결
        if self.radGray:
            self.radGray.toggled.connect(lambda on: self.stack.setCurrentIndex(0 if on else 1))
        if self.btnOk:     self.btnOk.clicked.connect(self.accept)
        if self.btnCancel: self.btnCancel.clicked.connect(self.reject)

        # 기본값
        if self.radGray: self.radGray.setChecked(True)
        if self.stack:   self.stack.setCurrentIndex(0)

    # --- 외부 API ---
    def set_wavelengths(self, wavelengths: Iterable[float], fmt: str = "{:.1f}") -> None:
        """
        콤보박스 항목을 파장 리스트로 세팅.
        fmt : 콤보 표시 포맷 (기본 소수1자리)
        """
        wl_list = [fmt.format(float(w)) for w in wavelengths] if wavelengths else []
        for cbo in (self.cboGray, self.cboR, self.cboG, self.cboB):
            if not cbo: continue
            cbo.clear()
            if wl_list:
                cbo.addItems(wl_list)
            cbo.setEditable(True)

        # 첫 값 기본 선택
        for cbo in (self.cboGray, self.cboR, self.cboG, self.cboB):
            if cbo and cbo.count() > 0:
                cbo.setCurrentIndex(0)

    def get_result(self) -> Dict[str, Any]:
        """
        exec_() == Accepted 후 호출.
        return:
          {"mode":"gray","gray":700.1}  또는
          {"mode":"rgb","r":700.1,"g":550.3,"b":450.1}
        """
        def _read_float(cbo: QtWidgets.QComboBox) -> Optional[float]:
            if not cbo: return None
            try: return float(cbo.currentText().strip())
            except Exception: return None

        if self.radGray and self.radGray.isChecked():
            return {"mode": "gray", "gray": _read_float(self.cboGray)}
        else:
            return {
                "mode": "rgb",
                "r": _read_float(self.cboR),
                "g": _read_float(self.cboG),
                "b": _read_float(self.cboB),
            }

    def _on_any_changed(self, *args):
        # 라디오 전환 시 스택 변경도 유지
        if self.radGray:
            self.stack.setCurrentIndex(0 if self.radGray.isChecked() else 1)
        self._debounce.start()
        
    def set_initial_selection(self, sel: dict, fmt: str = "{:.1f}"):
        """
        sel 예시:
        {"mode":"gray", "gray": 700.1}
        {"mode":"rgb",  "r": 700.1, "g": 550.3, "b": 450.1}
        wavelengths 콤보가 이미 set_wavelengths로 채워져 있다는 가정.
        """
        def _set_combo_text(cbo, v):
            if not cbo or v is None:
                return
            try:
                txt = fmt.format(float(v))
            except Exception:
                txt = str(v)
            if cbo.findText(txt) < 0:
                cbo.addItem(txt)
            cbo.setCurrentText(txt)

        mode = (sel or {}).get("mode", "gray")
        if mode == "gray":
            if self.radGray: self.radGray.setChecked(True)
            if self.stack:   self.stack.setCurrentIndex(0)
            _set_combo_text(self.cboGray, sel.get("gray"))
        else:
            if self.radRGB: self.radRGB.setChecked(True)
            if self.stack:  self.stack.setCurrentIndex(1)
            _set_combo_text(self.cboR, sel.get("r"))
            _set_combo_text(self.cboG, sel.get("g"))
            _set_combo_text(self.cboB, sel.get("b"))
