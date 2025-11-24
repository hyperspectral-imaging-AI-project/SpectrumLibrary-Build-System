# views/dialogs/camera_add.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtGui import QDoubleValidator
from data.db import add_camera, get_camera_list

# Qt5 정규식 검증기 호환
try:
    from PyQt5.QtCore import QRegularExpression
    from PyQt5.QtGui import QRegularExpressionValidator
    _HAS_QREGEX = True
except Exception:
    from PyQt5.QtCore import QRegExp
    from PyQt5.QtGui import QRegExpValidator
    _HAS_QREGEX = False


class CameraAddDialog(QtWidgets.QDialog):
    """
    'camera_add.ui' 기반 카메라 등록 다이얼로그

    - [카테고리 가져오기]: 서버에 저장된 카메라 목록을 **조회만** 함(입력창 값 변경/저장 X)
    - [저장]: 입력값(name, wavelength, fwhm)으로 API 호출(add_camera) → 성공시 cmr_cd 수신
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        ui_dir: Optional[Path] = None,
        ui_filename: str = "camera_add.ui",
    ):
        super().__init__(parent)
        self.app_dir: Path = Path(getattr(parent, "app_dir", Path.cwd()))
        self.ui_dir: Path = Path(ui_dir) if ui_dir else (self.app_dir / "ui")
        self.ui_path: Path = self.ui_dir / ui_filename
        if not self.ui_path.exists():
            raise FileNotFoundError(f"UI file not found: {self.ui_path}")

        uic.loadUi(str(self.ui_path), self)

        # 위젯 참조
        self.le_name: QtWidgets.QLineEdit = self.findChild(QtWidgets.QLineEdit, "lineEdit_camera_name")
        self.le_filter: QtWidgets.QLineEdit = self.findChild(QtWidgets.QLineEdit, "lineEdit_camera_filter")
        self.le_fwhm: QtWidgets.QLineEdit = self.findChild(QtWidgets.QLineEdit, "lineEdit_camera_fwhm")

        self.btn_load_cat: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "pushButton_load_category")
        self.btn_save: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "pushButton_save")
        self.btn_cancel: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "pushButton_cancel")

        # ★ 명시적으로 편집 가능하도록 설정
        if self.le_name:
            self.le_name.setReadOnly(False)
            self.le_name.setEnabled(True)
        if self.le_filter:
            self.le_filter.setReadOnly(False)
            self.le_filter.setEnabled(True)
        if self.le_fwhm:
            self.le_fwhm.setReadOnly(False)
            self.le_fwhm.setEnabled(True)

        # 플레이스홀더
        self.le_name.setPlaceholderText("예: MyCamera")
        self.le_filter.setPlaceholderText("예: 400,500,600 또는 [400,500,600]")
        self.le_fwhm.setPlaceholderText("예: 8.1,8.1,8.1 또는 [8.1,8.1,8.1]")

        # 검증기
        self._setup_validators()

        # 시그널
        self.btn_load_cat.clicked.connect(self._on_load_category)  # 조회만
        self.btn_save.clicked.connect(self._on_save)               # 저장만
        self.btn_cancel.clicked.connect(self.reject)

        # 엔터키 → 저장
        self.le_name.returnPressed.connect(self._on_save)
        self.le_filter.returnPressed.connect(self._on_save)
        self.le_fwhm.returnPressed.connect(self._on_save)

        # 결과
        self.result: Dict[str, Any] = {}

    # ---------------------------
    # Validators
    # ---------------------------
    def _setup_validators(self) -> None:
        # ★ validator를 제거하여 입력 중에도 자유롭게 입력 가능하도록 함
        # 검증은 저장 시(_on_save)에만 수행
        # fwhm: 양의 실수 (입력 편의를 위해 validator 제거, 저장 시 검증)
        # self.le_fwhm.setValidator(QDoubleValidator(0.0, 1e9, 6, self))

        # wavelength(filter): "num-num" 형식 (입력 편의를 위해 validator 제거, 저장 시 검증)
        # if _HAS_QREGEX:
        #     reg = QRegularExpression(r"^\s*\d+(\.\d+)?\s*-\s*\d+(\.\d+)?\s*$")
        #     self.le_filter.setValidator(QRegularExpressionValidator(reg, self))
        # else:
        #     reg = QRegExp(r"^\s*\d+(\.\d+)?\s*-\s*\d+(\.\d+)?\s*$")
        #     self.le_filter.setValidator(QRegExpValidator(reg, self))
        pass  # validator 제거로 인해 함수 본문이 비어있음

    # ---------------------------
    # Category (조회만)
    # ---------------------------
    def _on_load_category(self) -> None:
        """
        카메라 조회 전용: 새 .ui를 즉시 로드해서 모달로 띄우고,
        조회 버튼으로 표를 채운 뒤, 확인 누르면 닫는다.
        부모 폼의 입력값은 절대 변경하지 않는다.
        """
        # 1) .ui 로드
        list_ui_path = self.ui_dir / "camera_add_search_camera.ui"   # ← 파일명을 이렇게 저장해 둬
        if not list_ui_path.exists():
            QtWidgets.QMessageBox.critical(self, "오류", f"UI 파일이 없습니다:\n{list_ui_path}")
            return
        try:
            dlg = uic.loadUi(str(list_ui_path))  # QDialog 인스턴스 반환
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", f"카테고리 창 로딩 실패:\n{e}")
            return

        # 2) 위젯 참조
        le_start: QtWidgets.QLineEdit = dlg.findChild(QtWidgets.QLineEdit, "lineEdit_StartWavelength")
        le_end:   QtWidgets.QLineEdit  = dlg.findChild(QtWidgets.QLineEdit, "lineEdit_EndWavelength")
        btn_q:    QtWidgets.QPushButton = dlg.findChild(QtWidgets.QPushButton, "pushButton_Query")
        btn_ok:   QtWidgets.QPushButton = dlg.findChild(QtWidgets.QPushButton, "pushButton_ok")
        table:    QtWidgets.QTableWidget = dlg.findChild(QtWidgets.QTableWidget, "tableWidget")

        if not all([le_start, le_end, btn_q, btn_ok, table]):
            QtWidgets.QMessageBox.critical(self, "오류", "camera_list.ui의 objectName이 예상과 다릅니다.")
            return

        # 3) 숫자 검증기 적용
        from PyQt5.QtGui import QDoubleValidator
        le_start.setValidator(QDoubleValidator(0.0, 1e7, 6, dlg))
        le_end.setValidator(QDoubleValidator(0.0, 1e7, 6, dlg))

        # 4) 테이블 기본 세팅
        def init_table():
            table.clear()
            table.setColumnCount(3)
            table.setHorizontalHeaderLabels(["카메라 이름", "Wavelength", "FWHM"])
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            table.setAlternatingRowColors(True)
            table.horizontalHeader().setStretchLastSection(True)
            table.setRowCount(0)
        init_table()

        # 5) 유틸
        def _read_float(le: QtWidgets.QLineEdit):
            txt = (le.text() or "").strip()
            if not txt: return None
            try: return float(txt)
            except Exception: return None

        def _fill(rows: List[Dict[str, Any]]):
            table.setRowCount(len(rows))
            def pick(d, keys, default=""):
                for k in keys:
                    if k in d and d[k] is not None:
                        return str(d[k])
                return default
            for r, row in enumerate(rows):
                name = pick(row, ["CMR_NM"])
                wl   = pick(row, ["WV"])
                fwhm = pick(row, ["FWHM"])
                table.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
                table.setItem(r, 1, QtWidgets.QTableWidgetItem(wl))
                table.setItem(r, 2, QtWidgets.QTableWidgetItem(str(fwhm) if fwhm else ""))
            if rows:
                table.selectRow(0)

        # 6) 조회 클릭 핸들러
        def do_query():
            s = _read_float(le_start)
            e = _read_float(le_end)
            if (s is not None and e is not None) and s > e:
                QtWidgets.QMessageBox.warning(dlg, "입력 오류", "Start Wavelength가 End보다 클 수 없습니다.")
                return
            base_url = os.getenv("camera_list_url")
            rows: List[Dict[str, Any]] = []
            try:
                # data.db.get_camera_list 시그니처 다양성 대응
                try:
                    rows = get_camera_list(base_url=base_url, st_wv=s, ed_wv=e)  # type: ignore
                except TypeError:
                    rows = get_camera_list(base_url=base_url)  # type: ignore
                    # 서버가 필터 미지원이라면 간단한 클라이언트 필터(옵션)
                    if (s is not None) or (e is not None):
                        rows = rows  # 실제 필터는 서버가 처리하는 게 정확함. 여기선 그대로 둠.
            except Exception as ex:
                QtWidgets.QMessageBox.critical(dlg, "조회 실패", f"서버에서 목록을 가져오지 못했습니다.\n\n{ex}")
                return
            _fill(rows)

        # 7) 시그널 연결
        btn_q.clicked.connect(do_query)
        le_start.returnPressed.connect(do_query)
        le_end.returnPressed.connect(do_query)
        btn_ok.clicked.connect(dlg.accept)

        # 8) 모달 실행
        dlg.exec_()
        # ※ 여기서는 폼 필드 자동 채움/저장 등을 하지 않습니다(요구사항).

    # ---------------------------
    # Save (입력 → API 저장만)
    # ---------------------------
    def _on_save(self) -> None:
        name = self.le_name.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "입력 오류", "카메라 이름을 입력하세요.")
            self.le_name.setFocus()
            return

        # 파장대역: 리스트 형태로 파싱
        wavelength_text = self.le_filter.text().strip()
        if not wavelength_text:
            QtWidgets.QMessageBox.warning(self, "입력 오류", "파장대역을 입력하세요. 예: 400,500,600 또는 [400,500,600]")
            self.le_filter.setFocus()
            return
        
        wavelength_list = self._parse_number_list(wavelength_text)
        if wavelength_list is None or len(wavelength_list) == 0:
            QtWidgets.QMessageBox.warning(self, "입력 오류", "파장대역을 올바르게 입력하세요. 예: 400,500,600 또는 [400,500,600]")
            self.le_filter.setFocus()
            return
        
        # 모든 파장 값이 양수인지 확인
        if any(w <= 0 for w in wavelength_list):
            QtWidgets.QMessageBox.warning(self, "입력 오류", "파장 값은 모두 양수여야 합니다.")
            self.le_filter.setFocus()
            return

        # FWHM: 리스트 형태로 파싱
        fwhm_text = self.le_fwhm.text().strip()
        if not fwhm_text:
            QtWidgets.QMessageBox.warning(self, "입력 오류", "FWHM을 입력하세요. 예: 8.1,8.1,8.1 또는 [8.1,8.1,8.1]")
            self.le_fwhm.setFocus()
            return
        
        fwhm_list = self._parse_number_list(fwhm_text)
        if fwhm_list is None or len(fwhm_list) == 0:
            QtWidgets.QMessageBox.warning(self, "입력 오류", "FWHM을 올바르게 입력하세요. 예: 8.1,8.1,8.1 또는 [8.1,8.1,8.1]")
            self.le_fwhm.setFocus()
            return
        
        # 모든 FWHM 값이 양수인지 확인
        if any(f <= 0 for f in fwhm_list):
            QtWidgets.QMessageBox.warning(self, "입력 오류", "FWHM 값은 모두 양수여야 합니다.")
            self.le_fwhm.setFocus()
            return
        
        # 파장과 FWHM 길이 확인 (API 요구사항)
        if len(fwhm_list) > 0 and len(fwhm_list) != len(wavelength_list):
            QtWidgets.QMessageBox.warning(
                self, "입력 오류", 
                f"파장대역({len(wavelength_list)}개)과 FWHM({len(fwhm_list)}개)의 개수가 일치하지 않습니다."
            )
            return

        # 결과 구성(다이얼로그 결과)
        self.result = {
            "name": name,
            "wavelength": wavelength_list,
            "fwhm": fwhm_list,
        }

        # API 저장
        base_url = os.getenv("camera_add_url")
        try:
            cmr_cd = add_camera(cmr_nm=name, wv=wavelength_list, fwhm=fwhm_list, base_url=base_url)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "저장 실패", f"서버 저장에 실패했습니다.\n\n{e}")
            return

        # cmr_cd는 int로 반환됨
        self.result["cmr_cd"] = cmr_cd
        QtWidgets.QMessageBox.information(self, "저장 완료", f"저장되었습니다.\ncmr_cd: {cmr_cd}")
        self.accept()

    def get_result(self) -> Dict[str, Any]:
        """exec_() 이후, 저장 시 설정값 반환."""
        return self.result

    # ---------------------------
    # Utils
    # ---------------------------
    def _read_float(self, le: QtWidgets.QLineEdit) -> Optional[float]:
        """단일 실수 읽기 (레거시 호환용)"""
        txt = (le.text() or "").strip()
        if not txt:
            return None
        try:
            return float(txt)
        except Exception:
            return None

    def _parse_number_list(self, text: str) -> Optional[List[float]]:
        """
        문자열을 숫자 리스트로 파싱
        지원 형식:
        - "400,500,600" 또는 "400, 500, 600" (쉼표 구분)
        - "[400,500,600]" (JSON 배열 형식)
        """
        if not text:
            return None
        
        text = text.strip()
        if not text:
            return None
        
        # JSON 배열 형식 처리
        if text.startswith('[') and text.endswith(']'):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [float(x) for x in parsed]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        
        # 쉼표 구분 형식 처리
        try:
            # 쉼표로 분리하고 각 요소를 실수로 변환
            parts = [p.strip() for p in text.split(',') if p.strip()]
            if not parts:
                return None
            return [float(p) for p in parts]
        except (ValueError, TypeError):
            return None
