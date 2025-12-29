# views/dialogs/image_load.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from data.db import get_camera_list
try:
    import requests  # FastAPI 호출용
except Exception:
    requests = None
from dataio.sidecar import read_info, write_info   # ★ 추가


class ImageLoadDialog(QtWidgets.QDialog):
    """
    - .ui의 오브젝트명에 맞춤
    - '영상 확인'(pushButton_2): 파일 검사 → shape 추출 → H/W/C 콤보 채움
    - '조회'(pushButton_Query): Start/End Wavelength로 FastAPI 호출 → tableWidget 채움
    - Load: 기존 결과 dict 반환 (axis_map 포함)
    """
    configReady = pyqtSignal(dict)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        ui_dir: Optional[Path] = None,
        ui_filename: str = "image_load.ui",
        last_dir: Optional[Path] = None,
    ):
        super().__init__(parent)

        # 경로
        self.app_dir: Path = Path(getattr(parent, "app_dir", Path.cwd()))
        self.ui_dir: Path = Path(ui_dir) if ui_dir else (self.app_dir / "ui")
        self.ui_path: Path = self.ui_dir / ui_filename

        if not self.ui_path.exists():
            raise FileNotFoundError(f"UI file not found: {self.ui_path}")
        uic.loadUi(str(self.ui_path), self)

        # 상태
        self.result: Dict[str, Any] = {}
        self._last_dir: Path = Path(last_dir) if last_dir else self.app_dir
        self._mat_selected: Optional[Path] = None
        self._mat_arr_name: Optional[str] = None
        self._mat_shape: Optional[Tuple[int, ...]] = None
        self._hdr_shape: Optional[Tuple[int, int, int]] = None  # HDR 헤더에서 읽은 (H,W,C)
        self._mat_ndim: int = 0
        self._api_rows: List[Dict[str, Any]] = []      # ← 조회 결과 원본 보관
        self._selected_camera: Optional[Dict[str, Any]] = None  # ← 표에서 선택된 카메라
        self._loaded_array: Optional[np.ndarray] = None   # ← 영상 확인/로드 시 읽은 배열을 캐시

        # ---- 위젯 참조 ----
        # 파일들
        self.le_hdr: QtWidgets.QLineEdit = self.lineEdit_hdrFile
        self.btn_hdr: QtWidgets.QPushButton = self.pushButton_findHdr
        self.le_raw: QtWidgets.QLineEdit = self.lineEdit_rawFile
        self.btn_raw: QtWidgets.QPushButton = self.pushButton_findRaw

        self.le_mat = self.findChild(QtWidgets.QLineEdit, "lineEdit")       # mat 경로
        self.btn_mat = self.findChild(QtWidgets.QPushButton, "pushButton")  # '찾아보기 ...'

        # 모드 토글
        self.chk_mat: QtWidgets.QCheckBox = self.checkBox_mat
        self.rad_hdr: QtWidgets.QRadioButton = self.radioButton_hdr_raw

        # 사용자 타입 라디오 버튼 (상호 배타적)
        self.rad_personal: QtWidgets.QRadioButton = self.findChild(QtWidgets.QRadioButton, "radioButton")  # 개인 사용자
        self.rad_server: QtWidgets.QRadioButton = self.findChild(QtWidgets.QRadioButton, "radioButton_2")   # 서버 사용자
        
        # QButtonGroup으로 상호 배타적 설정
        self.user_type_group = QtWidgets.QButtonGroup(self)
        if self.rad_personal:
            self.user_type_group.addButton(self.rad_personal, 0)  # 0 = 개인 사용자
        if self.rad_server:
            self.user_type_group.addButton(self.rad_server, 1)    # 1 = 서버 사용자
        
        # 기본값: 개인 사용자
        if self.rad_personal:
            self.rad_personal.setChecked(True)

        # 파장
        self.le_wl_start: QtWidgets.QLineEdit = self.lineEdit_StartWavelength
        self.le_wl_end: QtWidgets.QLineEdit = self.lineEdit_EndWavelength
        self.le_wl_start.setValidator(QDoubleValidator(0.0, 1e7, 6, self))
        self.le_wl_end.setValidator(QDoubleValidator(0.0, 1e7, 6, self))

        # 축 매핑 콤보
        self.cb_H: QtWidgets.QComboBox = self.comboAxisH
        self.cb_W: QtWidgets.QComboBox = self.comboAxisW
        self.cb_C: QtWidgets.QComboBox = self.comboAxisC

        # 버튼/테이블
        self.btn_query: QtWidgets.QPushButton = self.pushButton_Query
        self.btn_load: QtWidgets.QPushButton = self.pushButton_Load
        self.btn_cancel: QtWidgets.QPushButton = self.pushButton_Cancel
        self.table: QtWidgets.QTableWidget = self.tableWidget

        # 새로 추가된 "영상 확인" 버튼
        self.btn_check: QtWidgets.QPushButton = self.findChild(QtWidgets.QPushButton, "pushButton_2")

        # 시그널 연결
        if self.btn_mat:
            self.btn_mat.clicked.connect(self._on_find_mat)
        self.btn_hdr.clicked.connect(self._on_find_hdr)
        self.btn_raw.clicked.connect(self._on_find_raw)
        self.btn_query.clicked.connect(self._on_query)
        self.btn_load.clicked.connect(self._on_load_clicked)
        self.btn_cancel.clicked.connect(self.reject)
        if self.btn_check:
            self.btn_check.clicked.connect(self._on_check_clicked)

        self.chk_mat.toggled.connect(self._on_mat_toggled)
        self.rad_hdr.toggled.connect(self._on_hdr_toggled)
        self.table.itemSelectionChanged.connect(self._on_table_select)  # ← 추가

        # 초기 상태
        self._apply_mode()
        self._init_table()

    # ---------------- 모드 전환 ----------------
    def _on_mat_toggled(self, checked: bool) -> None:
        if checked:
            if self.rad_hdr.isChecked():
                self.rad_hdr.blockSignals(True)
                self.rad_hdr.setChecked(False)
                self.rad_hdr.blockSignals(False)
        else:
            if not self.rad_hdr.isChecked():
                self.rad_hdr.blockSignals(True)
                self.rad_hdr.setChecked(True)
                self.rad_hdr.blockSignals(False)
        self._apply_mode()

    def _on_hdr_toggled(self, checked: bool) -> None:
        if checked and self.chk_mat.isChecked():
            self.chk_mat.blockSignals(True)
            self.chk_mat.setChecked(False)
            self.chk_mat.blockSignals(False)
        if not checked and not self.chk_mat.isChecked():
            self.rad_hdr.blockSignals(True)
            self.rad_hdr.setChecked(True)
            self.rad_hdr.blockSignals(False)
        self._apply_mode()

    def _apply_mode(self) -> None:
        use_mat = self.chk_mat.isChecked()
        # 스위치 항상 활성화
        self.chk_mat.setEnabled(True)
        self.rad_hdr.setEnabled(True)
        # H/W/C는 항상 활성
        for w in (self.cb_H, self.cb_W, self.cb_C):
            w.setEnabled(True)
        # 파일선택 위젯만 모드에 따라
        for w in (self.le_mat, self.btn_mat):
            if w is not None:
                w.setEnabled(use_mat)
        for w in (self.le_hdr, self.btn_hdr, self.le_raw, self.btn_raw):
            if w is not None:
                w.setEnabled(not use_mat)

    # ---------------- 파일 열기 버튼들 ----------------
    def _on_find_hdr(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select ENVI HDR", str(self._last_dir), "ENVI Header (*.hdr);;All Files (*.*)"
        )
        if not path:
            return
        p = Path(path)
        self._last_dir = p.parent
        self.le_hdr.setText(str(p))
        # .raw 추정
        raw_guess = p.with_suffix(".raw")
        if raw_guess.exists():
            self.le_raw.setText(str(raw_guess))

        # 0) 같은 경로의 .info가 있으면 그 값으로 즉시 UI 채움
        loaded = read_info(p)
        if loaded:
            self._apply_info_to_ui(loaded)
            return

        # 1) .info 없으면 기존처럼 헤더 파싱 → 콤보 채움
        try:
            hdr = self._parse_envi_hdr(p)
            H = int(hdr.get("lines", 0))
            W = int(hdr.get("samples", 0))
            C = int(hdr.get("bands", 0) or 1)
            if H > 0 and W > 0:
                self._hdr_shape = (H, W, C)
                self._populate_axis_combos(self._hdr_shape)
            else:
                QtWidgets.QMessageBox.warning(self, "HDR 오류", "HDR에서 lines/samples를 읽지 못했습니다.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "HDR 오류", f"HDR 파싱 실패:\n{e}")

    def _on_find_raw(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select RAW", str(self._last_dir), "RAW (*.raw);;All Files (*.*)"
        )
        if not path:
            return
        p = Path(path)
        self._last_dir = p.parent
        self.le_raw.setText(str(p))

    def _on_find_mat(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select MAT File", str(self._last_dir), "MAT (*.mat);;All Files (*.*)"
        )
        if not path:
            return
        p = Path(path)
        self._last_dir = p.parent
        self.le_mat.setText(str(p))
        self._mat_selected = p
        # 변수/shape 스캔
        try:
            best_name, best_shape = self._pick_best_mat_array(p)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "MAT 오류", f"MAT 파일 스캔 실패:\n{e}")
            return

        if not best_name:
            QtWidgets.QMessageBox.information(self, "안내", "배열(ndarray) 변수를 찾지 못했습니다.")
            return

        # .info가 있으면 그 값으로 UI 반영(= 재방문 시 자동 세팅)
        loaded = read_info(p)
        if loaded:
            self._apply_info_to_ui(loaded)
        else:
            # 기존 동작 유지
            self._mat_arr_name = best_name
            self._mat_shape = best_shape
            self._mat_ndim = len(best_shape)
            self._populate_axis_combos(best_shape)

    # ---------------- 영상 확인 ----------------
    def _on_check_clicked(self) -> None:
        """
        1) hdr+raw 또는 mat 경로 확인
        2) 파일을 열어(또는 헤더/shape 파악) shape 확보
        3) H/W/C 콤보 자동 세팅
        4) 사용자에게 shape 안내
        """
        if self.chk_mat.isChecked():
            # 0) 라인에딧에서 경로 확보
            if not self._mat_selected:
                txt = (self.le_mat.text() or "").strip()
                if txt:
                    self._mat_selected = Path(txt)
            if not self._mat_selected or not self._mat_selected.exists():
                QtWidgets.QMessageBox.warning(self, "입력 오류", "유효한 MAT 파일을 선택하세요.")
                return

            # 1) .info가 있으면 먼저 UI에 반영(축 선택/Start-End/카메라 표) ← 원하신 동작
            _ = self._try_apply_info_from_inputs()  # 성공/실패와 무관하게 이어서 파일 검사 진행

            # 2) 기존 동작: 실제 파일에서 배열/shape 확인
            try:
                arr_name, shape = self._load_mat_info(self._mat_selected)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "로딩 실패", f"MAT 로딩 실패:\n{e}")
                return

            self._mat_arr_name = arr_name
            self._mat_shape = shape
            self._mat_ndim = len(shape)

            # .info가 축 선택을 이미 복원했다면 그대로 두고,
            # 없었으면 자동/기존 방식으로 콤보 채움
            if self.cb_H.currentData() is None or self.cb_W.currentData() is None:
                self._populate_axis_combos(shape)

            # 캐시 로드
            self._loaded_array = self._load_mat_array_full(self._mat_selected, self._mat_arr_name)

            # QtWidgets.QMessageBox.information(self, "확인", f"MAT 배열: {arr_name}\nshape: {shape}")


        else:
            # HDR & RAW
            hdr_txt = (self.le_hdr.text() or "").strip()
            raw_txt = (self.le_raw.text() or "").strip()
            if not hdr_txt:
                QtWidgets.QMessageBox.warning(self, "입력 오류", "HDR 파일을 선택하세요.")
                return
            if not raw_txt:
                # 추정 시도
                guess = Path(hdr_txt).with_suffix(".raw")
                if guess.exists():
                    raw_txt = str(guess)
                    self.le_raw.setText(raw_txt)
                else:
                    QtWidgets.QMessageBox.warning(self, "입력 오류", "RAW 파일을 선택하세요.")
                    return

            # HDR & RAW ...
            # 1) .info 먼저 적용(있으면 축/Start-End/카메라 표 복원)
            _ = self._try_apply_info_from_inputs()

            # 2) 기존 동작: shape 확인/배열 로드
            shape = self._load_hdr_raw_info(Path(hdr_txt), Path(raw_txt))
            self._hdr_shape = shape

            # .info가 축 선택을 이미 복원했다면 그대로 두고,
            # 없었으면 자동/기존 방식으로 콤보 채움
            if self.cb_H.currentData() is None or self.cb_W.currentData() is None:
                self._populate_axis_combos(shape)

            self._loaded_array = self._load_hdr_raw_array(Path(hdr_txt), Path(raw_txt))
            # QtWidgets.QMessageBox.information(self, "확인", f"HDR/RAW shape: {shape}")


    # ---------------- 조회(FastAPI) ----------------
    def _on_query(self) -> None:
        s_wl = self._read_float(self.le_wl_start)
        e_wl = self._read_float(self.le_wl_end)
        if s_wl is None or e_wl is None:
            QtWidgets.QMessageBox.warning(self, "입력 오류", "Start/End Wavelength를 숫자로 입력하세요.")
            return
        if s_wl > e_wl:
            QtWidgets.QMessageBox.warning(self, "입력 오류", "Start Wavelength가 End보다 큽니다.")
            return
        url = os.getenv("camera_list_url")
        if requests is None:
            QtWidgets.QMessageBox.warning(self, "요청 불가", "requests 패키지가 없습니다. 설치 후 시도하세요.")
            return

        try:
            data = get_camera_list(base_url = url, st_wv = s_wl, ed_wv = e_wl)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "요청 실패", f"FastAPI 요청 실패:\n{e}")
            return

        # 표 채우기
        self._populate_table_from_api(data)

    def _populate_table_from_api(self, data: Any) -> None:
        # 원본을 리스트로 정규화
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
            rows = data["items"]
        else:
            rows = []

        self._api_rows = rows  # ← 그대로 보관 (원본 키 보존)
        self.table.clear()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["카메라 이름", "Wavelength", "FWHM"])
        self.table.setRowCount(len(rows))

        def pick(d: Dict[str, Any], keys: List[str], default="") -> str:
            for k in keys:
                if k in d and d[k] is not None:
                    return str(d[k])
            return default

        for r, row in enumerate(rows):
            # 다양한 키 케이스 대응(대/소문자 섞임)
            name = pick(row, ["CMR_NM"])
            wl   = pick(row, ["WV"])
            fwhm = pick(row, ["FWHM"])

            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(wl))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(fwhm))

        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        if rows:
            self.table.selectRow(0)  # 선택 트리거 → _on_table_select에서 self._selected_camera 세팅
            
    def _on_table_select(self) -> None:
        r = self.table.currentRow()
        if r < 0 or r >= len(self._api_rows):
            self._selected_camera = None
            return
        raw = self._api_rows[r]
        self._selected_camera = self._normalize_camera_row(raw)
        
    def _normalize_camera_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """다양한 키 이름을 통일해서 반환."""
        def g(*keys, default=None):
            for k in keys:
                if k in row and row[k] is not None:
                    return row[k]
            return default
        code = g("CMR_CD")
        name = g("CMR_NM")
        wl   = g("WV")
        fwhm = g("FWHM")

        # wl이 "390.8, 391.2, ..." 같은 문자열일 수도 있고 리스트일 수도 있음
        wl_list = None
        wl_min = wl_max = None
        if isinstance(wl, (list, tuple)) and len(wl) > 0:
            try:
                vals = [float(x) for x in wl]
                wl_list = vals
                wl_min, wl_max = min(vals), max(vals)
            except Exception:
                pass
        elif isinstance(wl, str):
            # 쉼표/공백 기준으로 숫자 부분만 추출
            try:
                parts = [p.strip() for p in wl.replace("[","").replace("]","").split(",")]
                vals = [float(p) for p in parts if p]
                if vals:
                    wl_list = vals
                    wl_min, wl_max = min(vals), max(vals)
            except Exception:
                pass

        return {
            "code": code,
            "name": str(name),
            "wavelength": wl,           # 원문
            "wavelength_list": wl_list, # 정규화된 리스트(있다면)
            "wavelength_min": wl_min,
            "wavelength_max": wl_max,
            "fwhm": fwhm,
            "raw": row,                 # 원본 그대로
        }
        
    # ---------------- Load (확정) ----------------
    def _on_load_clicked(self) -> None:
        # 공통: 콤보 선택 읽기
        h_ax = self.cb_H.currentData()
        w_ax = self.cb_W.currentData()
        c_ax = self.cb_C.currentData()  # None 허용

        if h_ax is None or w_ax is None:
            QtWidgets.QMessageBox.warning(self, "축 매핑 오류", "H/W 축을 올바르게 선택하세요.")
            return
        if c_ax is not None and (c_ax == h_ax or c_ax == w_ax):
            QtWidgets.QMessageBox.warning(self, "축 매핑 오류", "C 축은 H/W와 서로 달라야 합니다.")
            return

        s_wl = self._read_float(self.le_wl_start)
        e_wl = self._read_float(self.le_wl_end)
        if (s_wl is not None and e_wl is not None) and (s_wl > e_wl):
            QtWidgets.QMessageBox.warning(self, "입력 오류", "Start Wavelength가 End보다 클 수 없습니다.")
            return

        # MAT 모드
        if self.chk_mat.isChecked():
            if not self._mat_selected or not self._mat_selected.exists():
                QtWidgets.QMessageBox.warning(self, "입력 오류", "유효한 MAT 파일을 선택하세요.")
                return
            if not self._mat_shape:
                QtWidgets.QMessageBox.warning(self, "입력 오류", "MAT 내부 배열을 찾지 못했습니다.")
                return

            # 배열 캐시가 없으면 지금이라도 1회만 로드
            arr = self._loaded_array
            if arr is None:
                arr = self._load_mat_array_full(self._mat_selected, self._mat_arr_name)
            # axis_map 기준 (H,W,C)로 재배열
            arr_hwc = self._reorder_hwc_np(arr, {"H": int(h_ax), "W": int(w_ax), "C": (None if c_ax is None else int(c_ax))})

            # 사용자 타입 확인: .info 파일에서 읽거나 라디오 버튼에서 가져오기
            user_type = None
            try:
                # .info 파일에서 user_type 읽기 시도
                if self._mat_selected:
                    loaded_info = read_info(self._mat_selected)
                    if loaded_info and "user_type" in loaded_info:
                        user_type = loaded_info.get("user_type")
            except Exception:
                pass
            
            # .info 파일에 없으면 라디오 버튼에서 가져오기
            if user_type not in ("personal", "server"):
                user_type = "personal" if (self.rad_personal and self.rad_personal.isChecked()) else "server"
            
            cfg = {
                "type": "mat",
                "data_path": {"mat": str(self._mat_selected)},   # 경로
                "shape": list(self._mat_shape),
                "axis_map": {"H": int(h_ax), "W": int(w_ax), "C": (None if c_ax is None else int(c_ax))},
                "camera_code": self._selected_camera.get("code") if self._selected_camera else None,
                "camera_name": self._selected_camera.get("name") if self._selected_camera else None,
                "wavelength": self._selected_camera.get('wavelength') if self._selected_camera else None,  # 원문 (호환성)
                "wavelength_list": self._selected_camera.get('wavelength_list') if self._selected_camera else None,  # ★ 정규화된 리스트 (우선 사용)
                'fwhm': self._selected_camera.get('fwhm') if self._selected_camera else None,
                "selected_camera": self._pack_selected_camera(), # code/name/wavelength/fwhm
                "data": arr_hwc,                                 # ← ★ 실제 배열 동봉
                "user_type": user_type,                          # ← ★ 사용자 타입 추가
                # (선택) "array_name": self._mat_arr_name, "base_dir": str(self._mat_selected.parent),
            }

            # --- 현재 세팅을 .info로 저장 ---
            try:
                # 사용자 타입 확인 (0=개인 사용자, 1=서버 사용자)
                user_type = "personal" if (self.rad_personal and self.rad_personal.isChecked()) else "server"
                
                info_payload = {
                    "type": "mat",
                    "H": int(self._mat_shape[0]),
                    "W": int(self._mat_shape[1]),
                    "C": int(self._mat_shape[2] if len(self._mat_shape) > 2 else 1),
                    "axis_map": {"H": int(h_ax), "W": int(w_ax), "C": (None if c_ax is None else int(c_ax))},
                    "camera": {
                        "code": (self._selected_camera.get("code") if self._selected_camera else None),
                        "name": (self._selected_camera.get("name") if self._selected_camera else None),
                    },
                    # fastAPI 조회에서 파장 리스트가 있으면 저장(없으면 None)
                    "wavelength": self._selected_camera.get("wavelength_list"),
                    # fwhm은 API 포맷이 문자열일 수 있어 간단히 None 처리(필요 시 파싱 추가 가능)
                    "fwhm": self._selected_camera.get('fwhm'),
                    "user_type": user_type,  # ← ★ 사용자 타입 추가
                }
                write_info(self._mat_selected, info_payload)
            except Exception:
                pass

            # 완료
            self.result = cfg
            self.accept()
            return
        
        # HDR&RAW 모드
        hdr_path = Path(self.le_hdr.text().strip()) if self.le_hdr.text().strip() else None
        raw_path = Path(self.le_raw.text().strip()) if self.le_raw.text().strip() else None
        if not hdr_path or not hdr_path.exists():
            QtWidgets.QMessageBox.warning(self, "입력 오류", "유효한 HDR 파일을 선택하세요.")
            return
        if not raw_path or not raw_path.exists():
            guess = hdr_path.with_suffix(".raw")
            if guess.exists():
                raw_path = guess
                self.le_raw.setText(str(guess))
            else:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, "Select RAW", str(hdr_path.parent), "RAW (*.raw);;All Files (*.*)"
                )
                if not path:
                    return
                raw_path = Path(path)

        if not self._hdr_shape:
            # 마지막 보호: 헤더 다시 파싱
            try:
                hdr = self._parse_envi_hdr(hdr_path)
                H = int(hdr.get("lines", 0)); W = int(hdr.get("samples", 0)); C = int(hdr.get("bands", 0) or 1)
                if H > 0 and W > 0:
                    self._hdr_shape = (H, W, C)
                else:
                    QtWidgets.QMessageBox.warning(self, "HDR 오류", "HDR에서 lines/samples를 읽지 못했습니다.")
                    return
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "HDR 오류", f"HDR 파싱 실패:\n{e}")
                return

        # 배열 캐시가 없으면 지금이라도 1회만 로드
        arr = self._loaded_array
        if arr is None:
            arr = self._load_hdr_raw_array(hdr_path, raw_path)
        # axis_map 기준 (H,W,C)로 재배열(이미 HWC일 가능성 높지만 안전하게)
        arr_hwc = self._reorder_hwc_np(arr, {"H": int(h_ax), "W": int(w_ax), "C": (None if c_ax is None else int(c_ax))})

        # 사용자 타입 확인: .info 파일에서 읽거나 라디오 버튼에서 가져오기
        user_type = None
        try:
            # .info 파일에서 user_type 읽기 시도
            info_path = hdr_path.with_suffix(hdr_path.suffix + ".info")
            if info_path.exists():
                loaded_info = read_info(hdr_path)
                if loaded_info and "user_type" in loaded_info:
                    user_type = loaded_info.get("user_type")
        except Exception:
            pass
        
        # .info 파일에 없으면 라디오 버튼에서 가져오기
        if user_type not in ("personal", "server"):
            user_type = "personal" if (self.rad_personal and self.rad_personal.isChecked()) else "server"
        
        # HDR 메타에서 파장/FWHM 추출 (선택)
        hdr_meta = {}
        try:
            hdr_meta = self._parse_envi_hdr(hdr_path) or {}
        except Exception:
            pass
        
        cfg = {
            "type": "hdr_raw",
            "data_path": {"hdr": str(hdr_path), "raw": str(raw_path)},  # 경로
            "shape": list(self._hdr_shape),
            "axis_map": {"H": int(h_ax), "W": int(w_ax), "C": (None if c_ax is None else int(c_ax))},
            "camera_code": self._selected_camera.get("code") if self._selected_camera else None,
            "camera_name": self._selected_camera.get("name") if self._selected_camera else None,
            "wavelength": self._selected_camera.get('wavelength') if self._selected_camera else (hdr_meta.get("wavelength") if isinstance(hdr_meta.get("wavelength"), list) else None),  # 원문 (호환성)
            "wavelength_list": self._selected_camera.get('wavelength_list') if self._selected_camera else (hdr_meta.get("wavelength") if isinstance(hdr_meta.get("wavelength"), list) else None),  # ★ 정규화된 리스트 (우선 사용)
            "fwhm": self._selected_camera.get('fwhm') if self._selected_camera else (hdr_meta.get("fwhm") if isinstance(hdr_meta.get("fwhm"), list) else None),
            "selected_camera": self._pack_selected_camera(),  # code/name/wavelength/fwhm
            "data": arr_hwc,                                            # ← ★ 실제 배열 동봉
            "user_type": user_type,                                     # ← ★ 사용자 타입 추가
            # (선택) "base_dir": str(hdr_path.parent),
        }

        # --- 현재 세팅을 .info로 저장 ---
        try:
            info_payload = {
                "type": "hdr_raw",
                "H": int(self._hdr_shape[0]),
                "W": int(self._hdr_shape[1]),
                "C": int(self._hdr_shape[2]),
                "axis_map": {"H": int(h_ax), "W": int(w_ax), "C": (None if c_ax is None else int(c_ax))},
                "camera": {
                    "code": (self._selected_camera.get("code") if self._selected_camera else None),
                    "name": (self._selected_camera.get("name") if self._selected_camera else None) \
                            or (hdr_meta.get("sensor type") if isinstance(hdr_meta.get("sensor type"), str) else None),
                },
                "wavelength": self._selected_camera.get("wavelength_list") if self._selected_camera else (hdr_meta.get("wavelength") if isinstance(hdr_meta.get("wavelength"), list) else None),
                "fwhm": self._selected_camera.get('fwhm') if self._selected_camera else (hdr_meta.get("fwhm") if isinstance(hdr_meta.get("fwhm"), list) else None),
                "user_type": user_type,  # ← ★ 사용자 타입 추가
            }
            write_info(hdr_path, info_payload)
        except Exception:
            pass

        # 완료
        self.result = cfg
        self.accept()
        
    # ---------------- 내부 유틸 ----------------
    def _read_float(self, le: QtWidgets.QLineEdit) -> Optional[float]:
        txt = (le.text() or "").strip()
        if not txt:
            return None
        try:
            return float(txt)
        except Exception:
            return None

    def _populate_axis_combos(self, shape: Tuple[int, ...]) -> None:
        def fill_combo(cb: QtWidgets.QComboBox, allow_none: bool):
            cb.blockSignals(True)
            cb.clear()
            if allow_none:
                cb.addItem("없음", None)
            for i, sz in enumerate(shape):
                cb.addItem(f"dim{i} (size={sz})", i)
            cb.blockSignals(False)

        fill_combo(self.cb_H, allow_none=False)
        fill_combo(self.cb_W, allow_none=False)
        fill_combo(self.cb_C, allow_none=True)

        # 추천
        guess = self._auto_guess_axes(shape)
        self._set_combo_by_data(self.cb_H, guess.get("H"))
        w_ax = guess.get("W")
        if w_ax == guess.get("H"):
            candidates = [i for i in range(len(shape)) if i != guess.get("H")]
            w_ax = candidates[0] if candidates else 0
        self._set_combo_by_data(self.cb_W, w_ax)
        self._set_combo_by_data(self.cb_C, guess.get("C"))

    def _set_combo_by_data(self, cb: QtWidgets.QComboBox, value) -> None:
        for i in range(cb.count()):
            if cb.itemData(i) == value:
                cb.setCurrentIndex(i)
                return

    def _auto_guess_axes(self, shape: Tuple[int, ...]) -> Dict[str, Optional[int]]:
        n = len(shape)
        guess = {"H": None, "W": None, "C": None}
        if n == 2:
            guess["H"], guess["W"] = 0, 1
            return guess
        if n >= 3:
            sizes = list(shape)
            c_ax = int(np.argmin(sizes))
            guess["C"] = c_ax
            others = [i for i in range(n) if i != c_ax]
            h_ax = max(others, key=lambda i: shape[i])
            w_ax = min(others, key=lambda i: shape[i])
            guess["H"], guess["W"] = h_ax, w_ax
        return guess

    def _pick_best_mat_array(self, mat_path: Path) -> Tuple[Optional[str], Optional[Tuple[int, ...]]]:
        arrays: List[Tuple[str, Tuple[int, ...], str]] = []
        # 1) scipy.io.loadmat
        try:
            from scipy.io import loadmat  # type: ignore
            data = loadmat(str(mat_path))
            for k, v in data.items():
                if k.startswith("__"):
                    continue
                if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                    arrays.append((k, tuple(int(x) for x in v.shape), str(v.dtype)))
        except Exception:
            # 2) h5py
            try:
                import h5py  # type: ignore
                with h5py.File(str(mat_path), "r") as f:
                    def _collect(name, obj):
                        if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                            arrays.append((name, tuple(int(x) for x in obj.shape), str(obj.dtype)))
                    f.visititems(_collect)
            except Exception as e:
                raise e

        if not arrays:
            return None, None

        def _rank(shp: Tuple[int, ...]) -> Tuple[int, int]:
            ndim = len(shp)
            if ndim == 3:
                pri = 0
            elif ndim == 2:
                pri = 1
            else:
                pri = 2
            return (pri, -int(np.prod(shp)))

        arrays.sort(key=lambda t: _rank(t[1]))
        name, shape, _dtype = arrays[0]
        return name, shape

    def _parse_envi_hdr(self, hdr_path: Path) -> Dict[str, Any]:
        kv: Dict[str, Any] = {}
        text = hdr_path.read_text(encoding="utf-8", errors="ignore")
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                key = k.strip().lower()
                val = v.strip().strip("{}").strip()
                try:
                    if "." in val:
                        kv[key] = float(val)
                    else:
                        kv[key] = int(val)
                except Exception:
                    kv[key] = val
        return kv

    # 실제 데이터까지 읽어 shape 검증이 필요한 경우(경량화 위해 info만 사용)
    def _load_hdr_raw_info(self, hdr_path: Path, raw_path: Path) -> Tuple[int, int, int]:
        hdr = self._parse_envi_hdr(hdr_path)
        H = int(hdr.get("lines", 0)); W = int(hdr.get("samples", 0)); C = int(hdr.get("bands", 0) or 1)
        if H <= 0 or W <= 0:
            raise ValueError("HDR의 lines/samples가 유효하지 않습니다.")
        # raw 존재/크기 간단 점검(옵션)
        if not raw_path.exists():
            raise FileNotFoundError(f"RAW not found: {raw_path}")
        return (H, W, C)

    def _load_mat_info(self, mat_path: Path) -> Tuple[str, Tuple[int, ...]]:
        name, shape = self._pick_best_mat_array(mat_path)
        if not name or not shape:
            raise ValueError("MAT 내에서 적절한 배열을 찾지 못했습니다.")
        return name, shape

    def _init_table(self) -> None:
        """우측 표 초기화(헤더 세팅 + 빈 테이블)."""
        tw = self.table  # __init__에서 self.table = self.tableWidget 로 바인딩됨
        if tw is None:
            return
        tw.clear()
        tw.setColumnCount(3)
        tw.setHorizontalHeaderLabels(["카메라 이름", "Wavelength", "FWHM"])
        tw.setRowCount(0)
        tw.verticalHeader().setVisible(False)
        tw.horizontalHeader().setStretchLastSection(True)
        tw.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        tw.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            
    def _pack_selected_camera(self) -> Optional[Dict[str, Any]]:
        sc = getattr(self, "_selected_camera", None)
        if not sc:
            return None
        return {
            "code":       sc.get("code"),
            "name":       sc.get("name"),
            "wavelength": sc.get("wavelength"),
            "fwhm":       sc.get("fwhm"),
        }
        
    def _load_hdr_raw_array(self, hdr_path: Path, raw_path: Path) -> np.ndarray:
        hdr = self._parse_envi_hdr(hdr_path)
        H = int(hdr.get("lines")); W = int(hdr.get("samples")); C = int(hdr.get("bands", 1))
        inter = str(hdr.get("interleave", "bsq")).lower()
        dtype_code = int(hdr.get("data type", hdr.get("data_type", 4)))
        byte_order = int(hdr.get("byte order", hdr.get("byte_order", 0)))
        offset = int(hdr.get("header offset", hdr.get("header_offset", 0)))

        dtmap = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64,
                12: np.uint16, 13: np.uint32, 14: np.int64, 15: np.uint64}
        dt = np.dtype(dtmap[dtype_code]).newbyteorder("<" if byte_order == 0 else ">")

        mm = np.memmap(str(raw_path), mode="r", dtype=dt, offset=offset)
        if inter == "bsq":
            arr = mm.reshape(C, H, W).transpose(1, 2, 0)
        elif inter == "bil":
            arr = mm.reshape(H, C, W).transpose(0, 2, 1)
        elif inter == "bip":
            arr = mm.reshape(H, W, C)
        else:
            arr = mm.reshape(C, H, W).transpose(1, 2, 0)
        return np.asarray(arr)

    def _load_mat_array_full(self, mat_path: Path, array_name: Optional[str]) -> np.ndarray:
        try:
            from scipy.io import loadmat
            d = loadmat(str(mat_path))
            if array_name and array_name in d and isinstance(d[array_name], np.ndarray):
                return d[array_name]
            for k, v in d.items():
                if k.startswith("__"): continue
                if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                    return v
        except Exception:
            import h5py  # type: ignore
            with h5py.File(str(mat_path), "r") as f:
                if array_name and array_name in f:
                    return f[array_name][()]
                for name, ds in f.items():
                    if hasattr(ds, "dtype") and np.issubdtype(ds.dtype, np.number):
                        return ds[()]
        raise ValueError("MAT에서 숫자형 배열을 찾지 못했습니다.")

    def _reorder_hwc_np(self, arr: np.ndarray, axis_map: Dict[str, Optional[int]]) -> np.ndarray:
        if arr.ndim == 2:
            return arr[..., None]
        h = axis_map.get("H"); w = axis_map.get("W"); c = axis_map.get("C")
        if h is None or w is None:
            return arr
        order = [h, w] + ([c] if c is not None else [])
        target = [0, 1] + ([2] if c is not None else [])
        return np.moveaxis(arr, order, target)

    def _apply_info_to_ui(self, info: Dict[str, Any]) -> None:
        """sidecar .info 값을 현재 다이얼로그 UI에 완전 반영(H/W/C 선택+카메라 상태 보관 포함)"""
        # ---------- 1) H/W/C ----------
        H, W, C = info.get("H"), info.get("W"), info.get("C")
        axis_map = info.get("axis_map") or {}
        try:
            H = int(H) if H is not None else None
            W = int(W) if W is not None else None
            C = int(C) if C is not None else None
        except Exception:
            H = W = C = None

        if H and W:
            shape = (H, W, (C if C else 1))
            # 콤보 아이템 구성
            self._populate_axis_combos(shape)
            # 저장된 axis_map으로 "선택"을 복원(★ 핵심)
            try:
                def _set_combo(cb, val):
                    # val이 None이면 cb_C에서 "없음" 선택
                    for i in range(cb.count()):
                        if cb.itemData(i) == val:
                            cb.setCurrentIndex(i); return
                # axis_map은 원래 입력 배열의 축 인덱스 기준. 지금 콤보는 dim0..dimN 형태
                # 여기서는 저장된 그대로 세팅(저장 시 사용한 값 그대로라고 가정)
                _set_combo(self.cb_H, axis_map.get("H"))
                _set_combo(self.cb_W, axis_map.get("W"))
                _set_combo(self.cb_C, axis_map.get("C"))
            except Exception:
                pass

            # 내부 상태 업데이트
            self._hdr_shape = shape
            self._mat_shape = shape

        # ---------- 2) 파장(Start/End) ----------
        wl = info.get("wavelength")
        # 문자열일 수도 있으니 안전 파싱
        def _to_list(x):
            if isinstance(x, list): return x
            if isinstance(x, str):
                try:
                    parts = [p.strip() for p in x.strip("{}").replace("\n"," ").split(",")]
                    return [float(p) for p in parts if p]
                except Exception:
                    return None
            return None
        wl_list = _to_list(wl)
        if wl_list:
            try:
                self.le_wl_start.setText(f"{min(wl_list):.6f}")
                self.le_wl_end.setText(f"{max(wl_list):.6f}")
            except Exception:
                pass

        # ---------- 3) 카메라 표 시각화 + 내부 상태 ----------
        def _short(arr, nd=2, nhead=6, ntail=3):
            if not isinstance(arr, list) or not arr:
                return ""
            head = ", ".join(f"{x:.{nd}f}" for x in arr[:nhead])
            if len(arr) > nhead + ntail:
                tail = ", ".join(f"{x:.{nd}f}" for x in arr[-ntail:])
                return f"{head} ... {tail}"
            return head

        cam_name = ""
        cam_code = None
        cam = info.get("camera")
        if isinstance(cam, dict):
            cam_name = cam.get("name") or ""
            cam_code = cam.get("code")
        cam_name = info.get("camera_name", cam_name)

        fwhm_list = _to_list(info.get("fwhm"))

        # 테이블 갱신
        self.table.clear()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["카메라 이름", "Wavelength", "FWHM"])
        self.table.setRowCount(1)
        self.table.setItem(0, 0, QtWidgets.QTableWidgetItem(str(cam_name)))
        self.table.setItem(0, 1, QtWidgets.QTableWidgetItem(_short(wl_list, 2)))
        self.table.setItem(0, 2, QtWidgets.QTableWidgetItem(_short(fwhm_list, 3)))
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)

        # 내부 상태: 이후 저장 시에도 쓰일 수 있도록 "선택 카메라"를 만들어 둠
        self._selected_camera = {
            "code": cam_code,
            "name": cam_name,
            "wavelength": wl_list,        # 원문 호환
            "wavelength_list": wl_list,   # 정규화 리스트
            "fwhm": fwhm_list
        }
        
        # ---------- 4) user_type 설정 ----------
        user_type = info.get("user_type")
        if user_type in ("personal", "server"):
            try:
                if user_type == "personal" and self.rad_personal:
                    self.rad_personal.setChecked(True)
                elif user_type == "server" and self.rad_server:
                    self.rad_server.setChecked(True)
            except Exception:
                pass

    def _try_apply_info_from_inputs(self) -> bool:
        """
        현재 UI 입력(라인에딧)에 적힌 경로를 기준으로 .info를 찾아
        있으면 UI에 적용하고 True, 없으면 False 반환.
        MAT/HDR 모드 모두 처리.
        """
        try:
            if self.chk_mat.isChecked():
                ptxt = (self.le_mat.text() or "").strip()
                if not ptxt:
                    return False
                p = Path(ptxt)
            else:
                ptxt = (self.le_hdr.text() or "").strip()
                if not ptxt:
                    return False
                p = Path(ptxt)
            if not p.exists():
                return False
            loaded = read_info(p)
            if not loaded:
                return False
            self._apply_info_to_ui(loaded)
            return True
        except Exception:
            return False
