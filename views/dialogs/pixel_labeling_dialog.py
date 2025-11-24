# views/dialogs/pixel_labeling_dialog.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QPushButton, QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView, QComboBox
)
from PyQt5.QtCore import pyqtSignal
from data.db import send_label_add
from views.dialogs.labeling_candidate_review import LabelingCandidateReviewDialog

class PixelLabelingDialog(QDialog):
    """
    픽셀 라벨링 다이얼로그
    - .ui는 디자이너 파일을 그대로 사용 (레이아웃 sizeConstraint는 .ui의 레이아웃 속성으로 유지)
    - 이 파이썬 파일은 시그널 연결 및 UI 런타임 동작만 담당
    """
    user_labeling_requested = pyqtSignal()
    classmap_labeling_requested = pyqtSignal(dict)   # {"rows":[...]}
    recommend_labeling_requested = pyqtSignal(dict)  # {"rows":[...]}
    register_requested = pyqtSignal(dict)            # {"rows":[...]}
    canceled = pyqtSignal()
    verifyRequested = QtCore.pyqtSignal(dict)   # {'y': int, 'x': int, 'cid': int, 'row': int}
    
    # 
    labeling_requested = pyqtSignal(list)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, ui_dir: Optional[Path] = None):
        super().__init__(parent)

        if ui_dir is None:
            # 상위에서 넘겨준 app_dir/ui 경로가 없다면 현재 파일 기준 상대 경로로 폴백
            ui_dir = Path(__file__).resolve().parent / ".." / ".." / "ui"
        ui_path = (ui_dir / "pixel_labeling_dialog.ui").resolve()

        if not ui_path.exists():
            raise FileNotFoundError(f"UI 파일을 찾을 수 없습니다: {ui_path}")

        # .ui 로드 (여기서 QDialog 자체에 sizeConstraint를 설정하지 않음)
        self._root = uic.loadUi(str(ui_path), self)  # baseinstance=self 로 바로 self에 로드

        # === 위젯 핸들 ===
        self.btnUserLabeling: QPushButton = self.findChild(QPushButton, "btnUserLabeling")
        self.btnClassmapLabeling: QPushButton = self.findChild(QPushButton, "btnClassmapLabeling")
        self.btnRecommendLabeling: QPushButton = self.findChild(QPushButton, "btnRecommendLabeling")
        self.btnRegister: QPushButton = self.findChild(QPushButton, "btnRegister")
        self.btnCancel: QPushButton = self.findChild(QPushButton, "btnCancel")
        self.tableSelectedPixels: QTableWidget = self.findChild(QTableWidget, "tableSelectedPixels")

        # 필수 위젯 존재 확인(없으면 .ui와 이름 불일치)
        for name, w in [
            ("btnUserLabeling", self.btnUserLabeling),
            ("btnClassmapLabeling", self.btnClassmapLabeling),
            ("btnRecommendLabeling", self.btnRecommendLabeling),
            ("btnRegister", self.btnRegister),
            ("btnCancel", self.btnCancel),
            ("tableSelectedPixels", self.tableSelectedPixels),
        ]:
            if w is None:
                raise RuntimeError(f".ui에서 '{name}' 위젯을 찾지 못했습니다. objectName 확인 필요")

        # === 테이블 설정 ===
        self._init_table()

        # === 버튼 동작/시그널 연결 ===
        self._wire_signals()

        # === 다이얼로그 속성(선택사항) ===
        self.setModal(True)
        # self.resize(720, 520)
        self._center_to_parent()

        # ▼ 추가: 스펙트럼 프리뷰 캔버스 준비
        self._init_preview_canvas()
        self._class_options: list[tuple[int, str]] = []  # (cid, name)
        self._cid_to_name: dict[int, str] = {}  # cid -> mtrl_nm
        self._default_cid: Optional[int] = None  # user_labeling_dialog에서 선택한 기본 CID

    # -------------------------------
    # UI 초기화
    # -------------------------------
    def _init_table(self):
        tbl = self.tableSelectedPixels

        # 열 구성: □, #, Locate, Class, option
        tbl.setColumnCount(5)
        tbl.setHorizontalHeaderLabels(["□", "#", "Locate", "Class", "option"])

        # 헤더 크기 정책
        h: QHeaderView = tbl.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.Stretch)
        # 번호/체크는 고정폭으로 보이게 조정(선택)
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)

        v: QHeaderView = tbl.verticalHeader()
        v.setVisible(False)

        tbl.setColumnWidth(0, 30)   # 체크
        tbl.setColumnWidth(1, 40)   # 번호
        # 나머지는 Stretch에 의해 자동

        tbl.setAlternatingRowColors(True)
        tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        tbl.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)


        # 데모 행(선택): 실제 사용 시 제거하거나 데이터 바인딩 시 채우기
        # self._append_demo_rows(5)

    def _append_demo_rows(self, n: int = 3):
        """디버그용 예시 데이터"""
        tbl = self.tableSelectedPixels
        start = tbl.rowCount()
        for i in range(n):
            row = start + i
            tbl.insertRow(row)
            tbl.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            tbl.setItem(row, 1, QTableWidgetItem(f"(x={10+row}, y={20+row})"))
            tbl.setItem(row, 2, QTableWidgetItem("class-A"))
            tbl.setItem(row, 3, QTableWidgetItem("option"))

    def _wire_signals(self):
        self.btnUserLabeling.setCheckable(True)
        self.btnClassmapLabeling.setCheckable(True)
        # 초기 상태를 명시적으로 unchecked로 설정
        self.btnUserLabeling.setChecked(False)
        self.btnClassmapLabeling.setChecked(False)
        self.btnUserLabeling.toggled.connect(self._on_user_labeling_toggled)
        self.btnClassmapLabeling.toggled.connect(self._on_classmap_labeling_toggled)

        self.btnRecommendLabeling.clicked.connect(self._on_recommend_clicked)

        self.btnRegister.clicked.connect(self._on_register_clicked)
        self.btnCancel.clicked.connect(self.reject)

        self.tableSelectedPixels.itemSelectionChanged.connect(self.refresh_preview)
        self.tableSelectedPixels.cellClicked.connect(lambda *_: self.refresh_preview())

    # -------------------------------
    # 시그널 핸들러
    # -------------------------------
    def _on_user_labeling_toggled(self, checked: bool):
        if checked and self.btnClassmapLabeling.isChecked():
            # 서로 배타적
            self.btnClassmapLabeling.blockSignals(True)
            self.btnClassmapLabeling.setChecked(False)
            self.btnClassmapLabeling.blockSignals(False)
        # checked=True일 때 사용자 지정 라벨링 다이얼로그 열기
        if checked:
            self.user_labeling_requested.emit()

    def _on_classmap_labeling_toggled(self, checked: bool):
        if checked and self.btnUserLabeling.isChecked():
            self.btnUserLabeling.blockSignals(True)
            self.btnUserLabeling.setChecked(False)
            self.btnUserLabeling.blockSignals(False)
        # checked=True일 때 분류맵 기반 라벨링 다이얼로그 열기
        if checked:
            self.classmap_labeling_requested.emit({})

    def _on_recommend_clicked(self):
        """추천 픽셀 라벨링 버튼 클릭 시"""
        self.recommend_labeling_requested.emit({})

    def _on_label_selected_clicked(self):
        row = self.tableSelectedPixels.currentRow()
        if row < 0:
            QMessageBox.warning(self, "알림", "라벨링할 픽셀을 선택하세요.")
            return
        # 선택 행에 대한 라벨링 처리 로직 연동
        # TODO: 외부 콜백 또는 컨트롤러와 연결하여 실제 라벨링 실행
        QMessageBox.information(self, "라벨링", f"{row+1}번째 행 라벨링 실행")

    # payload 개발 완료 => API 만 연결하면 끝
    def _on_register_clicked(self):
        checked = self.get_checked_pixels()  # [{"y":..,"x":..,"cid":..,"row":..}, ...]
        if not checked:
            QtWidgets.QMessageBox.warning(self, "알림", "등록할 픽셀을 선택하세요.")
            return

        parent = self.parent()
        if not parent:
            QtWidgets.QMessageBox.critical(self, "오류", "부모 창 정보를 찾을 수 없습니다.")
            return

        try:
            img_cd = getattr(parent, "image_cd", None)
            cfg = getattr(parent, "cfg", {})
            cube = cfg.get("data", None)
            wavelength = cfg.get("wavelength", None)
            
            
            
            if img_cd is None or cube is None:
                QtWidgets.QMessageBox.warning(self, "알림", "이미지 또는 데이터가 로드되지 않았습니다.")
                return

            import numpy as np
            cube = np.asarray(cube, dtype=float)
            H, W, C = cube.shape

            # --- 좌표/중복/경계 정리 ---
            # (표시는 (x,y), 내부는 y,x) 규칙을 따른다고 가정
            uniq = {}
            for rec in checked:
                y, x = int(rec["y"]), int(rec["x"])
                if 0 <= y < H and 0 <= x < W:
                    uniq[(y, x)] = int(rec.get("cid", -1))  # 마지막 cid 기준으로 덮어쓰기

            if not uniq:
                QtWidgets.QMessageBox.warning(self, "알림", "유효한 좌표가 없습니다.")
                return

            ys = np.fromiter((p[0] for p in uniq.keys()), dtype=int)
            xs = np.fromiter((p[1] for p in uniq.keys()), dtype=int)
            cids = np.fromiter((uniq[p] for p in uniq.keys()), dtype=int)

            # --- 배치 스펙트럼 추출 ---
            spectra = cube[ys, xs, :]  # (N, C)
            # NaN/Inf 방어
            spectra = np.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0)

            # --- rows 페이로드 구성 ---
            rows = []
            for (y, x), cid, rfl in zip(uniq.keys(), cids.tolist(), spectra):
                rows.append({
                    "img_cd": int(img_cd),
                    "mtrl_cd": int(cid),
                    "img_x": int(x),
                    "img_y": int(y),
                    "rfl": rfl.astype(float).tolist(),
                })
                
            if not rows:
                QtWidgets.QMessageBox.warning(self, "알림", "등록할 유효한 픽셀이 없습니다.")
                return
            
            # personal 사용자는 API 저장 스킵
            user_type = getattr(parent, "user_type", "server")  # 기본값: server
            if user_type != "personal":
                api_base = os.getenv('label_add_url')
                if api_base:
                    send_label_add(api_base=api_base, targets=rows)
                else:
                    import logging
                    logging.warning("[PixelLabeling] label_add_url 환경변수가 설정되지 않았습니다.")
            else:
                import logging
                logging.info("[PixelLabeling] personal 사용자이므로 API 저장을 건너뜁니다.")

            self.register_requested.emit({"rows": rows})
            self.accept()

        except Exception as e:
            import logging
            logging.exception("[PixelLabeling] Register failed")
            QtWidgets.QMessageBox.critical(self, "오류", f"등록 처리 중 오류가 발생했습니다:\n{e}")


    # -------------------------------
    # 유틸
    # -------------------------------
    def _center_to_parent(self):
        """부모 중앙으로 배치(부모 없을 때는 스킵)"""
        if self.parent() and isinstance(self.parent(), QtWidgets.QWidget):
            parent_geom = self.parent().frameGeometry()
            self.move(parent_geom.center() - self.frameGeometry().center())

    # 외부에서 선택된 픽셀 데이터 바인딩 예시
    def bind_selected_pixels(self, rows: list[tuple]):
        """
        rows: [(index, (x,y), class_name, option_str), ...]
        """
        tbl = self.tableSelectedPixels
        tbl.setRowCount(0)
        for i, (idx, xy, cls, opt) in enumerate(rows):
            tbl.insertRow(i)
            # (0) 체크박스
            chk = QtWidgets.QCheckBox()
            chk.setChecked(True)
            # 체크박스 상태 변경 시 스펙트럼 업데이트
            chk.stateChanged.connect(lambda state, r=i: self._on_checkbox_changed(r, state))
            tbl.setCellWidget(i, 0, chk)
            # (1) 번호
            it_no = QTableWidgetItem(str(idx))
            it_no.setTextAlignment(Qt.AlignCenter)
            it_no.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            tbl.setItem(i, 1, it_no)
            # (2) 좌표
            tbl.setItem(i, 2, QTableWidgetItem(f"({xy[0]}, {xy[1]})"))
            # (3) 클래스
            tbl.setItem(i, 3, QTableWidgetItem(str(cls)))
            # (4) option → 검증 버튼
            btn = QtWidgets.QPushButton("검증")
            btn.setObjectName(f"btnVerify_{i}")
            btn.clicked.connect(lambda _=None, r=i: self._on_verify_clicked(r))
            tbl.setCellWidget(i, 4, btn)

    def append_selected_pixels(self, rows: list[dict]) -> None:
        """
        rows: [{"y": int, "x": int, "cid": int}, ...]
        - 표에 행을 추가하고, 체크박스/버튼을 세팅한다.
        - Class 컬럼은 '물질명'을 표시하고, 내부 데이터는 CID(int)로 유지한다.
        - 클래스 목록은 MainWindow에서 set_class_options() 로 주입된 self._class_options 를 사용한다.
        """
        tbl = self.tableSelectedPixels
        start = tbl.rowCount()

        # MainWindow에서 PixelLabelingDialog.set_class_options(...) 로 넣어준 전체 클래스 목록
        # 형태: [(cid, name), ...]
        class_options: list[tuple[int, str]] = getattr(self, "_class_options", []) or []

        for i, rec in enumerate(rows):
            y = int(rec.get("y", 0))
            x = int(rec.get("x", 0))
            cid = int(rec.get("cid", -1))

            row = start + i
            tbl.insertRow(row)

            # (0) 체크박스
            chk = QtWidgets.QCheckBox()
            chk.setChecked(True)
            # 체크 상태가 바뀌면 현재 규칙(체크 다중/선택 단일)에 따라 다시 그림
            chk.stateChanged.connect(lambda *_: self._plot_selected_row_spectrum())
            tbl.setCellWidget(row, 0, chk)

            # (1) 번호(연번)
            it_no = QTableWidgetItem(str(row + 1))
            it_no.setTextAlignment(Qt.AlignCenter)
            it_no.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            tbl.setItem(row, 1, it_no)

            # (2) 좌표 "(x, y)"
            tbl.setItem(row, 2, QTableWidgetItem(f"({x}, {y})"))

            # (3) 클래스 - 물질명 콤보박스
            combo_class = QComboBox()

            # 전체 클래스 넣기 (표시: 물질명, 데이터: CID)
            existed_cids = set()
            for cid_opt, name_opt in class_options:
                cid_i = int(cid_opt)
                existed_cids.add(cid_i)
                display_name = self._cid_to_name.get(cid_i, str(name_opt))
                combo_class.addItem(display_name, cid_i)

            # 현재 CID가 목록에 없으면 별도로 추가 (이름이 없으므로 cid 숫자로 표시)
            if cid not in existed_cids:
                display_name_missing = self._cid_to_name.get(cid, str(cid))
                combo_class.insertItem(0, display_name_missing, cid)

            # ★ 기본값 결정: user_labeling_dialog에서 선택한 값이 있으면 우선 사용, 없으면 현재 CID
            default_cid = self._default_cid if self._default_cid is not None and self._default_cid in existed_cids else cid
            
            # 기본 CID 선택
            idx = combo_class.findData(default_cid)
            combo_class.setCurrentIndex(idx if idx >= 0 else 0)

            # row 캡쳐 주의: 기본 인자로 row 고정
            combo_class.currentIndexChanged.connect(
                lambda _idx, r=row: self._on_class_combo_changed(r, combo_class)
            )
            combo_class.setProperty("row", row)
            tbl.setCellWidget(row, 3, combo_class)

            # (4) option → 검증 버튼
            btn = QtWidgets.QPushButton("검증")
            btn.setObjectName(f"btnVerify_{row}")
            btn.clicked.connect(lambda _=None, r=row: self._on_verify_clicked(r))
            tbl.setCellWidget(row, 4, btn)

        # 방금 추가된 첫 행 선택 + 프리뷰 갱신
        if rows:
            tbl.selectRow(start)
            self._plot_selected_row_spectrum()


            
    def get_checked_rows(self) -> list[int]:
        """체크된 행 인덱스 목록 반환."""
        tbl = self.tableSelectedPixels
        rows = []
        for r in range(tbl.rowCount()):
            w = tbl.cellWidget(r, 0)
            if isinstance(w, QtWidgets.QCheckBox) and w.isChecked():
                rows.append(r)
        return rows

    def _on_checkbox_changed(self, row: int, state: int):
        """체크 상태가 바뀌면 현재 체크/선택 규칙에 따라 다시 그림."""
        try:
            self._plot_selected_row_spectrum()
        except Exception:
            import logging
            logging.exception("[PixelLabeling] checkbox changed failed")

    def get_checked_pixels(self):
        """
        체크된 행만 수집하여 [{"y","x","cid","row"}...] 반환.
        - 좌표는 2열 "(x, y)"에서 파싱
        - 클래스는 3열: QComboBox면 currentData(), 아니면 item.text()
        """
        out = []
        tbl = self.tableSelectedPixels

        for r in range(tbl.rowCount()):
            # 체크박스 확인
            chk = tbl.cellWidget(r, 0)
            if not (isinstance(chk, QtWidgets.QCheckBox) and chk.isChecked()):
                continue

            # (x, y) 파싱
            loc_item = tbl.item(r, 2)
            if not loc_item:
                continue
            txt = (loc_item.text() or "").strip()
            if not txt:
                continue
            try:
                x_str, y_str = txt.strip("()").split(",")
                x, y = int(x_str), int(y_str)
            except Exception:
                # "(y, x)"가 들어오는 경우 대비
                y_str, x_str = txt.strip("()").split(",")
                y, x = int(y_str), int(x_str)

            # cid 추출 (콤보/아이템 둘 다 지원)
            cid = -1
            w = tbl.cellWidget(r, 3)
            if isinstance(w, QComboBox):
                data = w.currentData()
                cid = int(data) if data is not None else -1
            else:
                cid_item = tbl.item(r, 3)
                if cid_item:
                    txt_c = (cid_item.text() or "").strip()
                    if txt_c.lstrip("-").isdigit():
                        cid = int(txt_c)

            out.append({"y": int(y), "x": int(x), "cid": int(cid), "row": int(r)})

        return out

    def _on_verify_clicked(self, row: int):
        """
        '검증' 버튼 클릭 시:
        1) 해당 행의 (x,y,cid) 추출
        2) 부모(MainWindow)에서 cube/rgb 및 라이브러리 접근
        3) LabelingCandidateReviewDialog 로 전달하여 분석/표시
        """
        try:
            # 1) 좌표/클래스 읽기
            tbl = self.tableSelectedPixels
            if row < 0 or row >= tbl.rowCount():
                return
            loc_text = (tbl.item(row, 2).text() if tbl.item(row, 2) else "").strip()  # "(x, y)"
            try:
                x_str, y_str = loc_text.strip("()").split(",")
                x = int(x_str); y = int(y_str)
            except Exception:
                y_str, x_str = loc_text.strip("()").split(",")
                y = int(y_str); x = int(x_str)

            # 2) 부모 참조
            parent = self.parent()
            if parent is None:
                QtWidgets.QMessageBox.warning(self, "경고", "상위 창을 찾을 수 없습니다.")
                return
            # cube 존재 확인
            cfg = getattr(parent, "cfg", {})
            cube = cfg.get("data")
            if cube is None:
                QtWidgets.QMessageBox.warning(self, "경고", "HSI 데이터가 없습니다.")
                return

            app_dir = getattr(parent, "app_dir", None)
            ui_dir = (app_dir / "ui") if app_dir else None
            # 3) 검증 다이얼로그 실행
            from views.dialogs.labeling_candidate_review import LabelingCandidateReviewDialog
            self._review_dlg = LabelingCandidateReviewDialog(parent=parent, ui_dir=ui_dir)

            # ★ 시그널 연결(반드시 필요)
            self._review_dlg.class_changed.connect(self._on_review_class_changed)

            # 원본 행 정보와 함께 실행
            self._review_dlg.run_with_target(y=int(y), x=int(x), origin_row=row, origin_dialog=self)

        except Exception as e:
            import logging
            logging.exception("[PixelLabeling] verify clicked failed")
            QtWidgets.QMessageBox.critical(self, "오류", f"검증 중 오류가 발생했습니다:\n{e}")

    @QtCore.pyqtSlot(int, int)
    def _on_review_class_changed(self, origin_row: int, new_cid: int):
        """
        LabelingCandidateReviewDialog -> class_changed(origin_row, new_cid)
        표의 Class 컬럼(3번) QComboBox를 갱신한다. (y/x 등 다른 데이터는 변경하지 않음)
        """
        try:
            tbl = self.tableSelectedPixels
            if origin_row < 0 or origin_row >= tbl.rowCount():
                return

            # Class 컬럼은 QComboBox 위젯
            combo_widget = tbl.cellWidget(origin_row, 3)
            if not isinstance(combo_widget, QComboBox):
                # QComboBox가 아닌 경우(이상한 상태) 경고만 출력
                import logging
                logging.warning(f"[PixelLabeling] Row {origin_row} Class column is not QComboBox")
                return

            # 현재 선택된 CID 확인
            cur_cid = combo_widget.currentData()
            if cur_cid is not None and int(cur_cid) == int(new_cid):
                return  # 이미 동일한 값이면 무시

            # QComboBox에서 new_cid에 해당하는 인덱스 찾기
            target_idx = combo_widget.findData(int(new_cid))
            if target_idx >= 0:
                # 신호 차단하여 _on_class_combo_changed가 호출되지 않도록
                combo_widget.blockSignals(True)
                combo_widget.setCurrentIndex(target_idx)
                combo_widget.blockSignals(False)
            else:
                # new_cid가 목록에 없으면 추가 후 선택
                import logging
                logging.warning(f"[PixelLabeling] CID {new_cid} not found in combo, adding it")
                combo_widget.blockSignals(True)
                combo_widget.insertItem(0, str(new_cid), int(new_cid))
                combo_widget.setCurrentIndex(0)
                combo_widget.blockSignals(False)

        except Exception as e:
            import logging
            logging.exception("[PixelLabeling] class change apply failed")

    def show_spectrum_preview(self, row: Optional[int] = None) -> None:
        """
        표의 선택 행을 '스펙트럼 미리보기' 캔버스에 그린다.
        - row가 None이면 현재 선택된 행을 사용
        - parent(MainWindow).cfg의 'data'(HSI cube)와 'wavelength' 사용
        - 팝업을 띄우지 않고 내부 프리뷰 축(self._ax_sel)에만 그린다
        """
        try:
            # 프리뷰 캔버스가 없다면 지연 초기화
            if not hasattr(self, "_ax_sel") or self._ax_sel is None or not hasattr(self, "_canvas_sel") or self._canvas_sel is None:
                if hasattr(self, "_init_preview_canvas"):
                    self._init_preview_canvas()
                    # 재시도 체크
                    if not hasattr(self, "_ax_sel") or self._ax_sel is None or not hasattr(self, "_canvas_sel") or self._canvas_sel is None:
                        return
                else:
                    return

            tbl = self.tableSelectedPixels
            r = tbl.currentRow() if (row is None) else int(row)

            # 선택이 없으면 축 초기화
            if r < 0 or r >= tbl.rowCount():
                self._ax_sel.clear()
                self._ax_sel.grid(True, linestyle="--", alpha=0.35)
                self._ax_sel.set_title("선택 픽셀 스펙트럼")
                self._ax_sel.set_xlabel("Wavelength / Band")
                self._ax_sel.set_ylabel("Reflectance")
                self._canvas_sel.draw_idle()
                return

            # "(x, y)" 파싱 (x 먼저 표기)
            loc_item = tbl.item(r, 2)
            loc_text = loc_item.text().strip() if loc_item else ""
            if not loc_text:
                return
            try:
                x_str, y_str = loc_text.strip("()").split(",")
                x, y = int(x_str), int(y_str)
            except Exception:
                # 혹시 "(y, x)" 형태가 들어오는 경우 대비
                y_str, x_str = loc_text.strip("()").split(",")
                y, x = int(y_str), int(x_str)

            # 부모에서 cube / wavelength 확보
            parent = self.parent()
            if parent is None:
                return
            cfg = getattr(parent, "cfg", {}) or {}
            cube = cfg.get("data", None)
            wave = cfg.get("wavelength", None)
            if cube is None:
                return

            import numpy as _np
            arr = _np.asarray(cube)
            H, W, C = arr.shape
            if not (0 <= y < H and 0 <= x < W):
                return

            # 스펙트럼 추출 + NaN/Inf 방어
            spec = _np.nan_to_num(_np.asarray(arr[y, x, :], dtype=float),
                                nan=0.0, posinf=0.0, neginf=0.0)

            # x축(파장/밴드)
            use_wave = False
            if wave is not None:
                xs = _np.asarray(wave, dtype=float)
                use_wave = (xs.shape[0] == C) and _np.all(_np.isfinite(xs))
            if not use_wave:
                xs = _np.arange(C)

            # 그리기
            ax = self._ax_sel
            ax.clear()
            ax.plot(xs, spec, linewidth=1.6)
            ax.scatter(xs, spec, s=12)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_xlabel("Wavelength" if use_wave else "Band Index")
            ax.set_ylabel("Reflectance")
            ax.set_title(f"선택 픽셀 스펙트럼 · (x={x}, y={y})")

            self._canvas_sel.draw_idle()

        except Exception:
            import logging
            logging.exception("[PixelLabeling] show_spectrum_preview failed")
            
    def _init_preview_canvas(self) -> None:
        """
        다이얼로그 내 스펙트럼 프리뷰 Matplotlib 캔버스를 심는다.
        - .ui의 'spectrumPreview' 위젯을 호스트로 사용
        """
        try:
            # 1) 호스트 컨테이너 찾기 (.ui의 spectrumPreview 위젯)
            host = self.findChild(QtWidgets.QWidget, "spectrumPreview")
            if host is None:
                # fallback: 다른 가능한 이름들 시도
                for name in ("wPreview", "grpPreview", "framePreview", "previewArea", "widgetPreview", "spectrumWidget"):
                    host = self.findChild(QtWidgets.QWidget, name)
                    if host:
                        break
            
            if host is None:
                import logging
                logging.warning("[PixelLabeling] spectrumPreview 위젯을 찾을 수 없습니다. 스펙트럼 미리보기가 비활성화됩니다.")
                self._fig_sel = None
                self._ax_sel = None
                self._canvas_sel = None
                return

            # 2) 레이아웃 확인/생성
            lay = host.layout()
            if lay is None:
                lay = QtWidgets.QVBoxLayout(host)
                lay.setContentsMargins(6, 6, 6, 6)
                lay.setSpacing(4)

            # 3) Matplotlib 캔버스 생성 및 추가
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            self._fig_sel = Figure(figsize=(5.6, 3.2), dpi=100)
            self._ax_sel = self._fig_sel.add_subplot(111)
            self._ax_sel.set_title("Selected pixel spectrum")
            self._ax_sel.set_xlabel("Wavelength / Band")
            self._ax_sel.set_ylabel("Reflectance")
            self._ax_sel.grid(True, linestyle="--", alpha=0.35)

            self._canvas_sel = FigureCanvas(self._fig_sel)
            lay.addWidget(self._canvas_sel)
            
        except Exception as e:
            import logging
            logging.exception("[PixelLabeling] _init_preview_canvas failed")
            self._fig_sel = None
            self._ax_sel = None
            self._canvas_sel = None

    def _plot_selected_row_spectrum(self, row: Optional[int] = None) -> None:
        """
        - 체크박스가 하나 이상 체크되어 있으면: 체크된 모든 행을 겹쳐서 그림
        - 체크가 없으면: row(주어졌으면 그 행, 아니면 현재 선택된 행)만 그림
        """
        try:
            if not hasattr(self, "_ax_sel") or self._ax_sel is None or not hasattr(self, "_canvas_sel") or self._canvas_sel is None:
                return

            tbl = self.tableSelectedPixels

            # 1) rows_to_plot 수집
            rows_to_plot = []
            # (1) 체크된 행 우선
            for r in range(tbl.rowCount()):
                chk = tbl.cellWidget(r, 0)
                if isinstance(chk, QtWidgets.QCheckBox) and chk.isChecked():
                    rows_to_plot.append(r)

            # (2) 체크가 없으면 row(파라미터) 또는 현재 선택행
            if not rows_to_plot:
                if row is not None:
                    rows_to_plot = [int(row)]
                else:
                    cr = tbl.currentRow()
                    if 0 <= cr < tbl.rowCount():
                        rows_to_plot = [cr]

            # 2) 부모 데이터
            parent = self.parent()
            if not parent:
                return
            cfg = getattr(parent, "cfg", {}) or {}
            cube = cfg.get("data", None)
            wave = cfg.get("wavelength", None)

            import numpy as _np
            ax = self._ax_sel
            ax.clear()
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_title("Selected pixel spectrum")
            ax.set_xlabel("Wavelength / Band")
            ax.set_ylabel("Reflectance")

            if cube is None or not rows_to_plot:
                self._canvas_sel.draw_idle()
                return

            arr = _np.asarray(cube)
            H, W, C = arr.shape

            # x축
            use_wave = False
            if wave is not None:
                xs = _np.asarray(wave, dtype=float)
                use_wave = (xs.shape[0] == C) and _np.all(_np.isfinite(xs))
            if not use_wave:
                xs = _np.arange(C)

            # 3) 여러 행을 루프 돌며 겹쳐 그리기
            for r in rows_to_plot:
                loc_item = tbl.item(r, 2)
                if not loc_item:
                    continue
                txt = loc_item.text().strip()  # "(x, y)"
                try:
                    x_str, y_str = txt.strip("()").split(",")
                    x, y = int(x_str), int(y_str)
                except Exception:
                    y_str, x_str = txt.strip("()").split(",")
                    y, x = int(y_str), int(x_str)

                if not (0 <= y < H and 0 <= x < W):
                    continue

                spec = _np.nan_to_num(_np.asarray(arr[y, x, :], dtype=float),
                                    nan=0.0, posinf=0.0, neginf=0.0)

                label = f"#{r+1} (x={x}, y={y})"
                # 색상은 Matplotlib 기본 cycle로 자동 구분
                ax.plot(xs, spec, linewidth=1.6, label=label)
                ax.scatter(xs, spec, s=12)

            # 4) 범례 + 그리기
            try:
                if len(rows_to_plot) > 1:
                    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.85)
            except Exception:
                pass

            self._canvas_sel.draw_idle()

        except Exception:
            import logging
            logging.exception("[PixelLabeling] _plot_selected_row_spectrum failed")

    def _plot_rows_spectrum(self, rows: list[int]) -> None:
        """
        rows에 들어있는 행들의 (x,y) 스펙트럼을 프리뷰에 겹쳐서 그린다.
        rows가 비면 축만 초기화.
        """
        if not hasattr(self, "_ax_sel") or self._ax_sel is None or not hasattr(self, "_canvas_sel") or self._canvas_sel is None:
            return

        import numpy as _np

        ax = self._ax_sel
        ax.clear()
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_title("선택 픽셀 스펙트럼")
        ax.set_xlabel("Wavelength / Band")
        ax.set_ylabel("Reflectance")

        if not rows:
            self._canvas_sel.draw_idle()
            return

        # 부모 데이터
        parent = self.parent()
        if not parent:
            self._canvas_sel.draw_idle()
            return
        cfg = getattr(parent, "cfg", {}) or {}
        cube = cfg.get("data", None)
        wave = cfg.get("wavelength", None)
        if cube is None:
            self._canvas_sel.draw_idle()
            return

        arr = _np.asarray(cube)
        H, W, C = arr.shape
        # x축
        use_wave = False
        if wave is not None:
            xs = _np.asarray(wave, dtype=float)
            use_wave = (xs.shape[0] == C) and _np.all(_np.isfinite(xs))
        if not use_wave:
            xs = _np.arange(C)

        # 색상 순환(많이 그릴 때 구분)
        colors = None
        try:
            import itertools
            colors = itertools.cycle(["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"])
        except Exception:
            pass

        # 각 행 그리기
        for r in rows:
            loc_item = self.tableSelectedPixels.item(r, 2)
            if not loc_item: 
                continue
            txt = loc_item.text().strip()  # "(x, y)"
            try:
                x_str, y_str = txt.strip("()").split(",")
                x, y = int(x_str), int(y_str)
            except Exception:
                y_str, x_str = txt.strip("()").split(",")
                y, x = int(y_str), int(x_str)

            if not (0 <= y < H and 0 <= x < W):
                continue

            spec = _np.nan_to_num(_np.asarray(arr[y, x, :], dtype=float),
                                nan=0.0, posinf=0.0, neginf=0.0)

            label = f"#{r+1} (x={x}, y={y})"
            if colors is not None:
                c = next(colors)
                ax.plot(xs, spec, linewidth=1.6, label=label, color=c)
                ax.scatter(xs, spec, s=12, color=c)
            else:
                ax.plot(xs, spec, linewidth=1.6, label=label)
                ax.scatter(xs, spec, s=12)

        try:
            ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.8)
        except Exception:
            pass

        self._canvas_sel.draw_idle()

    def _collect_preview_rows(self) -> list[int]:
        """
        그래프에 그릴 행 인덱스 목록을 반환.
        - 체크된 행이 하나 이상: 체크된 행들을 반환(작은 번호 우선, 최대 20개 권장)
        - 체크가 없으면: 현재 선택된 행이 유효하면 [current_row]
        - 둘 다 없으면: []
        """
        tbl = self.tableSelectedPixels
        rows = []
        # 1) 체크된 행 우선
        for r in range(tbl.rowCount()):
            w = tbl.cellWidget(r, 0)
            if isinstance(w, QtWidgets.QCheckBox) and w.isChecked():
                rows.append(r)
        if rows:
            return rows[:20]  # 과도한 라인 방지(원하면 조절)

        # 2) 없으면 선택 행
        cr = tbl.currentRow()
        return [cr] if (0 <= cr < tbl.rowCount()) else []

    def refresh_preview(self) -> None:
        """체크/선택 상태를 읽어 그래프를 갱신한다."""
        rows = self._collect_preview_rows()
        self._plot_rows_spectrum(rows)
        
    def _collect_rows(self) -> list:
        """
        테이블 전 행 대상 수집(체크박스 무시).
        - Locate: 컬럼 2 "(x, y)"
        - Class : 컬럼 3
        """
        import re
        rows = []
        tv = self.tableSelectedPixels
        for r in range(tv.rowCount()):
            loc_item = tv.item(r, 2)
            cid_item = tv.item(r, 3)
            if not (loc_item and cid_item):
                continue
            loc = loc_item.text().strip()  # "(x, y)"
            cid_txt = cid_item.text().strip()
            if not cid_txt or not cid_txt.lstrip("-").isdigit():
                continue
            m = re.match(r"\s*\((\-?\d+)\s*,\s*(\-?\d+)\)\s*$", loc)
            if not m:
                continue
            x, y = int(m.group(1)), int(m.group(2))
            rows.append({"y": y, "x": x, "cid": int(cid_txt)})
        return rows

    def _on_click_register(self):
        rows = self._collect_rows()
        if not rows:
            QtWidgets.QMessageBox.information(self, "안내", "등록할 픽셀이 없습니다.")
            return
        self.labeling_requested.emit(rows)
        QtWidgets.QMessageBox.information(self, "안내", f"총 {len(rows)}개 픽셀 등록 요청을 보냈습니다.")
        
    # 클래스 내부에 추가
    def _on_class_combo_changed(self, row: int, combo_or_idx):
        """
        행(row)의 클래스 콤보 변경 시 UI만 갱신(내부 리스트 불필요)
        """
        try:
            # 콤보 위젯 확보
            if hasattr(combo_or_idx, "currentData"):   # QComboBox 자체가 넘어온 경우
                combo = combo_or_idx
            else:
                tbl = self.tableSelectedPixels
                w = tbl.cellWidget(row, 3)
                combo = w if isinstance(w, QtWidgets.QComboBox) else None
            if combo is None or combo.currentIndex() < 0:
                return

            # 필요 시 3열을 아이템 텍스트로도 동기화(선택사항)
            data = combo.currentData()
            if data is not None:
                # 3열이 콤보면 그대로 두고, 아이템으로 보여주고 싶을 땐 아래 한 줄 활성화
                # self.tableSelectedPixels.setItem(row, 3, QTableWidgetItem(str(int(data))))
                pass

            # 외부로 변경 알림이 필요하면 여기서 emit 호출
            # if hasattr(self, "_emit_table_changed"):
            #     self._emit_table_changed()

        except Exception:
            import logging
            logging.exception("[PixelLabelingDialog] _on_class_combo_changed failed")

    def set_class_options(self, items: list[tuple[int, str]]) -> None:
        """MainWindow에서 주입하는 전체 클래스 목록 저장."""
        try:
            cleaned: list[tuple[int, str]] = []
            self._cid_to_name.clear()
            for cid, name in (items or []):
                try:
                    cid_i = int(cid)
                    name_s = str(name)
                except Exception:
                    continue
                cleaned.append((cid_i, name_s))
                self._cid_to_name[cid_i] = name_s
            self._class_options = cleaned
        except Exception:
            self._class_options = []
            self._cid_to_name.clear()
    
    def set_default_cid(self, cid: Optional[int]) -> None:
        """user_labeling_dialog에서 선택한 기본 CID 설정."""
        self._default_cid = int(cid) if cid is not None else None