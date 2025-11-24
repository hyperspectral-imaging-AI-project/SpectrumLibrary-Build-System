# views/dialogs/diffusion_dialog.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QColor, QIcon, QPixmap


class DiffusionDialog(QtWidgets.QDialog):
    """
    유사도 확산 맵 생성 다이얼로그 (UI: diffusion_dialog.ui)
    - 기준 맵 선택(cboBaseMap): LayerManager에 등록된 top-level classmap 중에서 선택
    - 시드 테이블(tblSeedPixels): [색칩, #, Locate, Select Class, 삭제]
      * 색/클래스 콤보는 팔레트/클래스 옵션으로 구성
    - 임계값(dsbTauBase), 마진값(dsbTauGrow), 맵 이름(txtMapName)
    """
    diffusion_requested = pyqtSignal(dict)          # {tau, delta, map_name, pixel_seeds:[{y,x,cid}], classmap_name:str}
    analyze_requested   = pyqtSignal(list)          # [{y,x,cid}]
    seeds_changed       = pyqtSignal(list)          # [{y,x,cid}]
    base_lock_changed = pyqtSignal(bool)  # ★ 추가: 기준 맵 잠금 상태 신호

    def __init__(self, parent=None, ui_dir: Optional[Path] = None):
        super().__init__(parent)
        self.setWindowTitle("확산 맵 생성")

        app_dir = Path(getattr(parent, "app_dir", Path(__file__).resolve().parents[3]))
        ui_dir = Path(ui_dir) if ui_dir else (app_dir / "ui")
        self._root = uic.loadUi(str(ui_dir / "diffusion_dialog.ui"), self)

        # 위젯 핸들(디자이너 objectName 기준)
        self.cboBaseMap: QtWidgets.QComboBox = self.findChild(QtWidgets.QComboBox, "cboBaseMap")
        self.tbl: QtWidgets.QTableWidget    = self.findChild(QtWidgets.QTableWidget, "tblSeedPixels")
        self.dsbTauBase: QtWidgets.QDoubleSpinBox = self.findChild(QtWidgets.QDoubleSpinBox, "dsbTauBase")
        self.dsbTauGrow: QtWidgets.QDoubleSpinBox = self.findChild(QtWidgets.QDoubleSpinBox, "dsbTauGrow")
        self.txtMapName: QtWidgets.QLineEdit      = self.findChild(QtWidgets.QLineEdit, "txtMapName")
        self.btnNext: QtWidgets.QPushButton       = self.findChild(QtWidgets.QPushButton, "btnNext")
        self.btnSaveMap: QtWidgets.QPushButton    = self.findChild(QtWidgets.QPushButton, "btnSaveMap")
        self.btnCancel:  QtWidgets.QPushButton    = self.findChild(QtWidgets.QPushButton, "btnCancel")

        # 내부 상태
        self._pixel_seeds: List[Tuple[int, int, int]] = []  # (y, x, cid)
        self._class_options: List[Tuple[int, str]] = []     # [(cid, name)]
        self._cid_to_qcolor: Dict[int, QColor] = {}
        self._base_locked: bool = False

        # 테이블 초기화(디자이너는 3열이지만 런타임에서 5열로 확장)
        self._init_table()

        # 버튼
        if self.btnNext:
            self.btnNext.clicked.connect(self._on_next_clicked)
        self.btnSaveMap.clicked.connect(self._on_save_map)
        self.btnCancel.clicked.connect(self.reject)

    # ===== 외부 주입 =====
    def set_classmap_options(self, names: List[str]):
        self.cboBaseMap.blockSignals(True)
        self.cboBaseMap.clear()
        for nm in names:
            self.cboBaseMap.addItem(nm)
        self.cboBaseMap.blockSignals(False)

    def set_palette(self, cid_to_qcolor: Dict[int, QColor]):
        self._cid_to_qcolor = dict(cid_to_qcolor)
        self._refresh_color_all_rows()

    def set_class_options(self, items: List[Tuple[int, str]]):
        """Select Class 콤보 옵션."""
        self._class_options = list(items)
        self._refresh_combo_all_rows()

    # ===== getter =====
    def get_selected_classmap_name(self) -> Optional[str]:
        if self.cboBaseMap.count() == 0:
            return None
        s = self.cboBaseMap.currentText().strip()
        return s or None
    
    def is_base_locked(self) -> bool:
        """기준 맵이 잠금 상태인지 반환."""
        return getattr(self, "_base_locked", False)

    def get_parameters(self) -> dict:
        name = (self.txtMapName.text() or "Diffusion1").strip()
        return {
            "tau": float(self.dsbTauBase.value()),
            "delta": float(self.dsbTauGrow.value()),
            "map_name": name,
            "pixel_seeds": self.get_pixel_seeds(),
            "classmap_name": self.get_selected_classmap_name(),
        }

    # ===== 테이블 =====
    def _init_table(self):
        t = self.tbl
        t.clear()
        t.setColumnCount(5)
        t.setHorizontalHeaderLabels(["", "#", "Locate", "Select Class", ""])
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        t.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setStretchLastSection(False)
        t.setColumnWidth(0, 26)   # 색칩
        t.setColumnWidth(1, 46)   # 번호
        t.setColumnWidth(2, 140)  # 좌표
        t.setColumnWidth(3, 170)  # 콤보
        t.setColumnWidth(4, 36)   # 삭제

    def _color_icon(self, cid: int) -> QIcon:
        qc = self._cid_to_qcolor.get(int(cid), QColor(160, 160, 160))
        pm = QPixmap(14, 14); pm.fill(qc)
        return QIcon(pm)

    def add_pixel_seed(self, y: int, x: int, class_id: Optional[int] = None):
        """시드 행 추가: (y,x,cid). class_id 없으면 첫 옵션."""
        cid = int(class_id) if class_id is not None else (self._class_options[0][0] if self._class_options else -1)
        self._pixel_seeds.append((int(y), int(x), cid))
        self._append_row(len(self._pixel_seeds) - 1)
        self.seeds_changed.emit(self.get_pixel_seeds())

    def _append_row(self, idx: int):
        t = self.tbl
        t.insertRow(idx)
        y, x, cid = self._pixel_seeds[idx]

        # (0) 색칩
        it_chip = QtWidgets.QTableWidgetItem()
        it_chip.setFlags(Qt.ItemIsEnabled)
        it_chip.setIcon(self._color_icon(cid))
        t.setItem(idx, 0, it_chip)

        # (1) 번호
        it_no = QtWidgets.QTableWidgetItem(str(idx + 1))
        it_no.setTextAlignment(Qt.AlignCenter)
        it_no.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        t.setItem(idx, 1, it_no)

        # (2) 좌표
        it_loc = QtWidgets.QTableWidgetItem(f"{y}, {x}")  # 표시는 (y, x)
        it_loc.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        t.setItem(idx, 2, it_loc)

        # (3) Select Class 콤보
        combo = QtWidgets.QComboBox(t)
        for cid_opt, name in self._class_options:
            combo.addItem(f"Class {cid_opt}", cid_opt)
        # 현재값 지정
        if self._class_options:
            for i in range(combo.count()):
                if int(combo.itemData(i)) == int(cid):
                    combo.setCurrentIndex(i); break
        combo.currentIndexChanged.connect(
            lambda _=None, row=idx, cb=combo: self._on_combo_changed(row, cb)
        )
        t.setCellWidget(idx, 3, combo)

        # (4) 삭제 버튼
        btn = QtWidgets.QPushButton("✖")
        btn.setToolTip("삭제")
        btn.clicked.connect(lambda _=None, row=idx: self._remove_row(row))
        t.setCellWidget(idx, 4, btn)

    def _on_combo_changed(self, row: int, combo: QtWidgets.QComboBox):
        if 0 <= row < len(self._pixel_seeds):
            y, x, _ = self._pixel_seeds[row]
            cid = int(combo.currentData())
            self._pixel_seeds[row] = (y, x, cid)
            # 색칩 업데이트
            it = self.tbl.item(row, 0)
            if it:
                it.setIcon(self._color_icon(cid))
            self.seeds_changed.emit(self.get_pixel_seeds())

    def _remove_row(self, row: int):
        if not (0 <= row < len(self._pixel_seeds)):
            return
        self._pixel_seeds.pop(row)
        self.tbl.removeRow(row)
        # 번호 및 콜백 재바인딩
        for i in range(self.tbl.rowCount()):
            if self.tbl.item(i, 1):
                self.tbl.item(i, 1).setText(str(i + 1))
            # 삭제 버튼
            w_del = self.tbl.cellWidget(i, 4)
            if isinstance(w_del, QtWidgets.QPushButton):
                try: w_del.clicked.disconnect()
                except Exception: pass
                w_del.clicked.connect(lambda _=None, r=i: self._remove_row(r))
            # 콤보
            w_cb = self.tbl.cellWidget(i, 3)
            if isinstance(w_cb, QtWidgets.QComboBox):
                try: w_cb.currentIndexChanged.disconnect()
                except Exception: pass
                w_cb.currentIndexChanged.connect(lambda _=None, r=i, cb=w_cb: self._on_combo_changed(r, cb))
        self.seeds_changed.emit(self.get_pixel_seeds())

    def clear_seeds(self):
        self._pixel_seeds.clear()
        self.tbl.setRowCount(0)

    def _refresh_combo_all_rows(self):
        for i in range(self.tbl.rowCount()):
            cb = self.tbl.cellWidget(i, 3)
            if isinstance(cb, QtWidgets.QComboBox):
                cur_cid = int(self._pixel_seeds[i][2])
                cb.blockSignals(True)
                cb.clear()
                for cid_opt, name in self._class_options:
                    cb.addItem(f"Class {cid_opt}", cid_opt)
                for j in range(cb.count()):
                    if int(cb.itemData(j)) == cur_cid:
                        cb.setCurrentIndex(j); break
                cb.blockSignals(False)

    def _refresh_color_all_rows(self):
        for i in range(self.tbl.rowCount()):
            cid = int(self._pixel_seeds[i][2])
            it = self.tbl.item(i, 0)
            if it:
                it.setIcon(self._color_icon(cid))

    # ===== 수집/버튼 =====
    def get_pixel_seeds(self) -> List[Dict[str, int]]:
        return [{"y": y, "x": x, "cid": cid} for (y, x, cid) in self._pixel_seeds]

    def _on_save_map(self):
        if not self._pixel_seeds:
            QtWidgets.QMessageBox.warning(self, "경고", "기준 픽셀을 먼저 선택하세요.")
            return
        params = self.get_parameters()
        if not params.get("classmap_name"):
            QtWidgets.QMessageBox.warning(self, "경고", "기준 맵을 선택하세요.")
            return
        self.diffusion_requested.emit(params)
        self.accept()

    def _on_next_clicked(self):
        sel = self.get_selected_classmap_name()
        if not sel:
            QtWidgets.QMessageBox.information(self, "안내", "기준 맵을 먼저 선택하세요.")
            return
        self._base_locked = True
        if self.cboBaseMap: self.cboBaseMap.setEnabled(False)
        if self.btnNext:    self.btnNext.setEnabled(False)
        self.base_lock_changed.emit(True)  # ★ 잠금 알림