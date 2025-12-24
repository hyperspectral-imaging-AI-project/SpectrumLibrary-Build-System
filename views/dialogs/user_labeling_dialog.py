# views/dialogs/user_labeling_dialog.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import os
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QColor, QIcon, QPixmap
from controllers.pixel_click import ClickMode
import itertools  # ★ 누적 색상 순환용
from data.db import material_code_name, send_label_add
from services.resampling_cache import load_classes_from_info, resample_cache_paths

class UserLabelingDialog(QtWidgets.QDialog):
    """
    사용자 지정 라벨링 다이얼로그
    - .ui: user_labeling_dialog.ui
    - 테이블: [checkbox, #, Locate, Select Class, option, 삭제]
    """
    labeling_requested = pyqtSignal(list)  # [{y,x,cid}, ...] : 후보 등록 버튼 클릭 시 방출
    detail_requested = pyqtSignal(dict)  # ★ {"y":int,"x":int,"cid":int,"row":int}
    table_changed = pyqtSignal(list)  # ★ [{"y":..,"x":..,"cid":..}, ...] 현재 테이블 전체

    def __init__(self, parent=None, ui_dir: Optional[Path] = None):
        super().__init__(parent)
        self.setWindowTitle("사용자 지정 라벨링")

        app_dir = Path(getattr(parent, "app_dir", Path(__file__).resolve().parents[3]))
        ui_dir = Path(ui_dir) if ui_dir else (app_dir / "ui")
        self._root = uic.loadUi(str(ui_dir / "user_labeling_dialog.ui"))

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._root)

        # .ui 내부 레이아웃 마진 보정
        if hasattr(self._root, "verticalLayout_root"):
            try:
                self._root.verticalLayout_root.setContentsMargins(12, 12, 12, 12)
            except Exception:
                pass

        # 내부 상태
        # 내부 상태
        self._pixel_labels: List[Tuple[int, int, int]] = []  # (y, x, class_id)
        self._class_options: List[Tuple[int, str]] = []      # [(cid, name)]
        self._cid_to_qcolor: Dict[int, QColor] = {}          # 팔레트
        self._cid_to_name_cache: Dict[int, str] = {}         # API에서 가져온 cid별 이름 캐시
        self._names_fetched: bool = False                    # API 호출 플래그 (한 번만 호출)

        # ★ 추가: HSI 데이터/파장 보관
        self._cube: Optional[np.ndarray] = None              # (H,W,C)
        self._wavelength: Optional[np.ndarray] = None        # (C,)

        # ★ 추가: Matplotlib 캔버스 plotHolder에 장착
        self._fig = Figure(figsize=(5, 3), tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)
        
        # ★ 누적 그래프 관리용(라인 핸들과 색상 순환자)
        self._lines = []  # List[Line2D]
        self._line_by_row: Dict[int, object] = {}  # ★ row -> Line2D
        self._color_cycle = itertools.cycle(['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
        
        holder: QtWidgets.QWidget = getattr(self._root, "plotHolder", None)
        if holder is not None:
            lay = QtWidgets.QVBoxLayout(holder)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self._canvas)
        self._ax.set_xlabel("Wavelength")
        self._ax.set_ylabel("Reflectance / Intensity")
        self._ax.grid(True)
        self._canvas.draw()

        # 버튼 연결
        self._combo_global = self._root.comboBox   # 아래 콤보박스
        self._combo_global.clear()

        self._root.btnRegisterCandidates.clicked.connect(self._on_register_candidates)
        self._root.btnCancel.clicked.connect(self.reject)

        # 테이블 초기화
        self._init_table()
        
        # 다이얼로그 열릴 때, 스펙트럼으로 등록된 mtrl_nm을 콤보박스에 로드
        self._init_class_options_from_source()


    # ====== 외부 주입 API ======
    # 기존: set_class_options 안에서 material_code_name(api_base) 호출 → 삭제
    def set_class_options(self, items):
        """
        items: [(cid, name), ...] 형태 가정
        - 콤보박스에 넣기 전에 'Class(=cid) 오름차순'으로 정렬해서 넣음
        - name은 mtrl_nm(물질 이름)으로 간주하고 캐시에 저장
        """
        # 1) 안전 캐스팅 + 정렬
        cleaned = []
        for rec in (items or []):
            try:
                # tuple 길이에 따라 분기 (cid, name[, desc])
                if isinstance(rec, (list, tuple)):
                    if len(rec) >= 2:
                        cid_val, name_val = rec[0], rec[1]
                    else:
                        continue
                elif isinstance(rec, dict):
                    cid_val = rec.get("mtrl_cd") or rec.get("cid")
                    name_val = rec.get("mtrl_nm") or rec.get("name")
                else:
                    continue

                cid = int(cid_val)
                name = str(name_val)
                cleaned.append((cid, name))
                self._cid_to_name_cache[cid] = name
            except Exception:
                continue
        cleaned.sort(key=lambda x: x[0])  # ← Class(=cid) 기준 정렬

        # 2) 내부 보관
        self._class_options = cleaned

        # 3) 전역 콤보 갱신
        if hasattr(self, "_combo_global") and self._combo_global is not None:
            self._combo_global.blockSignals(True)
            self._combo_global.clear()
            for cid_opt, name_opt in self._class_options:
                # 캐시에서 mtrl_nm 가져오기 (없으면 name_opt 사용)
                display_name = self._cid_to_name_cache.get(int(cid_opt), name_opt)
                self._combo_global.addItem(display_name, cid_opt)  # mtrl_nm 기준으로 표시, 내부 데이터는 CID
            self._combo_global.blockSignals(False)


    def set_palette(self, cid_to_qcolor: Dict[int, QColor]):
        """팔레트 설정."""
        self._cid_to_qcolor = dict(cid_to_qcolor)
        self._refresh_color_all_rows()

    # ====== 테이블 구성 ======
    def _init_table(self):
        table = self._root.tableSelected
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["□", "#", "Locate", "option", "삭제"])
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setStretchLastSection(False)

        table.setColumnWidth(0, 30)   # checkbox
        table.setColumnWidth(1, 40)   # 번호
        table.setColumnWidth(2, 100)  # 좌표
        table.setColumnWidth(3, 80)   # option(상세)
        table.setColumnWidth(4, 40)   # 삭제

        try:
            table.verticalHeader().setVisible(False)
        except Exception:
            pass


    def add_pixel_label(self, y: int, x: int, class_id: Optional[int] = None):
        """행 추가: (y, x, class_id). class_id 없으면 첫 옵션으로."""
        if class_id is None:
            class_id = self._class_options[0][0] if self._class_options else -1
        self._pixel_labels.append((int(y), int(x), int(class_id)))
        self._append_row(len(self._pixel_labels) - 1)

    # 열: 0 체크박스, 1 번호, 2 좌표, 3 option(상세), 4 삭제
    def _append_row(self, idx: int):
        table: QtWidgets.QTableWidget = self._root.tableSelected
        table.insertRow(idx)
        y, x, cid = self._pixel_labels[idx]

        # (0) 체크박스
        checkbox = QtWidgets.QCheckBox()
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(lambda state, row=idx: self._on_checkbox_changed(row, state))
        table.setCellWidget(idx, 0, checkbox)

        # (1) 번호
        it_no = QtWidgets.QTableWidgetItem(str(idx + 1))
        it_no.setTextAlignment(Qt.AlignCenter)
        it_no.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        table.setItem(idx, 1, it_no)

        # (2) 좌표
        it_loc = QtWidgets.QTableWidgetItem(f"{x}, {y}")
        it_loc.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        table.setItem(idx, 2, it_loc)

        # (3) option: 상세분석 버튼
        btn_detail = QtWidgets.QPushButton("상세분석")
        btn_detail.setCursor(Qt.PointingHandCursor)
        btn_detail.setToolTip("해당 픽셀을 상세 분석")
        btn_detail.clicked.connect(lambda _=None, r=idx: self._on_detail_clicked(r))
        table.setCellWidget(idx, 3, btn_detail)

        # (4) 삭제 버튼
        btn = QtWidgets.QPushButton("✖")
        btn.setToolTip("삭제")
        btn.clicked.connect(lambda _=None, row=idx: self._remove_row(row))
        table.setCellWidget(idx, 4, btn)

        self._emit_table_changed()

    def _on_checkbox_changed(self, row: int, state: int):
        """체크박스 상태 변경 (필요시 구현)"""
        pass

    def _on_combo_changed(self, row: int, combo: QtWidgets.QComboBox):
        if 0 <= row < len(self._pixel_labels):
            y, x, _ = self._pixel_labels[row]
            cid = int(combo.currentData())
            self._pixel_labels[row] = (y, x, cid)

    def _remove_row(self, row: int):
        """행 삭제 + 번호 재정렬 + 그래프/버튼/체크박스 시그널 재바인딩"""
        if not (0 <= row < len(self._pixel_labels)):
            return

        # 1) 내부 데이터 삭제
        self._pixel_labels.pop(row)

        table: QtWidgets.QTableWidget = self._root.tableSelected

        # 2) 테이블에서 행 제거
        table.removeRow(row)

        # 3) (옵션) 그래프: 해당 row 라인 제거 및 재매핑
        try:
            ln = self._line_by_row.pop(row, None)
            if ln is not None:
                try:
                    ln.remove()  # matplotlib Line2D 제거
                except Exception:
                    pass
            # 인덱스가 하나씩 당겨지므로 키 재정렬
            new_map: Dict[int, object] = {}
            for old_row, h in self._line_by_row.items():
                new_row = old_row - 1 if old_row > row else old_row
                if new_row >= 0:
                    new_map[new_row] = h
            self._line_by_row = new_map

            # 축/범례 갱신
            self._ax.relim()
            self._ax.autoscale_view()
            if self._ax.get_legend() is not None:
                self._ax.legend(loc="best", fontsize=8)
            self._canvas.draw_idle()
        except Exception:
            pass

        # 4) 남은 행들 번호(열 1) 및 위젯 시그널 재바인딩
        for i in range(table.rowCount()):
            # 번호 셀 갱신
            it_no = table.item(i, 1)
            if it_no:
                it_no.setText(str(i + 1))

            # (0) 체크박스
            w_chk = table.cellWidget(i, 0)
            if isinstance(w_chk, QtWidgets.QCheckBox):
                try:
                    w_chk.stateChanged.disconnect()
                except Exception:
                    pass
                w_chk.stateChanged.connect(lambda state, r=i: self._on_checkbox_changed(r, state))

            # (3) 상세분석 버튼
            w_detail = table.cellWidget(i, 3)
            if isinstance(w_detail, QtWidgets.QPushButton):
                try:
                    w_detail.clicked.disconnect()
                except Exception:
                    pass
                w_detail.clicked.connect(lambda _=None, r=i: self._on_detail_clicked(r))

            # (4) 삭제 버튼
            w_del = table.cellWidget(i, 4)
            if isinstance(w_del, QtWidgets.QPushButton):
                try:
                    w_del.clicked.disconnect()
                except Exception:
                    pass
                w_del.clicked.connect(lambda _=None, r=i: self._remove_row(r))

        # 5) 변경사항 반영 이벤트
        self._emit_table_changed()
                               
    def _on_detail_clicked(self, row: int):
        """
        '상세분석' 버튼 클릭 처리: 해당 행의 (y,x,cid)를 읽어 시그널로 방출하거나
        여기서 직접 다이얼로그를 띄워도 됨.
        """
        if not (0 <= row < len(self._pixel_labels)):
            return
        y, x, cid = self._pixel_labels[row]
        payload = {"y": int(y), "x": int(x), "cid": int(cid), "row": int(row)}
        # 1) 메인에서 분석 UI를 띄울 수 있도록 시그널로 전달 (권장)
        try:
            self.detail_requested.emit(payload)
        except Exception:
            pass
        # 2) 또는 여기서 즉시 스펙트럼을 강조/확대하는 등 직접 처리도 가능:
        # self.plot_spectrum(y, x)  # 누적 플롯 사용 중이면 하이라이트 추가 가능

    def clear_labels(self):
        """테이블 초기화"""
        self._pixel_labels.clear()
        self._root.tableSelected.setRowCount(0)

    def _refresh_combo_all_rows(self):
        """클래스 옵션 변경 시 모든 행의 콤보 갱신."""
        table: QtWidgets.QTableWidget = self._root.tableSelected
        for i in range(table.rowCount()):
            if i >= len(self._pixel_labels):
                continue
                
            cb = table.cellWidget(i, 3)
            if isinstance(cb, QtWidgets.QComboBox):
                cur_cid = int(self._pixel_labels[i][2])
                cb.blockSignals(True)
                cb.clear()
                
                # 콤보박스에 클래스 옵션 추가
                for cid_opt, name_opt in self._class_options:
                    # 캐시에서 mtrl_nm 가져오기 (없으면 name_opt 사용)
                    display_name = self._cid_to_name_cache.get(int(cid_opt), name_opt)
                    cb.addItem(display_name, cid_opt)  # mtrl_nm 기준으로 표시, 내부 데이터는 CID
                
                # 기존 cid 선택 복원
                found = False
                for j in range(cb.count()):
                    if int(cb.itemData(j)) == cur_cid:
                        cb.setCurrentIndex(j)
                        found = True
                        break
                
                # cid를 찾지 못한 경우 로그 출력 (이미 위에서 기존 cid를 추가했으므로 발생하지 않아야 함)
                if not found and cur_cid >= 0:
                    import logging
                    logging.warning(f"[UserLabeling] 콤보박스에서 cid={cur_cid}를 찾을 수 없습니다. 행={i}")
                
                cb.blockSignals(False)

    def _refresh_color_all_rows(self):
        """팔레트 변경 시 갱신."""
        pass

    # ====== 데이터 수집/버튼 핸들러 ======
    def get_selected_pixel_labels(self) -> List[Dict[str, int]]:
        """체크된 픽셀들만 반환."""
        out = []
        table: QtWidgets.QTableWidget = self._root.tableSelected
        for i, (y, x, cid) in enumerate(self._pixel_labels):
            checkbox = table.cellWidget(i, 0)
            if isinstance(checkbox, QtWidgets.QCheckBox) and checkbox.isChecked():
                out.append({"y": int(y), "x": int(x), "cid": int(cid)})
        return out

    def _on_register_candidates(self):
        table: QtWidgets.QTableWidget = self._root.tableSelected
        if table.rowCount() == 0:
            QtWidgets.QMessageBox.warning(self, "경고", "등록할 픽셀이 없습니다.")
            return

        # 전역 콤보에서 CID 가져와 전체 행에 일괄 적용
        cid = self._get_global_cid()
        if cid == -1:
            QtWidgets.QMessageBox.warning(self, "경고", "하단 콤보에서 클래스를 먼저 선택하세요.")
            return

        # 전체 행에 적용
        for i in range(len(self._pixel_labels)):
            y, x, _ = self._pixel_labels[i]
            self._pixel_labels[i] = (y, x, cid)
        self._emit_table_changed()

        # 체크된 행만 수집
        selected = self.get_selected_pixel_labels()
        if not selected:
            QtWidgets.QMessageBox.warning(self, "경고", "등록할 픽셀을 먼저 선택하세요.")
            return

        # personal/server 분기 처리: API 호출 (pixel_labeling_dialog와 동일한 방식)
        # 주의: UserLabelingDialog의 parent는 MainWindow입니다 (main_window.py:1536)
        # 따라서 self.parent()가 바로 MainWindow입니다.
        parent = self.parent()
        user_type = None
        main_window = None
        
        # MainWindow에서 user_type 가져오기
        if parent:
            try:
                # parent가 MainWindow인지 확인 (hasattr로 체크)
                if hasattr(parent, "_cache_primary_path") or hasattr(parent, "_extract_src_path"):
                    main_window = parent  # parent가 바로 MainWindow
                else:
                    # parent가 다른 위젯인 경우, parent의 parent를 확인
                    main_window = parent.parent()
                    if main_window and not (hasattr(main_window, "_cache_primary_path") or hasattr(main_window, "_extract_src_path")):
                        main_window = None
                
                if main_window:
                    user_type = getattr(main_window, "user_type", None)
            except Exception as e:
                import logging
                logging.debug(f"[UserLabeling] MainWindow 찾기 실패: {e}")
                pass
        
        # server 사용자만 API 호출 (personal 사용자는 API 호출 안 함)
        import logging
        if user_type == "server":
            logging.info("[UserLabeling] server 사용자: API 호출 시도")
            try:
                img_cd = getattr(main_window, "image_cd", None) if main_window else None
                cfg = getattr(main_window, "cfg", {}) if main_window else {}
                cube = cfg.get("data", None)
                wavelength = cfg.get("wavelength", None)
                
                if img_cd is not None and cube is not None:
                    import numpy as np
                    cube = np.asarray(cube, dtype=float)
                    H, W, C = cube.shape
                    
                    # 좌표/중복/경계 정리
                    uniq = {}
                    for rec in selected:
                        y, x = int(rec["y"]), int(rec["x"])
                        if 0 <= y < H and 0 <= x < W:
                            uniq[(y, x)] = int(rec.get("cid", -1))
                    
                    if uniq:
                        ys = np.fromiter((p[0] for p in uniq.keys()), dtype=int)
                        xs = np.fromiter((p[1] for p in uniq.keys()), dtype=int)
                        cids = np.fromiter((uniq[p] for p in uniq.keys()), dtype=int)
                        
                        # 배치 스펙트럼 추출
                        spectra = cube[ys, xs, :]  # (N, C)
                        spectra = np.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # rows 페이로드 구성
                        rows = []
                        for (y, x), cid, rfl in zip(uniq.keys(), cids.tolist(), spectra):
                            rows.append({
                                "img_cd": int(img_cd),
                                "mtrl_cd": int(cid),
                                "img_x": int(x),
                                "img_y": int(y),
                                "rfl": rfl.astype(float).tolist(),
                            })
                        
                        # # API 호출
                        # api_base = os.getenv('label_add_url')
                        # if api_base:
                        #     try:
                        #         send_label_add(api_base=api_base, targets=rows)
                        #         logging.info(f"[UserLabeling] API 호출 완료: {len(rows)}개 픽셀 등록")
                        #     except Exception as e:
                        #         logging.warning(f"[UserLabeling] API 호출 실패: {e}")
                        # else:
                        #     logging.warning("[UserLabeling] label_add_url 환경변수가 설정되지 않았습니다.")
            except Exception as e:
                logging.debug(f"[UserLabeling] API 호출 중 오류 (무시): {e}")
        elif user_type == "personal":
            logging.info("[UserLabeling] personal 사용자: API 저장을 건너뜁니다.")
        else:
            # user_type이 None이거나 다른 값인 경우
            logging.warning(f"[UserLabeling] user_type을 확인할 수 없습니다 (user_type={user_type}). API 호출을 건너뜁니다.")

        self.labeling_requested.emit(selected)
        self.accept()

    def set_map_name(self, name: str):
        """맵 이름 설정"""
        self._root.editMapName.setText(name)

    def set_data(self, cube: np.ndarray, wavelength: Optional[np.ndarray] = None):
        self._cube = np.asarray(cube)
        self._wavelength = None if wavelength is None else np.asarray(wavelength)
        
    # ★ 추가: (y,x) 스펙트럼 플롯
    def plot_spectrum(self, y: int, x: int):
        if self._cube is None:
            return
        H, W, C = self._cube.shape
        if not (0 <= y < H and 0 <= x < W):
            return

        spec = self._cube[int(y), int(x), :].astype(float)
        # X축
        if self._wavelength is not None and isinstance(self._wavelength, np.ndarray) \
        and self._wavelength.shape[0] == spec.shape[0]:
            xs = self._wavelength
            xlab = "Wavelength"
        else:
            xs = np.arange(spec.size)
            xlab = "Band index"

        # ★ 누적: 기존 축/라인 유지, 새 라인만 추가
        color = next(self._color_cycle)
        (ln,) = self._ax.plot(xs, spec, marker=".", linewidth=1.0,
                            label=f"y={int(y)}, x={int(x)}", color=color)
        self._lines.append(ln)
        # ★ 방금 추가된 테이블 행(row)에 라인 매핑
        row_idx = len(self._pixel_labels) - 1
        self._line_by_row[row_idx] = ln

        # 레이블/그리드/범례 업데이트
        self._ax.set_xlabel(xlab)
        self._ax.set_ylabel("Reflectance / Intensity")
        self._ax.grid(True)
        # 범례가 너무 커지면 필요시 loc/cols 조절 가능
        self._ax.legend(loc="best", fontsize=8)

        # 자동 스케일
        self._ax.relim()
        self._ax.autoscale_view()

        self._canvas.draw_idle()

    def _emit_table_changed(self):
        rows = [{"y": int(y), "x": int(x), "cid": int(cid)} for (y,x,cid) in self._pixel_labels]
        try:
            self.table_changed.emit(rows)
        except Exception:
            pass
        
    def _get_global_cid(self) -> int:
        try:
            if self._combo_global is None or self._combo_global.currentIndex() < 0:
                return -1
            return int(self._combo_global.currentData())
        except Exception:
            return -1
        
    def _init_class_options_from_source(self):
        """
        다이얼로그가 처음 열릴 때, personal/server 타입에 따라
        스펙트럼으로 등록된 모든 mtrl_nm을 읽어와서
        self._class_options / 아래 콤보박스(_combo_global)를 채운다.
        """
        # 이미 한 번 불러왔다면 재호출 안 함 (원하면 제거 가능)
        if getattr(self, "_names_fetched", False):
            return

        parent = self.parent()
        main_window = None
        user_type = None
        primary_path = None

        try:
            # parent 가 MainWindow 이거나 그 parent 인지 확인 (기존 패턴 재사용)
            if parent:
                if hasattr(parent, "_cache_primary_path") or hasattr(parent, "_extract_src_path"):
                    main_window = parent
                else:
                    main_window = parent.parent()
                    if main_window and not (hasattr(main_window, "_cache_primary_path") or hasattr(main_window, "_extract_src_path")):
                        main_window = None

            if main_window:
                user_type = getattr(main_window, "user_type", None)
                cfg = getattr(main_window, "cfg", {})
                if hasattr(main_window, "_cache_primary_path"):
                    primary_path = main_window._cache_primary_path(cfg)
                elif hasattr(main_window, "_extract_src_path"):
                    primary_path = main_window._extract_src_path(cfg)
        except Exception:
            main_window = None
            user_type = None
            primary_path = None

        import logging
        class_options: list[tuple[int, str]] = []

        try:
            if user_type == "personal":
                # personal: .info 파일에서 클래스 목록 로드
                if primary_path:
                    try:
                        loaded = load_classes_from_info(primary_path)
                        # load_classes_from_info 가 (cid, name) 또는 (cid, name, desc) 형태를 줄 수 있으므로 정규화
                        for rec in (loaded or []):
                            try:
                                if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                                    cid_i = int(rec[0])
                                    name_s = str(rec[1])
                                    class_options.append((cid_i, name_s))
                            except Exception:
                                continue
                        logging.info(f"[UserLabeling] 초기 클래스 로드(personal): {len(class_options)}개")
                    except Exception as e:
                        logging.exception(f"[UserLabeling] 초기 클래스 로드 실패(personal): {e}")
                else:
                    logging.warning("[UserLabeling] personal: primary_path 없음 → .info에서 클래스 로드 불가")

            elif user_type == "server":
                # server: API(material_code_name) 에서 클래스 목록 로드
                try:
                    api_base = os.getenv('material_code_name_url') or os.getenv("MATERIAL_API_BASE")
                    if api_base:
                        api_result = material_code_name(api_base)
                        for rec in api_result:
                            try:
                                cid = int(rec.get("mtrl_cd", -1))
                                name = str(rec.get("mtrl_nm", f"Class {cid}")).strip()
                                if cid >= 0 and name:
                                    class_options.append((cid, name))
                            except Exception:
                                continue
                        logging.info(f"[UserLabeling] 초기 클래스 로드(server): {len(class_options)}개")
                    else:
                        logging.warning("[UserLabeling] server: material_code_name_url/MATERIAL_API_BASE 환경변수 없음")
                except Exception as e:
                    logging.exception(f"[UserLabeling] 초기 클래스 로드 실패(server): {e}")

            else:
                logging.warning(f"[UserLabeling] user_type 미확인(user_type={user_type}) → 초기 클래스 로드 생략")

        except Exception as e:
            logging.exception(f"[UserLabeling] 초기 클래스 로드 중 예외: {e}")
            class_options = []

        # 정리 & 세팅
        if class_options:
            # 중복 제거 + cid 기준 정렬
            uniq_map: Dict[int, str] = {}
            for cid_i, name_s in class_options:
                uniq_map[int(cid_i)] = str(name_s)
            norm_list = sorted(uniq_map.items(), key=lambda x: x[0])

            # 내부 캐시 및 콤보 갱신
            self.set_class_options(norm_list)
            self._names_fetched = True
        else:
            # 아무 것도 못 가져온 경우: 기존 상태 유지 (콤보는 비어있음)
            logging.warning("[UserLabeling] 초기 클래스 옵션을 로드하지 못했습니다.")
