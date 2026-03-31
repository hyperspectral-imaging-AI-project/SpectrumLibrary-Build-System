# views/dialogs/search_labeling_database.py
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from PyQt5 import uic, QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap, QImage, QPainter, QPen, QBrush
from PyQt5.QtWidgets import (
    QDialog, QTableWidget, QTableWidgetItem, QPushButton, 
    QComboBox, QHeaderView, QWidget, QGraphicsView, QLabel
)
from services.resampling_cache import load_classes_from_info

# API 호출 함수 import
try:
    from data.db import search_material_filtering_list
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    logging.warning("[SearchLabelingDatabase] API functions not available")

# matplotlib imports
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    MATPLOTLIB_AVAILABLE = True
    
    # 한글 폰트 설정
    def _setup_korean_font():
        try:
            korean_fonts = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 
                          'Gulim', 'Batang', 'Gungsuh', 'Dotum']
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            font_found = None
            for font in korean_fonts:
                if font in available_fonts:
                    font_found = font
                    break
            if font_found:
                matplotlib.rcParams['font.family'] = font_found
                matplotlib.rcParams['axes.unicode_minus'] = False
                logging.debug(f"[matplotlib] Korean font set to: {font_found}")
            else:
                matplotlib.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            logging.exception(f"[matplotlib] Failed to setup Korean font: {e}")
    
    _setup_korean_font()
    
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not available")


class SpectrumWidget(FigureCanvasQTAgg):
    """matplotlib 기반 스펙트럼 그래프 위젯"""
    
    def __init__(self, parent=None):
        self._figure = Figure(figsize=(6, 4))
        super().__init__(self._figure)
        self.setParent(parent)
        self._ax = self._figure.add_subplot(111)
        self._spectra: List[np.ndarray] = []
        self._wavelengths: Optional[np.ndarray] = None
        self._selected_index: Optional[int] = None
        self._spectrum_types: List[str] = []   # 기존
        self._class_ids: List[Optional[int]] = []  # ★ 추가: 각 스펙트럼의 cid
        

    def set_spectra(self, spectra: List[np.ndarray], 
                    wavelengths: Optional[np.ndarray] = None,
                    spectrum_types: Optional[List[str]] = None,
                    class_ids: Optional[List[int]] = None):
        """스펙트럼 데이터 설정"""
        self._spectra = spectra or []
        self._wavelengths = wavelengths
        self._spectrum_types = spectrum_types or []
        self._class_ids = class_ids or []
        self._selected_index = None
        self._update_plot()
    
    def set_selected_index(self, index: Optional[int]):
        """선택된 스펙트럼 인덱스 설정"""
        self._selected_index = index
        self._update_plot()
    
    def _update_plot(self):
        """그래프 업데이트"""
        self._ax.clear()
        
        if not self._spectra:
            self._ax.text(0.5, 0.5, '스펙트럼 데이터 없음', 
                         ha='center', va='center',
                         transform=self._ax.transAxes, fontsize=12)
            self._ax.set_xlabel('Wavelength (nm)')
            self._ax.set_ylabel('Reflectance')
            self.draw()
            return
        
        # ---- class별 색상 팔레트 준비 ----
        # 고정 팔레트 (원하면 색상 바꿔도 됨)
        base_colors = [
            (0.2, 0.7, 0.7),   # 청록 계열
            (0.3, 0.6, 0.9),   # 파랑 계열
            (0.3, 0.8, 0.4),   # 초록 계열
            (0.8, 0.5, 0.1),   # 주황 계열
            (0.6, 0.3, 0.8),   # 보라 계열
            (0.9, 0.3, 0.3),   # 빨강 계열
        ]
        unique_cids = []
        for cid in self._class_ids:
            if cid is not None and cid not in unique_cids:
                unique_cids.append(cid)
        cid_to_color = {}
        for i, cid in enumerate(unique_cids):
            cid_to_color[cid] = base_colors[i % len(base_colors)]
        
        # X축 준비
        if self._wavelengths is not None and len(self._wavelengths) > 0:
            x_data_base = self._wavelengths
        else:
            max_len = max(len(s) for s in self._spectra) if self._spectra else 0
            x_data_base = np.arange(max_len, dtype=float)
        
        # 각 스펙트럼 그리기
        for idx, spec in enumerate(self._spectra):
            y = np.asarray(spec).ravel()
            spec_len = len(y)
            
            if self._wavelengths is not None and len(self._wavelengths) == spec_len:
                x = self._wavelengths
            elif len(x_data_base) == spec_len:
                x = x_data_base
            else:
                x = np.arange(spec_len, dtype=float)
            
            # 선택된 스펙트럼은 빨간색으로 강조
            if idx == self._selected_index:
                color = (1.0, 0.0, 0.0)
                lw, alpha = 2.5, 1.0
            else:
                # ★ class별 색상 사용
                cid = None
                if idx < len(self._class_ids):
                    cid = self._class_ids[idx]
                if cid is not None and cid in cid_to_color:
                    color = cid_to_color[cid]
                else:
                    # fallback (타입 기반 or 회색)
                    spec_type = self._spectrum_types[idx] if idx < len(self._spectrum_types) else ""
                    if "label" in spec_type.lower():
                        color = (0.2, 0.4, 0.9)
                    elif "splib" in spec_type.lower():
                        color = (0.9, 0.8, 0.2)
                    else:
                        color = (0.5, 0.5, 0.5)
                lw, alpha = 1.0, 0.8
            
            self._ax.plot(x, y, color=color, linewidth=lw, alpha=alpha)
        
        if self._wavelengths is not None and len(self._wavelengths) > 0:
            self._ax.set_xlabel('파장 (nm)')
        else:
            self._ax.set_xlabel('밴드 인덱스')
        self._ax.set_ylabel('반사율')
        self._ax.grid(True, alpha=0.3, linestyle='--')
        self._figure.tight_layout()
        self.draw()



class SearchLabelingDatabaseDialog(QtWidgets.QDialog):
    """
    라벨링 데이터베이스 탐색 다이얼로그
    """
    
    def __init__(self, parent=None, ui_dir: Optional[Path] = None):
        super().__init__(parent)
        self.setWindowTitle("라벨링 데이터베이스 탐색")
        
        app_dir = Path(getattr(parent, "app_dir", Path(__file__).resolve().parents[3]))
        ui_dir = Path(ui_dir) if ui_dir else (app_dir / "ui")
        self._root = uic.loadUi(str(ui_dir / "search_labeling_database.ui"))
        
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self._root)
        
        # UI 파일에서 contentsMargins 설정
        if hasattr(self._root, 'verticalLayout_Main'):
            self._root.verticalLayout_Main.setContentsMargins(10, 10, 10, 10)
        
        # MainWindow에서 데이터 가져오기
        self._main_window = parent
        self._wavelength = None
        
        # MainWindow의 cfg에서 wavelength 가져오기 (wavelength_list 우선)
        if hasattr(parent, 'cfg') and isinstance(parent.cfg, dict):
            cfg = parent.cfg
            wl_raw = cfg.get("wavelength_list") or cfg.get("wavelength")
            if wl_raw is not None:
                try:
                    if isinstance(wl_raw, (list, tuple, np.ndarray)):
                        # 이미 리스트/배열 형태
                        self._wavelength = np.asarray(wl_raw, dtype=np.float32)
                    elif isinstance(wl_raw, str):
                        # 문자열 파싱 (예: "400,500,600" 또는 "[400,500,600]")
                        import json
                        try:
                            # JSON 배열 형태 시도
                            parsed = json.loads(wl_raw)
                            if isinstance(parsed, list):
                                self._wavelength = np.asarray(parsed, dtype=np.float32)
                        except (json.JSONDecodeError, ValueError):
                            # 쉼표로 구분된 문자열 파싱
                            parts = [p.strip() for p in wl_raw.split(',')]
                            self._wavelength = np.asarray([float(p) for p in parts], dtype=np.float32)
                    else:
                        self._wavelength = np.asarray(wl_raw, dtype=np.float32)
                except (ValueError, TypeError) as e:
                    logging.warning(f"[SearchLabelingDatabase] wavelength 파싱 실패: {e}")
                    self._wavelength = None
        elif hasattr(parent, 'wavelength') and parent.wavelength is not None:
            # 폴백: parent.wavelength 직접 사용
            self._wavelength = np.asarray(parent.wavelength, dtype=np.float32)
        
        # 데이터 저장소
        self._spectrum_data: Dict[int, List[Dict]] = {}  # cid -> [{"spectrum": ..., "from": ..., "meta": ...}]
        self._current_class_id: Optional[int] = None
        
        # 테이블 전체 행을 순서대로 들고 있을 리스트
        self._table_rows: List[Dict] = []

        # Material 정보 캐시 (API 호출 결과 저장)
        self._material_cache: Dict[int, Dict[str, str]] = {}  # {cid: {"name": ..., "description": ...}}
        self._personal_class_meta: Optional[Dict[int, Dict[str, str]]] = None
        
        # 스펙트럼 위젯 설정 (QGraphicsView를 matplotlib 위젯으로 교체)
        if MATPLOTLIB_AVAILABLE:
            graphics_view = self._root.graphicsView_Plot
            # QGraphicsView를 레이아웃에서 제거
            plot_container = self._root.widget_PlotContainer
            if plot_container:
                layout = plot_container.layout()
                if layout:
                    # 기존 QGraphicsView 제거
                    layout.removeWidget(graphics_view)
                    graphics_view.setParent(None)
                    graphics_view.deleteLater()
                    
                    # matplotlib 위젯 추가
                    spectrum_widget = SpectrumWidget(parent=plot_container)
                    layout.addWidget(spectrum_widget)
                    self._spectrum_widget = spectrum_widget
                else:
                    self._spectrum_widget = None
            else:
                self._spectrum_widget = None
        else:
            self._spectrum_widget = None
        
        # 테이블 초기화
        self._init_table()
        
        # 콤보박스 초기화 및 채우기
        self._init_combobox()
        
        # 버튼 연결
        self._root.pushButton_OK.clicked.connect(self.accept)
        self._root.comboBox_Class.currentIndexChanged.connect(self._on_class_selected)
        self._root.tableWidget_Spectra.itemSelectionChanged.connect(self._on_table_selection_changed)
        self._root.pushButton_Add.clicked.connect(self._on_add_clicked)
        self._root.pushButton_Reset.clicked.connect(self._on_reset_clicked)
        
        # 테이블 선택 하이라이트 스타일 설정 (연두색)
        self._setup_table_style()

    def _init_table(self):
        """스펙트럼 테이블 초기화"""
        table = self._root.tableWidget_Spectra
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["#", "Class", "From", "Material Name"])
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.setColumnWidth(0, 50)
        table.setColumnWidth(1, 80)
        table.setColumnWidth(2, 120)
        table.horizontalHeader().setStretchLastSection(True)
        # 테이블 선택 스타일 설정 (연두색 하이라이트)
        table.setStyleSheet("""
            QTableWidget {
                gridline-color: rgb(200, 200, 200);
                background-color: rgb(255, 255, 255);
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: rgb(144, 238, 144);  /* 연두색 하이라이트 */
                color: rgb(0, 0, 0);
            }
            QHeaderView::section {
                background-color: rgb(240, 240, 240);
                padding: 6px;
                border: 1px solid rgb(200, 200, 200);
                font-weight: bold;
            }
        """)
    
    def _setup_table_style(self):
        """테이블 선택 스타일 설정 (추가 스타일링 필요시)"""
        pass
    
    def _fetch_material_info(self, cid: int) -> Dict[str, str]:
        """
        API를 통해 Material 정보 가져오기 (캐시 사용)
        
        Args:
            cid: Material 코드 (mtrl_cd)
            
        Returns:
            {"name": str, "description": str} 또는 {"name": f"Class {cid}", "description": ""}
        """
        # 캐시에 있으면 반환
        if cid in self._material_cache:
            return self._material_cache[cid]
        
        # 기본값
        result = {"name": f"Class {cid}", "description": ""}

        user_type = getattr(self._main_window, "user_type", "server") if self._main_window else "server"
        if user_type == "personal":
            meta = self._get_personal_class_meta().get(int(cid))
            if meta:
                result = {
                    "name": meta.get("name", result["name"]),
                    "description": meta.get("description", result["description"]),
                }
                self._material_cache[cid] = result
                return result
        
        # API 호출
        if not API_AVAILABLE:
            self._material_cache[cid] = result
            return result
        
        try:
            api_base = os.getenv('material_filtering_url')
            if not api_base:
                logging.warning("[SearchLabelingDatabase] material_filtering_url 환경 변수가 설정되지 않았습니다.")
                self._material_cache[cid] = result
                return result
            
            # API 호출
            api_result = search_material_filtering_list(base_url=api_base, mtrl_ids=[cid])
            
            if api_result and len(api_result) > 0:
                material_data = api_result[0]
                result = {
                    "name": material_data.get("mtrl_nm", f"Class {cid}"),
                    "description": material_data.get("desc", "")
                }
            
            # 캐시에 저장
            self._material_cache[cid] = result
            
        except Exception as e:
            logging.exception(f"[SearchLabelingDatabase] Material 정보 가져오기 실패 (cid={cid}): {e}")
            # 실패해도 기본값 반환
            self._material_cache[cid] = result
        
        return result
    
    def _fetch_material_info_batch(self, cids: List[int]) -> Dict[int, Dict[str, str]]:
        """
        여러 Material 정보를 한 번에 가져오기 (배치 API 호출)
        
        Args:
            cids: Material 코드 리스트
            
        Returns:
            {cid: {"name": str, "description": str}}
        """
        if not cids:
            return {}
        
        result = {}

        user_type = getattr(self._main_window, "user_type", "server") if self._main_window else "server"
        if user_type == "personal":
            meta = self._get_personal_class_meta()
            for cid in cids:
                info = meta.get(int(cid), {})
                name = info.get("name", f"Class {cid}")
                desc = info.get("description", "")
                result[cid] = {"name": name, "description": desc}
                self._material_cache[cid] = result[cid]
            return result
        
        # MainWindow의 캐시를 먼저 확인 (server 모드)
        if self._main_window:
            # 1) _build_id_to_name_for_cids 메서드 사용 (가장 우선)
            if hasattr(self._main_window, '_build_id_to_name_for_cids'):
                try:
                    id_to_name = self._main_window._build_id_to_name_for_cids(cids)
                    found_count = 0
                    for cid in cids:
                        name = id_to_name.get(cid)
                        if name and name != str(cid):  # 기본값이 아닌 경우만 사용
                            result[cid] = {"name": name, "description": ""}
                            self._material_cache[cid] = result[cid]
                            found_count += 1
                    if found_count > 0:
                        logging.debug(f"[SearchLabelingDatabase] MainWindow _build_id_to_name_for_cids에서 {found_count}개 클래스명 찾음")
                except Exception as e:
                    logging.exception(f"[SearchLabelingDatabase] MainWindow _build_id_to_name_for_cids 실패: {e}")
            
            # 2) _mtrl_meta 직접 확인
            if hasattr(self._main_window, '_mtrl_meta'):
                try:
                    mtrl_meta = getattr(self._main_window, '_mtrl_meta', {}) or {}
                    if isinstance(mtrl_meta, dict):
                        found_count = 0
                        for cid in cids:
                            if cid in result:  # 이미 설정된 경우 스킵
                                continue
                            meta = mtrl_meta.get(cid) or mtrl_meta.get(str(cid)) or {}
                            if isinstance(meta, dict):
                                name = meta.get("mtrl_nm") or meta.get("name")
                                if name:
                                    desc = meta.get("desc") or meta.get("description") or ""
                                    result[cid] = {"name": str(name), "description": str(desc)}
                                    self._material_cache[cid] = result[cid]
                                    found_count += 1
                        if found_count > 0:
                            logging.debug(f"[SearchLabelingDatabase] MainWindow _mtrl_meta에서 {found_count}개 클래스명 찾음")
                except Exception as e:
                    logging.exception(f"[SearchLabelingDatabase] MainWindow _mtrl_meta 확인 실패: {e}")
        
        # 내부 캐시에 있는 것 추가
        for cid in cids:
            if cid not in result and cid in self._material_cache:
                result[cid] = self._material_cache[cid]
        
        # 캐시에 없는 cid만 필터링
        missing_cids = [cid for cid in cids if cid not in result]
        
        # 기본값 설정 (아직 설정되지 않은 것만)
        for cid in missing_cids:
            result[cid] = {"name": f"Class {cid}", "description": ""}
        
        # API 호출 (캐시에 없는 것만)
        if missing_cids:
            if API_AVAILABLE:
                try:
                    api_base = os.getenv('material_filtering_url')
                    if api_base:
                        logging.debug(f"[SearchLabelingDatabase] API 호출 시도: {len(missing_cids)}개 클래스 (CIDs: {missing_cids})")
                        api_result = search_material_filtering_list(base_url=api_base, mtrl_ids=missing_cids)
                        
                        # API 결과를 딕셔너리로 변환
                        api_dict = {}
                        for item in api_result:
                            mtrl_cd = item.get("mtrl_cd")
                            if mtrl_cd is not None:
                                api_dict[int(mtrl_cd)] = {
                                    "name": item.get("mtrl_nm", f"Class {mtrl_cd}"),
                                    "description": item.get("desc", "")
                                }
                        
                        # 결과 업데이트 및 캐시 저장
                        found_count = 0
                        for cid in missing_cids:
                            if cid in api_dict:
                                result[cid] = api_dict[cid]
                                self._material_cache[cid] = api_dict[cid]
                                found_count += 1
                            else:
                                # API에서 못 찾은 경우 기본값 유지 및 캐시 저장
                                self._material_cache[cid] = result[cid]
                        
                        if found_count > 0:
                            logging.debug(f"[SearchLabelingDatabase] API에서 {found_count}개 클래스명 찾음")
                    else:
                        logging.warning(f"[SearchLabelingDatabase] material_filtering_url 환경 변수가 설정되지 않음")
                except Exception as e:
                    logging.exception(f"[SearchLabelingDatabase] 배치 Material 정보 가져오기 실패: {e}")
                    # 실패한 경우 기본값 캐시에 저장
                    for cid in missing_cids:
                        if cid not in self._material_cache:
                            self._material_cache[cid] = result[cid]
            else:
                logging.debug(f"[SearchLabelingDatabase] API 사용 불가능, 기본값 사용: {len(missing_cids)}개 클래스")
        
        return result

    def _get_personal_class_meta(self) -> Dict[int, Dict[str, str]]:
        """personal 모드에서 .info 기반 클래스 메타를 캐시"""
        if self._personal_class_meta is not None:
            return self._personal_class_meta
        self._personal_class_meta = {}
        main_window = self._main_window
        if not main_window:
            return self._personal_class_meta

        try:
            primary = None
            cfg = getattr(main_window, "cfg", {}) if hasattr(main_window, "cfg") else {}
            if hasattr(main_window, "_cache_primary_path"):
                primary = main_window._cache_primary_path(cfg)
            if not primary and hasattr(main_window, "_extract_src_path"):
                primary = main_window._extract_src_path(cfg)

            if primary:
                classes = load_classes_from_info(primary) or []
                for rec in classes:
                    if len(rec) == 3:
                        cid, name, desc = rec
                    else:
                        cid, name = rec
                        desc = ""
                    self._personal_class_meta[int(cid)] = {
                        "name": str(name),
                        "description": str(desc) if desc not in (None, "") else "",
                    }
        except Exception:
            logging.exception("[SearchLabelingDatabase] personal class meta load failed")
        return self._personal_class_meta
    
    def _init_combobox(self):
        """콤보박스 초기화: personal은 .info, server는 기존 방식(+타입 방어)."""
        combo = self._root.comboBox_Class
        combo.clear()
        
        if not self._main_window:
            return

        user_type = getattr(self._main_window, "user_type", "server")

        # ------------------------
        # 1) personal 모드: .info 기반
        # ------------------------
        if user_type == "personal":
            options = getattr(self._main_window, "_resolve_current_class_options", None)
            class_options = options() if callable(options) else []
            # _resolve_current_class_options(personal) → [(cid, name, desc), ...] 형태
            for cid_opt, name_opt, _ in class_options:
                combo.addItem(f"{name_opt}", int(cid_opt))
            if combo.count() > 0:
                combo.setCurrentIndex(0)
            return

        # ------------------------
        # 2) server 모드: splib*/label*/lib 에서 cid 수집
        # ------------------------
        class_ids = set()

        def _extend_class_ids(src: Any):
            """splib_raw/splib_cr/label_raw/label_cr/lib가 dict 또는 list일 때 안전하게 cid를 수집."""
            nonlocal class_ids
            if src is None:
                return

            # dict 형태: {cid: ...}
            if isinstance(src, dict):
                for k in src.keys():
                    try:
                        class_ids.add(int(k))
                    except Exception:
                        continue
                return

            # list/tuple/ndarray: [ {mtrl_cd/ cid / class_id: ...}, ... ] 형식일 수 있음
            if isinstance(src, (list, tuple, np.ndarray)):
                for rec in src:
                    if not isinstance(rec, dict):
                        continue
                    cid = rec.get("mtrl_cd") or rec.get("cid") or rec.get("class_id")
                    try:
                        class_ids.add(int(cid))
                    except Exception:
                        continue

        # 각 소스에서 cid 수집 (타입에 상관 없이 안전하게)
        _extend_class_ids(getattr(self._main_window, "splib_raw", None))
        _extend_class_ids(getattr(self._main_window, "splib_cr", None))
        _extend_class_ids(getattr(self._main_window, "label_raw", None))
        _extend_class_ids(getattr(self._main_window, "label_cr", None))
        _extend_class_ids(getattr(self._main_window, "lib", None))

        sorted_ids = sorted(class_ids)

        # Material 정보 한 번에 가져오기
        material_info_dict = self._fetch_material_info_batch(sorted_ids)

        for cid in sorted_ids:
            material_info = material_info_dict.get(cid, {})
            print(material_info)
            name = material_info.get("name", f"Class {cid}")
            combo.addItem(name, cid)

        if combo.count() > 0:
            combo.setCurrentIndex(0)

    
    def _on_class_selected(self, index: int):
        """콤보박스에서 클래스 선택 시: 현재 클래스 ID만 저장"""
        if index < 0:
            self._current_class_id = None
            return

        combo = self._root.comboBox_Class
        class_id = combo.itemData(index)
        if class_id is None:
            self._current_class_id = None
            return

        self._current_class_id = int(class_id)
    
    def _populate_table(self):
        """선택된 클래스의 스펙트럼 데이터로 테이블 채우기"""
        table = self._root.tableWidget_Spectra
        table.setRowCount(0)
        
        if self._current_class_id is None:
            return
        
        cid = self._current_class_id
        
        # Material 정보 가져오기 (API 호출)
        material_info = self._fetch_material_info(cid)
        material_name = material_info.get("name", f"Class {cid}")
        material_description = material_info.get("description", "")
        
        spectra_list = []
        
        # 스펙트럼 라이브러리에서 데이터 수집
        if hasattr(self._main_window, 'splib_raw') and self._main_window.splib_raw:
            if cid in self._main_window.splib_raw:
                arr = self._main_window.splib_raw[cid]  # (N, C)
                if arr.ndim == 2:
                    for i in range(arr.shape[0]):
                        spectra_list.append({
                            "spectrum": arr[i],
                            "from": "Spectrum Library (Raw)",
                            "material_name": material_name,
                            "description": material_description,
                            "index": i,
                            "type": "splib_raw",
                            "cid": cid,
                        })
        
        if hasattr(self._main_window, 'splib_cr') and self._main_window.splib_cr:
            if cid in self._main_window.splib_cr:
                arr = self._main_window.splib_cr[cid]  # (N, C)
                if arr.ndim == 2:
                    for i in range(arr.shape[0]):
                        spectra_list.append({
                            "spectrum": arr[i],
                            "from": "Spectrum Library (CR)",
                            "material_name": material_name,
                            "description": material_description,
                            "index": i,
                            "type": "splib_cr",
                            "cid": cid,
                        })
        
        # 라벨링 데이터에서 데이터 수집
        # 좌표 정보는 MainWindow의 label_coords에서 가져오기
        label_coords = getattr(self._main_window, 'label_coords', {})
        
        if hasattr(self._main_window, 'label_raw') and self._main_window.label_raw:
            if cid in self._main_window.label_raw:
                arr = self._main_window.label_raw[cid]  # (N, C)
                if arr.ndim == 2:
                    # 해당 클래스의 좌표 정보 가져오기
                    coords_list = label_coords.get(cid, [])
                    for i in range(arr.shape[0]):
                        # 좌표 정보 찾기 (인덱스 매칭)
                        img_x, img_y, img_cd = None, None, None
                        if i < len(coords_list):
                            coord_data = coords_list[i]
                            if len(coord_data) >= 2:
                                img_x, img_y = coord_data[0], coord_data[1]
                            if len(coord_data) >= 3:
                                img_cd = coord_data[2]
                        elif coords_list:
                            # 인덱스가 다를 경우 첫 번째 좌표 사용 (임시)
                            coord_data = coords_list[0]
                            if len(coord_data) >= 2:
                                img_x, img_y = coord_data[0], coord_data[1]
                            if len(coord_data) >= 3:
                                img_cd = coord_data[2]
                        
                        spectra_list.append({
                            "spectrum": arr[i],
                            "from": "Labeling",
                            "material_name": material_name,
                            "description": material_description,
                            "index": i,
                            "type": "label_raw",
                            "label_index": i,
                            "img_x": img_x,
                            "img_y": img_y,
                            "img_cd": img_cd,
                            "cid": cid,
                        })
        
        if hasattr(self._main_window, 'label_cr') and self._main_window.label_cr:
            if cid in self._main_window.label_cr:
                arr = self._main_window.label_cr[cid]  # (N, C)
                if arr.ndim == 2:
                    # 해당 클래스의 좌표 정보 가져오기
                    coords_list = label_coords.get(cid, [])
                    for i in range(arr.shape[0]):
                        # 좌표 정보 찾기 (인덱스 매칭)
                        img_x, img_y, img_cd = None, None, None
                        if i < len(coords_list):
                            coord_data = coords_list[i]
                            if len(coord_data) >= 2:
                                img_x, img_y = coord_data[0], coord_data[1]
                            if len(coord_data) >= 3:
                                img_cd = coord_data[2]
                        elif coords_list:
                            # 인덱스가 다를 경우 첫 번째 좌표 사용 (임시)
                            coord_data = coords_list[0]
                            if len(coord_data) >= 2:
                                img_x, img_y = coord_data[0], coord_data[1]
                            if len(coord_data) >= 3:
                                img_cd = coord_data[2]
                        
                        spectra_list.append({
                            "spectrum": arr[i],
                            "from": "Labeling",
                            "material_name": material_name,
                            "description": material_description,
                            "index": i,
                            "type": "label_cr",
                            "label_index": i,
                            "img_x": img_x,
                            "img_y": img_y,
                            "img_cd": img_cd,
                            "cid": cid,
                        })
        
        # 테이블에 데이터 추가
        self._spectrum_data[cid] = spectra_list
        table.setRowCount(len(spectra_list))
        
        for row, spec_data in enumerate(spectra_list):
            table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            table.setItem(row, 1, QTableWidgetItem(str(cid)))
            table.setItem(row, 2, QTableWidgetItem(spec_data["from"]))
            table.setItem(row, 3, QTableWidgetItem(spec_data["material_name"]))
        
        # 스펙트럼 그래프 업데이트
        if self._spectrum_widget:
            spectra = [s["spectrum"] for s in spectra_list]
            spectrum_types = [s.get("type", "") for s in spectra_list]
            self._spectrum_widget.set_spectra(spectra, self._wavelength, spectrum_types)
    
    def _on_table_selection_changed(self):
        """테이블 선택 변경 시 호출"""
        table = self._root.tableWidget_Spectra
        selected_rows = table.selectedIndexes()

        if not selected_rows:
            return

        row = selected_rows[0].row()
        if row < 0 or row >= len(self._table_rows):
            return

        spec_data = self._table_rows[row]

        # 스펙트럼 그래프에서 선택된 항목 강조
        if self._spectrum_widget:
            self._spectrum_widget.set_selected_index(row)

        # 라벨링 데이터면 패치 이미지 표시
        if spec_data.get("type") in ("label_raw", "label_cr"):
            self._show_label_patch(spec_data, row)
        else:
            self._root.label_Image.clear()
            self._root.label_Image.setText("Image Placeholder")

        # 상세 정보 업데이트
        self._update_details(spec_data)

    def _show_label_patch(self, spec_data: Dict, row: int):
        """라벨링 데이터의 패치 이미지 표시 (중간에 빨간 점)"""
        try:
            # MainWindow에서 원본 이미지 데이터 가져오기
            if not hasattr(self._main_window, 'rgb_image') or self._main_window.rgb_image is None:
                self._root.label_Image.clear()
                self._root.label_Image.setText("RGB 이미지 없음")
                return
            
            rgb_image = self._main_window.rgb_image  # (H, W, 3)
            H, W = rgb_image.shape[:2]
            
            # 라벨링 데이터의 좌표 정보 가져오기
            patch_size = 30  # 패치 크기 (픽셀)
            
            # 좌표 정보가 있으면 사용, 없으면 중앙 좌표 사용
            img_x = spec_data.get('img_x')
            img_y = spec_data.get('img_y')
            
            if img_x is not None and img_y is not None:
                center_x = int(img_x)
                center_y = int(img_y)
            else:
                # 좌표 정보가 없으면 중앙 좌표 사용
                center_x = W // 2
                center_y = H // 2
            
            # 패치 영역 계산 (이미지 경계 체크)
            x0 = max(0, center_x - patch_size // 2)
            x1 = min(W, center_x + patch_size // 2 + 1)
            y0 = max(0, center_y - patch_size // 2)
            y1 = min(H, center_y + patch_size // 2 + 1)
            
            # 패치 추출
            patch = rgb_image[y0:y1, x0:x1].copy()
            
            # 패치가 비어있으면 안내 메시지
            if patch.size == 0:
                self._root.label_Image.clear()
                self._root.label_Image.setText("패치 추출 실패")
                return
            
            # QImage로 변환
            h_patch, w_patch = patch.shape[:2]
            qimg = QImage(w_patch, h_patch, QImage.Format_RGB888)
            bytes_per_line = qimg.bytesPerLine()
            expected_bpl = w_patch * 3
            
            ptr = qimg.bits()
            ptr.setsize(h_patch * bytes_per_line)
            
            # 패치 데이터 복사
            if bytes_per_line == expected_bpl:
                if patch.flags["C_CONTIGUOUS"]:
                    ptr[:] = patch.ravel().tobytes()[:h_patch * bytes_per_line]
                else:
                    patch_contiguous = np.ascontiguousarray(patch)
                    ptr[:] = patch_contiguous.ravel().tobytes()[:h_patch * bytes_per_line]
            else:
                # stride가 다를 경우 행 단위로 복사
                patch_contiguous = np.ascontiguousarray(patch)
                for y in range(h_patch):
                    row_data = patch_contiguous[y, :, :].tobytes()
                    offset = y * bytes_per_line
                    ptr[offset:offset + expected_bpl] = row_data
            
            # QPainter로 빨간 점 그리기
            pixmap = QPixmap.fromImage(qimg)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # ★ 패치 내부 기준에서 (x, y) 위치 계산
            #   원본 좌표 (center_x, center_y)가 패치 안에서는 (center_x - x0, center_y - y0)
            center_patch_x = center_x - x0
            center_patch_y = center_y - y0

            # 경계 체크 (혹시라도 범위 벗어나면 클램핑)
            center_patch_x = max(0, min(w_patch - 1, center_patch_x))
            center_patch_y = max(0, min(h_patch - 1, center_patch_y))
            
            # 빨간 점 그리기 (실제 (x, y)에 해당하는 위치)
            pen = QPen(QColor(255, 0, 0), 3)
            painter.setPen(pen)
            brush = QBrush(QColor(255, 0, 0))
            painter.setBrush(brush)
            painter.drawEllipse(
                int(center_patch_x) - 3,
                int(center_patch_y) - 3,
                6,
                6,
            )

            painter.end()
            
            # 라벨에 표시 (150x150으로 스케일)
            label = self._root.label_Image
            scaled_pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
            label.setText("")
            
        except Exception as e:
            logging.exception(f"[SearchLabelingDatabase] 패치 표시 실패: {e}")
            self._root.label_Image.clear()
            self._root.label_Image.setText("이미지 로드 실패")
    
    def _update_details(self, spec_data: Dict):
        """상세 정보 텍스트 에디터 업데이트"""
        details = self._root.textEdit_Details
        pk = spec_data.get('index', 'N/A')
        name = spec_data.get('material_name', 'N/A')
        from_source = spec_data.get('from', 'N/A')
        desc = spec_data.get('description', '')
        img_cd = spec_data.get('img_cd')
        img_x = spec_data.get('img_x')
        img_y = spec_data.get('img_y')

        user_type = getattr(self._main_window, "user_type", "server") if self._main_window else "server"
        cid = spec_data.get("cid")
        if user_type == "personal" and cid is not None:
            meta = self._get_personal_class_meta().get(int(cid))
            if meta:
                name = meta.get("name", name) or name
                desc = meta.get("description", desc) or desc
        
        text = f"PK : {pk}\n"
        text += f"Name : {name}\n"
        if img_cd is not None:
            text += f"Image Code : {img_cd}\n"
        if img_x is not None and img_y is not None:
            text += f"Coord : ({img_x}, {img_y})\n"
        text += f"Description : {desc}" if desc else "Description : "
        details.setPlainText(text)
    
    def _on_add_clicked(self):
        """추가 버튼 클릭 시: 현재 선택된 클래스의 스펙트럼을 테이블에 적재"""
        combo = self._root.comboBox_Class
        index = combo.currentIndex()
        if index < 0:
            return

        cid = combo.itemData(index)
        if cid is None:
            return

        cid = int(cid)

        # 해당 클래스의 스펙트럼 수집
        spectra_list = self._collect_spectra_for_class(cid)
        if not spectra_list:
            return

        # 기존 테이블 뒤에 이어서 추가
        start_len = len(self._table_rows)
        self._table_rows.extend(spectra_list)

        # 테이블/그래프 다시 그림
        self._rebuild_table_and_plot()

        # 방금 추가한 첫 번째 행 선택
        table = self._root.tableWidget_Spectra
        if start_len < table.rowCount():
            table.selectRow(start_len)

    
    def _on_reset_clicked(self):
        """초기화 버튼 클릭 시 호출"""
        self._root.comboBox_Class.setCurrentIndex(0)

        # 내부 데이터 초기화
        self._table_rows = []

        table = self._root.tableWidget_Spectra
        table.setRowCount(0)
        table.clearSelection()

        if self._spectrum_widget:
            self._spectrum_widget.set_spectra([])

        self._root.label_Image.clear()
        self._root.label_Image.setText("Image Placeholder")
        self._root.textEdit_Details.setPlainText("PK : \nName : \nDescription : ")


    def _collect_spectra_for_class(self, cid: int) -> List[Dict]:
        """
        주어진 클래스 ID에 대한 스펙트럼 리스트 생성 (테이블/그래프용)
        - splib_raw/splib_cr/label_raw/label_cr가 dict 또는 list/ndarray 모두 올 수 있으므로
          타입을 방어적으로 처리한다.
        """
        # Material 정보 가져오기
        material_info = self._fetch_material_info(cid)
        material_name = material_info.get("name", f"Class {cid}")
        material_description = material_info.get("description", "")

        spectra_list: List[Dict] = []
        label_coords = getattr(self._main_window, 'label_coords', {})

        def _get_array_for_cid(container: Any, cid_val: int, prefer_key: Optional[str] = None) -> Optional[np.ndarray]:
            """
            container가 dict 또는 list[dict] 등일 때, 해당 cid의 (N,C) 배열을 반환.
            prefer_key가 있으면 먼저 그 키(ref/rfl)를 우선.
            """
            if container is None:
                return None

            # dict 형태: {cid: (N,C) or 1D}
            if isinstance(container, dict):
                for k, v in container.items():
                    try:
                        if int(k) != int(cid_val):
                            continue
                    except Exception:
                        continue
                    arr = np.asarray(v)
                    if arr.ndim == 1:
                        arr = arr[None, :]
                    if arr.ndim != 2:
                        return None
                    return arr
                return None

            # list/tuple/ndarray: [ {mtrl_cd/cid/class_id, ref/rfl: ...}, ... ]
            if isinstance(container, (list, tuple, np.ndarray)):
                blocks = []
                for rec in container:
                    if not isinstance(rec, dict):
                        continue
                    c = rec.get("mtrl_cd") or rec.get("cid") or rec.get("class_id")
                    try:
                        if int(c) != int(cid_val):
                            continue
                    except Exception:
                        continue

                    spec = None
                    if prefer_key:
                        spec = rec.get(prefer_key)
                    if spec is None:
                        spec = rec.get("ref") or rec.get("rfl")
                    if spec is None:
                        continue

                    arr = np.asarray(spec)
                    if arr.ndim == 1:
                        arr = arr[None, :]
                    if arr.ndim != 2:
                        continue
                    blocks.append(arr)

                if not blocks:
                    return None
                return np.vstack(blocks)

            return None

        # 1) 스펙트럼 라이브러리 (Raw)
        arr = _get_array_for_cid(getattr(self._main_window, 'splib_raw', None), cid, prefer_key="ref")
        if arr is not None and arr.ndim == 2:
            for i in range(arr.shape[0]):
                spectra_list.append({
                    "cid": cid,
                    "spectrum": arr[i],
                    "from": "Spectrum Library (Raw)",
                    "material_name": material_name,
                    "description": material_description,
                    "index": i,
                    "type": "splib_raw",
                })

        # 2) 스펙트럼 라이브러리 (CR)
        arr = _get_array_for_cid(getattr(self._main_window, 'splib_cr', None), cid, prefer_key="ref")
        if arr is not None and arr.ndim == 2:
            for i in range(arr.shape[0]):
                spectra_list.append({
                    "cid": cid,
                    "spectrum": arr[i],
                    "from": "Spectrum Library (CR)",
                    "material_name": material_name,
                    "description": material_description,
                    "index": i,
                    "type": "splib_cr",
                })

        # 3) 라벨링 데이터 (Raw)
        arr = _get_array_for_cid(getattr(self._main_window, 'label_raw', None), cid, prefer_key="rfl")
        if arr is not None and arr.ndim == 2:
            coords_list = label_coords.get(cid, [])
            for i in range(arr.shape[0]):
                img_x, img_y, img_cd = None, None, None
                if i < len(coords_list):
                    coord_data = coords_list[i]
                elif coords_list:
                    coord_data = coords_list[0]
                else:
                    coord_data = []

                if len(coord_data) >= 2:
                    img_x, img_y = coord_data[0], coord_data[1]
                if len(coord_data) >= 3:
                    img_cd = coord_data[2]

                spectra_list.append({
                    "cid": cid,
                    "spectrum": arr[i],
                    "from": "Labeling",
                    "material_name": material_name,
                    "description": material_description,
                    "index": i,
                    "type": "label_raw",
                    "label_index": i,
                    "img_x": img_x,
                    "img_y": img_y,
                    "img_cd": img_cd,
                })

        # 4) 라벨링 데이터 (CR)
        arr = _get_array_for_cid(getattr(self._main_window, 'label_cr', None), cid, prefer_key="rfl")
        if arr is not None and arr.ndim == 2:
            coords_list = label_coords.get(cid, [])
            for i in range(arr.shape[0]):
                img_x, img_y, img_cd = None, None, None
                if i < len(coords_list):
                    coord_data = coords_list[i]
                elif coords_list:
                    coord_data = coords_list[0]
                else:
                    coord_data = []

                if len(coord_data) >= 2:
                    img_x, img_y = coord_data[0], coord_data[1]
                if len(coord_data) >= 3:
                    img_cd = coord_data[2]

                spectra_list.append({
                    "cid": cid,
                    "spectrum": arr[i],
                    "from": "Labeling",
                    "material_name": material_name,
                    "description": material_description,
                    "index": i,
                    "type": "label_cr",
                    "label_index": i,
                    "img_x": img_x,
                    "img_y": img_y,
                    "img_cd": img_cd,
                })

        return spectra_list


    def _rebuild_table_and_plot(self):
        """self._table_rows 기준으로 테이블과 스펙트럼 그래프를 다시 그림"""
        table = self._root.tableWidget_Spectra
        table.setRowCount(len(self._table_rows))

        for row, spec_data in enumerate(self._table_rows):
            cid = spec_data.get("cid", "")
            from_source = spec_data.get("from", "")
            material_name = spec_data.get("material_name", "")

            table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            table.setItem(row, 1, QTableWidgetItem(str(cid)))
            table.setItem(row, 2, QTableWidgetItem(from_source))
            table.setItem(row, 3, QTableWidgetItem(material_name))

        # 스펙트럼 그래프 업데이트
        if self._spectrum_widget:
            spectra = [s["spectrum"] for s in self._table_rows]
            spectrum_types = [s.get("type", "") for s in self._table_rows]
            class_ids = [s.get("cid") for s in self._table_rows]   # ★ 추가
            self._spectrum_widget.set_spectra(
                spectra,
                self._wavelength,
                spectrum_types,
                class_ids=class_ids,
            )
