# views/main_window.py
from __future__ import annotations
import os
import numpy as np
import logging
import json, hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
from PyQt5 import QtWidgets, QtCore, uic, QtGui
from PyQt5.QtCore import Qt  # ★ 추가: 오버레이 클릭 통과 설정에 사용
from PyQt5.QtGui import QIcon
from itertools import chain
from views.dialogs.image_load import ImageLoadDialog
from views.dialogs.camera_add import CameraAddDialog
from views.docks.layer_dock import LayersDock
from views.docks.unmixing_dock import UnmixingDock
from views.mapview import MapView
from data.db import image_check_and_save, search_material_filtering_list, send_label_add, call_unmixing
from core.vis import make_rgb
from core.resampling_service import resampling, perform_continuum_removal
from core.autoclass import classify_cube, UNKNOWN, MULTIPLE           # ★ 추가
from services.layer_manager import LayerManager           # ★ 추가
from services.paletteService import PaletteService               # ★ 추가
from services.classmap_render import ClassmapRenderer   # ★ 추가
from controllers.roi_controller import ROIController   # ★ 추가
from views.docks.pixel_classification_dock import PixelClassificationDock
from services.pixel_knn import topk_for_pixel  # ★ 상단 import
from controllers.pixel_click import PixelClickController, PixelClickDeps, ClickMode  # ★ 추가
from services.opacity import compute_alpha_from_gmin, build_opacity_overlay_rgba
from services.transparency_service import TransparencyService
from services.region_growing_service import RegionGrowingService
from controllers.class_roi_controller import ClassROIController
from views.dialogs.diffusion_dialog import DiffusionDialog
from views.dialogs.pixel_classification import PixelClassificationDialog
from services.opacity import build_target_only_overlay_rgba, build_target_only_trafficlight_overlay_rgba
from services.resampling_cache import try_load_cache, save_cache, resample_cache_paths, load_classes_from_info
from services.overlay_temp import TempOverlayService
from dataio.sidecar import read_info, write_info
from controllers.analysis_selection_controller import AnalysisSelectionController
from enum import Enum
from views.dialogs.pixel_labeling_dialog import PixelLabelingDialog
from views.dialogs.user_labeling_dialog import UserLabelingDialog
from views.dialogs.classmap_labeling_dialog import ClassmapLabelingDialog
from views.dialogs.analysis_selection_dialog import AnalysisSelectionDialog
from views.dialogs.workspace_dialog import WorkspaceDialog
from core.region_growing import perform_region_growing
from typing import List, Dict, Tuple, Union
from services.spec_library import build_library_from_spec_libs
from dataclasses import dataclass
from PyQt5 import QtCore
from views.dialogs.viewer_band_dialog import ViewerBandDialog
from views.dialogs.unmixing_dialog import UnmixingDialog
# ★ LABEL STORE / LABEL CODE 추가
from services.label_store import LabelStore
from models.labels import LabelRow
from services.label_code import make_label_code
import pandas as pd  # 패치 추출 헬퍼에서 NaN 판별용

ANALYSIS_LAYER_NAME = "analysis.mask"  # 한 장만 쓴다
RGB_LAYER_NAME = "image.rgb"  # RGB 베이스 레이어 내부 식별자
RGB_LAYER_DISPLAY_NAME = "image.rgb"  # Layers Dock에 표시할 이름 (원하는 이름으로 변경 가능)

def _fallback_color_for_cid(cid: int) -> QtGui.QColor:
    """CID 기반 HSV 규칙색(지도와 100% 일치 보장은 안 됨 → 최후 수단)."""
    phi = (1 + 5 ** 0.5) / 2  # golden ratio 약 1.618
    h = int((cid * 137.508) % 360)  # 137.508 ~= golden angle (도 단위)
    s = 210 + int((cid * 13) % 40)  # 210~249 사이 순환
    v = 210 + int((cid * 7) % 45)   # 210~254 사이 순환
    s = max(150, min(s, 255))
    v = max(180, min(v, 255))
    return QtGui.QColor.fromHsv(h, s, v, 255)


class AnalysisMode(str, Enum):
    NONE="none"; PIXEL="pixel"; RECT="rect"

@dataclass
class AnalysisState:
    mode: AnalysisMode = AnalysisMode.NONE     # 'none' | 'pixel' | 'rect'
    op: str = "union"                          # 'union' | 'subtract'
    mask: Optional[np.ndarray] = None          # (H,W) bool, SSOT
    rect: Optional[QtCore.QRect] = None        # 최근 bbox(유도)

class ROIInputOwner(str, Enum):
    WORK = "work"
    CLASS = "class"
    DIFFUSION = "diffusion"
    NONE = "none"

class MainWindow(QtWidgets.QMainWindow):    
    """
    주어진 ui/main_window.ui 의 액션(objectName):
      - actionCamera_Add
      - actionImage_Load
      - actionSave
      - actionExit
    각각에 대응하는 슬롯 메서드:
      - on_camera_add
      - on_image_load
      - on_save
      - on_exit
    """
    def __init__(self, app_dir: Optional[Path] = None):
        super().__init__()
        self.app_dir = Path(app_dir) if app_dir else Path(__file__).resolve().parents[1]
        self.ui_path = self.app_dir / "ui" / "main_window.ui"

        # 상태 변수(예: 최근 경로)
        self._last_open_dir: Path = self.app_dir
        
        # UI 로드
        self._load_ui()
        self._wire_actions()

        # 1) LayersDock → LayerManager 순
        self._init_layers_dock()
        self.layer_manager = LayerManager(
            map_add_layer   = lambda n,arr,op,uniq: self._map_view.add_temporal_layer(n,arr,op,uniq),
            map_set_visible = self._map_view.set_layer_visible,
            map_remove_layer= self._map_view.remove_temporal_layer,
            map_reorder     = self._map_view.reorder_layers,
            dock_add_layer  = self.layersDock.add_layer,
            dock_has_layer  = getattr(self.layersDock, "has_layer", None),
            dock_remove_layer = getattr(self.layersDock, "remove_layer", None),
            map_remove_rgb  = self._map_view.remove_rgb_image,  # RGB 이미지 제거 콜백
        )

        # 2) ★ SSOT 팔레트 1회 생성 (아래 새 함수)
        self._init_single_palette()

        # 3) ★ 서비스 생성·주입 + 렌더러에 팔레트 주입
        ps = PaletteService()
        cr = ClassmapRenderer(strict=True)
        self.transparency_service = TransparencyService()  # ★ 투명도 서비스 추가
        # 수정 후
        self.region_growing_service = RegionGrowingService()  # ★ Region Growing 서비스 추가
        # MapView가 만들어진 뒤( _load_ui() 이후 ) 바로 참조 주입
        # 예: __init__ 맨 아래 또는 _load_ui() 호출 직후
        try:
            self.region_growing_service.set_mapview_refs(self._map_view)
        except Exception:
            logging.exception("[RegionGrowing] set_mapview_refs failed")
        self.class_roi_controller = None  # ★ 클래스 ROI 컨트롤러 (지연 초기화)
        self.layer_manager.set_services(ps, cr)

        # ★ SSOT 팔레트 → 모든 컴포넌트에 통합 설정
        self._setup_palette_services(cr)

        # 작업 영역(Work Area) 마스크
        self._work_mask: Optional[np.ndarray] = None
        # (옵션) 클래스별 ROI 마스크를 따로 관리하고 싶을 때 사용
        self._class_roi_masks: Dict[int, np.ndarray] = {}

        # ROIController 생성 후에만 ROIController의 이벤트를 듣는다
        if self._ensure_roi_controller():
            # 작업 영역 ROI가 추가되면 여기서만 레이어 반영/상태 갱신
            try:
                self.roi.roiAdded.disconnect()
            except Exception:
                pass
            self.roi.roiAdded.connect(self._on_work_roi_added)

        # 상태바 메시지
        self.statusBar().showMessage("Ready")
        
        # ★ 최근 파일 저장소(QSettings)
        self._settings = QtCore.QSettings("third_pixel_tool", "hsi_app")
        # ★ Recent Files 메뉴 구성
        self._init_recent_menu()
        self._refresh_recent_menu()
        
        self.clsDock = None
        self.unmixingDock = None
        ## ROI 지정 구분
        self._active_roi_owner = ROIInputOwner.NONE
        self._label_local_rows = []   # 세션 메모리 캐시 (리스트<dict>)

        # ★ PixelClickController 의존성 주입
        deps = PixelClickDeps(
            get_cube        = lambda: self.cfg["data"],
            get_lib         = lambda: self.lib,
            get_metric_name = lambda: self.clsDock.get_params().get("metric", "SAD"),
            get_image_code  = lambda: getattr(self, "image_cd", None),

            # ★ 컨트롤러는 좌표만 통지 → 메인이 캐시 기반 다이얼로그 표시
            present_topk_for_pixel = lambda y, x: QtCore.QTimer.singleShot(0, lambda: self._show_pixel_topk(y, x)),

            analysis_callback = lambda y, x: self._handle_analysis_pixel(y, x),

            # (옵션) 메타 조회는 컨트롤러의 on-demand 계산용으로 유지 가능
            # get_meta_for_cid = lambda cid: (self._mtrl_meta or {}).get(cid, {}),

            label_callback  = lambda y,x,cid: self._handle_label_pick(y,x,cid),
            seed_callback   = lambda y,x,cid: self._handle_seed_pick(y,x,cid),
        )

        # 수정 후 - PixelClickController 초기화 (clsDock 초기화 후에 실행)
        try:
            self.pixelClick = PixelClickController(self._map_view, deps, cache_cap = 10)
            self._safe_set_click_mode(ClickMode.NONE)   # ← 기본은 일반 포인터(Arrow)
        except Exception as e:
            logging.exception("[PixelClick] 초기화 실패: %s", e)
            # 초기화 실패 시에도 앱이 계속 실행되도록 함
        
        # ===== 여기서부터 추가 =====
        # 분석 임시 오버레이 서비스 핸들
        self._analysis_overlay = TempOverlayService(self._map_view)
        self.analysisCtrl = None  # 지연 생성
        self._analysis_op:str = "union"
        self._analysis_mask:Optional[np.ndarray] = None
        self._analysis_rect:Optional[QtCore.QRect] = None
        self._map_view.inspectPixelPicked.connect(self._on_map_inspect_click)

        # ROI 컨트롤러가 준비되면 분석 컨트롤러 생성
        if self._ensure_roi_controller():
            def _set_click(mode_str: str):
                try:
                    mode = getattr(ClickMode, mode_str)
                    self._safe_set_click_mode(mode)
                except Exception:
                    pass

            self.analysisCtrl = AnalysisSelectionController(
                map_view=self._map_view,
                roi_controller=self.roi,
                overlay_service=self._analysis_overlay,
                set_click_mode=_set_click,
                get_op=self._get_analysis_op,          # ★ 추가: 최신 op 즉시 조회자
                parent=self
            )
            self.analysisCtrl.rectFinished.connect(self._on_analysis_rect_finished)

            # ★ 분석 마스크 확정되면 플래그를 work로 복귀
            logging.info("[MainWindow] connecting analysisMaskChanged signal")
            result = self.analysisCtrl.analysisMaskChanged.connect(self._on_analysis_mask_changed_from_ctrl)
            logging.info(f"[MainWindow] analysisMaskChanged.connect returned: {result}")
            self.clsDock.analysisStateChanged.connect(self._apply_dock_analysis_state)

            
        # Focus 강조 상태 백업 저장
        self._focus_backup: Optional[Dict[str, Dict[str, bool]]] = None
        self._class_topk_cids: Optional[np.ndarray] = None
        self._class_topk_vals: Optional[np.ndarray] = None
        self._click_mode_from_dock = False
        self._roi_capture_target = "work"   # 'work' | 'analysis'
        self._analysis_mask: Optional[np.ndarray] = None
        self._analysis_rect: Optional[QtCore.QRect] = None
        self._analysis_item = None  # 맵뷰 임시 오버레이 핸들 (변수명 통일)
        self._diff_seed_maps: dict[str, np.ndarray] = {}
        self.current_selection_rect = None  # ★ 추가: 작업 영역 사각형 기본값
        self._topk_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._map_semantics:dict[str, dict] = {}

        self._workspace_temp_overlay = None  # TempOverlayService 핸들 (열릴 때 보장)
        self._workspace_temp_mask: Optional[np.ndarray] = None
        self._workspace_temp_rect: Optional[QtCore.QRect] = None
        self.label_store: Optional[LabelStore] = None
        
        # --- Unmixing 상태 ---
        self.unmixing_endmembers = None      # np.ndarray (K, C) 등
        self.unmixing_abundance_map = None   # np.ndarray (H, W, K)
        self.unmixing_threshold = None       # float (0~1)
        self.unmixing_class_mapping: Dict[int, int] = {}  # endmember index -> class id

    # -----------------------------
    # UI
    # -----------------------------
    def _load_ui(self) -> None:
        uic.loadUi(str(self.ui_path), self)

        central = self.centralWidget()

        # 중앙에 레이아웃 생성(여백 0)
        lay = central.layout()
        if lay is None:
            lay = QtWidgets.QVBoxLayout(central)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)

        # MapView를 레이아웃에 추가(확장 정책)
        from PyQt5.QtWidgets import QSizePolicy
        self._map_view = MapView(self)
        self._map_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._map_view)


    # ROI switch
    def _switch_roi_owner(self, owner: ROIInputOwner):
        """MapView ROI 입력의 소유권을 단 하나의 컨트롤러에만 부여."""
        self._active_roi_owner = owner

        # 세 컨트롤러가 모두 accept_events를 가짐(아래 2번 패치 필요)
        try:
            # 기본값 모두 OFF
            if getattr(self, "roi", None) and hasattr(self.roi, "set_accept_events"):
                self.roi.set_accept_events(False)
            if getattr(self, "class_roi_controller", None) and hasattr(self.class_roi_controller, "set_accept_events"):
                self.class_roi_controller.set_accept_events(False)
            if getattr(self, "diffusion_roi_controller", None) and hasattr(self.diffusion_roi_controller, "set_accept_events"):
                self.diffusion_roi_controller.set_accept_events(False)

            # 해당 오너만 ON
            if owner == ROIInputOwner.WORK and getattr(self, "roi", None):
                self.roi.set_accept_events(True)
            elif owner == ROIInputOwner.CLASS and getattr(self, "class_roi_controller", None):
                self.class_roi_controller.set_accept_events(True)
            elif owner == ROIInputOwner.DIFFUSION and getattr(self, "diffusion_roi_controller", None):
                self.diffusion_roi_controller.set_accept_events(True)
        except Exception:
            logging.exception("[ROI] owner switch failed")


    # -----------------------------
    # Actions wiring
    # -----------------------------
    def _wire_actions(self) -> None:
        # 각 액션이 ui에 존재하는지 확인 후 슬롯에 연결
        self._bind_action("actionCamera_Add", self.on_camera_add)
        self._bind_action("actionImage_Load", self.on_image_load)
        self._bind_action("actionSave", self.on_save)
        self._bind_action("actionExit", self.on_exit)
        
        self._bind_action("action", self.on_classification_dialog_run)  # ★ 분류 맵 생성
        self._bind_action("action_2", self.on_diffusion_dialog_run)  # ★ 유사도 확산 맵 생성
        self._bind_action("action_4", self.on_pixel_labeling_dialog_run)  # ★ 선택 픽셀 라벨링
        self._bind_action("action_5", self.on_classmap_labeling_dialog_run)  # ★ 라벨링 데이터베이스 탐색
        self._bind_action("action_8", self.on_unmixing_dialog_run)  # ★ 분광 혼합 분석 맵 생성

    def _wire_manager_signals(self):
        lm = self.layer_manager
        # MapView 반영
        lm.sig_layer_added.connect(lambda name, payload: self._map_view.add_or_update(name, payload))
        lm.sig_layer_removed.connect(lambda name: self._map_view.remove_temporal_layer(name))
        lm.sig_visibility_changed.connect(lambda name, v: self._map_view.set_layer_visible(lm.map_name(name), v))
        lm.sig_order_changed.connect(lambda top_to_bottom: self._map_view.reorder_layers(lm.order_for_mapview(top_to_bottom)))
        lm.sig_data_updated.connect(lambda name, payload: self._map_view.add_or_update(name, payload))
        # LayersDock 반영
        lm.sig_layer_added.connect(lambda name, _: self.layersDock.add_layer(name=name, checked=True, parent=lm.parent_of(name)))
        lm.sig_layer_removed.connect(lambda name: self.layersDock.remove_layer(name))
        lm.sig_visibility_changed.connect(lambda name, v: self.layersDock.set_checked(name, v))
        lm.sig_order_changed.connect(lambda top_to_bottom: self.layersDock.set_order(top_to_bottom))

    def _bind_action(self, name: str, slot) -> None:
        act = getattr(self, name, None)
        if isinstance(act, QtWidgets.QAction):
            act.triggered.connect(slot)
        else:
            # 액션이 없으면 경고만 띄우고 계속 진행
            QtCore.qWarning(f"[WARN] QAction '{name}' not found in UI")
    def _info(self, msg: str):
        self.statusBar().showMessage(msg, 5000)
        logging.info(msg)
    # -----------------------------
    # Slots (각 버튼/메뉴용 def)
    # -----------------------------
    @QtCore.pyqtSlot()
    def on_camera_add(self) -> None:
        dlg = CameraAddDialog(parent=self, ui_dir=getattr(self, "ui_dir", self.app_dir / "ui"))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            cfg = dlg.get_result()
            # TODO: cfg를 저장/테이블 반영/프로젝트 설정 갱신 등
            # 수정 필요: 카메라 등록 결과를 데이터베이스에 저장하고 UI에 반영하는 로직 구현
            QtWidgets.QMessageBox.information(self, "등록 완료", f"카메라 등록:\n{cfg}")

    @QtCore.pyqtSlot()
    def on_image_load(self) -> None:
        """
        Image Load 다이얼로그 실행 → 결과(dict) 수신
        controllers/ingest_controller가 있으면 거기로 위임, 없으면 메시지로 확인.
        """
        last_dir = getattr(self, "_last_open_dir", self.app_dir)
        dlg = ImageLoadDialog(
            parent=self,
            ui_dir=getattr(self, "ui_dir", self.app_dir / "ui"),
            ui_filename="image_load.ui",
            last_dir=last_dir,
        )

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        cfg = getattr(dlg, "result", None)   # ← 다이얼로그가 저장해둔 결과
        if not cfg:
            QtWidgets.QMessageBox.warning(self, "Load 실패", "로드 결과가 비어 있습니다.")
            return
        
        # 통합된 HSI 로드 파이프라인 사용
        self._apply_loaded_hsi_cfg(cfg)
        
        # Region Growing 서비스 초기화
        if "data" in cfg:
            self.setup_region_growing(cfg["data"])
        
    @QtCore.pyqtSlot()
    def on_save(self) -> None:
        """
        Save: 결과/클래스맵/프로젝트 저장 등.
        현재는 파일 저장 예시만 제공.
        """
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save File",
            str(self._last_open_dir / "result.npy"),
            "NumPy Array (*.npy);;PNG Image (*.png);;All Files (*.*)",
        )
        if not path:
            return

        # TODO: 실제 저장 로직(services/io.py 등)을 호출
        # 수정 필요: 분류 결과, 클래스맵, ROI 등을 저장하는 서비스 로직 구현
        # 예시: np.save(path, some_array)
        self.statusBar().showMessage(f"Saved: {path}", 4000)

    @QtCore.pyqtSlot()
    def on_exit(self) -> None:
        """
        Exit: 애플리케이션 종료.
        """
        self.close()
        
    def _init_layers_dock(self) -> None:
        """좌측에 Layers 도크를 항상 표시 상태로 부착."""
        self.layersDock = LayersDock(
            parent=self,
            on_visible_change=self._on_layer_visible_changed,
            on_reorder=self._on_layers_reordered,
        )
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.layersDock)

        # 닫기 버튼을 비활성화 → 사용자가 닫을 수 없게 (항상 켜짐)
        feats = self.layersDock.features()
        self.layersDock.setFeatures(feats & ~QtWidgets.QDockWidget.DockWidgetClosable)
        self.layersDock.show()

        # 컨텍스트 메뉴 시그널 연결(보유한 콜백만 연결)
        self.layersDock.requestDeleteLayer.connect(self._on_delete_layer_req)
        self.layersDock.requestEditLayer.connect(self._on_edit_layer_req)
        self.layersDock.requestClassMap.connect(self._on_merge_classmap_req)
        self.layersDock.requestOpenWorkspace.connect(self._on_open_workspace_ui)  # ★ 추가
        self.layersDock.requestOpenViewerBand.connect(self._on_open_viewer_band)  # ← 🖥

        self.layersDock.requestSaveLayer.connect(self._on_save_layer_req)     # ★ 저장 버튼
        self.layersDock.requestLoadPaths.connect(self._on_load_files_req)     # ★ DnD 로드
    # -----------------------------
    # LayersDock 콜백 (stub)  (# NEW)
    # -----------------------------
    def _on_layer_visible_changed(self, name: str, visible: bool):
        try:
            self.layer_manager.set_visible(name, visible)  # ✅ 한 줄만
        except Exception:
            logging.exception("[Layers] manager visible failed")

    def _on_layers_reordered(self, names_top_to_bottom: list[str]):
        try:
            self.layer_manager.reorder_top_to_bottom(names_top_to_bottom)  # ✅ 단일 진입점
        except Exception:
            logging.exception("[Layers] reorder apply failed")

    def _on_delete_layer_req(self, name: str):
        # LayerManager가 실제 MapView/Dock 상태를 알고 있으므로 위임
        try:
            self.layer_manager.remove(name)          # 부모 또는 자식 이름 모두 처리
            self.layersDock.remove_layer(name)       # 트리에서 즉시 제거
        except Exception as e:
            logging.exception(e)
            
    def _on_edit_layer_req(self, name: str):
        print(f"[layers] edit request: {name}")
        # TODO: 수정 모드 진입
        # 수정 필요: 레이어 수정을 위한 다이얼로그나 인라인 편집 기능 구현

    @QtCore.pyqtSlot(list)
    def _on_merge_classmap_req(self, names: list[str]):
        try:
            # LayerManager가 병합/등록을 전부 담당
            out_name = self.layer_manager.merge_classmaps(names)

            # (선택) 저장할지 묻기
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "합친 클래스맵 저장",
                str(self._last_open_dir / f"{out_name}.npz"),
                "NumPy Zip (*.npz)"
            )
            if path:
                # LM 안쪽 저장소에서 바로 꺼내 저장
                cm = self.layer_manager.get_classmap(out_name)
                if cm is not None:
                    self._save_npz_with_meta(path, data=cm.astype(np.int32, copy=False), kind="classmap")
                    self.statusBar().showMessage(f"합친 클래스맵 저장: {path}", 4000)
                else:
                    QtWidgets.QMessageBox.warning(self, "경고", "결과 클래스맵을 찾을 수 없습니다.")
            else:
                self.statusBar().showMessage(f"합친 클래스맵 생성: {out_name}", 3000)

        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "경고", str(e))
        except Exception:
            logging.exception("[Layers] classmap merge failed")
            QtWidgets.QMessageBox.critical(self, "오류", "클래스맵 병합 중 오류가 발생했습니다.")


    def register_layer(self, name, data, *, type="overlay", visible=True, meta=None, **kwargs):
        """
        MainWindow는 오케스트레이션만 담당.
        실제 변환/정렬/시각화는 LayerManager.add가 전담한다.
        """
        meta = meta or {}
        # ROI면 image_code 검증·저장 등을 위해 LM 내부 ROI도 함께 보관(저장 메뉴용)
        if type == "mask.roi" and hasattr(self.layer_manager, "set_roi"):
            try:
                mask_bool = (data.astype(bool) if isinstance(data, np.ndarray) else np.asarray(data).astype(bool))
                self.layer_manager.set_roi(mask_bool)
            except Exception:
                pass

        try:
            # 나머지는 LM.add가 모두 처리(마스크→RGBA, 등록 시 최상단, 가시성 반영, 정렬 반영)
            self.layer_manager.add(
                name=name,
                data=data,            # mask/roi면 bool 2D, overlay면 RGBA/RGB
                type=type,            # "mask" | "mask.roi" | "overlay" | (옵션) "heatmap"
                visible=visible,
                meta=meta             # {"color": (r,g,b,a), "opacity": 0~1, ...}
            )
        except Exception:
            logging.exception("[Layers] register_layer failed")

    # ✅ 교체: on_classification_dialog_run 전체
    @QtCore.pyqtSlot()
    def on_classification_dialog_run(self) -> None:
        """
        메뉴 [분류 맵 생성] → classmap.ui(모달) 표시 → 사용자 입력 읽어 분류 실행 → 레이어/지도 반영 → PixelClassificationDock 표시
        """
        try:
            # 이미지 로드 체크
            if not hasattr(self, "rgb_image") or self.rgb_image is None:
                QtWidgets.QMessageBox.information(self, "안내", "먼저 이미지를 로드하세요.")
                return

            dlg = PixelClassificationDialog(parent=self, ui_dir=self.app_dir / "ui")

            # 현재 도크가 있다면 도크의 설정을 기본값으로 주입 (동기화)
            if hasattr(self, "clsDock") and self.clsDock and hasattr(self.clsDock, "get_params"):
                params = self.clsDock.get_params()
                # classmap.ui는 map_name을 기본 "classification1"로 가지고 있음
                params.setdefault("map_name", "classification1")
                dlg.set_parameters(params)

            dlg.classification_requested.connect(self._on_classification_requested)
            dlg.exec_()
        except Exception as e:
            logging.exception("[Classification] dialog run failed")
            QtWidgets.QMessageBox.critical(self, "오류", f"분류 다이얼로그 실행 중 오류가 발생했습니다: {e}")

    @QtCore.pyqtSlot(dict)
    def _on_classification_requested(self, params: dict):
        """
        PixelClassificationDialog의 '저장' 버튼 클릭 시 호출됨.
        - 분류 수행이 정상 완료되면 Dock을 표시한다.
        """
        try:
            self._perform_pixel_classification(params)  # 내부에서 _on_pixel_classify 호출

            # (선택) 맵 이름 적용이 필요하면 여기에 반영
            # if params.get("map_name") and params["map_name"] != "classification":
            #     try:
            #         self.layer_manager.rename("classification", params["map_name"])
            #     except Exception:
            #         logging.exception("[Classification] map rename failed")

            # ✅ Dialog에서 '저장'을 눌러 분류 맵이 저장(생성)되었으므로 Dock을 켠다.
            self._show_pixel_classification_dock()

        except Exception as e:
            logging.exception("[Classification] request failed")
            QtWidgets.QMessageBox.critical(self, "오류", f"분류 맵 생성 중 오류가 발생했습니다: {e}")

                
    # ✅ 교체/보강: _perform_pixel_classification (원래 로직은 그대로 두고, param을 그대로 전달)
    def _perform_pixel_classification(self, params: dict) -> None:
        """
        실제 분류 수행. params = {'metric','tau','delta','map_name'}.
        내부적으로 _on_pixel_classify를 호출해 Layer 등록/TopK/Map 반영까지 수행.
        """
        self._ensure_cls_dock_initialized()

        metric = params.get("metric", "SAD")
        tau    = float(params.get("tau", 0.05))
        delta  = float(params.get("delta", 0.03))

        # ★ Dialog에서 받은 tau/delta 값을 Dock에 반영하고 수정 불가 처리
        if hasattr(self, "clsDock") and self.clsDock:
            self.clsDock.apply_dialog_params({
                "metric": metric,
                "tau": tau,
                "delta": delta,
            })

        # 기존 분류 함수 재사용
        self._on_pixel_classify({
            "metric": metric,
            "tau":    tau,
            "delta":  delta,
        })

        # (선택) params['map_name'] 로 레이어 이름 바꿔 붙이려면 여기서 LayerManager.rename 사용
        # if params.get("map_name") and params["map_name"] != "classification":
        #     try:
        #         self.layer_manager.rename("classification", params["map_name"])
        #     except Exception:
        #         logging.exception("[Classification] map rename failed")

    def _ensure_cls_dock_initialized(self):
        """PixelClassificationDock 객체를 '숨긴 상태로' 미리 생성만 보장."""
        if getattr(self, "clsDock", None) is None:
            self._init_pixel_classification_dock()  # ← 여기서 생성만 하고 hide() 상태로 둠

    def _show_pixel_classification_dock(self):
        """Pixel Classification Dock 표시"""
        try:
            # 도크 보장(없으면 생성)
            if not hasattr(self, "clsDock") or self.clsDock is None:
                self._init_pixel_classification_dock()
                
            # 도크 표시
            self.clsDock.setVisible(True)
            if self.clsDock.parent() is None or self.clsDock.isFloating():
                self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.clsDock)
                self.clsDock.setFloating(False)

            self.clsDock.raise_()
            self.clsDock.activateWindow()
            
            self.statusBar().showMessage("Pixel Classification Dock 활성화", 2000)
            
        except Exception as e:
            logging.exception("[Dock] show failed")
            QtWidgets.QMessageBox.warning(self, "경고", "Pixel Classification Dock 표시에 실패했습니다.")

    # ✅ 추가: MainWindow 내부에 새 헬퍼 메서드
    def _sync_classification_params_to_dock(self, params: Dict[str, Any]) -> None:
        try:
            if not getattr(self, "clsDock", None):
                return
            root = getattr(self.clsDock, "_root", None)
            if not root:
                return

            # τ/δ: SpinBox가 기준 (없으면 무시)
            tau   = float(params.get("tau", 0.05))
            delta = float(params.get("delta", 0.03))
            if hasattr(root, "tauSpinBox"):   root.tauSpinBox.setValue(tau)
            if hasattr(root, "deltaSpinBox"): root.deltaSpinBox.setValue(delta)

            # metric 라디오가 Dock UI에 있다면만 반영
            metric = str(params.get("metric", "SAD")).upper()
            if hasattr(root, "radioSAD"): root.radioSAD.setChecked(metric == "SAD")
            if hasattr(root, "radioSID"): root.radioSID.setChecked(metric == "SID")
            if hasattr(root, "radioSCC"): root.radioSCC.setChecked(metric == "SCC")

        except Exception:
            logging.exception("[Classification] dock param sync failed")


    # --- (1) 다이얼로그 실행 시: Map 클릭을 '시드 선택 모드'로 전환하고,
    #       클래스 옵션/팔레트 주입, 종료 시 모드 복구
    @QtCore.pyqtSlot()
    def on_diffusion_dialog_run(self):
        """
        확산 맵 생성 다이얼로그 실행
        - 기준 classmap 선택(cboBaseMap)
        - 맵 클릭으로 시드 추가(선택한 classmap의 CID 자동 반영)
        - 임계/마진 값으로 region growing 수행
        - workspace(_work_mask)는 고정
        """
        try:
            # 0) 이미지 로드 가드
            if not hasattr(self, "rgb_image") or self.rgb_image is None:
                QtWidgets.QMessageBox.information(self, "안내", "먼저 이미지를 로드하세요.")
                return

            # 1) 다이얼로그 생성
            from views.dialogs.diffusion_dialog import DiffusionDialog
            dlg = DiffusionDialog(parent=self, ui_dir=self.app_dir / "ui")
            self._diff_dialog = dlg

            # 2) RegionGrowingService 보강: HSI/ROI 세팅
            try:
                if hasattr(self, "region_growing_service") and self.region_growing_service:
                    if self.region_growing_service.get_hsi_data_shape() is None and hasattr(self, "cfg"):
                        self.region_growing_service.set_hsi_data(self.cfg["data"])
                    # workspace 고정
                    self.region_growing_service.set_allowed_mask(getattr(self, "_work_mask", None))
            except Exception:
                logging.exception("[Diffusion] ensure HSI/ROI for region growing failed")

            # 3) 클래스맵 목록 주입
            try:
                if hasattr(self, "_cb_classmap_names"):
                    dlg.set_classmap_options(self._cb_classmap_names())
            except Exception:
                logging.exception("[Diffusion] set_classmap_options failed")

            # 4) 클래스 옵션/팔레트 주입
            try:
                # id->name 맵 정규화
                raw = getattr(self, "_last_id_to_name", {}) or {}
                id_to_name = {int(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
                lib = getattr(self, "lib", {}) or {}
                class_ids = sorted(map(int, lib.keys()))
                class_opts = [(cid, id_to_name.get(cid, str(cid))) for cid in class_ids]
                dlg.set_class_options(class_opts)
                dlg.set_palette(getattr(self, "class_palette_qcolor", {}) or {})
            except Exception:
                logging.exception("[Diffusion] set_class_options/palette failed")

            # 5) 신호 연결
            dlg.diffusion_requested.connect(self._on_diffusion_requested)
            if hasattr(dlg, "analyze_requested"):
                dlg.analyze_requested.connect(self._on_diffusion_analyze_requested)
            if hasattr(dlg, "seeds_changed"):
                # 맵 마커 갱신(지원 함수가 있을 때만)
                dlg.seeds_changed.connect(lambda seeds: getattr(self._map_view, "rebuild_seed_markers", lambda *_: None)(seeds))

            # 6) 픽셀 클릭 모드: 시드 선택
            self._enter_pixel_click_mode(ClickMode.NONE, mute_roi=True)
            
        # Next(기준 맵 확정) 후에만 클릭 허용
            def _on_lock_changed(locked: bool):
                if locked:
                    self._enter_pixel_click_mode(ClickMode.DIFFUSION_SEED, mute_roi=True)
                    self.statusBar().showMessage("기준 맵이 확정되었습니다. 맵을 클릭해 시드를 추가하세요.", 3000)
            if hasattr(dlg, "base_lock_changed"):
                dlg.base_lock_changed.connect(_on_lock_changed)
                
            # 7) 종료 시 정리
            def _on_closed(_code):
                try:
                    self._leave_pixel_click_mode(restore=True)
                finally:
                    self._diff_dialog = None
                    # 시드 마커 정리(지원 시)
                    if hasattr(self._map_view, "clear_seed_markers"):
                        self._map_view.clear_seed_markers()
            dlg.finished.connect(_on_closed)

            # 8) 초기 진입 시 마커 초기화(선택)
            if hasattr(self._map_view, "clear_seed_markers"):
                self._map_view.clear_seed_markers()

            # 9) 모델리스로 표시
            dlg.setModal(False)
            dlg.setWindowModality(Qt.NonModal)
            dlg.show()

        except Exception as e:
            logging.exception("[Diffusion] dialog run failed")
            QtWidgets.QMessageBox.critical(self, "오류", f"확산 다이얼로그 실행 중 오류가 발생했습니다: {e}")

    @QtCore.pyqtSlot()
    def on_pixel_labeling_dialog_run(self):
        """선택 픽셀 라벨링 다이얼로그 실행"""
        try:
            if not hasattr(self, "rgb_image") or self.rgb_image is None:
                QtWidgets.QMessageBox.information(self, "안내", "먼저 이미지를 로드하세요.")
                return
            
            dlg = PixelLabelingDialog(parent=self, ui_dir=self.app_dir / "ui")
            self._labeling_dialog = dlg  # 참조 보관

            # --- 클래스 옵션/팔레트 주입 ---
            try:
                class_options = self._resolve_current_class_options()
                if class_options and hasattr(dlg, "set_class_options"):
                    dlg.set_class_options(class_options)  # [(cid, name), ...]
                if hasattr(dlg, "set_palette"):
                    dlg.set_palette(getattr(self, "class_palette_qcolor", {}) or {})
                
                # ★ user_labeling_dialog에서 선택한 기본 CID 가져오기
                default_cid = None
                if hasattr(self, "_user_labeling_dialog") and self._user_labeling_dialog is not None:
                    try:
                        # user_labeling_dialog의 전역 콤보박스에서 선택된 CID 가져오기
                        if hasattr(self._user_labeling_dialog, "_get_global_cid"):
                            default_cid = self._user_labeling_dialog._get_global_cid()
                            if default_cid is not None and default_cid >= 0:
                                if hasattr(dlg, "set_default_cid"):
                                    dlg.set_default_cid(default_cid)
                                    import logging
                                    logging.info(f"[PixelLabeling] user_labeling_dialog에서 기본 CID 설정: {default_cid}")
                    except Exception as e:
                        import logging
                        logging.debug(f"[PixelLabeling] user_labeling_dialog에서 기본 CID 가져오기 실패: {e}")
            except Exception:
                logging.exception("[Labeling] set_class_options/palette failed")

            # --- 신호 연결 ---
            # dlg.labeling_requested.connect(self._on_labeling_requested)
            dlg.user_labeling_requested.connect(self.on_user_labeling_dialog_run)
            dlg.classmap_labeling_requested.connect(self.on_classmap_labeling_dialog_run_from_button)
            dlg.recommend_labeling_requested.connect(self.on_recommend_labeling_wizard_run)
            dlg.register_requested.connect(self._on_pixel_labels_registered)

            # --- 픽셀 클릭 모드 진입 (라벨링 모드) ---
            self._enter_pixel_click_mode(ClickMode.LABEL, mute_roi=True)

            # --- 종료 시 정리 ---
            def _on_labeling_closed(_code):
                try:
                    self._leave_pixel_click_mode(restore=True)
                finally:
                    self._labeling_dialog = None

            dlg.finished.connect(_on_labeling_closed)

            # --- 모델리스로 띄우기 ---
            dlg.setModal(False)
            dlg.setWindowModality(Qt.NonModal)
            dlg.show()

        except Exception as e:
            logging.exception("[Labeling] dialog run failed")
            QtWidgets.QMessageBox.critical(self, "오류", f"라벨링 다이얼로그 실행 중 오류가 발생했습니다: {e}")

    @QtCore.pyqtSlot()
    def on_unmixing_dialog_run(self):
        """분광 혼합 분석 맵 생성 다이얼로그 실행"""
        try:
            # 0) 이미지 로드 가드
            if not hasattr(self, "rgb_image") or self.rgb_image is None:
                QtWidgets.QMessageBox.information(self, "안내", "먼저 이미지를 로드하세요.")
                return

            # 1) 다이얼로그 생성
            dlg = UnmixingDialog(parent=self, ui_dir=self.app_dir / "ui")
            self._unmixing_dialog = dlg

            # 2) 신호 연결
            dlg.unmixing_requested.connect(self._on_unmixing_requested)

            # 3) 모델리스로 표시
            dlg.setModal(False)
            dlg.setWindowModality(Qt.NonModal)
            dlg.show()

        except Exception as e:
            logging.exception("[Unmixing] dialog run failed")
            QtWidgets.QMessageBox.critical(self, "오류", f"분광 혼합 분석 다이얼로그 실행 중 오류가 발생했습니다: {e}")

    def _on_unmixing_requested(self, params: dict):
        """분광 혼합 분석 실행 요청 처리 + Classmap 생성"""
        try:
            endmember_count = int(params.get("endmember_count", 3))
            # Dialog에서는 0.0~1.0으로 받지만, 혹시 80 같은 값이 들어와도 퍼센트로 처리
            raw_thr = params.get("threshold", 0.8)
            try:
                raw_thr = float(raw_thr)
            except Exception:
                raw_thr = 0.8

            # 0~1 범위로 정규화
            threshold = raw_thr / 100.0 if raw_thr > 1.0 else raw_thr
            threshold = max(0.0, min(1.0, threshold))

            logging.info(
                f"[Unmixing] Requested: endmember_count={endmember_count}, threshold={threshold:.3f}"
            )

            # 0) 이미지/ROI 체크
            if not hasattr(self, "cfg") or "data" not in self.cfg:
                QtWidgets.QMessageBox.information(self, "안내", "먼저 이미지를 로드하세요.")
                return

            cube = self.cfg["data"]
            H, W = cube.shape[:2]

            # 작업 영역 사각형이 있으면 그 영역 사용, 없으면 전체
            rect = getattr(self, "current_selection_rect", None)
            if rect is not None and rect.isValid():
                st_x = int(rect.x())
                st_y = int(rect.y())
                ed_x = st_x + int(rect.width())
                ed_y = st_y + int(rect.height())
            else:
                st_x, st_y = 0, 0
                ed_x, ed_y = W, H

            # 1) image_cd, base_url 준비
            img_cd = getattr(self, "image_cd", None)
            if img_cd is None:
                QtWidgets.QMessageBox.warning(self, "경고", "image_cd가 없습니다. 먼저 HSI를 로드하세요.")
                return

            base_url = os.getenv("unmixing_inference_url")
            if not base_url:
                QtWidgets.QMessageBox.warning(
                    self, "경고",
                    "UNMIXING_API_URL 환경변수가 설정되어 있지 않습니다.\n"
                    "분광 혼합 분석 API URL을 설정해 주세요."
                )
                return

            # 2) API 호출
            self.statusBar().showMessage("분광 혼합 분석 API 호출 중...", 3000)
            endmembers, abundance_map = call_unmixing(
                base_url=base_url,
                img_cd=int(img_cd),
                st_x=st_x, st_y=st_y,
                ed_x=ed_x, ed_y=ed_y,
                num_endmembers=endmember_count,
                # timeout=30,
            )

            # 3) 결과를 numpy로 변환해서 보관
            self.unmixing_endmembers = np.asarray(endmembers, dtype=np.float32)
            A = np.asarray(abundance_map, dtype=np.float32)

            # abundance_map shape 정규화: (H, W, K)로 맞추기
            if A.ndim == 3:
                if A.shape[0] == H and A.shape[1] == W:
                    # (H, W, K)
                    self.unmixing_abundance_map = A
                elif A.shape[1] == H and A.shape[2] == W:
                    # (K, H, W) → (H, W, K)
                    self.unmixing_abundance_map = np.moveaxis(A, 0, 2)
                else:
                    raise RuntimeError(f"abundance_map shape 예상과 다름: {A.shape}, 이미지=({H},{W})")
            else:
                raise RuntimeError(f"abundance_map ndim=3이 아님: {A.shape}")

            self.unmixing_threshold = threshold

            logging.info(
                "[Unmixing] API done: endmembers.shape=%s, abundance_map.shape=%s",
                getattr(self.unmixing_endmembers, "shape", None),
                getattr(self.unmixing_abundance_map, "shape", None),
            )

            # 4) Unmixing classmap 생성/등록
            # 4) Unmixing classmap 생성/등록
            self._build_unmixing_classmap_from_abundance(threshold)

            # 5) Unmixing Dock 표시 + 클래스 옵션 + 결과 전달
            self._show_unmixing_dock(endmember_count, threshold)

            try:
                if hasattr(self, "unmixingDock") and self.unmixingDock:
                    # 5-1) 클래스 옵션 (CID, 물질명) 생성
                    try:
                        class_options = self._build_class_options_for_labeling()  # [(cid, name), ...]
                    except Exception:
                        logging.exception("[Unmixing] build_class_options_for_labeling failed")
                        class_options = []

                    # 5-2) Dock에 클래스 옵션 넘기기 (Class 콤보박스 채우기)
                    if class_options and hasattr(self.unmixingDock, "set_class_options"):
                        self.unmixingDock.set_class_options(class_options)

                    # 5-3) 파장 정보도 Dock으로 전달 (EndmemberDetailDialog에서 사용)
                    wl = self.cfg.get("wavelength") or self.cfg.get("wavelength_list")
                    if wl is not None and hasattr(self.unmixingDock, "set_wavelengths"):
                        try:
                            self.unmixingDock.set_wavelengths(np.asarray(wl))
                        except Exception:
                            logging.exception("[Unmixing] set_wavelengths to dock failed")

                    # 5-4) Unmixing 결과 전달
                    if hasattr(self.unmixingDock, "set_results"):
                        self.unmixingDock.set_results(
                            endmembers=self.unmixing_endmembers,
                            abundance_map=self.unmixing_abundance_map,
                            threshold=threshold,
                        )

            except Exception:
                logging.exception("[Unmixing] set_results/set_class_options to dock failed")

            self.statusBar().showMessage(
                f"분광 혼합 분석 완료 — endmembers: {self.unmixing_endmembers.shape}, "
                f"abundance_map: {self.unmixing_abundance_map.shape}, "
                f"threshold={threshold:.3f}",
                4000,
            )

        except Exception as e:
            logging.exception("[Unmixing] request handling failed")
            QtWidgets.QMessageBox.critical(
                self, "오류",
                f"분광 혼합 분석 실행 중 오류가 발생했습니다:\n{e}"
            )

    def _init_unmixing_dock(self):
        """UnmixingDock 생성 및 초기화"""
        if self.unmixingDock is not None:
            return
        
        self.unmixingDock = UnmixingDock(
            parent=self,
            ui_dir=self.app_dir / "ui",
        )
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.unmixingDock)
        
        # 닫기 기능 활성화
        feats = self.unmixingDock.features()
        self.unmixingDock.setFeatures(feats | QtWidgets.QDockWidget.DockWidgetClosable)
        
        # View 메뉴에 토글 액션 추가
        self._add_dock_to_view_menu(self.unmixingDock, "Unmixing Analysis")
        
        # 초기에는 숨김 상태
        self.unmixingDock.hide()

        # ★ 임계값 변경 시그널 연결 (UnmixingDock 쪽에 thresholdChanged 시그널이 있다고 가정)
        try:
            self.unmixingDock.thresholdChanged.connect(self._on_unmixing_threshold_changed)
            self.unmixingDock.classMappingApplied.connect(self._on_unmixing_class_mapping_applied)
        except Exception:
            logging.exception("[Unmixing] connect thresholdChanged failed")


    def _show_unmixing_dock(self, endmember_count: int, threshold: float):
        """Unmixing Dock 표시"""
        try:
            # Dock 보장(없으면 생성)
            if not hasattr(self, "unmixingDock") or self.unmixingDock is None:
                self._init_unmixing_dock()
            
            # 파라미터 설정
            self.unmixingDock.set_endmember_count(endmember_count)
            self.unmixingDock.set_threshold(threshold)
            
            # Dock 표시
            self.unmixingDock.setVisible(True)
            if self.unmixingDock.parent() is None or self.unmixingDock.isFloating():
                self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.unmixingDock)
                self.unmixingDock.setFloating(False)
            
            self.unmixingDock.raise_()
            self.unmixingDock.activateWindow()
            
            self.statusBar().showMessage("Unmixing Analysis Dock 활성화", 2000)
            
        except Exception as e:
            logging.exception("[Unmixing] dock show failed")
            QtWidgets.QMessageBox.warning(self, "경고", "Unmixing Analysis Dock 표시에 실패했습니다.")
            
    def _merge_label_sources(
        self,
        base: Optional[dict[int, np.ndarray]],
        extra: Optional[dict[int, Any]],
        *,
        expect_c: Optional[int] = None,
        tag: str = "[LabelMerge]",
    ) -> dict[int, np.ndarray]:
        """dict 기반 라벨 스펙트럼을 병합해 cid -> (N,C) float32 로 반환."""
        out: dict[int, np.ndarray] = {}

        def _as_matrix(payload: Any) -> Optional[np.ndarray]:
            if payload is None:
                return None
            arr = np.asarray(payload, dtype=np.float32)
            if arr.size == 0:
                return None
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.ndim != 2:
                return None
            return arr

        if isinstance(base, dict):
            for cid, block in base.items():
                mat = _as_matrix(block)
                if mat is None:
                    continue
                out[int(cid)] = mat.copy()

        if isinstance(extra, dict):
            for cid, block in extra.items():
                mat = _as_matrix(block)
                if mat is None:
                    continue
                if expect_c is not None and expect_c > 0 and mat.shape[1] != expect_c:
                    logging.warning(
                        f"{tag} skip cid={cid}: C={mat.shape[1]} != expect_c={expect_c}"
                    )
                    continue

                cid_i = int(cid)
                prev = out.get(cid_i)
                if prev is None:
                    out[cid_i] = mat
                    continue

                if prev.shape[1] != mat.shape[1]:
                    logging.warning(
                        f"{tag} channel mismatch cid={cid}: prev C={prev.shape[1]} vs new C={mat.shape[1]} (skip)"
                    )
                    continue

                out[cid_i] = np.vstack([prev, mat]).astype(np.float32)

        return out

    # --- [추가] 공용 헬퍼: 스펙트럼 라이브러리 재구성 ---
    def _rebuild_lib(self) -> None:
        try:
            expect_c = int(self.cfg["data"].shape[2]) if hasattr(self, "cfg") and isinstance(self.cfg, dict) and "data" in self.cfg else None

            # ---- 공통 유틸 -------------------------------------------------------
            def _to_2d(a):
                try:
                    arr = np.asarray(a, dtype=np.float32)
                except Exception:
                    return None
                if arr.ndim == 1: arr = arr[None, :]
                if arr.ndim != 2: return None
                return arr

            def _unwrap_deep(e):
                """'spec_lib' 같은 래퍼가 여러 겹일 때까지 깊게 언래핑."""
                while isinstance(e, dict) and any(k in e for k in ("spec_lib","spec","library","entries")):
                    for k in ("spec_lib","spec","library","entries"):
                        if k in e:
                            e = e[k]
                            break
                return e

            def _resample_2d_to(mat: np.ndarray, dst_c: int) -> np.ndarray:
                """파장 정보가 없을 때 균등축 선형 보간으로 (N, C) → (N, dst_c)."""
                if mat.shape[1] == dst_c: return mat.astype(np.float32, copy=False)
                x_old = np.linspace(0.0, 1.0, num=mat.shape[1], dtype=np.float32)
                x_new = np.linspace(0.0, 1.0, num=dst_c,        dtype=np.float32)
                out = np.empty((mat.shape[0], dst_c), dtype=np.float32)
                for i in range(mat.shape[0]):
                    out[i, :] = np.interp(x_new, x_old, mat[i, :])
                return out

            def _norm_entries(entries, prefer_key: str, src_name: str = "unknown") -> dict[int, np.ndarray]:
                """
                entries: dict / list[dict] / ndarray
                prefer_key: 'ref'(스펙) or 'rfl'(라벨)
                src_name: 로깅용 소스 이름 (예: "label_raw", "label_cr")
                """
                # 입력 데이터 타입 확인
                if isinstance(entries, dict):
                    logging.info("[_norm_entries] %s: input dict with %d keys", src_name, len(entries))
                elif isinstance(entries, (list, tuple)):
                    logging.info("[_norm_entries] %s: input list/tuple with %d items", src_name, len(entries))
                else:
                    logging.info("[_norm_entries] %s: input type=%s", src_name, type(entries).__name__)
                
                e = _unwrap_deep(entries)

                # tuple 형태((lib_dict, id_to_name), ...) 대비: 스펙트럼 블록(dict)만 추출
                if isinstance(e, tuple):
                    picked = None
                    for candidate in e:
                        if isinstance(candidate, dict) and candidate:
                            sample = next(iter(candidate.values()), None)
                            if isinstance(sample, (list, tuple, np.ndarray)):
                                picked = candidate
                                break
                            if isinstance(sample, dict):
                                probe = sample.get(prefer_key) or sample.get("ref") or sample.get("rfl")
                                if isinstance(probe, (list, tuple, np.ndarray)):
                                    picked = candidate
                                    break
                    if picked is not None:
                        e = picked
                    else:
                        e = list(e)

                out: dict[int, list[np.ndarray]] = {}

                if isinstance(e, dict):
                    for k, v in e.items():
                        # cid 결정
                        try:
                            cid = int(k)
                        except Exception:
                            # 값이 dict이면 그 안에서 cid를 꺼냄
                            if isinstance(v, dict):
                                cid = v.get("mtrl_cd") or v.get("cid") or v.get("class_id")
                                try: cid = int(cid)
                                except Exception: continue
                            else:
                                continue

                        # 데이터 추출
                        a = v
                        if isinstance(v, dict):
                            a = v.get(prefer_key, None)
                            if a is None:
                                a = v.get("ref", None)
                            if a is None:
                                a = v.get("rfl", None)

                        a2 = _to_2d(a)
                        if a2 is None:
                            logging.warning("[rebuild] skip cid=%s: _to_2d failed (src=%s, prefer_key=%s)", cid, src_name, prefer_key)
                            continue

                        # 밴드 불일치 보정
                        if (expect_c is not None) and (a2.shape[1] != expect_c):
                            logging.info("[rebuild] resample cid=%s: C=%s -> %s (src=%s)", cid, a2.shape[1], expect_c, src_name)
                            try:
                                a2 = _resample_2d_to(a2, expect_c)
                            except Exception as ex:
                                logging.warning("[rebuild] skip cid=%s: resample failed (src=%s, C=%s -> %s, error=%s)", 
                                              cid, src_name, a2.shape[1], expect_c, str(ex))
                                continue

                        out.setdefault(int(cid), []).append(a2)

                else:
                    if isinstance(e, np.ndarray):
                        e = e.tolist()
                    if isinstance(e, (list, tuple)):
                        for rec in (e or []):
                            if not isinstance(rec, dict): continue
                            cid = rec.get("mtrl_cd") or rec.get("cid") or rec.get("_norm_entriesclass_id")
                            try: cid = int(cid)
                            except Exception: continue
                            a = rec.get(prefer_key, None)
                            if a is None:
                                a = rec.get("ref", None)
                            if a is None:
                                a = rec.get("rfl", None)
                            a2 = _to_2d(a)
                            if a2 is None:
                                logging.warning("[rebuild] skip cid=%s: _to_2d failed (src=%s, prefer_key=%s)", cid, src_name, prefer_key)
                                continue
                            if (expect_c is not None) and (a2.shape[1] != expect_c):
                                logging.info("[rebuild] resample cid=%s: C=%s -> %s (src=%s)", cid, a2.shape[1], expect_c, src_name)
                                try:
                                    a2 = _resample_2d_to(a2, expect_c)
                                except Exception as ex:
                                    logging.warning("[rebuild] skip cid=%s: resample failed (src=%s, C=%s -> %s, error=%s)", 
                                                  cid, src_name, a2.shape[1], expect_c, str(ex))
                                    continue
                            out.setdefault(int(cid), []).append(a2)

                result = {cid: np.vstack(mats).astype(np.float32) for cid, mats in out.items()}
                if result:
                    total_samples = sum(arr.shape[0] for arr in result.values())
                    logging.info("[_norm_entries] %s: output %d classes, %d total samples", src_name, len(result), total_samples)
                else:
                    logging.warning("[_norm_entries] %s: output is empty!", src_name)
                return result

            def _stack_by_class(*libs: dict | None) -> dict[int, np.ndarray]:
                buckets: dict[int, list[np.ndarray]] = {}
                for L in libs:
                    if not isinstance(L, dict) or not L: continue
                    for cid, arr in L.items():
                        a = _to_2d(arr)
                        if a is None:
                            continue
                        if (expect_c is not None) and (a.shape[1] != expect_c):
                            logging.warning("[rebuild] skip on stack cid=%s: C=%s != %s", cid, a.shape[1], expect_c)
                            continue
                        buckets.setdefault(int(cid), []).append(a)
                return {cid: np.vstack(mats).astype(np.float32) for cid, mats in buckets.items()}

            # 1) 정규화 (스펙=ref 우선, 라벨=rfl 우선)
            # 입력 데이터 확인
            label_raw_before = getattr(self, "label_raw", {})
            label_cr_before = getattr(self, "label_cr", {})
            if isinstance(label_raw_before, dict):
                logging.info("[rebuild] label_raw input: %d classes, samples=%s", 
                           len(label_raw_before), {cid: arr.shape[0] if hasattr(arr, 'shape') else '?' for cid, arr in list(label_raw_before.items())[:5]})
            if isinstance(label_cr_before, dict):
                logging.info("[rebuild] label_cr input: %d classes, samples=%s", 
                           len(label_cr_before), {cid: arr.shape[0] if hasattr(arr, 'shape') else '?' for cid, arr in list(label_cr_before.items())[:5]})
            
            splib_raw_norm = _norm_entries(getattr(self, "splib_raw"), prefer_key="ref", src_name="splib_raw")
            splib_cr_norm  = _norm_entries(getattr(self, "splib_cr"), prefer_key="ref", src_name="splib_cr")
            label_raw_norm = _norm_entries(getattr(self, "label_raw"), prefer_key="rfl", src_name="label_raw")
            label_cr_norm  = _norm_entries(getattr(self, "label_cr"), prefer_key="rfl", src_name="label_cr")
            
            # 정규화 후 결과 확인
            logging.info("[rebuild] after _norm_entries: label_raw_norm=%d classes, label_cr_norm=%d classes", 
                        len(label_raw_norm), len(label_cr_norm))

            # 2) 라벨 전용
            self.labeling_lib = _stack_by_class(label_raw_norm, label_cr_norm)
            
            # ✅ raw 라벨링만 따로 보관 (BR에서 사용할 전용 dict)
            self.labeling_raw_lib = label_raw_norm
            
            # spectrum library의 raw 데이터만 저장
            self.splib_raw_norm = splib_raw_norm
            
            # 3) 통합 라이브러리(스펙 + 라벨)
            self.lib = _stack_by_class(splib_raw_norm, splib_cr_norm, label_raw_norm, label_cr_norm)

            splib_raw_c, splib_raw_n = self._count_classes_and_samples(splib_raw_norm)
            splib_raw_c, splib_raw_n = self._count_classes_and_samples(splib_cr_norm)

            lab_raw_c, lab_raw_n = self._count_classes_and_samples(label_raw_norm)
            lab_cr_c,  lab_cr_n  = self._count_classes_and_samples(label_cr_norm)
            lib_c,     lib_n     = self._count_classes_and_samples(self.lib)
            logging.info("[rebuild] classes: splib_raw=%d, splib_cr=%d, label_raw=%d, label_cr=%d, lib=%d",
                        len(splib_raw_norm), len(splib_cr_norm), len(label_raw_norm), len(label_cr_norm), len(self.lib))

            logging.info(
                "[rebuild] label_raw=%d classes / %d samples, label_cr=%d classes / %d samples, lib=%d classes / %d samples",
                lab_raw_c, lab_raw_n, lab_cr_c, lab_cr_n, lib_c, lib_n
            )
            self.statusBar().showMessage(
                f"라이브러리 갱신: 라벨 {lab_raw_c+lab_cr_c} 클래스 / {lab_raw_n+lab_cr_n} 샘플",
                3000
            )
            # 4) 팔레트 보강
            try:
                all_cids = sorted(map(int, self.lib.keys()))
                self._augment_palette_for(all_cids, push_renderer=True, push_dock=True)
            except Exception:
                logging.exception("[Palette] augment after rebuild failed")

            self.statusBar().showMessage(f"라이브러리 갱신 완료: {len(self.lib)} classes", 2500)

        except Exception:
            logging.exception("[Library] rebuild failed")

    @QtCore.pyqtSlot(dict)
    def _on_pixel_labels_registered(self, payload: dict):
        """
        라벨 등록 (엄격 모드)
        - 저장/메모리/rows 입력을 모두 표준 포맷 {cid: (N,C) float32}로 정규화
        - 캐시는 프로세스 최초 1회만 하이드레이션(이후 재병합 금지) → 중복 2배 증가 방지
        - 병합 시 동일/근사 동일 스펙트럼 행만 제거(여러 스펙트럼은 정상 보존)
        - 최종 상태(self.*)로 재구성(_rebuild_lib) 후 캐시에 저장
        payload = {"rows": [{"img_cd":..., "mtrl_cd"|"cid": int, "img_x": x, "img_y": y, "rfl": [...]}, ...]}
        """
        import os, logging
        from services.resampling_cache import resample_cache_paths, save_cache
        from services.spec_library import norm_any_to_dict, detect_shape_info, FormatError

        # --- 로컬 유틸 -------------------------------------------------------------
        def _npz_get(z, key):
            """np.load로 읽은 npz에서 0-D object 안전 언패킹."""
            if key not in z:
                return {}
            obj = z[key]
            try:
                if getattr(obj, "ndim", None) == 0 and getattr(obj, "dtype", None) == object:
                    obj = obj.item()
            except Exception:
                pass
            return obj

        def _uniq_rows_by_round(mat: np.ndarray, decimals: int = 5) -> np.ndarray:
            """스펙트럼(행) 배열에서 라운딩 기반 중복 제거(첫 등장만 유지)."""
            if mat.ndim == 1:
                mat = mat[None, :]
            if mat.size == 0:
                return mat.astype(np.float32, copy=False)
            r = np.round(mat.astype(np.float32), decimals=decimals)
            keys = [r[i, :].tobytes() for i in range(r.shape[0])]
            seen = set()
            keep = []
            for i, k in enumerate(keys):
                if k in seen:
                    continue
                seen.add(k)
                keep.append(i)
            return mat[np.asarray(keep, dtype=int), :].astype(np.float32, copy=False)

        def _dedup_vstack(a: np.ndarray, b: np.ndarray, *, decimals: int = 5) -> np.ndarray:
            """두 (N,C)를 vstack 후 라운딩 기반으로 중복 행만 제거."""
            if a.ndim == 1: a = a[None, :]
            if b.ndim == 1: b = b[None, :]
            if a.size == 0: return _uniq_rows_by_round(b, decimals=decimals)
            if b.size == 0: return _uniq_rows_by_round(a, decimals=decimals)
            cat = np.vstack([a.astype(np.float32), b.astype(np.float32)])
            return _uniq_rows_by_round(cat, decimals=decimals)

        def _merge_dict(dst: dict[int, np.ndarray], src: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
            """
            {cid: (N,C)} 병합. 채널 불일치 시 예외.
            같은 cid에서 스펙트럼 행 중복만 제거(여러 스펙트럼은 보존).
            """
            if not src:
                return dst
            out = dict(dst)
            for k, v in src.items():
                b = np.asarray(v, dtype=np.float32)
                if b.ndim == 1: b = b[None, :]
                if k in out:
                    a = np.asarray(out[k], dtype=np.float32)
                    if a.ndim == 1: a = a[None, :]
                    if a.shape[1] != b.shape[1]:
                        raise FormatError(f"[Merge] 채널 불일치 cid={k}: {a.shape[1]} vs {b.shape[1]}")
                    out[k] = _dedup_vstack(a, b, decimals=5)  # ★ 중복만 제거
                else:
                    out[k] = b
            return out

        try:
            rows = (payload or {}).get("rows", None)
            if not isinstance(rows, list) or len(rows) == 0:
                raise ValueError("[LabelRegister] payload.rows 가 비어있거나 list 가 아닙니다.")

            # --- 기본 환경/형상 확인 ---
            cfg = getattr(self, "cfg", None)
            if not isinstance(cfg, dict) or "data" not in cfg:
                raise RuntimeError("[LabelRegister] cfg/data 가 없습니다. 먼저 이미지를 로드하세요.")

            cube = cfg["data"]
            C = detect_shape_info(cube)  # (H,W,C) → C

            cr_cube = getattr(self, "cr_cube", None)
            crC = detect_shape_info(cr_cube) if isinstance(cr_cube, np.ndarray) and cr_cube.ndim == 3 else None

            # --- 0) 메모리 정규화 ---
            mem_splib_raw = norm_any_to_dict(getattr(self, "splib_raw", {}), expect_c=C,            cube=cube,             prefer_key="ref", src_name="mem.splib_raw")
            mem_splib_cr  = norm_any_to_dict(getattr(self, "splib_cr",  {}), expect_c=(crC or C),   cube=(cr_cube or cube), prefer_key="ref", src_name="mem.splib_cr")
            mem_label_raw = norm_any_to_dict(getattr(self, "label_raw", {}), expect_c=C,            cube=cube,             prefer_key="rfl", src_name="mem.label_raw")
            mem_label_cr  = norm_any_to_dict(getattr(self, "label_cr",  {}), expect_c=(crC or C),   cube=(cr_cube or cube), prefer_key="rfl", src_name="mem.label_cr")

            # --- 0-1) 캐시 하이드레이션 (최초 1회만) ---
            cache_splib_raw = cache_splib_cr = cache_label_raw = cache_label_cr = {}
            primary = self._cache_primary_path(cfg)
            if not getattr(self, "_cache_hydrated", False):
                if primary:
                    info_p, npz_p = resample_cache_paths(primary)
                    if os.path.isfile(npz_p):
                        z = np.load(npz_p, allow_pickle=True)
                        if "splib_raw" in z:
                            cache_splib_raw = norm_any_to_dict(_npz_get(z, "splib_raw"), expect_c=C,          cube=cube,             prefer_key="ref", src_name="cache.splib_raw")
                        if "splib_cr" in z:
                            cache_splib_cr  = norm_any_to_dict(_npz_get(z, "splib_cr"),  expect_c=(crC or C), cube=(cr_cube or cube), prefer_key="ref", src_name="cache.splib_cr")
                        if "label_raw" in z:
                            cache_label_raw = norm_any_to_dict(_npz_get(z, "label_raw"), expect_c=C,          cube=cube,             prefer_key="rfl", src_name="cache.label_raw")
                        if "label_cr" in z:
                            cache_label_cr  = norm_any_to_dict(_npz_get(z, "label_cr"),  expect_c=(crC or C), cube=(cr_cube or cube), prefer_key="rfl", src_name="cache.label_cr")
                        if "label_coords" in z:
                            try:
                                self.label_coords = _npz_get(z, "label_coords") or {}
                            except Exception:
                                self.label_coords = {}
                        else:
                            self.label_coords = {}
                    else:
                        # .npz 파일이 없는 경우: user_type에 따라 처리
                        user_type = getattr(self, "user_type", "personal")  # 기본값: personal
                        if user_type == "personal":
                            # personal: DB에서 가져오지 않고 빈 .npz 파일 생성
                            logging.info("[Cache] .npz 파일이 없고 user_type=personal → 빈 .npz 파일 생성")
                            try:
                                image_cd = getattr(self, "image_cd", None)
                                from services.resampling_cache import save_cache, make_cache_key
                                import time
                                import json
                                
                                # 빈 딕셔너리로 .npz 파일 생성
                                empty_splib_raw = {}
                                empty_splib_cr = {}
                                empty_label_raw = {}
                                empty_label_cr = {}
                                
                                # 클래스 메타데이터 수집 (빈 dict이므로 빈 메타데이터)
                                classes_meta = self._collect_class_metadata_for_cache(
                                    empty_label_raw, empty_label_cr, primary_path=primary
                                )
                                
                                save_cache(
                                    primary, cfg,
                                    empty_splib_raw,
                                    empty_splib_cr,
                                    empty_label_raw,
                                    empty_label_cr,
                                    {},
                                    meta_extra={"mode": "empty_init", "user_type": "personal"},
                                    classes_metadata=classes_meta,
                                )
                                logging.info("[Cache] 빈 .npz 파일 생성 완료: %s", npz_p)
                            except Exception:
                                logging.exception("[Cache] 빈 .npz 파일 생성 실패")
                        else:
                            # personal이 아님: DB에서 가져와서 .npz 파일 생성
                            logging.info("[Cache] .npz 파일이 없고 user_type=%s → DB에서 가져와서 .npz 파일 생성", user_type)
                            try:
                                from services.resampling_cache import save_cache, make_cache_key
                                from data.db import search_material_filtering_list
                                import os
                                
                                # DB에서 라벨링 데이터 가져오기
                                db_splib_raw = {}
                                db_splib_cr = {}
                                db_label_raw = {}
                                db_label_cr = {}
                                
                                # image_cd가 있으면 DB에서 라벨링 데이터 조회
                                image_cd = getattr(self, "image_cd", None)
                                if image_cd is not None:
                                    try:
                                        base_url = os.getenv('base_url')
                                        if base_url:
                                            # TODO: 실제 DB 조회 로직 구현 필요
                                            # 현재는 빈 딕셔너리로 생성 (DB 조회 API가 명확하지 않음)
                                            logging.warning("[Cache] DB 조회 로직이 구현되지 않았습니다. 빈 .npz 파일을 생성합니다.")
                                    except Exception:
                                        logging.exception("[Cache] DB 조회 실패")
                                
                                # 클래스 메타데이터 수집
                                classes_meta = self._collect_class_metadata_for_cache(
                                    db_label_raw, db_label_cr, primary_path=primary
                                )
                                
                                # DB 데이터로 .npz 파일 생성
                                save_cache(
                                    primary, cfg,
                                    db_splib_raw,
                                    db_splib_cr,
                                    db_label_raw,
                                    db_label_cr,
                                    {},
                                    meta_extra={"mode": "db_init", "user_type": user_type},
                                    classes_metadata=classes_meta,
                                )
                                logging.info("[Cache] DB 데이터로 .npz 파일 생성 완료: %s", npz_p)
                            except Exception:
                                logging.exception("[Cache] DB 데이터로 .npz 파일 생성 실패")
                self._cache_hydrated = True  # ★ 재병합 금지

            # --- 0-2) 메모리 우선 + 캐시 병합(중복 스펙트럼 제거) ---
            splib_raw = _merge_dict(mem_splib_raw, cache_splib_raw)
            splib_cr  = _merge_dict(mem_splib_cr,  cache_splib_cr)
            label_raw = _merge_dict(mem_label_raw, cache_label_raw)
            label_cr  = _merge_dict(mem_label_cr,  cache_label_cr)

            # self에 반영(정규화된 합본)
            self.splib_raw = splib_raw
            self.splib_cr  = splib_cr
            self.label_raw = label_raw
            self.label_cr  = label_cr

            # --- 1) 이번 등록 rows → 표준 포맷 ---
            # 무효 CID(-1, -9999) 필터링 + 좌표/스펙 NaN 방어는 norm_any_to_dict 안에서 처리
            logging.info("[LabelRegister] processing rows: count=%d, crC=%s, C=%s", len(rows), crC, C)
            add_raw = norm_any_to_dict(rows, expect_c=C,          cube=cube,             prefer_key="rfl", src_name="rows.raw")
            
            # cr_cube가 없어도 cube에서 스펙트럼을 추출하여 CR 적용
            if crC:
                # cr_cube가 있으면 그대로 사용
                add_cr = norm_any_to_dict(rows, expect_c=crC, cube=cr_cube, prefer_key="rfl", src_name="rows.cr")
            else:
                # cr_cube가 없으면 add_raw에서 스펙트럼을 가져와서 CR 적용
                add_cr = {}
                if add_raw:
                    wavelength = cfg.get("wavelength")
                    if wavelength is not None:
                        wavelength = np.asarray(wavelength, dtype=np.float32)
                        for cid, spec_arr in add_raw.items():
                            # spec_arr: (N, C) 형태
                            if spec_arr.ndim != 2 or spec_arr.shape[0] == 0:
                                continue
                            cr_specs = []
                            for i in range(spec_arr.shape[0]):
                                spec = spec_arr[i, :]  # (C,)
                                try:
                                    cr_spec, _ = perform_continuum_removal(spec, wavelength, mode='reflectance')
                                    cr_specs.append(cr_spec)
                                except Exception as e:
                                    logging.warning("[LabelRegister] CR failed for cid=%s sample[%d]: %s", cid, i, e)
                                    continue
                            if cr_specs:
                                add_cr[cid] = np.vstack(cr_specs).astype(np.float32)
                    else:
                        logging.warning("[LabelRegister] wavelength not found, cannot apply CR")
            
            logging.info("[LabelRegister] add_raw: %d classes, add_cr: %d classes (crC=%s)", 
                        len(add_raw) if isinstance(add_raw, dict) else 0,
                        len(add_cr) if isinstance(add_cr, dict) else 0, crC)

            # --- 2) 누적(append) : 중복 스펙트럼 제거하며 병합 ---
            self.label_raw = _merge_dict(self.label_raw, add_raw)
            if add_cr:
                self.label_cr = _merge_dict(self.label_cr, add_cr)
            
            # --- 2-1) 라벨링 데이터 좌표 정보 저장 (패치 이미지 표시용) ---
            # 좌표 정보를 별도로 저장: {cid: [(img_x, img_y, img_cd), ...]} - 배열 순서와 동일
            if not hasattr(self, 'label_coords'):
                self.label_coords = {}  # {cid: [(img_x, img_y, img_cd), ...]}
            
            # rows에서 좌표 정보 추출하여 저장 (add_raw와 동일한 순서로)
            for row in rows:
                cid = row.get("mtrl_cd") or row.get("cid")
                img_x = row.get("img_x")
                img_y = row.get("img_y")
                img_cd = row.get("img_cd")  # 이미지 코드 추가
                
                if cid is not None and img_x is not None and img_y is not None:
                    try:
                        cid = int(cid)
                        img_x = int(img_x)
                        img_y = int(img_y)
                        img_cd = int(img_cd) if img_cd is not None else None
                        
                        # 음수 CID는 제외
                        if cid < 0:
                            continue
                        
                        if cid not in self.label_coords:
                            self.label_coords[cid] = []
                        # 배열에 추가되는 순서대로 좌표 정보 저장 (img_cd 포함)
                        self.label_coords[cid].append((img_x, img_y, img_cd))
                    except (ValueError, TypeError):
                        pass  # 좌표 정보가 유효하지 않으면 스킵

            # --- 3) 재구성(스펙+라벨 → self.lib / 팔레트 보강 포함) ---
            # ★ 주의: 사용자 라벨링 데이터는 .npz에 저장됨 (LabelStore는 사용하지 않음)
            # self.label_raw, self.label_cr에 이미 추가된 데이터가 .npz에 저장됨
            # LabelStore는 더 이상 사용하지 않으므로 초기화하지 않음
            self.label_store = None

            # --- 3) 재구성 ---
            self._rebuild_lib()
            # --- 4) 저장(최종 상태만) ---
            if primary:
                # 클래스 메타데이터 수집
                classes_meta = self._collect_class_metadata_for_cache(
                    self.label_raw, self.label_cr, primary_path=primary
                )
                
                # 저장 전 상태 로그
                lr_c_before = len(self.label_raw or {})
                lc_c_before = len(self.label_cr  or {})
                logging.info("[LabelRegister] 저장 시도: primary=%s, label_raw=%d classes, label_cr=%d classes", 
                           primary, lr_c_before, lc_c_before)
                
                try:
                    save_cache(
                        primary, cfg,
                        self.splib_raw,  # dict[int->(N,C)]
                        self.splib_cr,
                        self.label_raw,
                        self.label_cr,
                        getattr(self, "label_coords", None),
                        meta_extra={"mode": "overwrite"},
                        classes_metadata=classes_meta,
                        write_info=False,
                    )
                    # 저장 성공 확인
                    import os
                    info_path, npz_path = resample_cache_paths(primary)
                    if os.path.isfile(npz_path):
                        logging.info("[LabelRegister] 저장 성공: %s", npz_path)
                    else:
                        logging.error("[LabelRegister] 저장 실패: .npz 파일이 생성되지 않았습니다. %s", npz_path)
                except Exception as e:
                    logging.exception("[LabelRegister] save_cache 호출 중 예외 발생: %s", e)
            else:
                logging.warning("[LabelRegister] primary_path가 None이어서 .npz 파일에 저장하지 않습니다. cfg=%s", 
                             {k: v for k, v in list(cfg.items())[:3]} if isinstance(cfg, dict) else cfg)

            # (선택) 상태 로그
            lr_c = len(self.label_raw or {})
            lc_c = len(self.label_cr  or {})
            lib_c = len(self.lib or {})
            logging.info("[LabelRegister] done: label_raw=%d, label_cr=%d, lib=%d", lr_c, lc_c, lib_c)

        except FormatError as e:
            logging.exception(e)
            QtWidgets.QMessageBox.critical(self, "라벨 포맷 오류", str(e))
            return
        except Exception as e:
            logging.exception("[LabelRegister] unexpected error")
            QtWidgets.QMessageBox.critical(self, "라벨 등록 오류", f"라벨 등록 중 오류가 발생했습니다.\n{e}")
            return

    def _handle_label_pick(self, y: int, x: int, cid: Optional[int]):
        try:
            # UserLabelingDialog 처리
            dlg_user = getattr(self, "_user_labeling_dialog", None)
            if dlg_user is not None:
                guess_cid = self._label_at(y, x)
                if guess_cid is None or guess_cid < 0:
                    guess_cid = int(cid) if cid is not None else -1
                dlg_user.add_pixel_label(int(y), int(x), int(guess_cid))
                dlg_user.plot_spectrum(int(y), int(x))  # ★

                try:
                    number = dlg_user._root.tableSelected.rowCount()  # 1부터 증가
                    if hasattr(self._map_view, "add_ctx_marker"):
                        self._map_view.add_ctx_marker("LABEL", int(y), int(x), int(number))
                except Exception:
                    logging.exception("[Label] add_ctx_marker failed")

            # ClassmapLabelingDialog 처리
            dlg_classmap = getattr(self, "_classmap_labeling_dialog", None)
            if dlg_classmap is not None:
                guess_cid = self._label_at(y, x)
                if guess_cid is None or guess_cid < 0:
                    guess_cid = int(cid) if cid is not None else -1
                dlg_classmap.add_pixel_label(int(y), int(x), int(guess_cid))
                self.statusBar().showMessage(f"라벨 추가: ({y},{x}) → class {guess_cid}", 1500)
                return

            # 모든 라벨링/분석 다이얼로그가 없거나 보이지 않을 때 Top-K 유사도 표시
            dlg_user_check = getattr(self, "_user_labeling_dialog", None)
            dlg_analysis = getattr(self, "_analysis_selection_dialog", None)
            dlg_classmap_check = getattr(self, "_classmap_labeling_dialog", None)
            
            # 다이얼로그가 없거나 보이지 않는지 확인
            user_dlg_visible = dlg_user_check is not None and dlg_user_check.isVisible() if dlg_user_check else False
            analysis_dlg_visible = dlg_analysis is not None and dlg_analysis.isVisible() if dlg_analysis else False
            classmap_dlg_visible = dlg_classmap_check is not None and dlg_classmap_check.isVisible() if dlg_classmap_check else False
            
            # 모든 다이얼로그가 없거나 보이지 않을 때 Top-K 유사도 표시
            if not (user_dlg_visible or analysis_dlg_visible or classmap_dlg_visible):
                try:
                    tkc = getattr(self, "_class_topk_cids", None)
                    tkv = getattr(self, "_class_topk_vals", None)
                    if tkc is not None and tkv is not None:
                        # top-k 유사도 다이얼로그 표시
                        QtCore.QTimer.singleShot(100, lambda: self._show_pixel_topk(y, x))
                        self.statusBar().showMessage(f"픽셀 ({y},{x})의 Top-K 유사도 표시", 1500)
                        return
                except Exception:
                    logging.exception("[Labeling] top-k 표시 실패")

            # 다이얼로그 없으면 기본 메시지
            # classmap이 있는 경우에만 메시지 표시 (라벨링 다이얼로그가 열려있지 않을 때)
            has_classmap = False
            try:
                classmap_names = self._cb_classmap_names()
                has_classmap = len(classmap_names) > 0
            except Exception:
                pass
            
            if has_classmap:
                # classmap이 있지만 다이얼로그가 열려있지 않은 경우, 메시지 숨김
                # (사용자가 직접 라벨링 다이얼로그를 열 때까지 불필요한 메시지 표시 방지)
                pass
            else:
                # classmap이 없는 경우에만 기본 메시지 표시
                self.statusBar().showMessage(f"Label pixel: ({y},{x}), class={cid}", 2000)

        except Exception:
            logging.exception("[Labeling] label pick failed")
            QtWidgets.QMessageBox.information(self, "안내", "라벨을 추가하지 못했습니다.")

    @QtCore.pyqtSlot(list)
    def _on_labeling_requested(self, labels: list):
        """라벨링 요청 처리"""
        try:
            # TODO: 실제 라벨링 로직 구현
            QtWidgets.QMessageBox.information(
                self, "라벨링 완료",
                f"픽셀 라벨링 요청:\n"
                f"총 {len(labels)}개 픽셀\n"
            )
            
            # 여기에 실제 라벨링 로직을 구현
            # self._apply_labels_to_classmap(labels)
            
        except Exception as e:
            logging.exception("[Labeling] request failed")
            QtWidgets.QMessageBox.critical(self, "오류", f"라벨링 중 오류가 발생했습니다: {e}")

    @QtCore.pyqtSlot()
    def on_user_labeling_dialog_run(self):
        """사용자 지정 라벨링 다이얼로그 실행"""
        try:
            if not hasattr(self, "rgb_image") or self.rgb_image is None:
                QtWidgets.QMessageBox.information(self, "안내", "먼저 이미지를 로드하세요.")
                return

            # 이미 열려있으면 재사용(보이기만 보장)
            if getattr(self, "_user_labeling_dialog", None):
                if not self._user_labeling_dialog.isVisible():
                    self._user_labeling_dialog.show()
                self._user_labeling_dialog.raise_()
                self._user_labeling_dialog.activateWindow()
                # ★ 이미 열려있어도 데이터가 없으면 set_data 호출 (이미지 로드 후 다이얼로그 재사용 시)
                if hasattr(self, "cfg") and self.cfg and "data" in self.cfg:
                    if hasattr(self._user_labeling_dialog, "_cube") and self._user_labeling_dialog._cube is None:
                        self._user_labeling_dialog.set_data(self.cfg["data"], self.cfg.get("wavelength"))
                return

            dlg = UserLabelingDialog(parent=self, ui_dir=self.app_dir / "ui")
            self._user_labeling_dialog = dlg  # 참조 보관

            # ---------- 클래스 옵션 정규화 시작 ----------
            def _normalize_class_options(items) -> list[tuple[int, str]]:
                """
                허용 입력:
                - [(cid, name), ...]
                - [{"mtrl_cd":..,"name"/"mtrl_nm"/"label":..}, ...]
                - {"cid":..,"mtrl_nm":..} 섞임
                - 잡스러운 값/헤더 문자열은 스킵
                출력: [(int cid, str name)] (CID 중복은 선등장 우선)
                """
                if items is None:
                    return []
                if isinstance(items, dict):
                    items = [items]

                out = []
                for rec in items:
                    cid = name = None
                    if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                        cid, name = rec[0], rec[1]
                    elif isinstance(rec, dict):
                        cid = rec.get("mtrl_cd") or rec.get("cid") or rec.get("class_id")
                        name = rec.get("name") or rec.get("mtrl_nm") or rec.get("label")
                    else:
                        # 문자열(헤더 등) 또는 기타형식은 스킵
                        continue

                    try:
                        cid_int = int(str(cid).strip())
                    except Exception:
                        continue
                    name_str = (str(name).strip() if name not in (None, "") else f"Class {cid_int}")
                    out.append((cid_int, name_str))

                # CID 중복 제거(선등장 우선)
                seen, dedup = set(), []
                for cid_int, name_str in out:
                    if cid_int in seen:
                        continue
                    seen.add(cid_int)
                    dedup.append((cid_int, name_str))
                return dedup

            # 1) 클래스 옵션 가져오기 (user_type에 따라 분기)
            user_type = getattr(self, "user_type", "server")  # 기본값: server
            class_options = []
            
            if user_type == "personal":
                # personal 사용자: .info 파일에서 클래스 정보 읽기
                try:
                    primary = self._cache_primary_path(self.cfg) if hasattr(self, "cfg") else None
                    if not primary:
                        primary = self._extract_src_path(self.cfg) if hasattr(self, "cfg") else None
                    
                    if primary:
                        from services.resampling_cache import load_classes_from_info
                        class_options = load_classes_from_info(primary)
                        if class_options:
                            logging.info(f"[UserLabeling] personal: .info 파일에서 {len(class_options)}개 클래스 로드 완료")
                        else:
                            logging.info("[UserLabeling] personal: .info 파일에서 클래스 정보가 없거나 비어있습니다.")
                except Exception as e:
                    logging.exception("[UserLabeling] failed to read classes from .info file")
                    class_options = []
            else:
                # server 사용자: API에서 가져오기 (실패/타임아웃 시 빈 리스트)
                api_base = os.getenv("material_code_name_url") or os.getenv("MATERIAL_API_BASE")
                raw = self._fetch_class_options_safe(api_base)  # 어떤 형식이든 올 수 있음
                class_options = _normalize_class_options(raw)

            # 2) 라이브러리에 있는 CID도 이름 매핑해서 보강(옵션)
            if getattr(self, "lib", None):
                class_ids = sorted(int(c) for c in self.lib.keys())
                # _last_id_to_name가 dict 또는 [{mtrl_cd, mtrl_nm}, ...]일 수 있음
                id_to_name = {}
                raw_map = getattr(self, "_last_id_to_name", {})
                if isinstance(raw_map, dict):
                    for k, v in raw_map.items():
                        try:
                            id_to_name[int(k)] = str(v)
                        except Exception:
                            pass
                elif isinstance(raw_map, (list, tuple)):
                    for rec in raw_map:
                        if isinstance(rec, dict) and ("mtrl_cd" in rec):
                            try:
                                id_to_name[int(rec.get("mtrl_cd"))] = str(rec.get("mtrl_nm", rec.get("name", "")))
                            except Exception:
                                pass
                lib_opts = [(cid, id_to_name.get(cid, str(cid))) for cid in class_ids]
                # API 결과와 합치되, 중복 CID는 기존(class_options)의 선등장 유지
                # class_options는 (cid, name) 또는 (cid, name, desc) 형태일 수 있으므로 안전하게 처리
                have = set()
                for opt in class_options:
                    if isinstance(opt, (list, tuple)) and len(opt) >= 1:
                        have.add(int(opt[0]))
                class_options.extend([(cid, nm) for cid, nm in lib_opts if cid not in have])

            # 3) 다이얼로그에 전달 (이제 튜플 리스트만 넘어감)
            dlg.set_class_options(class_options)
            # ---------- 클래스 옵션 정규화 끝 ----------

            # 데이터/팔레트
            if hasattr(self, "cfg") and self.cfg and "data" in self.cfg:
                dlg.set_data(self.cfg["data"], self.cfg.get("wavelength"))
            if hasattr(self, "class_palette_qcolor"):
                dlg.set_palette(self.class_palette_qcolor)
            else:
                dlg.set_palette({})

            # 신호 연결
            dlg.labeling_requested.connect(self._on_user_labeling_candidates_registered)
            if hasattr(dlg, "table_changed"):
                dlg.table_changed.connect(self._on_user_labeling_table_changed)

            # 라벨링 모드 진입
            self._enter_pixel_click_mode(ClickMode.LABEL, mute_roi=True)

            # 종료/정리
            def _on_user_labeling_closed(_code):
                try:
                    self._leave_pixel_click_mode(restore=True)
                finally:
                    self._user_labeling_dialog = None
                    if hasattr(self._map_view, "clear_ctx_markers"):
                        self._map_view.clear_ctx_markers("LABEL")
                    # user_labeling_dialog가 닫힌 후 classmap이 있으면 INSPECT 모드로 설정
                    QtCore.QTimer.singleShot(10, lambda: self._sync_click_mode_with_analysis_state())

            dlg.finished.connect(_on_user_labeling_closed)

            # 모델리스로 표시
            dlg.setModal(False)
            dlg.setWindowModality(Qt.NonModal)
            dlg.show()

        except Exception as e:
            logging.exception("[User Labeling] dialog run failed")
            QtWidgets.QMessageBox.critical(self, "오류", f"사용자 지정 라벨링 다이얼로그 실행 중 오류가 발생했습니다: {e}")

            
    @QtCore.pyqtSlot(list)
    def _on_user_labeling_table_changed(self, rows: list):
        """
        사용자 지정 라벨링 테이블 변경 → Map의 LABEL 컨텍스트 마커 재구성
        rows: [{"y":..,"x":..,"cid":..}, ...] (현재 남아있는 순서대로)
        """
        try:
            # 1) 기존 LABEL 마커 비우기
            if hasattr(self._map_view, "clear_ctx_markers"):
                self._map_view.clear_ctx_markers("LABEL")
            # 2) 현재 rows를 순서대로 다시 마킹(번호 1부터 재부여)
            if hasattr(self._map_view, "add_ctx_marker"):
                for i, rec in enumerate(rows, start=1):
                    y, x = int(rec["y"]), int(rec["x"])
                    self._map_view.add_ctx_marker("LABEL", y, x, i)
            # 상태 메시지
            self.statusBar().showMessage(f"라벨 마커 갱신: {len(rows)}개", 2000)
        except Exception:
            logging.exception("[UserLabeling] update label markers failed")

    @QtCore.pyqtSlot(list)
    def _on_user_labeling_requested(self, labels: list):
        """사용자 지정 라벨링 요청 처리"""
        try:
            QtWidgets.QMessageBox.information(
                self, "라벨링 완료",
                f"사용자 지정 라벨링 요청:\n"
                f"총 {len(labels)}개 픽셀\n"
            )
        except Exception as e:
            logging.exception("[User Labeling] request failed")
            QtWidgets.QMessageBox.critical(self, "오류", f"라벨링 중 오류가 발생했습니다: {e}")

    @QtCore.pyqtSlot(dict)
    def on_recommend_labeling_wizard_run(self, payload: dict):
        """추천 픽셀 라벨링 위저드 실행"""
        try:
            from views.dialogs.recommend_label_wizard import RecommendationWizard
            ui_path = self.app_dir / 'ui' / 'recommend_label_wizard.ui'
            dlg = RecommendationWizard(parent=self, ui_path = ui_path)
            
            # 모달 다이얼로그로 표시
            dlg.setModal(True)
            dlg.exec_()
            
        except Exception as e:
            logging.exception("[RecommendLabelWizard] dialog run failed")
            QtWidgets.QMessageBox.critical(self, "오류", f"추천 픽셀 라벨링 위저드 실행 중 오류가 발생했습니다: {e}")
    
    def on_classmap_labeling_dialog_run_from_button(self):
        """분류맵 기반 라벨링 다이얼로그 실행(재오픈 안전)."""
        # 이미지 로드 확인
        if not hasattr(self, "rgb_image") or self.rgb_image is None:
            QtWidgets.QMessageBox.information(self, "안내", "먼저 이미지를 로드하세요.")
            return

        # 이미 떠있는 다이얼로그 처리 (앞으로 가져오거나 포인터 정리)
        dlg = getattr(self, "_classmap_labeling_dialog", None)
        if dlg is not None:
            try:
                if dlg.isVisible():
                    dlg.raise_(); dlg.activateWindow()
                    return
            except Exception:
                pass
            try:
                dlg.deleteLater()
            except Exception:
                pass
            self._classmap_labeling_dialog = None

        # 다이얼로그 생성
        dlg = ClassmapLabelingDialog(parent=self, ui_dir=self.app_dir / "ui")
        
        # ★ user_labeling_dialog에서 선택한 기본 CID 가져오기
        default_cid = None
        if hasattr(self, "_user_labeling_dialog") and self._user_labeling_dialog is not None:
            try:
                # user_labeling_dialog의 전역 콤보박스에서 선택된 CID 가져오기
                if hasattr(self._user_labeling_dialog, "_get_global_cid"):
                    default_cid = self._user_labeling_dialog._get_global_cid()
                    if default_cid is not None and default_cid >= 0:
                        if hasattr(dlg, "set_default_cid"):
                            dlg.set_default_cid(default_cid)
                            import logging
                            logging.info(f"[ClassmapLabeling] user_labeling_dialog에서 기본 CID 설정: {default_cid}")
            except Exception as e:
                import logging
                logging.debug(f"[ClassmapLabeling] user_labeling_dialog에서 기본 CID 가져오기 실패: {e}")
        try:
            dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        except Exception:
            pass
        self._classmap_labeling_dialog = dlg

        # 종료시 포인터 정리
        def _release(_code=None):
            try:
                pass
            finally:
                self._classmap_labeling_dialog = None
        dlg.finished.connect(_release)
        dlg.destroyed.connect(lambda *_: setattr(self, "_classmap_labeling_dialog", None))

        # 후보 등록 → PixelLabelingDialog로 라우팅
        dlg.candidates_to_user_labeling.connect(self._on_user_labeling_candidates_registered)

        # === Class/Labeling 스펙 콜백 주입 ===
        def _class_library(cid: int):
            """스펙트럼 라이브러리에서 해당 class 스펙 반환(list of 1D)."""
            out = []
            try:
                lib = getattr(self, "lib", {}) or {}
                arr = lib.get(int(cid))
                if arr is None: return out
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    out.append(arr.astype(float, copy=False))
                elif arr.ndim == 2:
                    for i in range(arr.shape[0]):
                        out.append(arr[i, :].astype(float, copy=False))
            except Exception:
                logging.exception("[DialogCB] class_library failed")
            return out

        def _class_labeling(cid: int):
            """저장된 라벨링 데이터에서 해당 class 스펙 반환(list of 1D)."""
            out = []
            try:
                lab = getattr(self, "labeling_lib", {}) or {}
                arr = lab.get(int(cid))
                if arr is None: return out
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    out.append(arr.astype(float, copy=False))
                elif arr.ndim == 2:
                    for i in range(arr.shape[0]):
                        out.append(arr[i, :].astype(float, copy=False))
            except Exception:
                logging.exception("[DialogCB] class_labeling failed")
            return out

        dlg.set_class_library_callback(_class_library)
        dlg.set_class_labeling_callback(_class_labeling)

        # === 전역 캐시 주입 (HSI cube / id_to_name만) ===
        cube = self.cfg["data"] if hasattr(self, "cfg") else None
        id2n = getattr(self, "_last_id_to_name", {}) or {}
        dlg.set_global_caches(cube=cube, classmap=None, topk_cids=None, topk_vals=None, id_to_name=id2n)

        # === classmap 목록/데이터/TopK 콜백 주입 ===
        dlg.set_classmap_options(self._cb_classmap_names())
        dlg.set_classmap_data_callback(self._cb_classmap_data)
        dlg._get_topk_for_map = self._cb_topk_for_map            # 다이얼로그 내부에서 호출
        dlg.set_class_pixels_callback(self._cb_class_pixels)      # 하위호환 세터
        dlg.set_bin_spectra_callback(lambda name, cid, lo, hi, n: self._bin_spectra_by_range(name, cid, lo, hi, n))
        dlg.set_spectrum_samples_callback(self._cb_spectra_for)   # (spec,(y,x)) 리스트

        # === 맵 의미(거리/유사도) ===
        dlg.set_map_semantics_callback(self._is_distance_map)

        # === 팔레트/클래스 옵션 ===
        id_to_name_raw = getattr(self, "_last_id_to_name", {}) or {}
        id_to_name = id_to_name_raw if isinstance(id_to_name_raw, dict) else {}

        options = []
        for k, v in id_to_name.items():
            try:
                cid = int(k)         # 키가 str 이여도 int 로 변환
            except Exception:
                continue
            options.append((cid, str(v)))   # (클래스ID, 물질명)

        # 클래스 ID 기준 정렬
        options.sort(key=lambda x: x[0])

        dlg.set_class_options(options)
        dlg.set_palette(getattr(self, "class_palette_qcolor", {}) or {})

        # 모델리스 표시
        dlg.setModal(False)
        dlg.setWindowModality(QtCore.Qt.NonModal)
        dlg.show()
        
    @QtCore.pyqtSlot()
    def on_classmap_labeling_dialog_run(self):
        """라벨링 데이터베이스 탐색 다이얼로그 실행"""
        try:
            from views.dialogs.search_labeling_database import SearchLabelingDatabaseDialog
            
            # 이미 SearchLabelingDatabaseDialog가 열려있으면 무시
            if hasattr(self, "_search_labeling_dialog") and self._search_labeling_dialog is not None:
                return
            
            dlg = SearchLabelingDatabaseDialog(parent=self, ui_dir=self.app_dir / "ui")
            self._search_labeling_dialog = dlg  # 참조 보관

            # --- 종료 시 정리 ---
            def _on_search_database_closed(_code):
                try:
                    pass
                finally:
                    self._search_labeling_dialog = None

            dlg.finished.connect(_on_search_database_closed)

            # --- 모델리스로 띄우기 ---
            dlg.setModal(False)
            dlg.setWindowModality(Qt.NonModal)
            dlg.show()

        except Exception as e:
            logging.exception("[Search Labeling Database] dialog run failed")
            QtWidgets.QMessageBox.critical(self, "오류", f"라벨링 데이터베이스 탐색 다이얼로그 실행 중 오류가 발생했습니다: {e}")

    @QtCore.pyqtSlot(list)
    def _on_classmap_labeling_requested(self, labels: list):
        """분류 맵 기반 라벨링 요청 처리"""
        try:
            QtWidgets.QMessageBox.information(
                self, "라벨링 완료",
                f"분류 맵 기반 라벨링 요청:\n"
                f"총 {len(labels)}개 픽셀\n"
            )
        except Exception as e:
            logging.exception("[Classmap Labeling] request failed")
            QtWidgets.QMessageBox.critical(self, "오류", f"라벨링 중 오류가 발생했습니다: {e}")
   
    def _handle_seed_pick(self, y: int, x: int, cid: Optional[int]):
        try:
            dlg = getattr(self, "_diff_dialog", None)
            if dlg is None:
                return

            # 0) 기준 맵 선택/확정 여부 확인
            map_name = dlg.get_selected_classmap_name()
            if not map_name or not dlg.is_base_locked():
                # 원하는 안내 방식: 상태바/메시지 박스/토스트(사용중인 위젯에 맞추세요)
                self.statusBar().showMessage("classmap을 선택해 주세요 (기준 맵 선택 후 '다음'을 누르세요)", 3000)
                return

            # 1) workspace(ROI) 내 클릭인지 확인(고정 워크스페이스 유지)
            m = getattr(self, "_work_mask", None)
            H, W = int(getattr(self._map_view, "img_h", 0) or 0), int(getattr(self._map_view, "img_w", 0) or 0)
            if not (0 <= y < H and 0 <= x < W):
                return
            if m is not None:
                try:
                    if not bool(np.asarray(m, dtype=bool)[y, x]):
                        self.statusBar().showMessage("작업 영역(ROI) 밖은 선택할 수 없습니다.", 2500)
                        return
                except Exception:
                    pass

            # 2) 선택된 기준 classmap에서 CID 읽기
            cm = self.layer_manager.get_classmap(map_name)
            guess_cid = None
            if isinstance(cm, np.ndarray) and cm.ndim == 2 and (0 <= y < cm.shape[0]) and (0 <= x < cm.shape[1]):
                try: guess_cid = int(cm[y, x])
                except Exception: guess_cid = None

            if guess_cid is None or guess_cid < 0:
                # fallback
                guess_cid = self._label_at(y, x)
                if guess_cid is None or guess_cid < 0:
                    guess_cid = int(cid) if cid is not None else -1

            # 3) 다이얼로그에 시드 추가
            dlg.add_pixel_seed(int(y), int(x), int(guess_cid))

            # 4) 맵 마커 갱신(선택)
            try:
                number = dlg.tbl.rowCount()
                if hasattr(self._map_view, "add_seed_marker"):
                    self._map_view.add_seed_marker(int(y), int(x), int(number),
                                                color=self.class_palette_qcolor.get(int(guess_cid)))
            except Exception:
                pass

            self.statusBar().showMessage(f"시드 추가: ({y},{x}) → class {guess_cid} @ {map_name}", 1500)

        except Exception:
            logging.exception("[Diffusion] seed pick failed")


    # --- (3) '선택 픽셀 상세분석' 신호 수신(임시 스텁)
    @QtCore.pyqtSlot(list)
    def _on_diffusion_analyze_requested(self, seeds: list):
        """
        seeds: [{y:int, x:int, cid:int}, ...]
        TODO: 어떤 정보를 표출할지 확정되면 여기서 구현.
        현재는 개수/일부 샘플만 메시지로 확인.
        """
        try:
            msg = [f"[선택 픽셀 상세분석] 총 {len(seeds)}개"]
            for i, s in enumerate(seeds[:5]):
                msg.append(f"  - #{i+1}: (y={s['y']}, x={s['x']}), cid={s['cid']}")
            QtWidgets.QMessageBox.information(self, "선택 픽셀 상세분석", "\n".join(msg))
        except Exception:
            logging.exception("[Diffusion] analyze stub failed")
           
    @QtCore.pyqtSlot(dict)
    def _on_diffusion_requested(self, params: dict):
        """
        params = {
        "tau": float, "delta": float, "map_name": str,
        "pixel_seeds": [{"y","x","cid"}, ...],
        "classmap_name": str
        }
        """
        try:
            tau   = float(params.get("tau", 0.05))
            delta = float(params.get("delta", 0.03))
            name  = str(params.get("map_name") or "diffusion1")
            seeds = params.get("pixel_seeds") or []
            base_map_name = params.get("classmap_name") or "classification"

            if not seeds:
                QtWidgets.QMessageBox.information(self, "안내", "시드를 먼저 선택하세요."); return

            cube = self.cfg["data"] if hasattr(self, "cfg") else None
            if cube is None:
                QtWidgets.QMessageBox.warning(self, "경고", "HSI 이미지가 로드되지 않았습니다."); return

            H, W, _ = cube.shape
            if not hasattr(self, "region_growing_service") or self.region_growing_service is None:
                QtWidgets.QMessageBox.warning(self, "경고", "Region Growing 서비스가 없습니다."); return

            # ROI 제한 유지(Workspace 고정)
            self.region_growing_service.set_allowed_mask(getattr(self, "_work_mask", None))
            self.region_growing_service.set_config(metric="SAM",
                                                threshold=float(tau),
                                                margin=float(delta),
                                                min_region_size=1,
                                                max_region_size=H*W,
                                                connectivity=8)

            # 시드별 확산
            seed_regions: list[tuple[np.ndarray, int]] = []
            for s in seeds:
                y = int(s.get("y",-1)); x = int(s.get("x",-1)); cid = int(s.get("cid",-1))
                if not (0 <= y < H and 0 <= x < W): continue
                res = None
                try:
                    if "grow_region_from_click" in dir(self.region_growing_service):
                        res = self.region_growing_service.grow_region_from_click(x=x, y=y)
                except Exception:
                    import logging; logging.exception("[Diffusion] grow_region_from_click failed")
                    res = None

                if res is None or getattr(res, "region_mask", None) is None:
                    from core.region_growing import perform_region_growing
                    try:
                        rg_res = perform_region_growing(
                            hsi_data=cube,
                            seed_x=x, seed_y=y,
                            metric="SAM",
                            threshold=float(tau), margin=float(delta),
                            min_region_size=1, max_region_size=H*W,
                            connectivity=8,
                            allowed_mask=getattr(self, "_work_mask", None)
                        )
                        res = rg_res
                    except Exception:
                        import logging; logging.exception("[Diffusion] core fallback failed"); res = None

                if res is None or getattr(res, "region_mask", None) is None:
                    continue

                m = np.asarray(res.region_mask, dtype=bool)
                if np.any(m) and cid >= 0:
                    seed_regions.append((m, cid))

            if not seed_regions:
                QtWidgets.QMessageBox.information(self, "안내", "확산된 영역이 없습니다(임계/마진/ROI 확인)."); return

            # 클래스맵 합성
            from core.autoclass import UNKNOWN as _UNK
            classmap = np.full((H, W), _UNK, dtype=np.int32)
            for mask_i, cid_i in seed_regions:
                classmap[mask_i] = int(cid_i)

            # 팔레트 보강/등록
            used_cids = sorted({int(c) for _, c in seed_regions})
            self._augment_palette_for(used_cids, push_renderer=True, push_dock=True)

            # 결과 클래스맵 등록
            classmap_name = f"{name}.classmap"
            self._register_map_semantics(classmap_name, metric="SAM", distance=True)
            self.layer_manager.register_classmap(
                name=classmap_name,
                classmap_i32=classmap,
                class_ids=used_cids,
                visible=True,
                id_to_name=self._norm_id_to_name(),
            )

            # 선택 base map 기준의 distance/top1 캐시 작성(있으면 덮어씀)
            try:
                base_c1, base_v1 = self._cb_topk_for_map(base_map_name)
                if isinstance(base_c1, np.ndarray) and isinstance(base_v1, np.ndarray):
                    # 동일 해상도 가정. 필요한 경우 ROI 확장 유틸 사용
                    self._topk_cache[classmap_name] = (base_c1.astype(np.int32, copy=False)[..., :1],
                                                    base_v1.astype(np.float32, copy=False)[..., :1])
            except Exception:
                import logging; logging.exception("[TopK] cache link failed")

            area = int(np.count_nonzero(classmap >= 0))
            self.statusBar().showMessage(
                f"[확산] '{name}' 완료 — seeds={len(seeds)}, classes={used_cids}, area={area}px (τ={tau}, δ={delta}, base={base_map_name})",
                4000
            )

            # 클릭 모드 해제
            try:
                self._safe_set_click_mode(ClickMode.NONE)
            except Exception:
                pass

        except Exception as e:
            import logging; logging.exception("[Diffusion] request failed")
            QtWidgets.QMessageBox.critical(self, "오류", f"확산 맵 생성 중 오류가 발생했습니다: {e}")


        
    def _on_roi_rect_done(self, rect: QtCore.QRect, mask_obj):
        # ★ 분석 캡처 중이면 '작업 영역' 처리를 하지 않는다(조용히 무시).
        if getattr(self, "_roi_capture_target", "work") == "analysis":
            return  # 분석 영역은 AnalysisSelectionController가 처리하므로 여기서는 무시

        try:
            m = np.asarray(mask_obj).astype(bool)
            self._work_mask = m                               # ← 작업 영역 전용
            if hasattr(self.layer_manager, "set_roi"):
                self.layer_manager.set_roi(m)

            self.register_layer(
                "작업 영역", m, type="mask.roi", visible=True,
                meta={"color": (0, 170, 255, 120)}
            )

            clamped = self._clamp_rect_to_image(rect)
            self.current_selection_rect = clamped

            if hasattr(self, "clsDock") and self.clsDock and hasattr(self.clsDock, "set_roi_drawing_state"):
                self.clsDock.set_roi_drawing_state(False)

            self.statusBar().showMessage(
                f"ROI 지정: {int(m.sum())} px, 선택영역=({clamped.x()},{clamped.y()},{clamped.width()}x{clamped.height()})",
                3000
            )
        except Exception:
            logging.exception("ROI rect mask save failed")


    @QtCore.pyqtSlot(object, object)
    def _on_roi_free_done(self, poly_img, mask_obj):
        # ★ 작업 영역 전용: 분석 영역 캡처 중이면 무시
        if getattr(self, "_roi_capture_target", "work") == "analysis":
            return  # 분석 영역은 AnalysisSelectionController가 처리하므로 여기서는 무시
        
        try:
            m = np.asarray(mask_obj).astype(bool)
            self._work_mask = m  # ← 작업 영역 전용
            if hasattr(self.layer_manager, "set_roi"):
                self.layer_manager.set_roi(m)

            self.register_layer(
                "작업 영역", m, type="mask.roi", visible=True,
                meta={"color": (0, 170, 255, 120)}
            )

            # 자유형은 '선택 사각형' 개념이 모호하므로 비워두는 편이 안전
            self.current_selection_rect = None
            
            if hasattr(self, "clsDock") and self.clsDock and hasattr(self.clsDock, "set_roi_drawing_state"):
                self.clsDock.set_roi_drawing_state(False)

            self.statusBar().showMessage(
                f"작업 영역 지정: {int(m.sum())} px",
                3000
            )
        except Exception:
            logging.exception("ROI free mask save failed")


    def _ensure_roi_mask_shape(self, H: int, W: int) -> Optional[np.ndarray]:
        m = getattr(self, "_work_mask", None)  # ← 명확화
        if m is None: return None
        try:
            m = np.asarray(m, dtype=bool)
            return m if m.shape == (H, W) and m.any() else None
        except Exception:
            logging.exception("ROI mask shape check failed")
            return None
 
    @QtCore.pyqtSlot(str)
    def _on_open_workspace_ui(self, parent_name: str):
        """
        ★ '작업 영역 선택' Dialog를 열고, 사용자가 그린 ROI는 캐시에만 저장.
        '완료'를 누를 때만 실제 작업 영역으로 커밋한다.
        - image.rgb의 '+' 버튼을 눌렀을 때만 작동
        """
        try:
            # ★ image.rgb 레이어의 '+' 버튼을 눌렀을 때만 작동
            if parent_name != "image.rgb":
                return  # image.rgb가 아니면 무시
            
            if not self._ensure_roi_controller():
                QtWidgets.QMessageBox.information(self, "안내", "이미지를 먼저 로드하세요.")
                return

            # Dialog 생성/보장
            if not hasattr(self, "_workspace_dialog") or self._workspace_dialog is None:
                self._workspace_dialog = WorkspaceDialog(parent=self, ui_dir=self.app_dir / "ui")
                self._workspace_dialog.roi_mode_requested.connect(self._set_roi_mode_from_dialog)
                self._workspace_dialog.register_workspace_requested.connect(self._on_register_workspace)
                self._workspace_dialog.finished.connect(self._on_workspace_dialog_finished)

            # --- 중요: 캐시 초기화 ---
            self._workspace_temp_mask = None
            self._workspace_temp_rect = None

            # ROI는 다이얼로그 동안 '자동 등록 금지'
            if hasattr(self, "roi") and self.roi:
                self.roi.register_on_finish = False
                if hasattr(self.roi, "set_accept_events"):
                    self.roi.set_accept_events(True)

            # 임시 오버레이 핸들 보장
            if self._workspace_temp_overlay is None:
                self._workspace_temp_overlay = TempOverlayService(self._map_view)

            # Dialog 표시
            self._workspace_dialog.show()
            self._workspace_dialog.raise_()
            self._workspace_dialog.activateWindow()

        except Exception as e:
            logging.exception(e)
    
    @QtCore.pyqtSlot()
    def _on_register_workspace(self):
        """★ '작업 영역 선택' 다이얼로그에서 '완료' 버튼 → 캐시를 실제 작업 영역으로 커밋"""
        try:
            # 1) 우선 캐시 우선 사용
            temp_mask = getattr(self, "_workspace_temp_mask", None)
            temp_rect = getattr(self, "_workspace_temp_rect", None)

            if temp_mask is None:
                # 캐시가 없다면 ROIController의 마지막 항목으로 보조
                if not hasattr(self, "roi") or self.roi is None or not self.roi.list():
                    QtWidgets.QMessageBox.information(self, "안내", "등록할 작업 영역이 없습니다. 먼저 ROI를 그려주세요.")
                    return
                last = self.roi.list()[-1]
                temp_mask = np.asarray(last.mask, dtype=bool)
                temp_rect = getattr(last, "rect", None)

            # 2) 여기서만 실제 작업 영역 커밋
            self._work_mask = np.asarray(temp_mask, dtype=bool)

            if hasattr(self.layer_manager, "set_roi"):
                self.layer_manager.set_roi(self._work_mask)

            # RegionGrowing/Analysis 허용 마스크 동기화
            try:
                if hasattr(self, "region_growing_service") and self.region_growing_service:
                    self.region_growing_service.set_allowed_mask(self._work_mask)
            except Exception:
                logging.exception("[RegionGrowing] set_allowed_mask failed")

            if hasattr(self, "analysisCtrl") and self.analysisCtrl:
                try:
                    self.analysisCtrl.set_allowed_mask(self._work_mask)
                except Exception:
                    logging.exception("[AnalysisSelection] set_allowed_mask failed")

            # 레이어 등록
            self.register_layer("작업 영역", self._work_mask, type="mask.roi", visible=True,
                                meta={"color": (0, 170, 255, 120)})

            # 선택 사각형 보관
            if temp_rect is not None:
                self.current_selection_rect = temp_rect

            # 3) 임시 오버레이/캐시 정리
            if self._workspace_temp_overlay:
                self._workspace_temp_overlay.clear()
            self._workspace_temp_mask = None
            self._workspace_temp_rect = None

            # ROI 자동등록 복귀(필요 시)
            if hasattr(self, "roi") and self.roi:
                self.roi.register_on_finish = False  # Dialog 흐름 유지(다음도 캐시)

            self.statusBar().showMessage("작업 영역이 등록되었습니다.", 3000)

        except Exception as e:
            logging.exception("[Workspace] register failed")
            QtWidgets.QMessageBox.warning(self, "오류", f"작업 영역 등록 중 오류가 발생했습니다: {e}")
    
    @QtCore.pyqtSlot(int)
    def _on_workspace_dialog_finished(self, result: int):
        """★ Dialog 닫힐 때 임시 상태 정리 (취소 시 커밋 금지)"""
        try:
            # 취소면: 최근 ROI 도형 제거(시각 잔상 방지), 캐시/오버레이 정리
            if result == QtWidgets.QDialog.Rejected:
                if hasattr(self, "roi") and self.roi:
                    roi_list = self.roi.list()
                    if roi_list:
                        latest = roi_list[-1]
                        self.roi.remove(latest.id)
                if self._workspace_temp_overlay:
                    self._workspace_temp_overlay.clear()
                self._workspace_temp_mask = None
                self._workspace_temp_rect = None
                self.statusBar().showMessage("작업 영역 선택이 취소되었습니다.", 2000)
            else:
                # (완료 버튼의 커밋은 _on_register_workspace에서 이미 수행)
                if self._workspace_temp_overlay:
                    self._workspace_temp_overlay.clear()
                self._workspace_temp_mask = None
                self._workspace_temp_rect = None

        except Exception as e:
            logging.exception("[Workspace] dialog finished cleanup failed")

    def _ensure_roi_controller(self) -> bool:
        """★ ROIController를 지연 생성. 이미지 없으면 False."""
        if getattr(self, "roi", None) is not None:
            return True
        # 이미지 없으면 생성 불가
        if not hasattr(self, "rgb_image") or self.rgb_image is None:
            return False
        try:
            self.roi = ROIController(
                map_view=self._map_view,
                layer_register=self.register_layer,
                img_shape_fn=lambda: (self._map_view.img_h, self._map_view.img_w),  # ★ 뷰의 실제 크기 사용
                parent=self,
            )
            
            try:
                self.roi.roiAdded.disconnect()
                
            except Exception:
                pass
            
            self.roi.roiAdded.connect(self._on_work_roi_added)
                
            return True
        
        except Exception as e:
            logging.exception(e)
            return False

    def _set_roi_mode_from_dialog(self, which: str):
        """★ WorkspaceDialog에서 호출되는 ROI 모드 설정 메서드"""
        if which not in ('rect', 'poly', 'none'):
            return

        # 작업 영역 모드로 명시 설정
        self._roi_capture_target = "work"
        self._switch_roi_owner(ROIInputOwner.WORK)

        # 클래스/디퓨전 입력 확실히 음소거
        if getattr(self, "class_roi_controller", None):
            self.class_roi_controller.set_mode(None, -1)
            if hasattr(self.class_roi_controller, "set_accept_events"):
                self.class_roi_controller.set_accept_events(False)
        if getattr(self, "diffusion_roi_controller", None):
            self.diffusion_roi_controller.set_mode(None)
            if hasattr(self.diffusion_roi_controller, "set_accept_events"):
                self.diffusion_roi_controller.set_accept_events(False)

        # ROI 진입 시 클릭모드 OFF
        self._safe_set_click_mode(ClickMode.NONE)

        if not self._ensure_roi_controller():
            QtWidgets.QMessageBox.information(self, "안내", "이미지를 먼저 로드하세요.")
            if hasattr(self, "_workspace_dialog") and self._workspace_dialog:
                self._workspace_dialog.reset_shape_selection()
            return

        if which == 'none':
            self.roi.set_mode(None)
            self.statusBar().clearMessage()
            # 기본 클릭 복귀 (분석 모드 상태 확인)
            self._sync_click_mode_with_analysis_state()
            if hasattr(self, "_workspace_dialog") and self._workspace_dialog:
                self._workspace_dialog.reset_shape_selection()
            return

        if which == 'rect':
            self.roi.set_mode('rect')
            self.statusBar().showMessage("사각형 ROI: 드래그→놓기.", 3000)
        elif which == 'poly':
            self.roi.set_mode('poly')
            self.statusBar().showMessage("다각형/자유형 ROI: 클릭/드래그, 더블클릭 완료.", 4000)
            
    def _set_roi_mode_from_toolbar(self, which: str):
        # ★ 작업 영역 전용: 좌측 상단 툴바에서 호출 (레거시, Dialog 사용으로 대체)
        if which not in ('rect', 'poly', 'none'):
            return

        # 작업 영역 모드로 명시 설정
        self._roi_capture_target = "work"
        self._switch_roi_owner(ROIInputOwner.WORK)

        # 클래스/디퓨전 입력 확실히 음소거
        if getattr(self, "class_roi_controller", None):
            self.class_roi_controller.set_mode(None, -1)
            if hasattr(self.class_roi_controller, "set_accept_events"):
                self.class_roi_controller.set_accept_events(False)
        if getattr(self, "diffusion_roi_controller", None):
            self.diffusion_roi_controller.set_mode(None)
            if hasattr(self.diffusion_roi_controller, "set_accept_events"):
                self.diffusion_roi_controller.set_accept_events(False)

        # ROI 진입 시 클릭모드 OFF
        self._safe_set_click_mode(ClickMode.NONE)

        # 작업영역 ROI 모드 적용 (툴바 액션 체크)
        if hasattr(self, "act_roi_rect") and hasattr(self, "act_roi_poly"):
            self.act_roi_rect.setChecked(which == 'rect')
            self.act_roi_poly.setChecked(which == 'poly')

        if not self._ensure_roi_controller():
            QtWidgets.QMessageBox.information(self, "안내", "이미지를 먼저 로드하세요.")
            if hasattr(self, "act_roi_rect") and hasattr(self, "act_roi_poly"):
                self.act_roi_rect.setChecked(False)
                self.act_roi_poly.setChecked(False)
            return

        if which == 'none':
            self.roi.set_mode(None)
            self.statusBar().clearMessage()
            # 기본 클릭 복귀 (분석 모드 상태 확인)
            self._sync_click_mode_with_analysis_state()
            return

        if which == 'rect':
            self.roi.set_mode('rect')
            self.statusBar().showMessage("사각형 ROI: 드래그→놓기.", 3000)
        elif which == 'poly':
            self.roi.set_mode('poly')
            self.statusBar().showMessage("다각형/자유형 ROI: 클릭/드래그, 더블클릭 완료.", 4000)
            
    @QtCore.pyqtSlot(str)
    def _on_save_layer_req(self, name: str):
        """
        LayersDock 우클릭 '저장…' → .npy 저장
        (데이터 소스는 오직 LayerManager API에서만 가져온다)
        - '작업 영역'         : roi = layer_manager.get_roi() → bool(H,W)
        - 'parent/cid'        : mask = layer_manager.get_mask(parent, cid) → bool(H,W)
        - 'parent'(top-level) : cm   = layer_manager.get_classmap(parent)  → int32(H,W)
        """
        try:
            # ROI 저장
            if name == "작업 영역":
                roi = getattr(self.layer_manager, "get_roi", lambda: None)()
                if roi is None:
                    QtWidgets.QMessageBox.information(self, "안내", "저장할 작업 영역(ROI)이 없습니다.")
                    return
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, "ROI 저장", str(self._last_open_dir / "roi.npz"), "NumPy Zip (*.npz)"
                )
                if not path:
                    return
                self._save_npz_with_meta(path, data=roi.astype(np.bool_), kind="mask")
                self.statusBar().showMessage(f"저장됨: {path}", 3000)
                return

            # 자식(class) 저장: "parent/cid"
            if "/" in name:
                parent, cid_str = name.split("/", 1)
                try:
                    cid = int(cid_str)
                except Exception:
                    QtWidgets.QMessageBox.warning(self, "경고", f"잘못된 클래스 ID: {cid_str}")
                    return

                get_mask = getattr(self.layer_manager, "get_mask", None)
                if not callable(get_mask):
                    QtWidgets.QMessageBox.critical(self, "오류", "LayerManager.get_mask 가 없습니다.")
                    return
                mask = get_mask(parent, cid)  # 기대: bool(H,W)
                if mask is None:
                    QtWidgets.QMessageBox.warning(self, "경고", f"'{parent}/{cid}' 마스크를 찾을 수 없습니다.")
                    return

                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, "마스크 저장", str(self._last_open_dir / "mask.npz"), "NumPy Zip (*.npz)"
                )
                if not path:
                    return
                self._save_npz_with_meta(path, data=mask.astype(np.bool_), kind="mask")
                self.statusBar().showMessage(f"저장됨: {path}", 3000)
                return

            # 부모(top-level) 분류맵 저장
            get_cm = getattr(self.layer_manager, "get_classmap", None)
            if not callable(get_cm):
                QtWidgets.QMessageBox.critical(self, "오류", "LayerManager.get_classmap 가 없습니다.")
                return

            cm = get_cm(name)  # 기대: int32(H,W)
            if cm is None:
                QtWidgets.QMessageBox.warning(self, "경고", f"'{name}' 분류맵을 찾을 수 없습니다.")
                return

            H, W = int(self._map_view.img_h), int(self._map_view.img_w)
            cm2d = cm
            if cm.ndim == 1 and cm.size == H*W:
                cm2d = cm.reshape(H, W)
            if cm2d.ndim != 2 or cm2d.shape != (H, W):
                QtWidgets.QMessageBox.warning(self, "경고", f"분류맵 크기 불일치: {cm2d.shape} != {(H, W)}")
                return

            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "분류맵 저장", str(self._last_open_dir / f"{name}.npz"), "NumPy Zip (*.npz)"
            )
            if not path: return
            self._save_npz_with_meta(path, data=cm2d.astype(np.int32, copy=False), kind="classmap")
            self.statusBar().showMessage(f"저장됨: {path}", 3000)

        except Exception:
            logging.exception("[LayersDock] 저장 실패")
            QtWidgets.QMessageBox.critical(self, "오류", "저장 중 오류가 발생했습니다.")

    @QtCore.pyqtSlot(list)
    def _on_load_files_req(self, paths: list):
        """
        LayersDock에 .npz 파일 드롭 → 자동 로드
        - .npz: image_code 검증 필수 (self._load_npz_with_check 사용)
        kind: 'classmap' | 'roi' | 'mask'
        """
        try:
            # 이미지가 아직 안 올라온 경우 가드
            if not getattr(self._map_view, "img_h", None) or not getattr(self._map_view, "img_w", None):
                QtWidgets.QMessageBox.information(self, "안내", "먼저 이미지를 로드하세요.")
                return

            H, W = int(self._map_view.img_h), int(self._map_view.img_w)

            for p in paths:
                # 1) 확장자 검사
                if Path(p).suffix.lower() != ".npz":
                    QtWidgets.QMessageBox.warning(self, "경고", f"{p}: .npz만 지원합니다. (image_code 검증 필요)")
                    continue

                # 2) image_code 검증 + kind/data 로드
                try:
                    kind, arr = self._load_npz_with_check(p)  # ← 여기서 파일의 image_code == self.image_cd 검증
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "경고", f"{p}: 로드 거부 — {e}")
                    continue

                # 3) 1D → 2D 복원
                if arr.ndim == 1 and arr.size == H * W:
                    arr = arr.reshape(H, W)

                # 4) 2D 크기 검증
                if arr.ndim != 2 or arr.shape != (H, W):
                    QtWidgets.QMessageBox.warning(self, "경고", f"{p}: 2D 크기 불일치 {arr.shape} != {(H, W)}")
                    continue

                name = Path(p).stem

                # 5) kind 분기
                if kind == "classmap":
                    if not np.issubdtype(arr.dtype, np.integer):
                        QtWidgets.QMessageBox.warning(self, "경고", f"{p}: classmap은 정수형이어야 합니다.")
                        continue
                    cm = arr.astype(np.int32, copy=False)
                    present = sorted(int(i) for i in np.unique(cm) if i >= 0)  # UNKNOWN(-1) 제외
                    # ★ 팔레트에 없는 클래스 자동 추가 (모든 클래스가 레이어로 표시되도록)
                    self._augment_palette_for(present, push_renderer=True, push_dock=True)
                    self.layer_manager.register_classmap(
                        name=name,
                        classmap_i32=cm,
                        class_ids=present,
                        visible=True,
                        id_to_name=self._norm_id_to_name(),
                    )
                    self.statusBar().showMessage(f"분류맵 로드: {p}", 3000)
                    continue

                if kind in ("roi", "mask"):
                    mask = (arr.astype(bool) if arr.dtype != np.bool_ else arr)
                    if not np.any(mask):
                        QtWidgets.QMessageBox.information(self, "안내", f"{p}: 빈 {kind}입니다.")
                        continue

                    # ROI는 내부 상태에도 저장(저장 메뉴에서 사용)
                    if kind == "roi" and hasattr(self.layer_manager, "set_roi"):
                        self.layer_manager.set_roi(mask)

                    # 시각화 색상
                    r, g, b, a = (255, 255, 255, 180) if kind == "mask" else (0, 170, 255, 120)
                    rgba = np.zeros((H, W, 4), dtype=np.uint8)
                    rgba[mask, 0] = r; rgba[mask, 1] = g; rgba[mask, 2] = b; rgba[mask, 3] = a

                    self.layer_manager.register_overlay(
                        name=name, rgba_or_rgb_u8=rgba, visible=True, parent=None
                    )
                    self.statusBar().showMessage(f"{kind.upper()} 로드: {p}", 3000)
                    continue

                # 정의되지 않은 kind
                QtWidgets.QMessageBox.warning(self, "경고", f"{p}: 알 수 없는 kind='{kind}'")
        except Exception:
            logging.exception("[LayersDock] 파일 로드 실패")
            QtWidgets.QMessageBox.critical(self, "오류", "파일 로드 중 오류가 발생했습니다.")

    def _current_image_code(self):
        ic = getattr(self, "image_cd", None)
        return str(ic) if ic is not None else None

    def _save_npz_with_meta(self, path: str, *, data: np.ndarray, kind: str) -> None:
        """
        .npz로 저장: data + image_code + kind + shape
        kind: 'classmap' | 'roi' | 'mask'
        """
        img_code = self._current_image_code()
        if img_code is None:
            QtWidgets.QMessageBox.warning(self, "경고", "현재 이미지의 image_code가 없습니다. 먼저 HSI를 로드하세요.")
            return
        h = int(data.shape[0]) if data.ndim >= 2 else None
        w = int(data.shape[1]) if data.ndim >= 2 else None
        c = int(data.shape[2]) if data.ndim == 3 else None
        np.savez(path, data=data, image_code=img_code, kind=kind, height=h, width=w, channels=c)

    def _load_npz_with_check(self, path: str) -> tuple[str, np.ndarray]:
        """
        .npz 열고 image_code 검증. ok면 (kind, data) 반환.
        """
        z = np.load(path, allow_pickle=False)
        if "data" not in z or "image_code" not in z or "kind" not in z:
            raise ValueError("필수 키(data, image_code, kind) 누락")
        arr  = np.asarray(z["data"])
        code = str(z["image_code"])
        kind = str(z["kind"])
        cur  = self._current_image_code()
        if cur is None:
            raise ValueError("현재 image_code가 없습니다. 먼저 HSI를 로드하세요.")
        if code != cur:
            raise ValueError(f"image_code 불일치: 파일={code}, 현재={cur}")
        return kind, arr

    def _init_recent_menu(self):
        """
        File 메뉴 아래 'Recent Files' 서브메뉴를 보장.
        UI에 항목이 있든 없든 동작.
        """
        menubar = self.menuBar()
        file_menu = None
        for a in menubar.actions():
            if a.text().replace("&", "").lower() in ("file", "파일"):
                file_menu = a.menu()
                break
        if file_menu is None:
            file_menu = menubar.addMenu("File")

        # 이미 있는 'Recent Files'를 재사용하거나 새로 만든다
        self._recent_menu = None
        for act in file_menu.actions():
            if act.text().replace("&", "").lower() in ("recent files", "최근 작업", "최근 파일"):
                self._recent_menu = act.menu() if act.menu() else None
                if self._recent_menu is None:
                    self._recent_menu = QtWidgets.QMenu("Recent Files", self)
                    idx = file_menu.actions().index(act)
                    file_menu.removeAction(act)
                    file_menu.insertMenu(file_menu.actions()[idx] if idx < len(file_menu.actions()) else None,
                                        self._recent_menu)
                break
        if self._recent_menu is None:
            self._recent_menu = file_menu.addMenu("Recent Files")

        # Clear 버튼
        self._act_clear_recent = QtWidgets.QAction("Clear List", self)
        self._act_clear_recent.triggered.connect(self._clear_recent)

    def _refresh_recent_menu(self):
        """QSettings의 최근 목록으로 Recent Files 메뉴를 다시 그림."""
        self._recent_menu.clear()
        items = self._settings.value("recent_hsi_files", [], type=list) or []
        items = [p for p in items if isinstance(p, str) and os.path.isfile(p)]

        if not items:
            a = QtWidgets.QAction("(Empty)", self)
            a.setEnabled(False)
            self._recent_menu.addAction(a)
        else:
            for path in items:
                act = QtWidgets.QAction(path, self)  # 경로 그대로 표시
                act.triggered.connect(lambda _, p=path: self._open_recent_hsi(p))
                self._recent_menu.addAction(act)

        self._recent_menu.addSeparator()
        self._recent_menu.addAction(self._act_clear_recent)
        
    def _recent_add(self, path: str, max_items: int = 10):
        if not (path and os.path.isfile(path)):
            return
        p = os.path.abspath(path)
        norm = lambda s: os.path.normcase(os.path.abspath(s))  # 윈도우 중복 방지
        items = self._settings.value("recent_hsi_files", [], type=list) or []
        items = [x for x in items if norm(x) != norm(p)]       # 대소문자/슬래시 차이 제거
        items.insert(0, p)
        self._settings.setValue("recent_hsi_files", items[:max_items])
        self._refresh_recent_menu()

    def _clear_recent(self):
        self._settings.setValue("recent_hsi_files", [])
        self._refresh_recent_menu()

    def _open_recent_hsi(self, path: str):
        """
        Recent Files에서 원천 HSI(.hdr/.mat)를 기존 ImageLoadDialog 로직으로 그대로 로드한다.
        (.npy/.npz는 classmap용이므로 여기서는 지원하지 않음)
        """
        try:
            p = Path(path)
            if not p.exists():
                QtWidgets.QMessageBox.warning(self, "경고", f"파일이 없습니다:\n{path}")
                return
            ext = p.suffix.lower()

            if ext not in (".hdr", ".mat"):
                QtWidgets.QMessageBox.information(self, "안내", "Recent는 .HDR / .mat만 지원합니다.")
                return

            # 1) 대화상자 인스턴스 (표시는 안 함)
            dlg = ImageLoadDialog(
                parent=self,
                ui_dir=getattr(self, "ui_dir", self.app_dir / "ui"),
                ui_filename="image_load.ui",
                last_dir=p.parent,
            )

            # 2) 경로/모드 채우기
            if ext == ".hdr":
                dlg.rad_hdr.setChecked(True)
                dlg.chk_mat.setChecked(False)
                dlg.le_hdr.setText(str(p))
                # raw 추정(.raw/.RAW)
                raw_guess = p.with_suffix(".raw")
                if not raw_guess.exists():
                    rg = p.with_suffix(".RAW")
                    if rg.exists():
                        raw_guess = rg
                if raw_guess.exists():
                    dlg.le_raw.setText(str(raw_guess))
            else:  # .mat
                dlg.chk_mat.setChecked(True)
                dlg.rad_hdr.setChecked(False)
                if dlg.le_mat:
                    dlg.le_mat.setText(str(p))

            # 3) 기존 “영상 확인” 로직 재사용(축 콤보 자동 채움 + .info 반영)
            dlg._on_check_clicked()

            # 4) 기존 “Load” 로직 그대로 호출 → dlg.result에 cfg 생성
            dlg._on_load_clicked()
            cfg = getattr(dlg, "result", None)
            if not cfg:
                QtWidgets.QMessageBox.warning(self, "경고", "최근 파일 로드 실패(구성 생성 안 됨).")
                return

            # 5) on_image_load 성공경로와 동일 파이프라인
            self._apply_loaded_hsi_cfg(cfg)

            # 6) 최근 목록 갱신(대표 경로: .mat은 mat, .hdr은 hdr)
            data_path = cfg.get("data_path") or {}
            primary = (data_path.get("mat") or data_path.get("hdr") or path)
            if isinstance(primary, (str, Path)) and os.path.isfile(str(primary)):
                self._recent_add(str(primary))

        except Exception as e:
            logging.exception(e)
            QtWidgets.QMessageBox.critical(self, "오류", f"최근 파일 로드 실패: {e}")
    
    def _collect_class_metadata_for_cache(
        self,
        label_raw: Any,
        label_cr: Any,
        primary_path: Optional[str] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        클래스 메타데이터 수집 (personal: .info에서, server: API 호출)
        
        Returns:
            Dict[int, Dict[str, Any]]: {cid: {"cid": int, "mtrl_nm": str, "desc": str}}
        """
        # 모든 cid 수집
        cids = set()
        
        if isinstance(label_raw, dict):
            cids.update(int(k) for k in label_raw.keys() if isinstance(k, (int, str)))
        if isinstance(label_cr, dict):
            cids.update(int(k) for k in label_cr.keys() if isinstance(k, (int, str)))
        
        if not cids:
            return {}
        
        # 기본값으로 초기화
        class_meta: Dict[int, Dict[str, Any]] = {
            cid: {
                "cid": int(cid),
                "mtrl_nm": f"Class {cid}",
                "desc": "",
            }
            for cid in cids
        }
        
        user_type = getattr(self, "user_type", "server")
        
        # personal 사용자: .info 파일에서 읽기
        if user_type == "personal" and primary_path:
            try:
                info_path, _ = resample_cache_paths(primary_path)
                if os.path.isfile(info_path):
                    with open(info_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    
                    # .info 파일의 classes 필드에서 메타데이터 읽기
                    classes_info = meta.get("classes", {})
                    if isinstance(classes_info, dict):
                        for cid_str, info in classes_info.items():
                            try:
                                cid_int = int(cid_str)
                                if cid_int in class_meta:
                                    class_meta[cid_int] = {
                                        "cid": cid_int,
                                        "mtrl_nm": str(info.get("mtrl_nm", f"Class {cid_int}")),
                                        "desc": str(info.get("desc", "")),
                                    }
                            except (TypeError, ValueError):
                                continue
            except Exception:
                logging.debug("[MainWindow] failed to read class metadata from .info file")
        
        # server 사용자: API 호출
        elif user_type == "server":
            try:
                api_base = os.getenv("material_filtering_url")
                if api_base:
                    from data.db import search_material_filtering_list
                    cid_list = sorted(list(cids))
                    api_result = search_material_filtering_list(
                        base_url=api_base,
                        mtrl_ids=cid_list,
                        timeout=5.0,
                    )
                    
                    # API 결과로 업데이트
                    for rec in api_result:
                        if not isinstance(rec, dict):
                            continue
                        cid = rec.get("mtrl_cd")
                        if cid is not None:
                            try:
                                cid_int = int(cid)
                                if cid_int in class_meta:
                                    class_meta[cid_int] = {
                                        "cid": cid_int,
                                        "mtrl_nm": str(rec.get("mtrl_nm", f"Class {cid_int}")),
                                        "desc": str(rec.get("desc", rec.get("dsc", ""))),
                                    }
                            except (TypeError, ValueError):
                                continue
            except Exception:
                # API 호출 실패 시 기본값 유지 (조용히 실패)
                logging.debug("[MainWindow] class metadata API call failed, using defaults")
        
        return class_meta
    
    @staticmethod
    def _append_inplace(dst: dict[int, np.ndarray],
                        add: dict[int, np.ndarray],
                        expect_c: Optional[int] = None) -> None:
        """dst에 add를 in-place로 누적(vstack). 채널 불일치면 스킵."""
        if not isinstance(dst, dict):
            raise TypeError("dst must be dict[int, ndarray]")
        for cid, block in (add or {}).items():
            a = np.asarray(block, dtype=np.float32)
            if a.ndim == 1: a = a[None, :]
            if a.ndim != 2:
                logging.warning(f"[Append] skip cid={cid}: ndim={a.ndim}")
                continue
            if expect_c and a.shape[1] != expect_c:
                logging.warning(f"[Append] skip cid={cid}: C={a.shape[1]} != {expect_c}")
                continue
            prev = dst.get(int(cid))
            if prev is None:
                dst[int(cid)] = a
            else:
                p = np.asarray(prev, dtype=np.float32)
                if p.ndim == 1: p = p[None, :]
                if p.shape[1] != a.shape[1]:
                    logging.warning(f"[Append] skip cid={cid}: prevC={p.shape[1]} vs newC={a.shape[1]}")
                    continue
                dst[int(cid)] = np.vstack([p, a]).astype(np.float32)

    def _apply_loaded_hsi_cfg(self, cfg: dict):
        """
        HSI 로드 이후 공용 파이프라인
        - RGB 구성 및 맵 반영
        - image_cd 발급/저장
        - resample 캐시 로드 또는 API 호출
        - self.splib_* / self.label_* 바인딩
        - LabelStore 초기화 및 집계(★)
        - _rebuild_lib()로 통합
        """
        try:
            # ===== 0) 기본 바인딩 & RGB 표시 =====
            self.cfg = cfg
            data = cfg.get("data")
            if data is None:
                QtWidgets.QMessageBox.warning(self, "경고", "HSI 데이터가 없습니다.")
                return
            
            # 사용자 타입 저장 (개인 사용자/서버 사용자)
            user_type = cfg.get("user_type", "personal")  # 기본값: 개인 사용자
            self.user_type = user_type  # MainWindow 인스턴스에 저장

            # wavelength 파싱
            wavelength = None
            wl_raw = cfg.get("wavelength_list") or cfg.get("wavelength")
            if wl_raw is not None:
                if isinstance(wl_raw, (list, tuple, np.ndarray)):
                    try:
                        wavelength = [float(x) for x in wl_raw]
                    except (ValueError, TypeError):
                        wavelength = None
                elif isinstance(wl_raw, str):
                    try:
                        if wl_raw.strip().startswith('['):
                            parsed = json.loads(wl_raw)
                            if isinstance(parsed, list):
                                wavelength = [float(x) for x in parsed]
                        else:
                            parts = [p.strip() for p in wl_raw.split(',') if p.strip()]
                            if parts:
                                wavelength = [float(p) for p in parts]
                    except (json.JSONDecodeError, ValueError, TypeError):
                        wavelength = None

            if wavelength is not None:
                C = data.shape[2] if data.ndim >= 3 else 0
                if len(wavelength) != C:
                    logging.warning(f"[RGB] wavelength 길이({len(wavelength)}) != C({C})")
                    wavelength = None

            rgb_image = make_rgb(data, wavelength=wavelength)
            self.rgb_image = rgb_image

            # MapView / Layer 등록
            self._map_view.set_rgb_image(rgb_image, do_fit=True)
            if hasattr(self.layer_manager, "register_rgb_base"):
                self.layer_manager.register_rgb_base(RGB_LAYER_NAME, rgb_image, visible=True, display_label=RGB_LAYER_DISPLAY_NAME)
                
                # ✅ 초기 로드 시 현재 RGB 이미지에 해당하는 R/G/B 밴드 정보를 LayersDock에 표시
                try:
                    sel_init = self._guess_view_bands_from_current()
                    if sel_init:
                        # QTimer를 사용하여 레이어 등록 후 viewer bands 표시
                        fmt_str = "{:.1f}" if wavelength else "{:.0f}"
                        def _show_initial_bands():
                            self.layersDock.set_viewer_bands_from_selection(
                                sel_init, fmt=fmt_str, attach_to="image.rgb")
                        QtCore.QTimer.singleShot(100, _show_initial_bands)
                except Exception:
                    logging.exception("[RGB] 초기 viewer bands 표시 실패")

            expect_c = int(np.asarray(data).shape[2])
            
            if user_type != 'personal':

                # ===== 1) image_cd 발급 및 서버 저장(옵션) =====
                try:
                    check_url = os.getenv('image_exist_url')
                    image_add_url = os.getenv('image_add_url')
                    image_save_path = os.getenv('image_save_path')
                    image_cd = image_check_and_save(
                        check_url=check_url,
                        save_url=image_add_url,
                        raw_image=data,
                        rgb_image=rgb_image,
                        save_path=image_save_path,
                        cmr_cd=cfg.get('camera_code')
                    )
                    self.image_cd = image_cd
                except Exception:
                    logging.exception("image_check_and_save failed")
            else:
                self.image_cd = 10

            # ===== 2) 원천 경로 결정 (캐시 파일 위치) =====
            primary = (self._pick_path_from_datapath(cfg.get('data_path'))
                    or self._extract_src_path(cfg))
            if (not primary) and isinstance(cfg.get('data_path'), dict):
                for v in cfg['data_path'].values():
                    if isinstance(v, (str, Path)):
                        ap = os.path.abspath(str(v))
                        if os.path.isfile(ap):
                            primary = ap
                            break

            # ===== 2-1) server 사용자 타입일 때 image_cd를 .info 파일에 저장 =====
            if user_type != 'personal' and primary and hasattr(self, 'image_cd') and self.image_cd:
                try:
                    primary_path = Path(primary)
                    existing_info = read_info(primary_path) or {}
                    existing_info['image_cd'] = self.image_cd
                    write_info(primary_path, existing_info)
                    logging.info(f"[MainWindow] image_cd를 .info 파일에 저장: {primary_path} -> {self.image_cd}")
                except Exception:
                    logging.exception(f"[MainWindow] .info 파일에 image_cd 저장 실패: {primary}")

            # ===== 3) resample 캐시 → API 리샘플링 =====
            spec_raw_dict: dict[int, np.ndarray] = {}
            spec_cr_dict:  dict[int, np.ndarray] = {}
            label_raw_dict: dict[int, np.ndarray] = {}
            label_cr_dict:  dict[int, np.ndarray] = {}

            # user_type 에 따라 resampling 수행 여부 결정
            do_resample = (user_type != "personal")  # personal이면 False, 그 외는 True

            # 공통적으로 쓸 메타 정보
            cache_meta: dict[str, Any] = {}
            elapsed: Optional[float] = None

            # ★ 모든 사용자(personal 포함)에서 .npz 캐시 파일 로드 시도
            cached = try_load_cache(primary, cfg) if primary else None
            if cached is not None:
                (
                    spectrum_library_raw,
                    spectrum_library_cr,
                    labeling_result_raw,
                    labeling_result_cr,
                    cached_label_coords,
                ) = cached
                spec_raw_dict  = spectrum_library_raw or {}
                spec_cr_dict   = spectrum_library_cr  or {}
                label_raw_dict = labeling_result_raw  or {}
                label_cr_dict  = labeling_result_cr   or {}
                if cached_label_coords:
                    self.label_coords = cached_label_coords
                cache_meta["source"] = "cache"
                self._info("[ResampleCache] cached result loaded")
                logging.info(
                    "[ResampleCache] loaded: splib_raw=%d, splib_cr=%d, label_raw=%d, label_cr=%d",
                    len(spec_raw_dict), len(spec_cr_dict),
                    len(label_raw_dict), len(label_cr_dict),
                )
            else:
                # 캐시가 없는 경우
                logging.info("[ResampleCache] 캐시 파일이 없습니다. primary=%s", primary)

            if do_resample:
                # ---- (A) server 사용자: 캐시가 없으면 API resampling ----
                if cached is None:
                    # 캐시가 없으면 API resampling 수행
                    try:
                        spectrum_library_url = os.getenv('spectrum_library_url')
                        labeling_url = os.getenv('label_list_url')
                        sel = {
                            'CMR_CD': cfg.get('camera_code'),
                            'CMR_NM': cfg.get('camera_name'),
                            'WV':     cfg.get('wavelength'),
                            'ST_WV':  min(cfg.get('wavelength')) if cfg.get('wavelength') is not None else None,
                            'ED_WV':  max(cfg.get('wavelength')) if cfg.get('wavelength') is not None else None,
                            'FWHM':   cfg.get('fwhm'),
                        }
                        t0 = time.perf_counter()
                        spectrum_library_raw, spectrum_library_cr, labeling_result_raw, labeling_result_cr = resampling(
                            sel, spectrum_library_url, labeling_url
                        )
                        elapsed = time.perf_counter() - t0
                        self._info(f"[Resampling] API+compute {elapsed:.2f}s")

                        # API에서 가져온 데이터로 업데이트 (캐시에 있던 데이터는 이미 로드됨)
                        spec_raw_dict  = spectrum_library_raw or {}
                        spec_cr_dict   = spectrum_library_cr  or {}
                        # label_raw, label_cr은 캐시에서 로드된 것이 있으면 유지, 없으면 API 결과 사용
                        if not label_raw_dict:
                            label_raw_dict = labeling_result_raw or {}
                        if not label_cr_dict:
                            label_cr_dict  = labeling_result_cr  or {}

                        cache_meta["source"] = "resample"
                        if elapsed is not None:
                            cache_meta["resample_elapsed_sec"] = elapsed

                        # 리샘플링 결과 캐시 저장(원천 경로가 있을 때만)
                        if primary:
                            try:
                                # 클래스 메타데이터 수집
                                classes_meta = self._collect_class_metadata_for_cache(
                                    label_raw_dict, label_cr_dict, primary_path=primary
                                )
                                save_cache(
                                    primary, cfg,
                                    spec_raw_dict, spec_cr_dict,
                                    label_raw_dict, label_cr_dict,
                                    getattr(self, "label_coords", None),
                                    meta_extra=cache_meta,
                                    classes_metadata=classes_meta,
                                )
                                logging.info("[ResampleCache] resampling 결과를 캐시에 저장했습니다.")
                            except Exception:
                                logging.exception("[ResampleCache] save_cache failed after resampling")
                    except Exception:
                        logging.exception("[Resampling] API/caching failed")
            else:
                # ---- (B) personal 사용자: API resampling은 스킵, 캐시만 로드/저장 ----
                self._info("[Resampling] personal 사용자: 캐시 파일에서 데이터 로드 (API resampling 없음)")
                logging.info("[Resampling] personal user: loaded from cache (no API resampling)")
                cache_meta["source"] = "personal_cache"
                
                # 캐시가 없으면 빈 캐시 파일 생성
                if cached is None and primary:
                    try:
                        # 클래스 메타데이터 수집 (빈 dict이므로 메타데이터도 빈 dict)
                        classes_meta = self._collect_class_metadata_for_cache(
                            label_raw_dict, label_cr_dict, primary_path=primary
                        )
                        save_cache(
                            primary, cfg,
                            spec_raw_dict, spec_cr_dict,
                            label_raw_dict, label_cr_dict,
                            getattr(self, "label_coords", None),
                            meta_extra=cache_meta,
                            classes_metadata=classes_meta,
                        )
                        logging.info("[ResampleCache] 빈 캐시 파일 생성 완료 (personal user)")
                    except Exception:
                        logging.exception("[ResampleCache] save_cache failed for personal user")
                elif cached is not None:
                    logging.info("[ResampleCache] personal 사용자: 기존 캐시 파일에서 데이터 로드 완료")


            # ===== 4) 결과 바인딩 =====
            self.splib_raw = spec_raw_dict
            self.splib_cr  = spec_cr_dict
            self.label_raw = label_raw_dict
            self.label_cr  = label_cr_dict

            # ===== 5) ★ LabelStore는 사용하지 않음 (.npz가 주 저장소) =====
            # 주의: 사용자 라벨링 데이터는 .npz에 저장되므로, LabelStore는 더 이상 사용하지 않음
            # .npz에서 읽은 label_raw, label_cr에 이미 사용자 라벨링 데이터가 포함되어 있음
            self.label_store = None

            # ===== 6) 한 번에 재구성(정규화/보간 포함) =====
            self._rebuild_lib()
            logging.info(
                "[Rebuild] lib=%d classes, labeling_lib=%d classes",
                len(getattr(self, "lib", {}) or {}),
                len(getattr(self, "labeling_lib", {}) or {})
            )

            # ===== 7) PixelClick 캐시 초기화 =====
            try:
                if hasattr(self, "pixelClick") and self.pixelClick:
                    self.pixelClick.clear_cache()
            except Exception:
                logging.exception("[PixelClick] clear_cache failed")

            # ===== 8) Recent 목록 갱신 =====
            try:
                primary3 = (self._pick_path_from_datapath(cfg.get('data_path'))
                            or self._extract_src_path(cfg))
                if isinstance(primary3, (str, Path)) and os.path.isfile(str(primary3)):
                    self._recent_add(str(primary3))
            except Exception:
                logging.exception("[Recent] add after image load failed")

            # ===== 9) 상태 메시지 =====
            self._info(
                f"Resampling 완료 — splib_raw:{len(self.splib_raw)}, "
                f"splib_cr:{len(self.splib_cr)}, label_raw:{len(self.label_raw)}, label_cr:{len(self.label_cr)}"
            )

        except Exception:
            logging.exception("[HSI] _apply_loaded_hsi_cfg failed")
            QtWidgets.QMessageBox.critical(self, "오류", "HSI 로드 후 처리 중 오류가 발생했습니다.")



    def _extract_src_path(self, cfg: dict) -> Optional[str]:
        """
        다이얼로그/호출자별 다른 키를 넓게 커버해 실제 파일 경로를 뽑아낸다.
        우선순위: src_path, path, file, filepath, filename, input_path, hsi_path, npz_path
        list/tuple이면 첫 요소 사용.
        """
        cand_keys = ("src_path","path","file","filepath","filename","input_path","hsi_path","npz_path")
        for k in cand_keys:
            p = cfg.get(k)
            if isinstance(p, (list, tuple)) and p:
                p = p[0]
            if isinstance(p, (str, Path)):
                p = os.path.abspath(str(p))
                if os.path.isfile(p):
                    return p
        return None

    def _pick_path_from_datapath(self, data_path) -> Optional[str]:
        """
        cfg['data_path']에서 실제 파일이 존재하는 대표 경로 하나를 고른다.
        우선순위: MAT -> HDR -> RAW -> 그 외 값들 순.
        """
        if not isinstance(data_path, dict):
            return None
        order = ("mat", "hdr", "raw")  # 필요 시 'npz','npy' 등 추가 가능
        # 1) 우선순위 키 먼저 확인
        for k in order:
            p = data_path.get(k)
            if isinstance(p, (str, Path)):
                ap = os.path.abspath(str(p))
                if os.path.isfile(ap):
                    return ap
        # 2) 나머지 키들도 스캔
        for _, v in data_path.items():
            if isinstance(v, (str, Path)):
                ap = os.path.abspath(str(v))
                if os.path.isfile(ap):
                    return ap
        return None
        
    def _init_pixel_classification_dock(self):
        """PixelClassificationDock 생성 및 시그널 배선(작업영역 vs 클래스 ROI 분리)."""
        self.clsDock = PixelClassificationDock(
            parent=self,
            ui_dir=getattr(self, "ui_dir", self.app_dir / "ui"),
        )
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.clsDock)

        # 닫기 기능 활성화 (사용자가 껐다 켰다 할 수 있도록)
        feats = self.clsDock.features()
        self.clsDock.setFeatures(feats | QtWidgets.QDockWidget.DockWidgetClosable)
        
        # View 메뉴에 토글 액션 추가 (껐다 켰다 할 수 있도록)
        self._add_dock_to_view_menu(self.clsDock, "Pixel Classification")

        # ── 공통 기능 연결 ─────────────────────────────────────────
        # self.clsDock.requestClassify.connect(self._on_pixel_classify)
        # self.clsDock.requestFocusClass.connect(self._on_focus_class_from_viewer)
        # self.clsDock.requestFocusClass.connect(
            # lambda cid: self.pixelClick.set_focus_cid(cid if cid >= 0 else None)
        # )
        self.clsDock.requestApplyThresholds.connect(self._apply_transparency_overlay)
        self.clsDock.requestApplyThresholdsForClass.connect(self._on_apply_thresholds_for_class)
        self.clsDock.requestResetOpacity.connect(self._clear_confidence_overlay_mapview)
        self.clsDock.requestVizClassChanged.connect(self._on_viz_class_changed)
        self.clsDock.requestApplyThresholdsForClass.connect(self._on_apply_thresholds_for_class)
        
        # Dock 보이기/숨김에 따른 클릭 모드 전환(있으면 유용)
        try:
            self.clsDock.visibilityChanged.connect(self._on_cls_dock_visibility)
        except Exception:
            pass

        self.clsDock.requestClassRectROIStart.connect(self._on_start_class_rect_roi)

        # Dock 내 기타 기능 유지
        # self.clsDock.requestViewSelectionDetails.connect(self._on_view_selection_details)
        self.clsDock.requestViewAnalysisDetails.connect(self._on_view_analysis_details_from_dock)
        
        # 분석 관련 시그널 연결 (dock에서 정의된 시그널만 사용)
        self.clsDock.requestAnalysisOpChanged.connect(self._set_analysis_op)
        self.clsDock.requestAnalysisRectROIStart.connect(self._on_start_analysis_rect_from_dock)
        self.clsDock.requestAnalysisPixelStart.connect(self._on_start_analysis_pixel)
        self.clsDock.requestAnalysisStop.connect(self._on_analysis_stop)
        try:
            self.clsDock.requestClearAnalysisRegion.connect(self._on_clear_analysis_region)  # ★ 추가
        except Exception:
            import logging
            logging.exception("[MainWindow] connect requestClearAnalysisRegion failed")
    
    def _add_dock_to_view_menu(self, dock: QtWidgets.QDockWidget, menu_text: str):
        """View 메뉴에 Dock 토글 액션 추가"""
        try:
            menubar = self.menuBar()
            view_menu = None
            
            # View 메뉴 찾기
            for a in menubar.actions():
                if a.text().replace("&", "").lower() in ("view", "보기"):
                    view_menu = a.menu()
                    break
            
            # View 메뉴가 없으면 생성
            if view_menu is None:
                view_menu = menubar.addMenu("View")
            
            # Dock의 토글 액션 가져오기 (QDockWidget이 자동으로 제공)
            toggle_action = dock.toggleViewAction()
            toggle_action.setText(menu_text)
            toggle_action.setToolTip(f"{menu_text} Dock 표시/숨김")
            
            # View 메뉴에 추가 (이미 있으면 중복 추가 방지)
            if toggle_action not in view_menu.actions():
                view_menu.addAction(toggle_action)
        except Exception:
            import logging
            logging.exception("[MainWindow] add dock to view menu failed")
    
    def _on_start_class_rect_roi(self):
        if not self._ensure_class_roi_controller():
            QtWidgets.QMessageBox.information(self, "안내", "이미지를 먼저 로드하세요.")
            return

        # 1) PixelClick 완전 OFF (검사모드가 드로잉을 가로막지 않도록)
        self._safe_set_click_mode(ClickMode.NONE)

        # 2) 클래스 ROI 입력 소유권으로 전환
        self._switch_roi_owner(ROIInputOwner.CLASS)

        # 3) 작업영역 ROI는 비활성
        if getattr(self, "roi", None):
            self.roi.set_mode(None)
            if hasattr(self.roi, "set_accept_events"):
                self.roi.set_accept_events(False)

        # 4) 클래스 ROI '사각형' 모드 시작 (class_roi_mode 실행)
        self.class_roi_controller.set_mode('rect', None)

        if hasattr(self.clsDock, "set_roi_drawing_state"):
            self.clsDock.set_roi_drawing_state(True)

        self.statusBar().showMessage("클래스 ROI(사각형): 드래그→놓기.", 3000)

    @QtCore.pyqtSlot()
    def _on_start_diffusion_rect_roi(self):
        if not getattr(self, "diffusion_roi_controller", None):
            # 요청: 없으면 파일 보내주면 구현, 임시로는 ROIController로 대체 가능
            QtWidgets.QMessageBox.information(self, "안내", "디퓨전 ROI 컨트롤러가 없습니다.")
            return

        self._switch_roi_owner(ROIInputOwner.DIFFUSION)
        try:
            if getattr(self, "roi", None):
                self.roi.set_mode(None)
            if getattr(self, "class_roi_controller", None):
                self.class_roi_controller.set_mode(None, -1)
        except Exception:
            pass

        self.diffusion_roi_controller.set_mode('rect')  # 필요 파라미터에 맞게
        self.statusBar().showMessage("디퓨전 사각형 ROI: 드래그로 지정, 놓으면 완료.", 3000)

    @QtCore.pyqtSlot(dict)
    def _on_pixel_classify(self, params: dict):
        """
        도크에서 '분류' 클릭 → 기존 Pixel Classification 수행.
        params: {"metric":"SAD|SID|SCC","tau":float,"delta":float}
        """
       
        try:
            # 1) 입력
            cube = self.cfg["data"]     # (H,W,C)
            # 수정: lib 접근 전 hasattr 체크로 AttributeError 방지
            if not hasattr(self, "lib") or self.lib is None or not self.lib:
                QtWidgets.QMessageBox.warning(self, "경고", "라이브러리가 로드되지 않았습니다.")
                return
            
            lib  = self.lib

            H, W, C = cube.shape

            assert isinstance(lib, dict) and all(
                isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == C for v in lib.values()
            ), "library는 {class_id: (P_i, C)} 형태의 dict 여야 합니다."

            # 2) ROI
            roi_m = None
            m = getattr(self, "_work_mask", None)
            if isinstance(m, np.ndarray):
                m = np.asarray(m, dtype=bool)
                if m.shape == (H, W) and m.any():
                    roi_m = m

            # 3) 분류 실행
            classmap, where_note = self._execute_classification(cube, lib, roi_m, params)

            # 4) 레이어 등록
            layer_name = "classification"

            # ★ 포커스 대상 부모명/백업 초기화 (이름 통일)
            self._focus_parent = layer_name
            self._focus_backup = None
            # 수정: lib 접근 전 hasattr 체크로 AttributeError 방지
            if not hasattr(self, "lib") or self.lib is None or not self.lib:
                cls_ids = []
            else:
                cls_ids = list(self.lib.keys())
            values, counts = np.unique(classmap, return_counts=True)
            cnt_dict = {int(v): int(c) for v, c in zip(values, counts)}
            mtrl_ids = list(cnt_dict.keys())

            # 1) ★ 먼저 팔레트 보강 + 렌더러에 주입 (지도/Viewer 동기화)
            self._augment_palette_for(mtrl_ids, push_renderer=True, push_dock=True)

            base_url = os.getenv('material_filtering_url')
            user_type = getattr(self, "user_type", "server")
            id_to_name = None

            if user_type == "personal":
                # personal: .info 파일에서 클래스 이름을 읽어 viewer에 표시
                try:
                    primary = self._cache_primary_path(self.cfg) if hasattr(self, "cfg") else None
                    if not primary:
                        primary = self._extract_src_path(self.cfg) if hasattr(self, "cfg") else None

                    if primary:
                        from services.resampling_cache import load_classes_from_info
                        classes_from_info = load_classes_from_info(primary)
                        
                        print(classes_from_info)
                        if classes_from_info:
                            id_to_name = {
                                int(cid): {"mtrl_nm": str(name), 'desc':desc}
                                for cid, name, desc in classes_from_info
                            }
                            logging.info("[Classification] personal: .info 기반 id_to_name 적용 (%d개)", len(id_to_name))
                except Exception:
                    logging.exception("[Classification] personal: .info 기반 id_to_name 생성 실패")
                    id_to_name = None
            else:
                # server: 기존 API + lib 기반 이름 사용
                if base_url and mtrl_ids:
                    try:
                        id_to_name = search_material_filtering_list(base_url=base_url, mtrl_ids=mtrl_ids)
                    except Exception:
                        logging.exception("[Classification] class label fetch failed")
                        id_to_name = None

            if id_to_name is None:
                id_to_name = self._norm_id_to_name()

            # 2) ★ 그 다음 지도에 레이어 등록 → 최초 렌더부터 보강된 팔레트 사용
            self.layer_manager.register_classmap(
                name="classification",
                classmap_i32=classmap.astype(np.int32),
                class_ids=list(self.lib.keys()),
                visible=True,
                id_to_name=id_to_name,
            )

            # 3) ★ Viewer 갱신 (같은 팔레트 참조)
            self.clsDock.sync_viewer_with_palette(cnt_dict, self.class_palette_qcolor, id_to_name=id_to_name)
            
            # (선택) 이후 팔레트 변경 이벤트에 재사용할 수 있도록 보관
            self._last_counts = cnt_dict
            self._last_id_to_name = id_to_name        
            print(self._last_id_to_name)
            
            # 6) 상태바
            total = classmap.size
            u = int((classmap == UNKNOWN).sum())
            mlt = int((classmap == MULTIPLE).sum())
            sgl = total - u - mlt
            where_note = "ROI 내"
            self.statusBar().showMessage(
                f"{where_note} 분류 완료 — single:{sgl}, unknown:{u}, multiple:{mlt}", 4000
            )
        
        # 수정: 분류 완료 시 플래그 해제
        finally:
            self._is_classifying = False
        
        # ★ 분류 직후: ROI 잔상 해제 + 클릭 모드 INSPECT로 명시 복귀
        try:
            if hasattr(self._map_view, "set_roi_active"):
                self._map_view.set_roi_active(False)
            if hasattr(self._map_view, "set_roi_mode"):
                self._map_view.set_roi_mode(None)

            # 클릭 모드 복귀 (분석 모드 상태 확인)
            self._sync_click_mode_with_analysis_state()
        except Exception:
            logging.exception("[classify→click] restore failed")

    def _init_single_palette(self):
        base = {UNKNOWN:(128,128,128), MULTIPLE:(0,0,0), 0:(0,0,255), 1:(255,0,0,), 7:(0,200,0)}
        # RGB dict
        self.class_palette_rgb = {int(k):(int(r),int(g),int(b)) for k,(r,g,b) in base.items()}
        # QColor dict
        self.class_palette_qcolor = {
            cid: QtGui.QColor(r, g, b, 255) for cid,(r,g,b) in self.class_palette_rgb.items()
        }

    def _setup_palette_services(self, renderer):
        """팔레트를 모든 서비스에 통합 설정"""
        services = [
            ("ClassmapRenderer", renderer, "set_palette_rgb", self.class_palette_rgb),
            ("PaletteService", getattr(self.layer_manager, "palette_service", None), "set_palette_dict", self.class_palette_rgb),
            ("PixelClassificationDock", getattr(self, "clsDock", None), "set_class_palette_ref", self.class_palette_qcolor),
        ]
        
        for name, service, method, palette in services:
            if service and hasattr(service, method):
                try:
                    getattr(service, method)(palette)
                except Exception:
                    logging.exception(f"{name} {method} failed")

    def _augment_palette_for(self, cids: list[int], *, push_renderer=True, push_dock=True):
        """
        분류 결과에 등장한 CID 중 팔레트에 없는 항목을
        '결정적 HSV 해시' 색으로 자동 보강(랜덤 X)하고,
        Renderer/Dock에 즉시 반영한다.
        """
        missing = [int(c) for c in cids if int(c) not in self.class_palette_rgb]
        if not missing:
            return

        for cid in missing:
            qc = _fallback_color_for_cid(int(cid))
            self.class_palette_qcolor[int(cid)] = qc
            self.class_palette_rgb[int(cid)]   = (qc.red(), qc.green(), qc.blue())

        # ★ Renderer 업데이트
        if push_renderer:
            cr = getattr(self.layer_manager, "renderer", None)
            if cr and hasattr(cr, "set_palette_rgb"):
                try:
                    cr.set_palette_rgb(self.class_palette_rgb)
                except Exception:
                    logging.exception("ClassmapRenderer set_palette_rgb failed")

        # ★ 질문하신 코드: PaletteService도 함께 업데이트 (중요)
        try:
            ps = getattr(self.layer_manager, "palette_service", None)
            if ps and hasattr(ps, "set_palette_dict"):
                ps.set_palette_dict(self.class_palette_rgb)
        except Exception:
            logging.exception("PaletteService set_palette_dict failed")


        # Dock도 참조를 다시 세팅(이미 참조라도 안전)
        if push_dock and hasattr(self, "clsDock") and self.clsDock:
            try:
                self.clsDock.set_class_palette_ref(self.class_palette_qcolor)
            except Exception:
                logging.exception("Dock set_class_palette_ref failed")

    @QtCore.pyqtSlot(int)
    def _on_focus_class_from_viewer(self, cid: int):
        parent = getattr(self, "_focus_parent", "classification")  # ★ 통일된 이름 사용
        if cid == -9999:
            self._focus_restore(parent)   # ★ 이름 변경
            return
        self._focus_apply(parent, cid)    # ★ 이름 변경

    def _on_start_analysis_rect_from_dock(self):
        """분석 사각형 드로잉 시작: ROI 입력 ON, 픽셀 인터셉터 OFF, 작업영역은 건드리지 않음."""
        try:
            # ★ 사각형 버튼이 실제로 활성화되어 있는지 확인 (dock에서 호출될 때만)
            # Dialog에서 호출될 때는 이 체크를 건너뛰어야 함
            if (hasattr(self, "clsDock") and self.clsDock and 
                hasattr(self.clsDock, "_btn_rect") and self.clsDock._btn_rect):
                if not self.clsDock._btn_rect.isChecked():
                    return  # dock의 사각형 버튼이 활성화되어 있지 않으면 실행하지 않음
            
            if not self._ensure_roi_controller():
                QtWidgets.QMessageBox.information(self, "안내", "이미지를 먼저 로드하세요.")
                return

            # 분석 플래그: 작업영역 갱신 가드
            self._roi_capture_target = "analysis"

            # 임시 오버레이 핸들 보장
            if getattr(self, "_analysis_overlay", None) is None:
                self._analysis_overlay = TempOverlayService(self._map_view)

            # ★ 1) ROI 소유권을 WORK로 설정 (다른 컨트롤러 비활성화)
            self._switch_roi_owner(ROIInputOwner.WORK)
            
            # ★ 2) ROI 입력 확실히 ON
            try:
                if hasattr(self, "_mute_roi_inputs"):
                    self._mute_roi_inputs(False)            # ROI 이벤트 허용
                self.roi.register_on_finish = False         # 레이어 등록 방지(분석용)
                if hasattr(self.roi, "set_accept_events"):
                    self.roi.set_accept_events(True)
                if hasattr(self.roi, "set_mode"):
                    self.roi.set_mode(None)                 # 이전 모드 클리어
            except Exception:
                logging.exception("[AnalysisRect] enable ROI input failed")

            # 3) 픽셀 클릭 인터셉터 OFF (드래그 충돌 제거) - 사각형 모드일 때만
            try:
                if hasattr(self, "pixelClick") and self.pixelClick:
                    self._safe_set_click_mode(ClickMode.NONE)
            except Exception:
                pass

            # 4) 이미지 크기 가져오기
            try:
                cfg = getattr(self, "cfg", {})
                data = cfg.get("data")
                if data is None:
                    QtWidgets.QMessageBox.warning(self, "경고", "이미지 데이터가 없습니다.")
                    return
                H, W = data.shape[:2]
            except Exception:
                logging.exception("[AnalysisRect] get image shape failed")
                QtWidgets.QMessageBox.warning(self, "경고", "이미지 크기를 가져올 수 없습니다.")
                return

            # 5) 컨트롤러 생성/연결(없으면)
            if self.analysisCtrl is None:
                def _set_click(mode_str: str):
                    try:
                        mode = getattr(ClickMode, mode_str)
                        self._safe_set_click_mode(mode)
                    except Exception:
                        logging.exception("[AnalysisRect] set_click failed")

                self.analysisCtrl = AnalysisSelectionController(
                    map_view=self._map_view,
                    roi_controller=self.roi,
                    overlay_service=self._analysis_overlay,
                    set_click_mode=_set_click,
                    get_op=self._get_analysis_op,          # ★ 추가: 최신 op 즉시 조회자
                    parent=self
                )

                # SSOT에서 읽어서 주입
                self.analysisCtrl.set_op(self._get_analysis_op())
                # 분석 마스크 변경 시 동기화는 _on_analysis_mask_changed_from_ctrl에서 처리
                self.analysisCtrl.rectFinished.connect(self._on_analysis_rect_finished)
                logging.info("[MainWindow] (reinit) connecting analysisMaskChanged signal")
                self.analysisCtrl.analysisMaskChanged.connect(self._on_analysis_mask_changed_from_ctrl)
                logging.info("[MainWindow] (reinit) analysisMaskChanged signal connected")

            # ★ 6) AnalysisSelectionController 초기화 (shape, allowed_mask)
            self.analysisCtrl.set_shape(H, W)
            work_mask = getattr(self, "_work_mask", None)
            self.analysisCtrl.set_allowed_mask(work_mask)

            # 7) 사각형 드로잉 시작
            try:
                self.analysisCtrl.cancel()
            except Exception:
                pass
            current_op = self._get_analysis_op()
            self._rect_active_op = current_op
            self.analysisCtrl.set_op(current_op)
            self.analysisCtrl.start_rect_selection()

            # ROI 이벤트 확실히 허용
            if getattr(self, "roi", None) and hasattr(self.roi, "set_accept_events"):
                self.roi.set_accept_events(True)

            # 클릭 충돌 방지
            self._safe_set_click_mode(ClickMode.NONE)

            # 8) Dock 시각 상태
            if hasattr(self, "clsDock") and self.clsDock and hasattr(self.clsDock, "set_roi_drawing_state"):
                self.clsDock.set_roi_drawing_state(True)

            self.statusBar().showMessage("분석 사각형: 드래그로 지정, 놓으면 완료.", 3000)

        except Exception:
            logging.exception("[AnalysisRect] start failed")
            QtWidgets.QMessageBox.critical(self, "오류", "분석 사각형 모드를 시작하는 데 실패했습니다.")

        
    @QtCore.pyqtSlot(list, int)
    def _on_analysis_pixels_class_set(self, pixel_coords: List[Tuple[int, int]], new_cid: int):
        """분석 다이얼로그에서 픽셀 클래스 설정 요청"""
        try:
            classmap = self.layer_manager.get_classmap("classification")
            if classmap is None:
                QtWidgets.QMessageBox.warning(self, "경고", "분류맵이 없습니다.")
                return
            
            # classmap 업데이트
            for y, x in pixel_coords:
                if 0 <= y < classmap.shape[0] and 0 <= x < classmap.shape[1]:
                    classmap[y, x] = int(new_cid)
            
            # 클래스 ID 목록 업데이트
            unique_classes = sorted([int(c) for c in np.unique(classmap)])
            
            # LayerManager에 다시 등록하여 레이어 갱신
            self.layer_manager.register_classmap(
                name="classification",
                classmap_i32=classmap,
                class_ids=unique_classes,
            visible=True,
            id_to_name=self._norm_id_to_name(),
            )
            
            logging.info(f"[Analysis] {len(pixel_coords)}개 픽셀의 클래스를 {new_cid}로 변경")
            self.statusBar().showMessage(
                f"{len(pixel_coords)}개 픽셀의 클래스를 {new_cid}로 변경했습니다.", 
                3000
            )
            
        except Exception:
            logging.exception("[Analysis] _on_analysis_pixels_class_set failed")
            QtWidgets.QMessageBox.critical(self, "오류", "클래스 설정 중 오류가 발생했습니다.")
    
    @QtCore.pyqtSlot()
    def _on_view_analysis_details_from_dock(self):
        """
        '분석 선택영역 상세보기' 요청 처리.
        - 다이얼로그가 없으면 생성하고, 있으면 재사용
        - classmap/hsi_data/id_to_name/wavelength/metric/topk 를 '최초 1회' set_data 로 주입
        - 이후 마스크 변화는 MainWindow->_on_analysis_mask_changed_from_ctrl 에서 dlg.update_mask(mask) 로만 갱신
        """
        try:
            # 0) 필수 데이터 검사
            classmap = self.layer_manager.get_classmap("classification")
            if classmap is None:
                QtWidgets.QMessageBox.information(self, "분석 상세", "분류맵이 없습니다. 먼저 분류를 수행하세요.")
                return

            if not hasattr(self, "cfg") or not isinstance(self.cfg, dict) or "data" not in self.cfg:
                QtWidgets.QMessageBox.warning(self, "경고", "HSI 데이터가 없습니다. 이미지를 먼저 로드하세요.")
                return
            hsi_data   = self.cfg["data"]
            wavelengths = self.cfg.get("wavelength")

            # 1) 보조 데이터 정규화
            id_to_name = self._norm_id_to_name() if hasattr(self, "_norm_id_to_name") else {}
            palette    = getattr(self, "class_palette_qcolor", {}) or {}
            metric     = "SAD"
            try:
                if getattr(self, "clsDock", None):
                    metric = self.clsDock.get_metric()
            except Exception:
                pass

            # 2) Top-K (없으면 캐시/콜백에서 보강 시도)
            topk_cids, topk_vals = getattr(self, "_class_topk_cids", None), getattr(self, "_class_topk_vals", None)
            if topk_cids is None or topk_vals is None:
                try:
                    c1, v1 = self._cb_topk_for_map("classification")
                    if c1 is not None and v1 is not None:
                        topk_cids, topk_vals = c1, v1
                except Exception:
                    logging.exception("[Analysis] get topk from cache failed")

            # 3) 초기 마스크 결정 (저장된 분석 마스크가 없으면 전체 True)
            H, W = classmap.shape[:2]
            init_mask = getattr(self, "_analysis_mask", None)
            if not (isinstance(init_mask, np.ndarray) and init_mask.shape == (H, W) and np.any(init_mask)):
                init_mask = np.zeros((H, W), dtype=bool)

            # 4) 다이얼로그 인스턴스 보장/신호 연결
            if not hasattr(self, "_analysis_selection_dialog") or self._analysis_selection_dialog is None:
                dlg = AnalysisSelectionDialog(parent=self, ui_dir=self.app_dir / "ui")
                self._analysis_selection_dialog = dlg
                # 픽셀 클래스 일괄 설정 콜백
                if hasattr(dlg, "pixels_class_set"):
                    dlg.pixels_class_set.connect(self._on_analysis_pixels_class_set)
                # 닫힘 처리(분석 상태 초기화)
                dlg.finished.connect(self._on_analysis_dialog_finished)
            else:
                dlg = self._analysis_selection_dialog
            
            # 스펙트럼 라이브러리 provider 설정 (파란색)
            def _spectrum_library_provider(cid: int):
                """
                스펙트럼 라이브러리에서 해당 class 스펙 반환(list of 1D).
                - self.spectrum_lib_raw / self.spectrum_lib_all 만 사용
                - 라벨링(self.label_raw, self.lib 등)은 절대 섞지 않음
                """
                out: list[np.ndarray] = []
                try:
                    # ✅ 우선순위: raw 라이브러리만 사용
                    splib = getattr(self, "spectrum_lib_raw", None)
                    if not isinstance(splib, dict) or not splib:
                        # 혹시나 raw가 비어 있고 CR까지 합쳐서 쓰고 싶다면 spectrum_lib_all 사용
                        splib = getattr(self, "spectrum_lib_all", None)
                        if not isinstance(splib, dict) or not splib:
                            logging.info("[Analysis] spectrum_lib_raw/all 이 비어 있습니다.")
                            return out

                    arr = splib.get(int(cid))
                    if arr is None:
                        logging.info("[Analysis] spectrum_lib_* 에 cid=%s 데이터가 없습니다.", cid)
                        return out

                    arr = np.asarray(arr, dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr[None, :]   # (C,) → (1,C)

                    if arr.ndim != 2:
                        logging.warning(
                            "[Analysis] spectrum_lib_*[%s] ndim=%s, 기대=(N, C).",
                            cid, arr.ndim
                        )
                        return out

                    # 현재 이미지 밴드 수와 맞는지 한 번 확인 (안 맞으면 표시 안 함)
                    expect_c = None
                    try:
                        if hasattr(self, "cfg") and isinstance(self.cfg, dict) and "data" in self.cfg:
                            expect_c = int(self.cfg["data"].shape[2])
                    except Exception:
                        expect_c = None

                    if expect_c is not None and arr.shape[1] != expect_c:
                        logging.warning(
                            "[Analysis] spectrum_lib_*[%s] C=%s != expect_c=%s, 사용하지 않습니다.",
                            cid, arr.shape[1], expect_c
                        )
                        return out

                    # 각 행을 개별 스펙트럼으로 반환
                    for i in range(arr.shape[0]):
                        spec = arr[i, :].copy()
                        if spec.size > 0:
                            out.append(spec)

                except Exception:
                    logging.exception("[Analysis] _spectrum_library_provider failed for cid=%s", cid)

                return out
   

            # 라벨링 데이터 provider 설정 (초록색, RAW만 사용)
            def _labeling_provider(cid: int):
                """
                라벨링 RAW 스펙트럼에서 해당 class의 스펙만 반환.
                - self.labeling_raw_lib: {cid: (N,C)} 형태라고 가정
                - 반환: [1D np.ndarray(C,), ...]
                """
                out: list[np.ndarray] = []
                try:
                    lab = getattr(self, "labeling_raw_lib", {}) or {}
                    if not isinstance(lab, dict):
                        logging.warning("[Analysis] labeling_raw_lib 가 dict가 아닙니다: %r", type(lab))
                        return out

                    arr = lab.get(int(cid))
                    if arr is None:
                        logging.info("[Analysis] labeling_raw_lib 에 cid=%s 데이터가 없습니다.", cid)
                        return out

                    arr = np.asarray(arr, dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr[None, :]    # (C,) → (1,C)

                    if arr.ndim != 2:
                        logging.warning("[Analysis] labeling_raw_lib[%s] ndim=%s, 기대=(N, C).", cid, arr.ndim)
                        return out

                    # 현재 이미지 밴드 수와 맞는지만 한 번 확인 (안 맞으면 그냥 쓰지 말고 버리기)
                    expect_c = None
                    try:
                        if hasattr(self, "cfg") and isinstance(self.cfg, dict) and "data" in self.cfg:
                            expect_c = int(self.cfg["data"].shape[2])
                    except Exception:
                        expect_c = None

                    if expect_c is not None and arr.shape[1] != expect_c:
                        logging.warning(
                            "[Analysis] labeling_raw_lib[%s] C=%s != expect_c=%s, 사용하지 않습니다.",
                            cid, arr.shape[1], expect_c
                        )
                        return out

                    # 각 행을 개별 스펙트럼으로 반환
                    for i in range(arr.shape[0]):
                        spec = arr[i, :].copy()
                        if spec.size > 0:
                            out.append(spec)

                except Exception:
                    logging.exception("[Analysis] _labeling_provider failed for cid=%s", cid)

                return out


            
            # Provider 설정
            if hasattr(dlg, "set_spectrum_library_provider"):
                dlg.set_spectrum_library_provider(_spectrum_library_provider)
            if hasattr(dlg, "set_label_provider"):
                dlg.set_label_provider(_labeling_provider)
            
            # Match Provider 설정 (우측 상단 그래프용)
            # 클로저로 hsi_data, metric, lib를 캡처
            def _create_match_provider(hsi_data_ref, metric_ref, lib_ref):
                """Match Provider 콜백 생성"""
                def _match_provider(y: int, x: int, cid: int) -> Optional[np.ndarray]:
                    """
                    해당 픽셀(y, x)에서 사용된 reference 스펙트럼을 반환
                    - 픽셀 스펙트럼과 라이브러리의 reference 스펙트럼을 비교하여 가장 유사한 것을 반환
                    """
                    try:
                        # 1) 픽셀 스펙트럼 가져오기
                        if hsi_data_ref is None:
                            return None
                        if not (0 <= y < hsi_data_ref.shape[0] and 0 <= x < hsi_data_ref.shape[1]):
                            return None
                        pixel_spec = np.asarray(hsi_data_ref[y, x, :], dtype=np.float32)
                        
                        # 2) 라이브러리에서 해당 cid의 reference 스펙트럼들 가져오기
                        if not lib_ref or cid not in lib_ref:
                            return None
                        
                        ref_specs = lib_ref[cid]  # (N, C)
                        ref_specs = np.asarray(ref_specs, dtype=np.float32)
                        if ref_specs.ndim == 1:
                            ref_specs = ref_specs[None, :]  # (1, C)
                        if ref_specs.ndim != 2 or ref_specs.shape[1] != pixel_spec.shape[0]:
                            return None
                        
                        # 3) 메트릭 함수 가져오기
                        from core.metrics import get_metric
                        metric_fn = get_metric(metric_ref.upper() if metric_ref else "SAD")
                        
                        # 4) 픽셀 스펙트럼과 가장 유사한 reference 스펙트럼 찾기
                        # pixel_spec: (C,), ref_specs: (N, C)
                        distances = metric_fn(ref_specs, pixel_spec)  # (N,)
                        if distances.size == 0:
                            return None
                        
                        # 가장 작은 거리(가장 유사한) reference 선택
                        best_idx = int(np.argmin(distances))
                        best_ref = ref_specs[best_idx, :]  # (C,)
                        
                        return best_ref.astype(np.float32, copy=False)
                        
                    except Exception as e:
                        logging.exception(f"[Analysis] _match_provider failed for (y={y}, x={x}, cid={cid}): {e}")
                        return None
                return _match_provider
            
            # Match Provider 생성 및 설정
            lib_for_match = getattr(self, "lib", {}) or {}
            match_provider = _create_match_provider(hsi_data, metric, lib_for_match)
            if hasattr(dlg, "set_match_provider"):
                dlg.set_match_provider(match_provider)

            # 5) 팔레트 먼저 주입
            try:
                if hasattr(dlg, "set_palette"):
                    dlg.set_palette(palette)
            except Exception:
                logging.exception("[Analysis] set_palette to dialog failed")

            # 6) Dialog를 먼저 표시 (사용자에게 즉시 피드백)
            dlg.setModal(False)
            dlg.setWindowModality(QtCore.Qt.NonModal)
            dlg.show(); dlg.raise_(); dlg.activateWindow()

            # 7) 무거운 데이터 처리는 비동기로 지연 (Dialog 표시 후)
            # set_data의 _update_pixel_table이 전체 이미지 픽셀을 처리하므로 시간이 걸림
            QtCore.QTimer.singleShot(50, lambda: self._load_analysis_data_async(
                dlg, init_mask, classmap, hsi_data, topk_cids, topk_vals, 
                id_to_name, wavelengths, metric
            ))

            QtCore.QTimer.singleShot(0, lambda: self._sync_click_mode_with_analysis_state())

            logging.info("[Analysis] dialog opened: HxW=%sx%s, mask_sum=%s",
                        H, W, int(init_mask.sum()) if isinstance(init_mask, np.ndarray) else -1)

        except Exception:
            logging.exception("[Analysis] show dialog failed")
            QtWidgets.QMessageBox.critical(self, "오류", "분석 다이얼로그를 표시하는 중 오류가 발생했습니다.")        
    
    def _load_analysis_data_async(self, dlg, mask, classmap, hsi_data, topk_cids, topk_vals, 
                                   id_to_name, wavelengths, metric):
        """
        분석 데이터를 비동기로 로드 (Dialog 표시 후 무거운 처리를 수행)
        """
        try:
            # 진행 상태 표시
            if hasattr(self, "statusBar"):
                self.statusBar().showMessage("분석 데이터 로딩 중...", 2000)
            
            # set_data 호출 (무거운 _update_pixel_table 포함)
            dlg.set_data(
                mask=mask,
                classmap=classmap,
                hsi_data=hsi_data,
                topk_cids=topk_cids,
                topk_vals=topk_vals,
                id_to_name=id_to_name,
                wavelengths=wavelengths,
                metric=metric
            )
            
            # 완료 메시지
            if hasattr(self, "statusBar"):
                self.statusBar().showMessage("분석 데이터 로딩 완료", 2000)
            
        except Exception:
            logging.exception("[Analysis] async data load failed")
            if hasattr(self, "statusBar"):
                self.statusBar().showMessage("분석 데이터 로딩 실패", 3000)
    
    @QtCore.pyqtSlot(int)
    def _on_analysis_dialog_finished(self, result: int):
        """
        AnalysisSelectionDialog 종료 시:
        1) 분석 컨트롤러 중단
        2) 분석 마스크/오버레이 완전 초기화
        3) 클릭 모드/ROI 입력 복구 (INSPECT, work)
        4) MapView의 빨간색 스펙트럼 마커 제거
        """
        try:
            # 0) MapView 마커 및 오버레이 제거
            if hasattr(self, "_map_view") and self._map_view:
                try:
                    self._map_view.clear_ctx_markers("ANALYSIS_TL_SELECTED")  # TL 그래프에서 선택된 스펙트럼
                    self._map_view.clear_ctx_markers("ANALYSIS_TR_REF_MATCH")  # TR 그래프에서 빨간색 reference와 일치하는 픽셀
                    # 오버레이 제거
                    if hasattr(self._map_view, "remove_temporal_layer"):
                        self._map_view.remove_temporal_layer("ANALYSIS_TL_SELECTED_OVERLAY")
                except Exception:
                    pass
            
            # 1) 컨트롤러 중단
            if getattr(self, "analysisCtrl", None):
                try:
                    self.analysisCtrl.cancel()
                except Exception:
                    pass

            # 2) 분석 영역 완전 초기화 (마스크/사각형/오버레이/독 버튼)
            if hasattr(self, "_on_clear_analysis_region"):
                self._on_clear_analysis_region()   # ← 내부에서 _analysis_mask/_analysis_rect, overlay, dock 버튼 모두 클리어

            # 3) 클릭 모드 복구(라벨/시드 모드가 아니면 INSPECT로)
            self._roi_capture_target = "work"
            current_mode = getattr(self.pixelClick, "mode", None)
            if current_mode not in (ClickMode.LABEL, ClickMode.DIFFUSION_SEED, ClickMode.INSPECT):
                self._enter_pixel_click_mode(ClickMode.INSPECT, mute_roi=False)
            else:
                # 라벨/시드 진행 중이면 ROI 입력은 다시 허용
                if getattr(self, "roi", None) and hasattr(self.roi, "set_accept_events"):
                    self.roi.set_accept_events(True)

            self.statusBar().showMessage("분석 영역이 초기화되었습니다.", 2500)

        except Exception:
            logging.exception("[Analysis] dialog finished cleanup failed")


    # === MainWindow 클래스 내부 아무 곳에 추가 ===
    @QtCore.pyqtSlot(QtCore.QRect, object)
    def _on_analysis_rect_finished(self, rect: QtCore.QRect, mask_obj):
        # 컨트롤러가 이미 내부 맵에 적용해서 analysisMaskChanged를 쏘므로,
        # 여기서는 fallback만 처리
        try:
            if self.analysisCtrl is not None:
                return
            m = np.asarray(mask_obj, dtype=bool)
            H = int(getattr(self._map_view, "img_h", 0) or 0)
            W = int(getattr(self._map_view, "img_w", 0) or 0)
            self._ensure_analysis_mask(H, W)
            full = np.zeros((H, W), dtype=bool)
            y0, x0, h, w = rect.y(), rect.x(), rect.height(), rect.width()
            full[y0:y0+h, x0:x0+w] = m[:h, :w]
            # 사각형 완료 시점의 최신 op 사용 (사각형 그리기 중 op 변경 반영)
            op = self._get_analysis_op()
            logging.debug(f"[AnalysisRect] finish: using current op={op} (not locked _rect_active_op)")
            self._analysis_update(mask_delta=full, rect=rect, absolute=False, op=op, publish=True)
        except Exception:
            logging.exception("[AnalysisRect] finish fallback failed")
        finally:
            QtCore.QTimer.singleShot(0, self._sync_click_mode_with_analysis_state)

    def _handle_analysis_pixel(self, y: int, x: int) -> None:
        try:
            # analysisCtrl이 없거나 초기화되지 않았으면 초기화
            if self.analysisCtrl is None or getattr(self.analysisCtrl, "_mask", None) is None:
                if not self._ensure_analysis_ctrl_ready():
                    logging.warning("[Analysis] pixel click: analysisCtrl 초기화 실패")
                    return
            
            # 도크의 현재 op를 컨트롤러에도 반영(혹시 변경됐을 수 있음)
            self.analysisCtrl.set_op(self._get_analysis_op())
            # 한 줄로 끝: 컨트롤러가 내부 맵 갱신 → analysisMaskChanged 신호 → 메인에서 _publish_analysis_overlay()
            logging.debug(f"[Analysis] adding pixel at ({y},{x}) with op={self._get_analysis_op()}")
            self.analysisCtrl.add_pixel(int(y), int(x))
        except Exception:
            logging.exception("[Analysis] pixel pick failed")

    def _update_analysis_rect_from_mask(self) -> None:
        """현재 분석 마스크의 최소 bounding rect를 계산해 `_analysis_rect`에 반영."""
        mask = getattr(self, "_analysis_mask", None)
        if mask is None:
            self._analysis_rect = None
            return
        try:
            coords = np.argwhere(mask)
            if coords.size == 0:
                self._analysis_rect = None
                return
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            self._analysis_rect = QtCore.QRect(
                int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)
            )
        except Exception:
            logging.exception("[Analysis] update rect failed")

    def _focus_apply(self, parent_name: str, cid: int):
        """focus: parent의 자식 중 선택 클래스만 보이고 나머지는 숨김."""
        try:
            children = self.layer_manager.get_children(parent_name)
            if not children:
                return

            # 최초 진입 시 현재 가시 상태 백업
            if not getattr(self, "_focus_backup", None):
                self._focus_backup = {}
            if parent_name not in self._focus_backup:
                self._focus_backup[parent_name] = {}
                for ch in children:
                    rec = self.layer_manager.recs.get(ch)
                    self._focus_backup[parent_name][ch] = bool(rec.visible) if rec else True

            # 모든 자식 숨김 → 대상만 보이기
            target_name = f"{parent_name}/{int(cid)}"
            for ch in children:
                self.layer_manager.set_visible(ch, ch == target_name)

            # 부모(합성)는 항상 꺼둠
            prec = self.layer_manager.recs.get(parent_name)
            if prec and prec.map_item_name and prec.map_item_name != "RGB":
                self.layer_manager.set_visible(parent_name, False)

        except Exception:
            logging.exception("[Focus] apply failed")

    def _focus_restore(self, parent_name: str):
        """focus: 이전 가시 상태로 원복."""
        try:
            if not getattr(self, "_focus_backup", None):
                return
            backup = self._focus_backup.get(parent_name)
            if not backup:
                return

            for ch, vis in backup.items():
                self.layer_manager.set_visible(ch, bool(vis))

            # 부모는 기본 정책: 꺼둠
            prec = self.layer_manager.recs.get(parent_name)
            if prec and prec.map_item_name and prec.map_item_name != "RGB":
                self.layer_manager.set_visible(parent_name, False)

            # 백업 제거/정리
            self._focus_backup.pop(parent_name, None)
            if not self._focus_backup:
                self._focus_backup = None
        except Exception as e:
            logging.exception("[Focus] restore failed: %s", e)

    def _label_at(self, y: int, x: int) -> Optional[int]:
        """현재 focus 부모의 라벨을 돌려준다."""
        try:
            parent = getattr(self, "_focus_parent", "classification")
            cm = self.layer_manager.get_classmap(parent)
            if cm is None: return None
            return int(cm[y, x])
        except Exception:
            return None

    def _execute_classification(self, cube, lib, roi_m, params) -> tuple[np.ndarray, str]:
        """분류 실행 로직 (작업 영역 ROI 완전 지원 + Top-K도 전역 크기로 확장 저장)"""
        H, W, C = cube.shape

        # ROI 유효성 정리
        if roi_m is not None:
            try:
                roi_m = np.asarray(roi_m, dtype=bool)
                if roi_m.shape != (H, W) or not roi_m.any():
                    roi_m = None  # 형상이 맞지 않거나 비어있으면 전체 분류로 처리
            except Exception:
                roi_m = None

        metric = params.get("metric", "SAD")
        tau    = float(params.get("tau", 0.05))
        delta  = float(params.get("delta", 0.03))

        if roi_m is None:
            # ========= 전체 분류 =========
            classmap, extras = classify_cube(
                cube, lib,
                metric=metric, tau=tau, delta=delta,
                prev_map=None, save_topk=True, topk_k=10
            )
            where_note = "전체"

            # Top-K 그대로 저장
            self._class_topk_cids = extras.get("topk_cids")
            self._class_topk_vals = extras.get("topk_vals")
            

            # ★ 추가: classification 맵을 _topk_cache에도 저장
            tkc = extras.get("topk_cids")
            tkv = extras.get("topk_vals")
            if tkc is not None and tkv is not None:
                try:
                    # K=1로 제한 (첫 번째 Top-1만 사용)
                    c1 = tkc.astype(np.int32, copy=False)[..., :1] if tkc.ndim >= 3 else tkc[..., None]
                    v1 = tkv.astype(np.float32, copy=False)[..., :1] if tkv.ndim >= 3 else tkv[..., None]
                    self._topk_cache["classification"] = (c1, v1)
                    # 메트릭 정보도 저장
                    self._register_map_semantics("classification", metric=metric, distance=True)
                except Exception:
                    logging.exception("[TopK] store classification into _topk_cache failed")
            
            return classmap, where_note

        # ========= ROI 분류 =========
        ys, xs = np.where(roi_m)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1

        cube_sub = cube[y0:y1, x0:x1, :]
        roi_sub  = roi_m[y0:y1, x0:x1]

        classmap_sub, extras = classify_cube(
            cube_sub, lib,
            metric=metric, tau=tau, delta=delta,
            prev_map=None, save_topk=True, topk_k=10
        )

        # 전역 classmap 만들기: ROI 밖은 UNKNOWN
        try:
            from core.autoclass import UNKNOWN
        except Exception:
            UNKNOWN = -1

        classmap = np.full((H, W), UNKNOWN, dtype=classmap_sub.dtype)
        # ROI 내부만 반영
        sub_view = classmap[y0:y1, x0:x1]
        np.copyto(sub_view, classmap_sub, where=roi_sub)
        classmap[y0:y1, x0:x1] = sub_view

        # === Top-K도 전역 크기로 확장 ===
        tkc = extras.get("topk_cids", None)   # (h, w, K)
        tkv = extras.get("topk_vals", None)   # (h, w, K)
        
        if tkc is not None and tkv is not None and tkc.ndim == 3 and tkv.ndim == 3:
            K = int(tkc.shape[2])

            # 기본값: ROI 밖은 Class=-1, Score=+inf
            full_cids = np.full((H, W, K), -1, dtype=tkc.dtype)
            full_vals = np.full((H, W, K), np.inf, dtype=tkv.dtype)

            mask3 = roi_sub[..., None]  # (h, w, 1) → 브로드캐스트로 (h,w,K)

            # ROI 내부만 복사
            tgt_cids = full_cids[y0:y1, x0:x1, :]
            tgt_vals = full_vals[y0:y1, x0:x1, :]
            np.copyto(tgt_cids, tkc, where=mask3)
            np.copyto(tgt_vals, tkv, where=mask3)

            full_cids[y0:y1, x0:x1, :] = tgt_cids
            full_vals[y0:y1, x0:x1, :] = tgt_vals

            self._class_topk_cids = full_cids
            self._class_topk_vals = full_vals
            
            # ★ 추가: classification 맵을 _topk_cache에도 저장
            try:
                # K=1로 제한 (첫 번째 Top-1만 사용)
                c1 = full_cids.astype(np.int32, copy=False)[..., :1]
                v1 = full_vals.astype(np.float32, copy=False)[..., :1]
                self._topk_cache["classification"] = (c1, v1)
                # 메트릭 정보도 저장
                self._register_map_semantics("classification", metric=metric, distance=True)
            except Exception:
                logging.exception("[TopK] store ROI classification into _topk_cache failed")
        else:
            # Top-K가 없거나 형상이 예상과 다르면 None 처리
            self._class_topk_cids = None
            self._class_topk_vals = None

        where_note = "ROI(작업 영역) 내"
        return classmap, where_note


    def _convert_to_cache_coordinates(self, y: int, x: int, cache_shape) -> tuple[int, int, Optional[str]]:
        """전역 좌표를 캐시 좌표로 변환하는 헬퍼 메서드"""
        Hc, Wc = int(cache_shape[0]), int(cache_shape[1])
        H_img = int(getattr(self._map_view, "img_h", Hc))
        W_img = int(getattr(self._map_view, "img_w", Wc))

        # 캐시가 전체 이미지인 경우
        if (Hc == H_img) and (Wc == W_img):
            return y, x, None

        # ROI 서브 캐시인 경우
        m = getattr(self, "_work_mask", None)
        if m is None or not np.any(m):
            return 0, 0, "ROI 영역을 먼저 지정하고 분류하세요."
        
        m = np.asarray(m, dtype=bool)
        ys, xs = np.where(m)
        if ys.size == 0:
            return 0, 0, "ROI가 비어 있습니다."
            
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1

        # 클릭 좌표가 ROI 바운딩 안인지 검사
        if not (y0 <= y < y1 and x0 <= x < x1):
            return 0, 0, "ROI 영역 밖 픽셀입니다. ROI 내부를 클릭하세요."

        # 전역 → 로컬 변환
        return y - y0, x - x0, None

    def _show_pixel_topk(self, y: int, x: int):
        """
        분류 시 저장한 Top-K 캐시에서 (y,x)의 후보들을 전부 표로 표시.
        캐시가 ROI 서브(h,w,K)인 경우 전역좌표→ROI 로컬좌표로 변환 후 접근.
        Score는 extras 저장된 원시 거리 그대로 사용.
        """
        try:
            if hasattr(self, "clsDock") and self.clsDock and \
            hasattr(self.clsDock, "is_analysis_mode_active") and \
            self.clsDock.is_analysis_mode_active():
                self.statusBar().showMessage("분석 모드에서는 유사도 팝업을 띄우지 않습니다.", 1500)
                return
        except Exception:
            pass

        try:
            tkc = getattr(self, "_class_topk_cids", None)
            tkv = getattr(self, "_class_topk_vals", None)
            if tkc is None or tkv is None:
                QtWidgets.QMessageBox.information(self, "안내", "Top-K 캐시가 없습니다. 분류를 먼저 수행하세요.")
                return

            K = int(tkc.shape[2])

            # 좌표 변환 헬퍼 메서드 사용
            yy, xx, error_msg = self._convert_to_cache_coordinates(y, x, tkc.shape)
            if error_msg:
                QtWidgets.QMessageBox.information(self, "안내", error_msg)
                return

            # 2) 이제 범위를 검사 (yy,xx는 캐시 기준 좌표)
            Hc, Wc = int(tkc.shape[0]), int(tkc.shape[1])
            if not (0 <= yy < Hc and 0 <= xx < Wc):
                QtWidgets.QMessageBox.information(self, "안내", "선택한 위치가 캐시 범위를 벗어났습니다.")
                return

            # 3) 캐시에서 읽기 (원시 거리 그대로)
            cids = tkc[yy, xx, :]
            vals = tkv[yy, xx, :]

            # 4) 메타 맵 준비
            meta_map = getattr(self, "_mtrl_meta", {}) or {}
            user_type = getattr(self, "user_type", "server")

            if user_type == "personal":
                # personal: .resample.info 의 classes 사용
                try:
                    if not meta_map:
                        primary = self._cache_primary_path(self.cfg) if hasattr(self, "cfg") else None
                        if not primary and hasattr(self, "_extract_src_path"):
                            primary = self._extract_src_path(self.cfg)
                        if primary:
                            classes_info = load_classes_from_info(primary)  # [(cid, name, desc), ...]
                            for cid_i, name_i, desc_i in classes_info:
                                meta_map[int(cid_i)] = {
                                    "name": str(name_i),
                                    "desc": str(desc_i) if desc_i is not None else "",
                                }
                    # 캐시 갱신
                    self._mtrl_meta = meta_map
                except Exception:
                    logging.exception("[TopK] personal: load_classes_from_info failed")
            else:
                # server: 기존 API 경로 유지
                missing = [int(c) for c in cids.tolist()
                        if (int(c) not in meta_map and str(int(c)) not in meta_map)]
                if missing:
                    try:
                        base_url = os.getenv('material_filtering_url')
                        recs = search_material_filtering_list(base_url=base_url, mtrl_ids=missing)
                        for rec in (recs or []):
                            try:
                                cid2 = int(rec.get('mtrl_cd'))
                                meta_map[cid2] = {
                                    'name': rec.get('mtrl_nm', str(cid2)),
                                    'desc': rec.get('desc', rec.get('dsc', '')),
                                }
                            except Exception:
                                pass
                        self._mtrl_meta = meta_map
                    except Exception:
                        # 메타 조회 실패해도 표시는 가능
                        logging.exception("[TopK] server: material_filtering_list failed")

            # 5) rows 구성 (원시 score 그대로 표시)
            rows = []
            for i in range(K):
                cid = int(cids[i])
                score = float(vals[i])  # RAW 그대로 (SAM은 라디안)
                meta = meta_map.get(cid) or meta_map.get(str(cid), {}) or {}
                name = meta.get('name', str(cid))
                desc = meta.get('desc', '')
                rows.append((i + 1, cid, f"{score:.6f}", name, desc))

            self._show_topk_dialog(rows, "선택 픽셀 분류 결과")

        except IndexError:
            QtWidgets.QMessageBox.information(self, "안내", "선택한 위치의 Top-K 정보를 표시할 수 없습니다(ROI/좌표 확인).")
        except Exception:
            logging.exception("[TopK] show failed")
            QtWidgets.QMessageBox.information(self, "안내", "Top-K 정보를 만들 수 없습니다. 먼저 분류를 수행해 주세요.")

            
    def _on_cls_dock_visibility(self, visible: bool):
        """PixelClassificationDock이 보이면 클릭 모드 ON, 숨기면 OFF."""
        # ★ 확산 시드 모드일 땐 건드리지 않음
        try:
            if hasattr(self, "pixelClick") and self.pixelClick.mode == ClickMode.DIFFUSION_SEED:
                return
        except Exception:
            pass

        if visible:
            self._enter_click_mode_from_dock()
        else:
            self._leave_click_mode_from_dock()


    def _enter_click_mode_from_dock(self):
        # ROI/분석 중이 아닐 때만 Inspect 켠다
        is_roi_class = (self._active_roi_owner == ROIInputOwner.CLASS)
        is_roi_work  = (self._active_roi_owner == ROIInputOwner.WORK)
        if is_roi_class or is_roi_work:
            return  # 현재 ROI 드로잉을 방해하지 않음

        # 분석 모드 상태를 확인하여 적절한 클릭 모드 설정
        self._sync_click_mode_with_analysis_state()

    def _leave_click_mode_from_dock(self):
        """Dock 숨김/이탈 시: 기본 상태로 복귀."""
        try:
            self._safe_set_click_mode(ClickMode.NONE)
        except Exception:
            logging.exception("pixelClick.set_mode reset failed")
        self.statusBar().clearMessage()
        
    # MainWindow 클래스 내부 어딘가(예: _show_pixel_topk 아래) 추가
    def _show_topk_dialog(self, rows, title="선택 픽셀 분류 결과"):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(780, 420)
        layout = QtWidgets.QVBoxLayout(dlg)

        table = QtWidgets.QTableWidget(dlg)
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Rank", "Class", "Score", "Material Name", "Description"])
        table.setRowCount(len(rows))
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        for r, (rank, cid, score_str, name, desc) in enumerate(rows):
            for c, val in enumerate([str(rank), str(cid), str(score_str), str(name), str(desc)]):
                it = QtWidgets.QTableWidgetItem(val)
                it.setFlags(it.flags() ^ QtCore.Qt.ItemIsEditable)
                table.setItem(r, c, it)

        layout.addWidget(table)
        btn = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close, parent=dlg)
        btn.rejected.connect(dlg.reject)
        layout.addWidget(btn)
        dlg.setLayout(layout)
        dlg.exec_()
        
    # MainWindow 클래스 내부에 추가
    def _apply_confidence_overlay_to_mapview(self, strict_val: float, base_val: float):
        """
        유사도(gmin) 기반 투명도 오버레이를 **MapView에만** 올린다.
        LayersDock/LayerManager에는 아무 것도 등록하지 않음.
        """
        gm = getattr(self, "_gmin_map", None)
        cm = getattr(self, "_last_classmap", None)
        if gm is None or cm is None:
            QtWidgets.QMessageBox.information(self, "안내", "분류 결과가 없습니다. 먼저 분류를 수행하세요.")
            return

        alpha = compute_alpha_from_gmin(gm, strict_val, base_val)
        if alpha is None:
            QtWidgets.QMessageBox.information(self, "안내", "g1 정보가 없습니다.")
            return

        pal = getattr(self, "class_palette_rgb", None)
        if not pal:
            QtWidgets.QMessageBox.information(self, "안내", "팔레트가 없습니다.")
            return

        def _fallback(cid: int):
            qc = _fallback_color_for_cid(int(cid))
            return (qc.red(), qc.green(), qc.blue())

        rgba = build_opacity_overlay_rgba(cm, pal, alpha, fallback_color_fn=_fallback)  # (H,W,4) uint8

        # numpy → QImage → QPixmap
        h, w, _ = rgba.shape
        qimg = QtGui.QImage(rgba.data, w, h, 4*w, QtGui.QImage.Format_RGBA8888)
        qimg = qimg.copy()  # numpy 버퍼 라이프사이클 보호
        pix = QtGui.QPixmap.fromImage(qimg)

        # 이전 임시 오버레이 제거
        self._clear_confidence_overlay_mapview()

        # MapView에 **직접** 추가(레이어 트리 비참여)
        try:
            scene = getattr(self._map_view, "scene", None)
            if callable(scene):
                scene = self._map_view.scene()
            if scene is None:
                QtWidgets.QMessageBox.information(self, "안내", "MapView scene을 찾을 수 없습니다.")
                return

            item = scene.addPixmap(pix)
            item.setZValue(1e6)
            item.setOpacity(1.0)
            item.setOffset(0, 0)
            try:
                item.setAcceptedMouseButtons(Qt.NoButton)             # ★ 클릭 통과
                item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
                item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
            except Exception:
                pass
            self._conf_item = item
            
            self.statusBar().showMessage(f"투명도 적용: Strict={strict_val}, Base={base_val}", 3000)
        except Exception:
            logging.exception("[Opacity] MapView overlay add failed")
            QtWidgets.QMessageBox.critical(self, "오류", "투명도 적용 중 오류가 발생했습니다.")

    def _clear_confidence_overlay_mapview(self):
        """
        임시 오버레이를 MapView에서 제거. 레이어 트리는 건드리지 않음.
        """
        try:
            item = getattr(self, "_conf_item", None)
            if item is not None:
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
            self._conf_item = None
        except Exception:
            logging.exception("[Opacity] clear failed")
        finally:
            self.statusBar().showMessage("투명도 초기화 완료", 2000)

    def _apply_transparency_overlay_for_class(self, cid: int, strict_val: float, base_val: float):
        """
        클래스 ID를 명시적으로 받아 해당 클래스만 '교통신호등' 오버레이로 시각화.
        (기존 _apply_transparency_overlay 로직을 cid 입력형으로 래핑)
        """
        # 1) 분류 결과/TopK 가드
        classmap = self.layer_manager.get_classmap("classification")
        if classmap is None:
            QtWidgets.QMessageBox.information(self, "안내", "분류맵을 찾을 수 없습니다.")
            return
        cids = getattr(self, "_class_topk_cids", None)
        vals = getattr(self, "_class_topk_vals", None)
        if cids is None or vals is None:
            QtWidgets.QMessageBox.information(self, "안내", "Top-K 결과가 없습니다. 먼저 분류하세요.")
            return

        # 2) 메트릭/임계 보정
        params = self.clsDock.get_params() if hasattr(self, "clsDock") and callable(self.clsDock.get_params) else {}
        metric = str(params.get("metric", "SAD")).upper()
        s, b = float(strict_val), float(base_val)
        if metric in ("SAD","SID","SCC","SAM") and s > b:
            s, b = b, s  # 거리형: s<=b 보정

        # 3) ROI 확장 포함 오버레이 생성 (기존 유틸 재사용)
        roi_mask = getattr(self, "_work_mask", None)
        overlay = build_target_only_trafficlight_overlay_rgba(
            classmap=classmap,
            topk_cids=self._class_topk_cids,
            topk_vals=self._class_topk_vals,
            target_cid=int(cid),
            strict=s,
            base=b,
            metric=metric,
            roi_mask=roi_mask,
            alpha=180,
        )

        # 4) MapView에만 적용 (레이어 트리 비참여)
        self._apply_overlay_to_mapview(overlay)

        # 5) 카드 숫자 갱신(선택): 기존 계산 함수를 그대로 사용하거나 간단히 재계산
        try:
            H, W = classmap.shape
            sel = (self._class_topk_cids.astype(np.int32) == int(cid))
            present = sel.any(axis=2)
            idx = sel.argmax(axis=2)
            scores = np.take_along_axis(self._class_topk_vals, idx[..., None], axis=2).squeeze(-1).astype(np.float32)
            scores[~present] = np.nan
            roi = roi_mask.astype(bool) if roi_mask is not None else np.ones((H, W), dtype=bool)
            valid = np.isfinite(scores) & (classmap == int(cid)) & roi
            yellow = int(np.sum(valid & (scores < s)))
            green  = int(np.sum(valid & (scores >= s) & (scores < b)))
            red    = int(np.sum(valid & (scores >= b)))
            if hasattr(self, "clsDock") and self.clsDock:
                self.clsDock.set_value_cards(great=yellow, mid=green, little=red)
        except Exception:
            pass

        self.statusBar().showMessage(
            f"임계 적용 — Class={cid}, Strict={s:.3f}, Base={b:.3f}", 3000
        )

            
    def _apply_overlay_to_mapview(self, overlay: np.ndarray):
        """
        RGBA 오버레이를 MapView에 적용 (기존 _apply_confidence_overlay_to_mapview 로직 재사용)
        
        Args:
            overlay: RGBA 오버레이 (H, W, 4)
        """
        try:
            # 이전 오버레이 제거
            self._clear_confidence_overlay_mapview()
            
            # 기존 로직과 동일한 numpy → QImage → QPixmap 변환
            h, w, _ = overlay.shape
            qimg = QtGui.QImage(overlay.data, w, h, 4*w, QtGui.QImage.Format_RGBA8888)
            qimg = qimg.copy()  # numpy 버퍼 라이프사이클 보호
            pix = QtGui.QPixmap.fromImage(qimg)
            
            # MapView에 직접 추가 (기존과 동일한 로직)
            scene = getattr(self._map_view, "scene", None)
            if callable(scene):
                scene = self._map_view.scene()
            if scene is None:
                QtWidgets.QMessageBox.information(self, "안내", "MapView scene을 찾을 수 없습니다.")
                return
                
            item = scene.addPixmap(pix)
            item.setZValue(1e6)
            item.setOpacity(1.0)
            item.setOffset(0, 0)
            try:
                item.setAcceptedMouseButtons(Qt.NoButton)             # ★ 클릭 통과
                item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
                item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
            except Exception:
                pass
            self._conf_item = item
            
        except Exception as e:
            logging.exception("[Overlay] apply to MapView failed")
            QtWidgets.QMessageBox.critical(self, "오류", "오버레이 적용 중 오류가 발생했습니다.")
            
    # ---- MainWindow 내부 공용 헬퍼 ----
    def _mute_roi_inputs(self, mute: bool = True):
        """ROI 계열 컨트롤러의 마우스 이벤트 수신을 일괄 on/off."""
        try:
            if getattr(self, "roi", None):
                if mute:
                    self.roi.set_mode(None)
                if hasattr(self.roi, "set_accept_events"):
                    self.roi.set_accept_events(not mute)
            if getattr(self, "class_roi_controller", None):
                if mute:
                    self.class_roi_controller.set_mode(None, -1)
                if hasattr(self.class_roi_controller, "set_accept_events"):
                    self.class_roi_controller.set_accept_events(not mute)
            if getattr(self, "diffusion_roi_controller", None):
                if mute:
                    self.diffusion_roi_controller.set_mode(None)
                if hasattr(self.diffusion_roi_controller, "set_accept_events"):
                    self.diffusion_roi_controller.set_accept_events(not mute)
        except Exception:
            logging.exception("[ClickMode] mute ROI controllers failed")

    def _safe_set_click_mode(self, mode: ClickMode):
        """
        ★ 안전한 클릭 모드 설정: analysis_selection_dialog가 열려있을 때는 NONE으로 변경 방지
        """
        try:
            # ★ analysis_selection_dialog가 열려있으면 분석 영역 지정만 가능
            dialog_open = (
                hasattr(self, "_analysis_selection_dialog") and 
                self._analysis_selection_dialog is not None and
                self._analysis_selection_dialog.isVisible()
            )
            
            if dialog_open:
                # Dialog가 열려있을 때는 NONE으로 변경 방지 (사각형 모드일 때만 허용)
                if mode == ClickMode.NONE:
                    # 사각형 모드가 활성화되어 있는지 확인
                    btn_rect_checked = False
                    if hasattr(self._analysis_selection_dialog, "btnRect") and self._analysis_selection_dialog.btnRect:
                        btn_rect_checked = self._analysis_selection_dialog.btnRect.isChecked()
                    if not btn_rect_checked:
                        logging.debug("[SafeSetClickMode] Dialog 열려있음 - NONE 모드 변경 방지 (사각형 모드가 아닐 때)")
                        return  # NONE으로 변경하지 않음
                # INSPECT 모드도 방지
                elif mode == ClickMode.INSPECT:
                    logging.debug("[SafeSetClickMode] Dialog 열려있음 - INSPECT 모드 변경 방지")
                    return  # INSPECT로 변경하지 않음
            
            # Dialog가 열려있지 않거나, 허용된 모드면 설정
            if hasattr(self, "pixelClick") and self.pixelClick:
                self.pixelClick.set_mode(mode)
            if hasattr(self._map_view, "set_click_mode"):
                self._map_view.set_click_mode(mode)
        except Exception:
            logging.exception("[SafeSetClickMode] failed")
    
    def _enter_pixel_click_mode(self, mode: ClickMode, *, mute_roi: bool = True):
        """
        공용: 픽셀 클릭 모드로 진입한다.
        - mode: ClickMode.LABEL / ClickMode.DIFFUSION_SEED / ClickMode.INSPECT /
                ClickMode.ANALYSIS / ClickMode.NONE
        - mute_roi=True면 ROI 계열 입력을 잠시 비활성화
        - ★ analysis_selection_dialog가 열려있으면 INSPECT나 NONE 모드로 전환하지 않음
        """
        try:
            # ★ analysis_selection_dialog가 열려있으면 분석 영역 지정만 가능
            dialog_open = (
                hasattr(self, "_analysis_selection_dialog") and 
                self._analysis_selection_dialog is not None and
                self._analysis_selection_dialog.isVisible()
            )
            
            if dialog_open:
                # Dialog가 열려있을 때는 ANALYSIS 모드만 허용
                if mode not in (ClickMode.ANALYSIS, ClickMode.NONE):
                    # ANALYSIS 모드가 아니면 무시 (다른 모드로 전환 방지)
                    logging.debug(f"[EnterClickMode] Dialog 열려있음 - {mode} 모드 무시 (ANALYSIS만 허용)")
                    return
                # NONE 모드는 사각형 모드일 때만 허용
                if mode == ClickMode.NONE:
                    # 사각형 모드가 활성화되어 있는지 확인
                    btn_rect_checked = False
                    if hasattr(self._analysis_selection_dialog, "btnRect") and self._analysis_selection_dialog.btnRect:
                        btn_rect_checked = self._analysis_selection_dialog.btnRect.isChecked()
                    if not btn_rect_checked:
                        logging.debug("[EnterClickMode] Dialog 열려있음 - NONE 모드 무시 (사각형 모드가 아닐 때)")
                        return
            
            if mute_roi:
                self._mute_roi_inputs(True)

            # 이전 모드 기억(복귀용)
            self._prev_click_mode = getattr(self, "_prev_click_mode", ClickMode.INSPECT)
            try:
                self._prev_click_mode = self.pixelClick.mode
            except Exception:
                pass

            # ★ 안전한 모드 설정 사용
            self._safe_set_click_mode(mode)

            self.statusBar().showMessage(f"픽셀 클릭 모드: {mode.name}", 1800)
        except Exception:
            logging.exception("[ClickMode] enter failed")

    def _leave_pixel_click_mode(self, *, restore: bool = True):
        """
        공용: 픽셀 클릭 모드 해제.
        - restore=True면 이전 모드로 복귀, 실패시 기본 INSPECT
        - ROI 입력도 함께 복구
        """
        try:
            target = ClickMode.INSPECT
            if restore and hasattr(self, "_prev_click_mode") and isinstance(self._prev_click_mode, ClickMode):
                target = self._prev_click_mode

            self._safe_set_click_mode(target)

            # ROI 입력 복구
            self._mute_roi_inputs(False)

            self.statusBar().showMessage("픽셀 클릭 모드 해제", 1500)
        except Exception:
            logging.exception("[ClickMode] leave failed")


    def _ensure_class_roi_controller(self) -> bool:
        """클래스 ROI 컨트롤러를 지연 생성. 이미지가 있어야 생성 가능."""
        if self.class_roi_controller is not None:
            return True
            
        # 이미지가 없으면 생성 불가
        if not hasattr(self, "rgb_image") or self.rgb_image is None:
            return False
            
        try:
            self.class_roi_controller = ClassROIController(
                map_view=self._map_view,
                layer_register=self.register_layer,
                img_shape_fn=lambda: (self._map_view.img_h, self._map_view.img_w),
                get_work_area_mask=self._get_work_area_mask,
                temp_overlay = self._analysis_overlay,
                parent=self
            )
            
            # 클래스 ROI 생성 시그널 연결
            self.class_roi_controller.classROIAdded.connect(self._on_class_roi_added)
            # ROI 모드 변경 시그널을 Dock과 연결하여 동기화
            self.class_roi_controller.classROIModeChanged.connect(self._on_class_roi_mode_changed)
            return True
        except Exception as e:
            logging.exception("[ClassROI] controller creation failed")
            return False
            
    def _get_work_area_mask(self) -> Optional[np.ndarray]:
        """작업 영역 마스크 반환"""
        return getattr(self, "_work_mask", None)
                
    def _on_class_roi_mode_requested(self, mode, class_id):
        self._switch_roi_owner(ROIInputOwner.CLASS)

        # 작업영역 ROI는 끔
        if self._ensure_roi_controller():
            self.roi.set_mode(None)
            if hasattr(self.roi, "set_accept_events"):
                self.roi.set_accept_events(False)

        if not self._ensure_class_roi_controller():
            QtWidgets.QMessageBox.information(self, "안내", "이미지를 먼저 로드하세요.")
            return

        # 클래스 ROI 모드 적용 및 수신 허용
        self.class_roi_controller.set_mode(mode, class_id)
        if hasattr(self.class_roi_controller, "set_accept_events"):
            self.class_roi_controller.set_accept_events(bool(mode))

        # 클릭 모드 OFF
        self._safe_set_click_mode(ClickMode.NONE)

        if mode:
            self.statusBar().showMessage(f"클래스 {class_id} {mode} ROI 모드 활성화", 3000)
        else:
            self.statusBar().showMessage("클래스 ROI 모드 해제", 2000)

            
    @QtCore.pyqtSlot(bool, int)
    def _on_class_roi_active_requested(self, active: bool, class_id: int):
        """클래스 ROI 활성화 요청 처리"""
        try:
            if not self._ensure_class_roi_controller():
                return
                
            self.class_roi_controller.set_active(active, class_id)
            
        except Exception as e:
            logging.exception("[ClassROI] active request failed")
            
    @QtCore.pyqtSlot(object)
    def _on_class_roi_mode_changed(self, mode):
        """ClassROIController의 모드 변경을 PixelClassificationDock에 동기화"""
        try:
            if hasattr(self, 'clsDock') and self.clsDock:
                # mode가 None이면 비활성화, 그렇지 않으면 활성화
                self.clsDock.set_roi_mode_active(mode is not None, mode)
        except Exception as e:
            logging.exception("[ClassROI] mode sync failed")
            
    def _on_class_roi_added(self, roi_record):
        # 1) ClassROI를 '분석 선택'으로도 확정해둔다 (상세보기의 입력 소스로 사용)
        self._analysis_mask = np.asarray(roi_record.mask).astype(bool)

        # rect가 None일 수도 있으니, 없으면 mask로부터 bounding rect를 계산한다
        if roi_record.rect is not None:
            self._analysis_rect = self._clamp_rect_to_image(roi_record.rect)
        else:
            yy, xx = np.where(self._analysis_mask)
            if yy.size > 0:
                x0, x1 = int(xx.min()), int(xx.max()) + 1
                y0, y1 = int(yy.min()), int(yy.max()) + 1
                self._analysis_rect = QtCore.QRect(x0, y0, x1 - x0, y1 - y0)
            else:
                self._analysis_rect = None  # 빈 ROI 보호

        # 2) 맵에 임시 오버레이 표시 (지도 전용)
        ORANGE = (255, 200, 0, 120)
        self._analysis_overlay.show_mask(
            self._analysis_mask,
            rgba=ORANGE,   # ★ 고정 색상으로 통일
            z=990000
        )

        # 3) 상태 표시
        px = int(self._analysis_mask.sum())
        msg = f"선택(분석) 영역 확정: {px} px"
        if self._analysis_rect:
            r = self._analysis_rect
            msg += f", rect=({r.x()},{r.y()},{r.width()}x{r.height()})"
        self.statusBar().showMessage(msg, 3000)

        # 4) 클릭 모드 복귀 (분석 모드 상태 확인)
        self._sync_click_mode_with_analysis_state()

    # ==================== Region Growing 관련 메서드 ====================    
    def setup_region_growing(self, hsi_data: np.ndarray) -> None:
        """Region Growing 서비스 초기화"""
        try:
            self.region_growing_service.set_hsi_data(hsi_data)
            logging.info("[RegionGrowing] Service initialized with HSI data")
        except Exception as e:
            logging.exception("[RegionGrowing] Setup failed")
    
    def configure_region_growing(
        self,
        metric: str = "SAD",
        threshold: float = 0.05,
        margin: float = 0.03,          # ★ 추가
        min_region_size: int = 10,
        max_region_size: int = 10000,
        connectivity: int = 4
    ) -> None:
        try:
            self.region_growing_service.set_config(
                metric=metric,
                threshold=threshold,
                margin=margin,          # ★ 전달
                min_region_size=min_region_size,
                max_region_size=max_region_size,
                connectivity=connectivity
            )
            logging.info(f"[RegionGrowing] Config updated: {metric}, τ={threshold}, δ={margin}")
        except Exception as e:
            logging.exception("[RegionGrowing] Config update failed")
        
    def perform_region_growing_from_spectrum(
        self, 
        x: int, 
        y: int, 
        spectrum: np.ndarray
    ) -> None:
        """특정 스펙트럼을 시드로 하여 Region Growing 수행"""
        try:
            result = self.region_growing_service.grow_region_from_spectrum(x, y, spectrum)
            
            if result is not None:
                self.statusBar().showMessage(
                    f"Region Growing 완료: 크기={result.region_size}, 커스텀 스펙트럼 사용", 
                    3000
                )
            else:
                QtWidgets.QMessageBox.warning(self, "경고", "Region Growing에 실패했습니다.")
                
        except Exception as e:
            logging.exception("[RegionGrowing] Spectrum handler failed")
            QtWidgets.QMessageBox.critical(self, "오류", f"Region Growing 중 오류가 발생했습니다: {e}")
    
    @QtCore.pyqtSlot(object)
    def _on_region_grown(self, result) -> None:
        """Region Growing 완료 시 호출되는 슬롯"""
        try:
            # 영역을 레이어로 추가
            self._add_region_as_layer(result)
            
            # 통계 정보 표시
            stats = result.statistics
            logging.info(
                f"[RegionGrowing] Region grown: size={stats['region_size']}, "
                f"compactness={stats['compactness']:.3f}, "
                f"spectral_variance={stats['spectral_variance']:.6f}"
            )
            
        except Exception as e:
            logging.exception("[RegionGrowing] Region grown handler failed")
    
    @QtCore.pyqtSlot(str)
    def _on_region_growing_failed(self, error_message: str) -> None:
        """Region Growing 실패 시 호출되는 슬롯"""
        logging.error(f"[RegionGrowing] Failed: {error_message}")
        QtWidgets.QMessageBox.warning(self, "Region Growing 실패", error_message)
    
    def _add_region_as_layer(self, result) -> None:
        """Region Growing 결과를 레이어로 추가"""
        try:
            region_mask = result.region_mask
            region_name = f"Region_{result.seed_coordinate[0]}_{result.seed_coordinate[1]}"
            
            # 마스크를 RGBA 오버레이로 변환
            rgba_overlay = np.zeros((*region_mask.shape, 4), dtype=np.uint8)
            rgba_overlay[region_mask, 0] = 255  # Red
            rgba_overlay[region_mask, 1] = 0    # Green
            rgba_overlay[region_mask, 2] = 0    # Blue
            rgba_overlay[region_mask, 3] = 128  # Alpha (반투명)
            
            # 레이어 매니저에 오버레이 등록
            self.layer_manager.register_overlay(
                name=region_name,
                rgba_or_rgb_u8=rgba_overlay,
                visible=True,
                parent=None
            )
            
            logging.info(f"[RegionGrowing] Region added as layer: {region_name}")
            
        except Exception as e:
            logging.exception("[RegionGrowing] Add layer failed")
    
    def _convert_screen_to_image_coords(self, screen_x: int, screen_y: int) -> Tuple[int, int]:
        """화면 좌표를 이미지 좌표로 변환"""
        try:
            # MapView의 좌표 변환 사용
            if hasattr(self, "_map_view") and self._map_view:
                # MapView의 mapToScene 또는 mapFromScene 메서드 사용
                # 현재는 단순히 그대로 반환 (실제 구현 시 MapView 참조 필요)
                return screen_x, screen_y
            else:
                return screen_x, screen_y
        except Exception as e:
            logging.exception("[RegionGrowing] Coordinate conversion failed")
            return screen_x, screen_y
        
    def _on_start_work_rect_roi(self):
        if not self._ensure_roi_controller():
            QtWidgets.QMessageBox.information(self, "안내", "이미지를 먼저 로드하세요.")
            return

        # ✅ 작업영역 모드로 명시 전환 (가드에서 막히지 않도록)
        self._roi_capture_target = "work"

        # (권장) ROIController가 자동 레이어 등록하지 않게 유지
        if getattr(self, "roi", None):
            self.roi.register_on_finish = False

        if hasattr(self, "clsDock") and self.clsDock and hasattr(self.clsDock, "set_roi_drawing_state"):
            self.clsDock.set_roi_drawing_state(True)

        self._set_roi_mode_from_toolbar('rect')
        self.statusBar().showMessage("사각형 ROI: 드래그로 지정, 놓으면 완료.", 3000)
       
    # MainWindow 클래스 내부에 추가(아무 위치 OK, 호출 전 정의되면 됨)
    def _clamp_rect_to_image(self, rect: QtCore.QRect) -> QtCore.QRect:
        """입력 rect를 현재 이미지 크기(H,W) 경계로 클램프."""
        try:
            H = int(getattr(self._map_view, "img_h", 0) or 0)
            W = int(getattr(self._map_view, "img_w", 0) or 0)
            if H <= 0 or W <= 0:
                return rect  # 이미지 정보가 없으면 원본 반환

            x = max(0, min(rect.x(), W-1))
            y = max(0, min(rect.y(), H-1))
            # width/height는 남은 영역 내에서 보정
            w = max(1, min(rect.width(),  W - x))
            h = max(1, min(rect.height(), H - y))
            return QtCore.QRect(x, y, w, h)
        except Exception:
            logging.exception("[RectClamp] failed")
            return rect

    def _on_work_roi_added(self, rec):
        # ★ 분석 캡처 중이면 무시
        if getattr(self, "_roi_capture_target", "work") == "analysis":
            return

        m = np.asarray(rec.mask, dtype=bool)
        rect = getattr(rec, "rect", None)

        # ★ WorkspaceDialog 열림: 캐시만
        if self._is_workspace_dialog_open():
            self._workspace_temp_mask = m
            self._workspace_temp_rect = rect
            if self._workspace_temp_overlay:
                self._workspace_temp_overlay.clear()
                self._workspace_temp_overlay.show_mask(m, rgba=(0,170,255,120), z=990000)
            self.statusBar().showMessage(
                f"작업 영역(임시) 저장 — '완료'를 누르면 반영 ({int(m.sum())} px)", 3000
            )
            return

        # ★ 다이얼로그 닫힘: 즉시 커밋 (기존 동일)
        self._work_mask = m
        if hasattr(self.layer_manager, "set_roi"):
            self.layer_manager.set_roi(m)
        try:
            if hasattr(self, "region_growing_service") and self.region_growing_service:
                self.region_growing_service.set_allowed_mask(self._work_mask)
            if hasattr(self, "analysisCtrl") and self.analysisCtrl:
                self.analysisCtrl.set_allowed_mask(self._work_mask)
        except Exception:
            logging.exception("[WorkROI] set_allowed_mask failed")

        self.register_layer("작업 영역", m, type="mask.roi", visible=True,
                            meta={"color": (0,170,255,120)})
        if rect is not None:
            self.current_selection_rect = rect
        self._sync_click_mode_with_analysis_state()
        self.statusBar().showMessage(f"작업 영역 지정: {int(m.sum())} px", 3000)


    @QtCore.pyqtSlot(list)
    def _on_user_labeling_candidates_registered(self, rows: list):
        """
        UserLabelingDialog → '후보 등록' 클릭 시 호출.
        rows: [{"y":int,"x":int,"cid":int}, ...]
        동작:
        1) PixelLabelingDialog 보장/표시 (반드시 on_pixel_labeling_dialog_run() 경로 사용)
        2) '선택된 픽셀' 테이블에 누적 추가
        """
        try:
            import logging

            # 1) PixelLabelingDialog 보장
            #    ❗ 직접 PixelLabelingDialog(...) 를 만들지 말고,
            #    항상 on_pixel_labeling_dialog_run() 을 통해 생성/초기화(시그널 연결까지) 한다.
            if not hasattr(self, "_labeling_dialog") or self._labeling_dialog is None:
                # 기존 메뉴/버튼에서 사용하는 것과 동일한 초기화 경로
                logging.info("[Labeling] PixelLabelingDialog가 없어 on_pixel_labeling_dialog_run()으로 생성합니다.")
                self.on_pixel_labeling_dialog_run()

            dlg = getattr(self, "_labeling_dialog", None)
            if dlg is None:
                logging.warning("[Labeling] PixelLabelingDialog 생성에 실패했습니다.")
                return

            # 다이얼로그 보이기/포커스
            if not dlg.isVisible():
                dlg.show()
            dlg.raise_()
            dlg.activateWindow()

            # 2) 클래스 옵션 재주입 (.info/API 최신 내용 반영)
            try:
                class_options = self._resolve_current_class_options()
                if class_options and hasattr(dlg, "set_class_options"):
                    dlg.set_class_options(class_options)
            except Exception:
                logging.exception("[PixelLabeling] 클래스 옵션 재주입 실패")

            # 3) 후보 픽셀 누적 추가
            if hasattr(dlg, "append_selected_pixels"):
                dlg.append_selected_pixels(rows)
            else:
                # (하위 버전 대비 fallback, 필요 없으면 삭제 가능)
                try:
                    tbl = dlg.tableSelectedPixels
                    exist = []
                    for r in range(tbl.rowCount()):
                        idx = tbl.item(r, 0).text() if tbl.item(r, 0) else str(r+1)
                        loc = tbl.item(r, 1).text() if tbl.item(r, 1) else ""
                        cls = tbl.item(r, 2).text() if tbl.item(r, 2) else ""
                        opt = tbl.item(r, 3).text() if tbl.item(r, 3) else "-"
                        exist.append((idx, loc, cls, opt))
                    start = len(exist)
                    for i, rec in enumerate(rows):
                        y, x, cid = int(rec["y"]), int(rec["x"]), int(rec["cid"])
                        exist.append((str(start+i+1), f"({x}, {y})", str(cid), "-"))
                    tbl.setRowCount(0)
                    for r, (idx, loc, cls, opt) in enumerate(exist):
                        tbl.insertRow(r)
                        tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(idx))
                        tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(loc))
                        tbl.setItem(r, 2, QtWidgets.QTableWidgetItem(cls))
                        tbl.setItem(r, 3, QtWidgets.QTableWidgetItem(opt))
                except Exception:
                    logging.exception("[Labeling] fallback append to PixelLabelingDialog failed")

            self.statusBar().showMessage(
                f"후보 {len(rows)}개가 픽셀 라벨링 테이블에 추가되었습니다.", 3000
            )

        except Exception:
            logging.exception("[Labeling] route candidates to PixelLabelingDialog failed")
            QtWidgets.QMessageBox.warning(self, "경고", "후보 등록 결과를 픽셀 라벨링에 반영하지 못했습니다.")

    def _store_topk_fullsize(self, map_name: str,
                            cm_full: np.ndarray,      # (H,W)
                            cids_sub: np.ndarray,     # (h,w,1) or (H,W,1)
                            vals_sub: np.ndarray,     # (h,w,1) or (H,W,1)
                            roi_mask: Optional[np.ndarray] = None) -> None:
        H, W = cm_full.shape
        # 이미 전역 크기면 그대로 저장
        if cids_sub.shape[:2] == (H, W):
            c1 = cids_sub.astype(np.int32,  copy=False)[..., :1]
            v1 = vals_sub.astype(np.float32, copy=False)[..., :1]
            self._topk_cache[map_name] = (c1, v1)
            return

        # ROI 서브 크기라면 전역으로 확장
        if roi_mask is None:
            raise ValueError("ROI 분류 결과는 roi_mask가 필요합니다.")
        m = np.asarray(roi_mask, dtype=bool)
        assert m.shape == (H, W)

        c1 = np.full((H, W, 1), -1,   dtype=np.int32)    # ROI 밖 클래스 없음
        v1 = np.full((H, W, 1), np.inf, dtype=np.float32) # ROI 밖 값은 히스토그램에서 제외

        ys, xs = np.where(m)
        y0, y1 = int(ys.min()), int(ys.max())+1
        x0, x1 = int(xs.min()), int(xs.max())+1

        c1[y0:y1, x0:x1, 0] = cids_sub[..., 0].astype(np.int32,  copy=False)
        v1[y0:y1, x0:x1, 0] = vals_sub[..., 0].astype(np.float32, copy=False)

        self._topk_cache[map_name] = (c1, v1)
        
    def _register_map_semantics(self, name: str, metric: str, *, distance: Optional[bool]=None):
        if distance is None:
            distance = metric.upper() in ("SAD", "SID", "SCC", "SAM")  # 전부 거리형(작을수록 유사)
        self._map_semantics[name] = {"metric": metric, "distance": bool(distance)}

    def _is_distance_map(self, name: str) -> bool:
        return bool(self._map_semantics.get(name, {}).get("distance", True))


    # === MainWindow 클래스 내부에 추가 (classmap_labeling_dialog.py)===
    def _cb_classmap_names(self) -> list[str]:
        """
        LayerManager에 등록된 모든 'top-level classmap' 이름을 반환.
        분류/디퓨전 모두 포함 (예: "classification", "diffusion1.classmap", ...)
        """
        names = []
        try:
            if hasattr(self.layer_manager, "_classmap_store"):
                names = list(self.layer_manager._classmap_store.keys())
        except Exception:
            logging.exception("[CB] classmap_names failed")
        return names

    def _cb_classmap_data(self, name: str) -> Optional[np.ndarray]:
        """
        클래스맵 데이터 반환: int32(H,W)
        """
        try:
            return self.layer_manager.get_classmap(name)
        except Exception:
            logging.exception("[CB] classmap_data failed: %s", name)
            return None

    def _cb_topk_for_map(self, name: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        (cids(H,W,K), vals(H,W,K)) 반환.
        - classification: 분류 직후 _topk_cache["classification"] 저장
        - diffusion 계열: 저장 시 _topk_cache[name] 저장 (이미 코드 있음)
        """
        tk = getattr(self, "_topk_cache", {})
        cids_vals = tk.get(name)
        if isinstance(cids_vals, tuple) and len(cids_vals) == 2:
            return cids_vals
        # 없으면 None
        return None, None

    def _cb_class_pixels(self, name: str, cid: int) -> tuple[np.ndarray, np.ndarray]:
        """
        (map_name, cid) → 해당 클래스의 모든 (ys, xs).
        """
        cm = self._cb_classmap_data(name)
        if cm is None:
            return np.array([], dtype=int), np.array([], dtype=int)
        ys, xs = np.where(cm == int(cid))
        return ys.astype(int, copy=False), xs.astype(int, copy=False)

    def _cb_spectra_for(self, name: str, cid: int, max_n: int) -> list[tuple[np.ndarray, tuple[int,int]]]:
        """
        (map_name, cid) → 최대 max_n 개의 (spec, (y,x)) 튜플 리스트.
        - 스펙트럼은 self.cfg["data"]에서 직접 추출 (HSI cube 기준)
        - 좌표는 해당 classmap의 클래스 위치에서 샘플링
        """
        out: list[tuple[np.ndarray, tuple[int,int]]] = []
        try:
            cube = self.cfg["data"] if hasattr(self, "cfg") else None
            cm = self._cb_classmap_data(name)
            if cube is None or cm is None:
                return out

            ys, xs = np.where(cm == int(cid))
            if ys.size == 0: return out

            take = min(int(max_n), ys.size)
            sel = np.random.choice(ys.size, take, replace=False)

            # (선택) 유사도 기준 샘플링을 원하면 _cb_topk_for_map(name) 값을 활용해 정렬 후 선택 가능
            # 여기선 단순 무작위 샘플링
            for i in sel:
                y, x = int(ys[i]), int(xs[i])
                spec = cube[y, x, :].astype(float, copy=False)
                out.append((spec, (y, x)))
        except Exception:
            logging.exception("[CB] spectra_for failed: %s / %s", name, cid)
        return out

    def _bin_spectra_by_range(self, map_name: str, cid: int, lo: float, hi: float, max_n: int):
        """
        히스토그램의 [lo, hi) 구간을 만족하는 픽셀 중에서 최대 max_n개를
        (spec,(y,x))로 반환. Top-1 값을 기준으로 정렬(거리형이면 오름차순, 유사도형이면 내림차순).
        """
        out = []
        try:
            cm = self._cb_classmap_data(map_name)
            cube = self.cfg["data"] if hasattr(self, "cfg") else None
            cids, vals = self._cb_topk_for_map(map_name)   # (H,W,K)
            if cm is None or cube is None or vals is None:
                return out
            # K=1 사용
            v1 = np.asarray(vals[..., 0], dtype=float)
            mask = (cm == int(cid)) & np.isfinite(v1) & (v1 >= float(lo)) & (v1 < float(hi))
            ys, xs = np.where(mask)
            if ys.size == 0: return out

            take = min(int(max_n), ys.size)
            vals_in_bin = v1[ys, xs]
            if self._is_distance_map(map_name):
                order = np.argsort(vals_in_bin)[:take]     # 작을수록 유사
            else:
                order = np.argsort(-vals_in_bin)[:take]    # 클수록 유사
            ys, xs = ys[order], xs[order]

            for y, x in zip(ys, xs):
                spec = cube[int(y), int(x), :].astype(float, copy=False)
                out.append((spec, (int(y), int(x))))
        except Exception:
            logging.exception("[CB] bin_spectra_by_range failed")
        return out
    
    def _prepare_spec_library(self, spec_lib_dicts, expect_c: int):
        from services.spec_library import build_library_from_spec_libs
        lib = {}
        id_to_name = {}
        try:
            lib, id_to_name = build_library_from_spec_libs(spec_lib_dicts, expect_c)
            if not lib:
                self._show_status("스펙트럼 라이브러리가 비어 있습니다.", level="warn")
                # 비어있어도 일단 빈 튜플 반환(호출부에서 판단)
            # UI 갱신
            self._refresh_class_combo(list(id_to_name.items()))
            self._show_status(f"라이브러리 로드 완료: {len(lib)} classes")
            return lib, id_to_name
        except Exception as e:
            self._show_status(f"라이브러리 로드 실패: {e}", level="error")
            # 실패 시 명시적으로 예외 재발생(호출부에서 try/except 권장)
            raise
        
    # 수정 후
    def _refresh_class_combo(self, cid_name_items):
        # 1) self.cboClass 우선
        combo = getattr(self, "cboClass", None)
        # 2) 못 찾으면 findChild로 탐색
        if combo is None:
            combo = self.findChild(QtWidgets.QComboBox, "cboClass")
        # 3) 그래도 없으면 건너뛰고 로그만
        if combo is None:
            logging.warning("[UI] QComboBox 'cboClass'를 찾을 수 없습니다. 콤보 갱신을 건너뜁니다.")
            return

        combo.blockSignals(True)
        combo.clear()
        for cid, name in sorted(cid_name_items, key=lambda x: x[0]):
            combo.addItem(f"{cid} - {name}", cid)
        combo.blockSignals(False)


    def _show_status(self, msg: str, level: str="info"):
        # 상태바/로그 공용
        self.statusBar().showMessage(msg, 5000)
        getattr(logging, level if level in ("info","warn","error","debug") else "info")(msg)
                
    def _on_analysis_op_changed(self, op: str):
        if op in ("union", "subtract"):
            self._analysis_op = op
            self.statusBar().showMessage(f"지정 영역 연산 모드: {op}", 1500)

    @QtCore.pyqtSlot(str, str)
    @QtCore.pyqtSlot()
    def _on_start_analysis_pixel(self):
        """분석 픽셀 모드 시작 - _sync_click_mode_with_analysis_state()로 통합 처리"""
        # ★ analysis_selection_dialog가 열려있을 때만 작동
        if not (hasattr(self, "_analysis_selection_dialog") and 
                self._analysis_selection_dialog is not None and
                self._analysis_selection_dialog.isVisible()):
            return  # Dialog가 열려있지 않으면 실행하지 않음
        
        # 모든 상태 변경은 _sync_click_mode_with_analysis_state()에서 통합 처리
        QtCore.QTimer.singleShot(10, lambda: self._sync_click_mode_with_analysis_state())
    
    @QtCore.pyqtSlot()
    def _on_analysis_stop(self):
        """분석 모드 종료 - _sync_click_mode_with_analysis_state()로 통합 처리"""
        # 모든 상태 변경은 _sync_click_mode_with_analysis_state()에서 통합 처리
        QtCore.QTimer.singleShot(10, lambda: self._sync_click_mode_with_analysis_state())
    
    def _cache_primary_path(self, cfg: dict) -> Optional[str]:
        """캐시 파일 기준 원천 경로를 일관되게 선택."""
        p = self._pick_path_from_datapath(cfg.get('data_path'))
        if p: return p
        p = self._extract_src_path(cfg)
        if p: return p
        dp = cfg.get('data_path')
        if isinstance(dp, dict):
            # 우선순위 고정
            for k in ("mat", "hdr", "raw"):
                v = dp.get(k)
                if isinstance(v, (str, Path)):
                    ap = os.path.abspath(str(v))
                    if os.path.isfile(ap): return ap
            # 그래도 없으면 아무 유효 파일
            for v in dp.values():
                if isinstance(v, (str, Path)):
                    ap = os.path.abspath(str(v))
                    if os.path.isfile(ap): return ap
        return None
           
    def _count_classes_and_samples(self, d: dict[int, np.ndarray]) -> tuple[int, int]:
        """
        d: {cid: (N,C)} 또는 (C,)
        return: (num_classes, num_samples)
        - num_classes: CID 개수
        - num_samples: 각 CID의 행 수 합계 (1D는 1로 간주)
        """
        if not isinstance(d, dict) or not d:
            return 0, 0
        classes = 0
        samples = 0
        for _, v in d.items():
            a = np.asarray(v)
            if a.ndim == 1:
                samples += 1
            elif a.ndim == 2:
                samples += int(a.shape[0])
            classes += 1
        return classes, samples

    def _set_analysis_op(self, op: str):
        self._analysis_op = op if op in ("union","subtract") else "union"
        # 버튼 시각만 업데이트(신호 내보내지 않음)
        if getattr(self, "clsDock", None) and hasattr(self.clsDock, "set_analysis_op"):
            self.clsDock.set_analysis_op(self._analysis_op)

    def _get_analysis_op(self) -> str:
        """
        분석 op를 반환. 버튼 상태를 직접 확인하여 최신 상태 반영.
        버튼 상태가 내부 상태보다 우선순위가 높음.
        """
        # 버튼 상태를 직접 확인 (최신 상태 보장)
        if hasattr(self, "clsDock") and self.clsDock:
            if hasattr(self.clsDock, "_btn_sub") and self.clsDock._btn_sub and self.clsDock._btn_sub.isChecked():
                # "-" 버튼이 활성화되어 있으면 subtract
                op = "subtract"
                # 내부 상태도 즉시 동기화
                if getattr(self, "_analysis_op", "union") != op:
                    self._analysis_op = op
                return op
            elif hasattr(self.clsDock, "_btn_union") and self.clsDock._btn_union and self.clsDock._btn_union.isChecked():
                # "+" 버튼이 활성화되어 있으면 union
                op = "union"
                # 내부 상태도 즉시 동기화
                if getattr(self, "_analysis_op", "union") != op:
                    self._analysis_op = op
                return op
        
        # 버튼이 모두 꺼져 있으면 내부 상태 반환
        return getattr(self, "_analysis_op", "union")

    # ① 전역 분석 마스크 보장
    def _ensure_analysis_ctrl_ready(self) -> bool:
        """analysisCtrl을 초기화하고 shape/allowed_mask를 설정 (픽셀 모드용)"""
        try:
            if not self._ensure_roi_controller():
                return False
            
            # 이미지 크기 확인
            try:
                cfg = getattr(self, "cfg", {})
                data = cfg.get("data")
                if data is None:
                    logging.warning("[AnalysisPixel] 이미지 데이터가 없습니다.")
                    return False
                H, W = data.shape[:2]
            except Exception:
                logging.exception("[AnalysisPixel] get image shape failed")
                return False
            
            # analysisCtrl 생성/연결
            if self.analysisCtrl is None:
                def _set_click(mode_str: str):
                    try:
                        mode = getattr(ClickMode, mode_str)
                        self._safe_set_click_mode(mode)
                    except Exception:
                        logging.exception("[AnalysisPixel] set_click failed")
                
                self.analysisCtrl = AnalysisSelectionController(
                    map_view=self._map_view,
                    roi_controller=self.roi,
                    overlay_service=self._analysis_overlay,
                    set_click_mode=_set_click,
                    get_op=self._get_analysis_op,
                    parent=self
                )
                
                self.analysisCtrl.set_op(self._get_analysis_op())
                self.analysisCtrl.rectFinished.connect(self._on_analysis_rect_finished)
                logging.info("[MainWindow] (pixel mode) connecting analysisMaskChanged signal")
                self.analysisCtrl.analysisMaskChanged.connect(self._on_analysis_mask_changed_from_ctrl)
                logging.info("[MainWindow] (pixel mode) analysisMaskChanged signal connected")
            
            # shape 및 allowed_mask 설정
            self.analysisCtrl.set_shape(H, W)
            work_mask = getattr(self, "_work_mask", None)
            self.analysisCtrl.set_allowed_mask(work_mask)
            logging.debug(f"[AnalysisPixel] analysisCtrl ready: shape=({H},{W})")
            return True
        except Exception:
            logging.exception("[AnalysisPixel] _ensure_analysis_ctrl_ready failed")
            return False
    
    def _ensure_analysis_mask(self, H: int, W: int) -> None:
        m = getattr(self, "_analysis_mask", None)
        if m is None or m.shape != (H, W):
            self._analysis_mask = np.zeros((H, W), dtype=bool)

    # 분석 표시는 오직 임시 오버레이 1장만 사용
    def _publish_analysis_overlay(self):
        m = getattr(self, "analysis", None).mask if hasattr(self, "analysis") else getattr(self, "_analysis_mask", None)
        if m is None or not np.any(m):
            self._clear_analysis_overlay_item()
            return

        H, W = int(self._map_view.img_h), int(self._map_view.img_w)
        if m.shape != (H, W):
            self._clear_analysis_overlay_item()
            return

        self._clear_analysis_overlay_item()
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[m, 0] = 255; rgba[m, 1] = 200; rgba[m, 2] = 0; rgba[m, 3] = 120
        qimg = QtGui.QImage(rgba.data, W, H, 4*W, QtGui.QImage.Format_RGBA8888).copy()
        pix  = QtGui.QPixmap.fromImage(qimg)
        item = (self._map_view.scene() if callable(self._map_view.scene) else self._map_view.scene).addPixmap(pix)
        item.setZValue(1_000_000); item.setOpacity(1.0); item.setPos(0, 0)
        item.setAcceptedMouseButtons(Qt.NoButton)
        item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
        item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
        self._analysis_item = item


    @QtCore.pyqtSlot(object, QtCore.QRect)
    def _on_analysis_mask_changed_from_ctrl(self, mask, rect):
        # ★ 분석 영역 전용: 컨트롤러 → 메인: 메모리 맵만 갱신하고 classmap 정보를 다이얼로그에 전달
        try:
            logging.info(f"[Analysis] _on_analysis_mask_changed_from_ctrl called: mask={mask is not None}, rect={rect}")
            # 분석 영역 처리 중임을 명시 (작업 영역과 구분)
            self._roi_capture_target = "analysis"
            if isinstance(mask, np.ndarray):
                self._ensure_analysis_mask(*mask.shape)
                self._analysis_mask[:] = mask.astype(bool, copy=False)
                self._analysis_rect = rect if rect and rect.isValid() else None
                
                # 임시 오버레이도 유지 (선택적)
                self._publish_analysis_overlay()

                # ★ 다이얼로그가 열려있다면 마스크 변경을 즉시 반영 (Map의 분석 영역 변경 → Dialog 업데이트)
                dlg = getattr(self, "_analysis_selection_dialog", None)
                if dlg is not None:
                    try:
                        # ★ 마스크가 변경될 때마다 즉시 dialog 업데이트
                        # 다이얼로그가 보이지 않아도 업데이트 (다음에 열릴 때 최신 상태 표시)
                        # 마스크만 전달 (classmap 정보는 set_data에서 이미 전달됨)
                        dlg.update_mask(mask)
                        logging.info(f"[Analysis] ★ dialog mask updated: visible={dlg.isVisible()}, mask_sum={int(mask.sum())}, shape={mask.shape}")
                    except Exception:
                        logging.exception("[Analysis] dialog update_mask failed")
            else:
                logging.warning("[Analysis] mask_changed received None mask")
        except Exception:
            logging.exception("[Analysis] mask_changed sync failed")

        # 사각형 버튼이 계속 켜져 있으면 자동 재진입
        QtCore.QTimer.singleShot(0, self._auto_reenter_if_needed)

    def _auto_reenter_if_needed(self):
        # 도크 버튼 상태 읽어, 여전히 '사각형'이면 다시 시작
        if getattr(self.clsDock, "_btn_rect", None) and self.clsDock._btn_rect.isChecked():
            self._on_start_analysis_rect_from_dock()
            
    def _apply_dock_analysis_state(self, mode: str, active: bool, op: str):
        """Dock 분석 상태 변경 처리 - 시그널로 받은 버튼 활성화 상태를 즉시 반영"""
        logging.debug(f"[AnalysisState] received: mode={mode}, active={active}, op={op}")
        
        # op 반영 (SSOT)
        op_norm = op if op in ("union","subtract") else "union"
        self._set_analysis_op(op_norm)
        
        # 사각형 모드가 이미 활성화된 경우 op를 즉시 반영
        if hasattr(self, "analysisCtrl") and self.analysisCtrl:
            if getattr(self.analysisCtrl, "_active", False):
                # 사각형 드로잉 중이면 op를 즉시 업데이트 (다음 사각형에 적용)
                self._rect_active_op = op_norm
                self.analysisCtrl.set_op(op_norm)
                logging.info(f"[AnalysisState] op updated during rect mode: {op_norm} (will apply to next rect)")
        
        # 사각형 모드인 경우 op를 즉시 반영 (활성화 여부와 관계없이)
        if mode == "rect" and active and hasattr(self, "analysisCtrl") and self.analysisCtrl:
            current_op = self._get_analysis_op()
            self._rect_active_op = current_op
            self.analysisCtrl.set_op(current_op)
            logging.debug(f"[AnalysisState] rect mode active, op set to: {current_op}")
        
        # 모든 상태 변경은 최종적으로 _sync_click_mode_with_analysis_state()에서 통합 처리
        # 다음 이벤트 루프에서 실행하여 버튼 상태 변경이 완료된 후 처리
        # 단, op 변경은 이미 위에서 처리했으므로 _sync_click_mode는 상태 동기화만 수행
        QtCore.QTimer.singleShot(10, lambda: self._sync_click_mode_with_analysis_state())

    def _sync_click_mode_with_analysis_state(self):
        """
        [단순화 버전] 분석 영역 지정은 'AnalysisSelectionDialog'만 책임지고,
        Dock 버튼 상태는 더 이상 참조하지 않는다.

        동작 요약
        - WorkspaceDialog 열림: ROI 드로잉 우선 → 클릭 모드 변경/해제 금지
        - AnalysisSelectionDialog 열림: (+/-) × (픽셀/사각형)만으로 분석 모드 강제
        - 둘 다 닫힘: 분석 모드 종료 → 일반 INSPECT로 복귀(라벨/시드 모드면 유지)
        """
        try:
            # 0) 작업 영역 다이얼로그가 열려있으면 ROI 드로잉만 허용 → 건드리지 않음
            if self._is_workspace_dialog_open():
                return

            # 1) 분석 다이얼로그가 열려있으면 ANALYSIS 전용
            if self._is_analysis_dialog_open():
                dlg = getattr(self, "_analysis_selection_dialog", None)

                # 버튼 상태 읽기
                btn_union_checked = bool(getattr(dlg, "btnPlus", None)  and dlg.btnPlus.isChecked())
                btn_sub_checked   = bool(getattr(dlg, "btnMinus", None) and dlg.btnMinus.isChecked())
                btn_pixel_checked = bool(getattr(dlg, "btnPixel", None) and dlg.btnPixel.isChecked())
                btn_rect_checked  = bool(getattr(dlg, "btnRect", None)  and dlg.btnRect.isChecked())

                # 아무 버튼도 켜져 있지 않다면 기본값: '+' & '픽셀'
                if not (btn_union_checked or btn_sub_checked or btn_pixel_checked or btn_rect_checked):
                    if getattr(dlg, "btnPlus", None):
                        dlg.btnPlus.setChecked(True);  btn_union_checked = True
                    if getattr(dlg, "btnPixel", None):
                        dlg.btnPixel.setChecked(True); btn_pixel_checked = True
                    # 최신 op로 동기화
                    self._current_op = "union"
                    self._update_analysis_op()

                # 사각형 모드 우선
                if btn_rect_checked:
                    self._roi_capture_target = "analysis"

                    # 컨트롤러/shape/allowed_mask 보장
                    if not self._ensure_analysis_ctrl_ready():
                        return

                    # 항상 깨끗이 재시작: cancel → set_op → start_rect_selection
                    if hasattr(self, "_restart_analysis_rect"):
                        self._restart_analysis_rect()
                    else:
                        try:
                            self.analysisCtrl.cancel()
                        except Exception:
                            pass
                        current_op = self._get_analysis_op()
                        self.analysisCtrl.set_op(current_op)
                        try:
                            self.analysisCtrl.start_rect_selection()
                        except Exception:
                            logging.exception("[SyncClick] start_rect_selection failed")

                        # ROI 이벤트 확실히 허용
                        if getattr(self, "roi", None) and hasattr(self.roi, "set_accept_events"):
                            self.roi.set_accept_events(True)

                        # 드로잉 충돌 방지: 클릭은 NONE
                        self._safe_set_click_mode(ClickMode.NONE)

                    # (선택) Dock에 ROI 드로잉 표시 연동
                    if getattr(self, "clsDock", None) and hasattr(self.clsDock, "set_roi_drawing_state"):
                        self.clsDock.set_roi_drawing_state(True)
                    return

                # 픽셀(+/-) 모드: ANALYSIS 클릭 모드 유지
                if btn_pixel_checked or btn_union_checked or btn_sub_checked:
                    self._roi_capture_target = "analysis"
                    if not self._ensure_analysis_ctrl_ready():
                        return
                    # 최신 op로 동기화(도중에 +/- 바뀌었을 수 있음)
                    self.analysisCtrl.set_op(self._get_analysis_op())
                    # 드로잉 아님 → 표시 끔
                    if getattr(self, "clsDock", None) and hasattr(self.clsDock, "set_roi_drawing_state"):
                        self.clsDock.set_roi_drawing_state(False)
                    # 클릭 모드를 ANALYSIS로 고정
                    if getattr(self.pixelClick, "mode", None) != ClickMode.ANALYSIS:
                        self._enter_pixel_click_mode(ClickMode.ANALYSIS, mute_roi=True)
                    return

                # (이례적) 여전히 어떤 버튼도 없지만 다이얼로그 열림 → 변경하지 않음
                return

            # 2) 여기로 오면 다이얼로그가 모두 닫힘 → 분석 모드 종료 후 일반 INSPECT 복귀
            self._roi_capture_target = "work"

            # 분석 컨트롤러가 살아있고 active면 cancel
            if getattr(self, "analysisCtrl", None) and getattr(self.analysisCtrl, "_active", False):
                try:
                    self.analysisCtrl.cancel()
                except Exception:
                    pass

            # Dock ROI 드로잉 표시도 끔
            if getattr(self, "clsDock", None) and hasattr(self.clsDock, "set_roi_drawing_state"):
                self.clsDock.set_roi_drawing_state(False)

            # 라벨/시드 모드로 이미 사용 중이면 건드리지 않음
            current_mode = getattr(self.pixelClick, "mode", None)
            if current_mode in (ClickMode.LABEL, ClickMode.DIFFUSION_SEED):
                return

            # user labeling 창 열림 여부
            dlg_user = getattr(self, "_user_labeling_dialog", None)
            is_user_labeling_active = bool(dlg_user and getattr(dlg_user, "isVisible", lambda: False)())

            # classmap 존재 여부(있으면 INSPECT가 유용)
            has_classmap = False
            try:
                names = self._cb_classmap_names()
                has_classmap = bool(names)
            except Exception:
                pass

            # 최종 INSPECT 복귀
            if current_mode != ClickMode.INSPECT:
                self._enter_pixel_click_mode(ClickMode.INSPECT, mute_roi=False)

        except Exception:
            logging.exception("[SyncClickMode] failed")

        
    def _analysis_update(self, *, mask_delta: np.ndarray = None, rect: QtCore.QRect = None,
                        op: str = None, absolute: bool = False, publish: bool = True):
        try:
            H = int(getattr(self._map_view, "img_h", 0) or 0)
            W = int(getattr(self._map_view, "img_w", 0) or 0)
            if H <= 0 or W <= 0: return
            self._ensure_analysis_mask(H, W)

            if absolute:
                if isinstance(mask_delta, np.ndarray) and mask_delta.shape == (H, W):
                    self._analysis_mask[:] = mask_delta.astype(bool, copy=False)
            else:
                op2 = (op or self._get_analysis_op())
                if isinstance(mask_delta, np.ndarray) and mask_delta.shape == (H, W):
                    if op2 == "subtract":
                        self._analysis_mask &= ~mask_delta.astype(bool, copy=False)
                    else:
                        wm = getattr(self, "_work_mask", None)
                        if isinstance(wm, np.ndarray) and wm.shape == (H, W):
                            mask_delta = mask_delta & wm
                        self._analysis_mask |= mask_delta.astype(bool, copy=False)

            self._analysis_rect = self._clamp_rect_to_image(rect) if rect is not None else None
            if self._analysis_rect is None:
                self._update_analysis_rect_from_mask()

            if publish:
                self._publish_analysis_overlay()
        except Exception:
            logging.exception("[Analysis] _analysis_update failed")

    def _mask_to_rgba(self, m: np.ndarray, rgba=(255, 200, 0, 120)) -> np.ndarray:
        H, W = m.shape
        out = np.zeros((H, W, 4), dtype=np.uint8)
        r, g, b, a = rgba
        out[m, 0] = r; out[m, 1] = g; out[m, 2] = b; out[m, 3] = a
        return out
    
    def _clear_analysis_overlay_item(self):
        try:
            item = getattr(self, "_analysis_item", None)
            if item is not None:
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
            self._analysis_item = None
        except Exception:
            logging.exception("[Analysis] clear overlay item failed")
            
    def _current_analysis_state(self):
        """Dock에서 현재 활성화된 분석 도구/연산자를 즉시 읽는다."""
        tool = None
        if hasattr(self, "clsDock") and self.clsDock:
            if getattr(self.clsDock, "_btn_rect", None) and self.clsDock._btn_rect.isChecked():
                tool = "rect"
            elif getattr(self.clsDock, "_btn_pick", None) and self.clsDock._btn_pick.isChecked():
                tool = "pixel"
        op = self._get_analysis_op()  # "union" | "subtract" (버튼 우선 반영)
        active = bool(tool) or (op in ("union","subtract") and (
            getattr(self.clsDock, "_btn_union", None) and self.clsDock._btn_union.isChecked() or
            getattr(self.clsDock, "_btn_sub",   None) and self.clsDock._btn_sub.isChecked()
        ))
        return {"active": active, "tool": tool, "op": op}
    
    @QtCore.pyqtSlot(int, int)
    def _on_map_inspect_click(self, y, x):
        # 1) 작업 영역 다이얼로그 중이면 ROI 드로잉만 허용 → 클릭 무시
        if self._is_workspace_dialog_open():
            return

        # 2) 분석 다이얼로그가 열려있으면 분석 전용 분기
        if self._is_analysis_dialog_open():
            st = self._current_analysis_state()
            self._set_analysis_op(st["op"])
            if st["tool"] == "rect":
                self._on_start_analysis_rect_from_dock()
            else:
                self._handle_analysis_pixel(y, x)
            return

        # 3) 일반 모드: 분석/작업 다이얼로그 모두 없음 → 기본 클릭 처리
        # ② 일반 모드: Top-K/라벨/시드/분석-픽셀 디스패치 (컨트롤러에 on_map_click 없음)
        try:
            mode = getattr(self.pixelClick, "mode", ClickMode.INSPECT)

            if mode == ClickMode.LABEL:
                # 라벨링 다이얼로그 열렸을 때 사용
                self._handle_label_pick(y, x, None)

            elif mode == ClickMode.DIFFUSION_SEED:
                # 확산(Region Growing) 시드 선택
                self._handle_seed_pick(y, x, None)

            elif mode == ClickMode.ANALYSIS:
                # 분석 픽셀(+/-) 모드
                self._handle_analysis_pixel(y, x)

            else:
                pass

        except Exception:
            logging.exception("[Click] default handler failed")
        
    def dispatch(self, action: dict):
        """
        action = {"type": str, ...}
        허용 type:
        - "SET_OP": {"op":"union"|"subtract"}
        - "ENTER_RECT"
        - "ENTER_PIXEL"
        - "EXIT"
        - "APPLY_PIXELS": {"coords":[(y,x),...]}
        - "APPLY_RECT": {"rect":QRect, "mask":np.bool_}
        - "SET_MASK": {"mask":np.bool_}  # 외부 적용용
        """
        t = action.get("type")
        if t == "SET_OP":
            self.analysis.op = "subtract" if action.get("op")=="subtract" else "union"
            if hasattr(self.analysisCtrl, "on_op_changed"):
                self.analysisCtrl.on_op_changed(self.analysis.op)

        elif t == "ENTER_RECT":
            self.analysis.mode = AnalysisMode.RECT
            self._ensure_analysis_ctrl()                   # 없으면 생성
            self.analysisCtrl.set_shape(self._map_view.img_h, self._map_view.img_w, emit=False)
            self.analysisCtrl.set_allowed_mask(getattr(self, "_work_mask", None))
            self.analysisCtrl.set_op(self.analysis.op)     # 현재 op 반영
            self.analysisCtrl.start_rect_selection()       # ★ 자동 진입
            self._mute_roi_inputs(True)
            self._safe_set_click_mode(ClickMode.NONE)
            self._status_bar_state()

        elif t == "ENTER_PIXEL":
            self.analysis.mode = AnalysisMode.PIXEL
            self._ensure_analysis_ctrl()
            self.analysisCtrl.set_shape(self._map_view.img_h, self._map_view.img_w, emit=False)
            self.analysisCtrl.set_allowed_mask(getattr(self, "_work_mask", None))
            self._safe_set_click_mode(ClickMode.ANALYSIS)
            self._mute_roi_inputs(True)
            self._status_bar_state()

        elif t == "EXIT":
            self.analysis.mode = AnalysisMode.NONE
            if self.analysisCtrl and getattr(self.analysisCtrl, "_active", False):
                self.analysisCtrl.cancel()
            self._safe_set_click_mode(ClickMode.INSPECT)
            self._mute_roi_inputs(False)
            self._status_bar_state()

        elif t == "APPLY_PIXELS":
            coords = action.get("coords") or []
            # 컨트롤러에 위임: 클릭 직전 최신 op를 읽어서 SSOT 갱신 → 시그널 → 아래 슬롯에서 렌더
            if self.analysisCtrl: self.analysisCtrl.add_pixels(coords)

        elif t == "APPLY_RECT":
            rect, m = action.get("rect"), action.get("mask")
            # 컨트롤러가 emit한 변경은 아래 슬롯에서 SSOT 반영되므로 여기선 패스(호환 시 사용)
            pass

        elif t == "SET_MASK":
            m = np.asarray(action.get("mask"), dtype=bool)
            self.analysis.mask = m
            self._update_analysis_rect_from_mask()
            self._publish_analysis_overlay()  # 임시 오버레이 1장

    @QtCore.pyqtSlot(list, int)
    def _on_pixels_class_set(self, pixel_coords, new_cid):
        """
        pixel_coords: [(y,x), ...]
        new_cid: int
        1) 실제 레이어(classmap) 데이터 갱신
        2) 맵(오버레이) 리렌더
        3) 픽셀 분류(top-k 캐시) 부분 갱신
        """
        if not pixel_coords:
            return

        # --- (1) 실제 classmap 레이어 갱신 ---
        # LayerManager에서 현재 활성 classmap 레이어를 얻는 방법에 맞춰 호출
        # 예시: active_layer = self.layer_manager.get_active("classmap")
        active_layer = getattr(self.layer_manager, "get_active", lambda t: None)("classmap")
        arr = None

        if active_layer is not None and hasattr(active_layer, "data"):
            arr = active_layer.data  # (H,W) ndarray
        else:
            # fallback: MainWindow가 직접 보유한 self.classmap 사용
            arr = getattr(self, "classmap", None)

        if arr is None:
            logging.error("[MainWindow] classmap 없음: 갱신 실패")
            return

        ys = np.fromiter((y for (y, _) in pixel_coords), dtype=int, count=len(pixel_coords))
        xs = np.fromiter((x for (_, x) in pixel_coords), dtype=int, count=len(pixel_coords))
        arr[ys, xs] = int(new_cid)

        # 레이어 객체가 별도 참조면 다시 할당
        if active_layer is not None and hasattr(active_layer, "data"):
            active_layer.data = arr
        else:
            self.classmap = arr

        # --- (2) 맵(오버레이) 리렌더 ---
        try:
            # ClassmapRenderer가 (H,W) classmap + 팔레트로 QImage/np.uint8 RGB 반환한다고 가정
            palette = self.paletteService.get_palette() if hasattr(self, "paletteService") else {}
            overlay_img = self.classmap_renderer.render(arr, palette)  # QImage 혹은 np.ndarray
            # MapView에 반영 (프로젝트 메서드명에 맞춰 호출)
            if hasattr(self.mapView, "set_classmap_overlay"):
                self.mapView.set_classmap_overlay(overlay_img)
            elif hasattr(self.mapView, "update_overlay"):
                self.mapView.update_overlay("classmap", overlay_img)
            self.mapView.update()
        except Exception:
            logging.exception("[MainWindow] classmap 오버레이 리렌더 실패")

        # --- (3) 픽셀 분류(top-k 캐시) 갱신 ---
        # 캐시가 있다면 바뀐 픽셀만 부분 재계산(성능 ↑). 없으면 전체 재계산.
        try:
            has_cache = hasattr(self, "_topk_cids") and hasattr(self, "_topk_vals") \
                        and self._topk_cids is not None and self._topk_vals is not None

            # 필요 리소스 체크
            lib = getattr(self, "lib", None)                # 분류 라이브러리(dict 등)
            cube = getattr(self, "hsi_cube", None)          # (H,W,C)
            metric = getattr(self, "metric", "SAD")
            K = getattr(self, "topk", 3)

            if (lib is not None) and (cube is not None):
                specs = cube[ys, xs, :].astype(np.float32)             # (N,C)

                if has_cache:
                    pass
                else:
                    pass

            # 우측 정보 패널/독 등을 쓰면 여기서도 리프레시 트리거
            if hasattr(self.layersDock, "refresh_counts_from"):
                self.layersDock.refresh_counts_from(arr)
        except Exception:
            logging.exception("[MainWindow] 부분 재분류/캐시 갱신 실패")
        
    @QtCore.pyqtSlot()
    def _on_clear_analysis_region(self):
        """
        분석 영역(마스크/ROI) 초기화:
        - 내부 마스크 변수/ROI 컨트롤러 정리
        - 맵 오버레이 갱신
        - 관련 독/툴 패널 초기 상태로
        """
        import logging
        try:
            # 1) AnalysisSelectionController의 분석 영역 초기화
            if hasattr(self, "analysisCtrl") and self.analysisCtrl is not None:
                # 시그널을 일시적으로 차단하여 순서 보장
                try:
                    self.analysisCtrl.analysisMaskChanged.disconnect(self._on_analysis_mask_changed_from_ctrl)
                except Exception:
                    pass  # 연결되어 있지 않으면 무시
                
                # 분석 영역 클리어
                self.analysisCtrl.clear_selection()
                
                # 시그널 재연결
                try:
                    self.analysisCtrl.analysisMaskChanged.connect(self._on_analysis_mask_changed_from_ctrl)
                except Exception:
                    pass

            # 2) MainWindow의 분석 마스크/상태 초기화
            self._analysis_mask = None
            self._analysis_rect = None
            
            # ★ 2-1) Dialog가 열려있으면 마스크 초기화를 반영
            # (clear_selection에서 이미 analysisMaskChanged 시그널이 발생하므로,
            #  _on_analysis_mask_changed_from_ctrl에서 자동으로 처리됨)
            # 하지만 확실하게 하기 위해 여기서도 직접 업데이트
            dlg = getattr(self, "_analysis_selection_dialog", None)
            if dlg is not None:
                try:
                    # 현재 마스크 크기 확인 (classmap이 있으면 그 크기 사용)
                    classmap = self.layer_manager.get_classmap("classification")
                    if classmap is not None:
                        H, W = classmap.shape[:2]
                    else:
                        H, W = int(self._map_view.img_h), int(self._map_view.img_w)
                    if H > 0 and W > 0:
                        empty_mask = np.zeros((H, W), dtype=bool)
                        dlg.update_mask(empty_mask)
                        logging.debug(f"[Analysis] dialog mask cleared after reset: shape=({H},{W})")
                except Exception:
                    logging.exception("[Analysis] dialog update_mask on clear failed")
            
            # 3) 화면에서 분석 오버레이 제거
            self._clear_analysis_overlay_item()

            # 4) 분석 모드 종료 (활성화된 분석 도구가 있으면 해제)
            if hasattr(self, "analysisCtrl") and self.analysisCtrl is not None:
                self.analysisCtrl.cancel()

            # 5) Dock의 분석 버튼 상태도 초기화
            if hasattr(self, "clsDock") and self.clsDock:
                # 사각형/픽셀 선택 모드 해제
                if hasattr(self.clsDock, "_btn_rect") and self.clsDock._btn_rect:
                    self.clsDock._btn_rect.blockSignals(True)
                    self.clsDock._btn_rect.setChecked(False)
                    self.clsDock._btn_rect.blockSignals(False)
                if hasattr(self.clsDock, "_btn_pick") and self.clsDock._btn_pick:
                    self.clsDock._btn_pick.blockSignals(True)
                    self.clsDock._btn_pick.setChecked(False)
                    self.clsDock._btn_pick.blockSignals(False)
                # 연산자 버튼도 해제
                if hasattr(self.clsDock, "_btn_union") and self.clsDock._btn_union:
                    self.clsDock._btn_union.blockSignals(True)
                    self.clsDock._btn_union.setChecked(False)
                    self.clsDock._btn_union.blockSignals(False)
                if hasattr(self.clsDock, "_btn_sub") and self.clsDock._btn_sub:
                    self.clsDock._btn_sub.blockSignals(True)
                    self.clsDock._btn_sub.setChecked(False)
                    self.clsDock._btn_sub.blockSignals(False)

            logging.info("[Analysis] 분석 영역 초기화 완료")

        except Exception:
            logging.exception("[MainWindow] _on_clear_analysis_region failed")

    def _fetch_class_options_safe(self, api_base: str, timeout: float = 3.0) -> list[tuple[int,str]]:
        try:
            from data.db import material_code_name
            return material_code_name(api_base=api_base, timeout=timeout)  # 구현은 아래 옵션 B 참고
        except Exception as e:
            import logging
            logging.exception("[ClassOptions] fetch failed")
            # 실패 시 최소한의 fallback
            return []
        
    def _is_workspace_dialog_open(self) -> bool:
        return bool(getattr(self, "_workspace_dialog", None) and self._workspace_dialog.isVisible())

    def _is_analysis_dialog_open(self) -> bool:
        return bool(getattr(self, "_analysis_selection_dialog", None) and self._analysis_selection_dialog.isVisible())
        
    def _restart_analysis_rect(self):
        """분석-사각형 모드를 항상 '깨끗이' 재시작한다."""
        try:
            self._roi_capture_target = "analysis"

            # 컨트롤러 준비/shape/allowed_mask 보장
            if not self._ensure_analysis_ctrl_ready():
                return

            # ROI 입력 반드시 허용(드로잉 이벤트 받도록)
            if getattr(self, "roi", None) and hasattr(self.roi, "set_accept_events"):
                self.roi.set_accept_events(True)

            # 이전 세션 깔끔히 종료 후 최신 op로 재설정
            try:
                self.analysisCtrl.cancel()
            except Exception:
                pass
            self.analysisCtrl.set_op(self._get_analysis_op())

            # 사각형 드로잉 시작
            self.analysisCtrl.start_rect_selection()

            # 드로잉 충돌 방지: 클릭은 NONE으로 고정
            self._safe_set_click_mode(ClickMode.NONE)

            # Dock/시각 상태 연동
            if getattr(self, "clsDock", None) and hasattr(self.clsDock, "set_roi_drawing_state"):
                self.clsDock.set_roi_drawing_state(True)

        except Exception:
            logging.exception("[AnalysisRect] restart failed")

    def _norm_id_to_name(self) -> Dict[int, str]:
        """self._last_id_to_name 형태가 dict 또는 list[dict] 등 섞여 올 수 있으므로 안전하게 정규화."""
        out: Dict[int, str] = {}
        raw = getattr(self, "_last_id_to_name", {}) or {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    cid = int(k)
                    nm = (v.get("mtrl_nm", v) if isinstance(v, dict) else v)
                    out[cid] = str(nm)
                except Exception:
                    pass
        elif isinstance(raw, (list, tuple)):
            for rec in raw:
                if isinstance(rec, dict) and "mtrl_cd" in rec:
                    try:
                        cid = int(rec["mtrl_cd"])
                        nm = rec.get("mtrl_nm", rec.get("name", str(cid)))
                        out[cid] = str(nm)
                    except Exception:
                        pass
        return out

    # ★ (신규) 라벨코드 → 패치/스펙트럼/그래프
    def _extract_patch(self, image_cd: int, y: int, x: int,
                    patch_size: int = 21,
                    bbox: Optional[Tuple[int,int,int,int]] = None) -> Optional[np.ndarray]:
        if image_cd != getattr(self, "image_cd", None):
            return None
        rgb = getattr(self, "rgb_image", None)
        if rgb is None: return None
        H, W, _ = rgb.shape
        if bbox:
            x0,y0,w,h = bbox
            x1,y1 = min(W, x0+w), min(H, y0+h)
            return rgb[y0:y1, x0:x1, :].copy()
        r = max(1, int(patch_size or 21))
        r2 = r // 2
        y0, y1 = max(0, y - r2), min(H, y + r - r2)
        x0, x1 = max(0, x - r2), min(W, x + r - r2)
        return rgb[y0:y1, x0:x1, :].copy()

    def get_label_patch(self, label_code: str, default_size: int = 21):
        if not self.label_store: return None
        row = self.label_store.get_row_by_code(label_code)
        if row is None: return None
        img_cd, y, x = int(row["image_cd"]), int(row["y"]), int(row["x"])
        size = int(row["patch_size"]) if not pd.isna(row["patch_size"]) else default_size
        bbox = None
        if not pd.isna(row.get("bbox_x", np.nan)):
            bbox = (int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_w"]), int(row["bbox_h"]))
        return self._extract_patch(img_cd, y, x, patch_size=size, bbox=bbox)

    def get_label_spectrum(self, label_code: str, use: str="raw"):
        if not self.label_store: return None, None
        s = self.label_store.get_spectrum_by_code(label_code, use=use)
        if s is None:
            hit = self.label_store.get_row_by_code(label_code)
            if hit is None: return None, None
            img_cd, y, x = int(hit["image_cd"]), int(hit["y"]), int(hit["x"])
            if img_cd != getattr(self, "image_cd", None):
                return None, None
            cube = self.cfg["data"]
            s = cube[y, x, :].astype(np.float32)
            if use == "cr":
                wl = np.asarray(self.cfg.get("wavelength"), dtype=np.float32) if self.cfg.get("wavelength") is not None else None
                if wl is not None:
                    s, _ = perform_continuum_removal(s, wl, mode="reflectance")
        wl = self.cfg.get("wavelength")
        return s, wl

    def plot_label_spectrum(self, label_code: str, use: str="raw", ax=None, title_prefix=""):
        s, wl = self.get_label_spectrum(label_code, use=use)
        if s is None:
            self.statusBar().showMessage(f"{label_code} 스펙트럼을 찾을 수 없습니다.", 2000)
            return
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        x = np.arange(len(s)) if wl is None else np.asarray(wl, dtype=float)
        ax.plot(x, s)
        ax.set_xlabel("Wavelength" if wl is not None else "Band")
        ax.set_ylabel("Reflectance" if use=="raw" else "CR-Reflectance")
        ax.set_title(f"{title_prefix}{label_code}")

    @QtCore.pyqtSlot(int)
    def _on_viz_class_changed(self, cid: int):
        """
        Viewer 선택 변화 시 Map에는 '선택/포커스'를 표시하지 않고,
        오직 '표시 클래스 필터'만 바꿉니다. (-9999면 해제)
        """
        try:
            # ClassmapRenderer/LayerManager가 있으면 거기에 표시 필터를 적용
            # 구현 예시: renderer가 특정 클래스만 보이게 하는 API가 있는 경우
            cr = getattr(self.layer_manager, "renderer", None)
            if cr and hasattr(cr, "set_visible_class_only"):
                if cid is not None and cid >= 0:
                    cr.set_visible_class_only(int(cid))
                else:
                    cr.clear_visible_class_filter()  # 해제
                # 지도 리렌더
                if hasattr(self.layer_manager, "refresh_active_classmap"):
                    self.layer_manager.refresh_active_classmap()
                return

            # 렌더러에 전용 API가 없다면, 임시 오버레이(투명도 오버레이)를 제거해
            # '시각화' 잔상을 없애는 정도만 수행
            self._clear_confidence_overlay_mapview()

        except Exception:
            logging.exception("[MainWindow] _on_viz_class_changed failed")


    @QtCore.pyqtSlot(int, float, float)
    def _on_apply_thresholds_for_class(self, cid: int, strict: float, base: float):
        """
        임계 적용 시 '해당 클래스만' 시각화.
        Dock에서 넘어온 클래스 ID를 그대로 사용하여 오버레이 생성.
        """
        try:
            if cid is None or cid < 0:
                # 해제 요청: 기존 오버레이만 제거
                self._clear_confidence_overlay_mapview()
                self.statusBar().showMessage("시각화 해제(클래스 선택 없음)", 2000)
                return

            # 기존 로직을 재활용하기 위해 별도 헬퍼 사용
            self._apply_transparency_overlay_for_class(int(cid), float(strict), float(base))

        except Exception:
            logging.exception("[MainWindow] _on_apply_thresholds_for_class failed")
            QtWidgets.QMessageBox.critical(self, "오류", "임계 적용 중 오류가 발생했습니다.")
            
    @QtCore.pyqtSlot(float, float)
    def _apply_transparency_overlay(self, strict_val: float, base_val: float):
        """
        도크의 '임계 적용' 기본 신호(두 개짜리)를 받는 래퍼.
        현재 도크에서 선택된 클래스 id를 읽어서
        _apply_transparency_overlay_for_class(cid, strict, base)로 위임한다.
        """
        try:
            # 도크에서 현재 선택된 클래스 ID 가져오기
            cid = None
            if hasattr(self, "clsDock") and self.clsDock and hasattr(self.clsDock, "get_selected_class_id"):
                cid = self.clsDock.get_selected_class_id()

            # 선택 클래스가 없으면 오버레이만 지우고 종료(정책에 맞게 조정 가능)
            if cid is None or int(cid) < 0:
                self._clear_confidence_overlay_mapview()
                self.statusBar().showMessage("선택된 클래스가 없습니다. Viewer에서 클래스를 선택하세요.", 2500)
                return

            # 클래스 지정 버전으로 위임
            self._apply_transparency_overlay_for_class(int(cid), float(strict_val), float(base_val))

        except Exception:
            import logging
            logging.exception("[Transparency] _apply_transparency_overlay wrapper failed")
            QtWidgets.QMessageBox.critical(self, "오류", "임계 적용 처리 중 오류가 발생했습니다.")
            
    @QtCore.pyqtSlot(str)
    def _on_open_viewer_band(self, parent_name: str = "image.rgb"):
        """뷰어 대역 선택 다이얼로그 열기. parent_name은 LayersDock에서 전달된 부모 레이어 이름."""
        if not hasattr(self, "cfg") or "data" not in self.cfg:
            QtWidgets.QMessageBox.information(self, "안내", "먼저 이미지를 로드하세요.")
            return

        # parent_name이 없거나 빈 문자열이면 기본값 사용
        if not parent_name or not isinstance(parent_name, str):
            parent_name = "image.rgb"

        dlg = ViewerBandDialog(parent=self, ui_dir=self.app_dir / "ui")

        # 파장목록 주입
        wl = self.cfg.get("wavelength")
        if wl is None:
            C = int(self.cfg["data"].shape[2])
            wl = list(range(1, C + 1))
            fmt = "{:.0f}"
        else:
            fmt = "{:.1f}"
        dlg.set_wavelengths(wl, fmt=fmt)

        # ✅ 1) 캐시에서 초기 선택 불러오기 (우선순위: 메모리 → QSettings)
        sel_init = getattr(self, "_viewerband_last_sel", None)
        if sel_init is None:
            try:
                st = QtCore.QSettings("third_pixel_tool", "hsi_app")
                raw = st.value("viewer_band/last_selection", {}, type=dict)
                sel_init = raw if isinstance(raw, dict) and raw else None
            except Exception:
                sel_init = None

        # ✅ 2) 캐시가 없으면 현재 rgb_image로부터 추정
        if sel_init is None:
            sel_init = self._guess_view_bands_from_current()  # 아래 헬퍼

        # ✅ 3) 초기값 적용
        if sel_init:
            dlg.set_initial_selection(sel_init, fmt=fmt)

        # ✅ 미커밋 프리뷰 준비: 현재 RGB 백업
        self._viewerband_backup_rgb = getattr(self, "rgb_image", None).copy() if getattr(self, "rgb_image", None) is not None else None
        
        # ✅ 현재 LayersDock의 viewer bands 선택값 백업 (취소 시 복원용)
        self._viewerband_backup_sel = dict(sel_init) if sel_init else None

        # parent_name을 프리뷰/커밋 메서드에서 사용할 수 있도록 저장
        self._viewerband_parent_name = parent_name

        # 라이브 프리뷰 연결
        dlg.selectionChanged.connect(self._on_viewerband_preview)

        code = dlg.exec_()

        try:
            dlg.selectionChanged.disconnect(self._on_viewerband_preview)
        except Exception:
            pass

        if code == QtWidgets.QDialog.Accepted:
            sel = dlg.get_result()
            # ✅ 커밋 전 레이어 순서 백업 (register_rgb_base가 순서를 변경하므로)
            try:
                layer_order_backup = self.layersDock._all_names_top_to_bottom()
            except Exception:
                layer_order_backup = None
            
            # 커밋 적용
            self._apply_view_bands_ui_selection(sel, commit=True, parent_name=parent_name)
            
            # ✅ 레이어 순서 복원 (Layers 규칙 유지)
            if layer_order_backup:
                try:
                    self.layer_manager.reorder_top_to_bottom(layer_order_backup)
                except Exception:
                    logging.exception("[ViewerBand] 레이어 순서 복원 실패")
            
            # ✅ 메모리 & QSettings 캐시 저장
            try:
                self._viewerband_last_sel = dict(sel)
                st = QtCore.QSettings("third_pixel_tool", "hsi_app")
                st.setValue("viewer_band/last_selection", self._viewerband_last_sel)
            except Exception:
                pass
        else:
            # 원복
            self._restore_viewerband_backup()
        
        # 정리
        if hasattr(self, "_viewerband_parent_name"):
            delattr(self, "_viewerband_parent_name")
        if hasattr(self, "_viewerband_backup_sel"):
            delattr(self, "_viewerband_backup_sel")
            
    # MainWindow 내부에 기존 함수를 아래로 교체
    def _apply_view_bands(self, *, gray_w: float = None, r_w: float = None, g_w: float = None, b_w: float = None, commit: bool = True):
        cube: np.ndarray = self.cfg["data"]
        wl = self.cfg.get("wavelength")
        H, W, C = cube.shape

        def _band_from_w(w_target: float) -> int:
            if wl is None:
                idx = int(round(float(w_target))) - 1
                return max(0, min(C-1, idx))
            arr = np.asarray(wl, dtype=float).ravel()
            return int(np.clip(np.argmin(np.abs(arr - float(w_target))), 0, C-1))

        def _norm_u8(x: np.ndarray) -> np.ndarray:
            x = x.astype(np.float32, copy=False)
            mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
            if mx <= mn + 1e-12:
                return np.zeros_like(x, dtype=np.uint8)
            y = (x - mn) / (mx - mn)
            return (255.0 * np.clip(y, 0, 1)).astype(np.uint8)

        # ---- 실제로 그릴 모드/대역 결정 ----
        if gray_w is not None:
            # 단일 밴드 Gray
            k = _band_from_w(gray_w)
            band = cube[:, :, k]
            g8 = _norm_u8(band)
            rgb_img = np.dstack([g8, g8, g8])
            current_sel = {"mode": "gray", "gray": float(gray_w)}
            msg = f"Gray(단일) {'적용' if commit else '프리뷰'}"
        else:
            if None in (r_w, g_w, b_w):
                QtWidgets.QMessageBox.information(self, "안내", "RGB 모드에는 R/G/B 파장이 모두 필요합니다.")
                return
            kr, kg, kb = _band_from_w(r_w), _band_from_w(g_w), _band_from_w(b_w)
            R, G, B = cube[:, :, kr], cube[:, :, kg], cube[:, :, kb]
            rgb_img = np.dstack([_norm_u8(R), _norm_u8(G), _norm_u8(B)])
            current_sel = {"mode": "rgb", "r": float(r_w), "g": float(g_w), "b": float(b_w)}
            msg = f"RGB {'적용' if commit else '프리뷰'}"

        # ---- MapView 갱신(프리뷰/커밋 공통) ----
        if hasattr(self._map_view, "set_rgb_image"):
            self._map_view.set_rgb_image(rgb_img, do_fit=False)

        # ✅ LayersDock: 지금 그린 모드/대역을 즉시 반영(SSOT)
        try:
            self.layersDock.set_viewer_bands_from_selection(current_sel, fmt="{:.1f}", attach_to="image.rgb")
        except Exception:
            pass

        if commit:
            # 커밋: image.rgb 레이어 교체 + 마지막 커밋 선택 캐시
            if hasattr(self.layer_manager, "register_rgb_base"):
                try:
                    self.layer_manager.register_rgb_base(RGB_LAYER_NAME, rgb_img, visible=True, display_label=RGB_LAYER_DISPLAY_NAME)
                except Exception:
                    pass
            self.rgb_image = rgb_img
            # 마지막 커밋 상태를 따로 보관(취소/복구 대비)
            self._viewerband_last_committed_sel = dict(current_sel)
            try:
                st = QtCore.QSettings("third_pixel_tool", "hsi_app")
                st.setValue("viewer_band/last_selection", self._viewerband_last_committed_sel)
            except Exception:
                pass

            self.statusBar().showMessage(f"뷰어 대역 {msg}", 2000)
        else:
            self.statusBar().showMessage(f"{msg}", 1200)

    def _on_viewerband_preview(self, sel: dict):
        # 기존 프리뷰 적용
        parent_name = getattr(self, "_viewerband_parent_name", "image.rgb")
        self._apply_view_bands_ui_selection(sel, commit=False, parent_name=parent_name)
        # ★ Layers에 현재 선택 표기 (parent_name을 명시적으로 전달)
        try:
            self.layersDock.set_viewer_bands_from_selection(sel, fmt="{:.1f}", attach_to=parent_name)
        except Exception:
            pass

    def _apply_view_bands_ui_selection(self, sel: dict, *, commit: bool, parent_name: str = "image.rgb"):
        mode = sel.get("mode")
        if mode == "gray":
            self._apply_view_bands(gray_w=sel.get("gray"), commit=commit)
        else:
            self._apply_view_bands(r_w=sel.get("r"), g_w=sel.get("g"), b_w=sel.get("b"), commit=commit)
        # ★ 커밋 시에도 최종 선택을 Layers에 표시(프리뷰와 동일, parent_name을 명시적으로 전달)
        if commit:
            try:
                self.layersDock.set_viewer_bands_from_selection(sel, fmt="{:.1f}", attach_to=parent_name)
            except Exception:
                pass

    def _restore_viewerband_backup(self):
        rgb = getattr(self, "_viewerband_backup_rgb", None)
        if rgb is None:
            return
        # 이미지 복원
        if hasattr(self._map_view, "set_rgb_image"):
            self._map_view.set_rgb_image(rgb, do_fit=False)
        if hasattr(self.layer_manager, "register_rgb_base"):
            try:
                self.layer_manager.register_rgb_base(RGB_LAYER_NAME, rgb, visible=True, display_label=RGB_LAYER_DISPLAY_NAME)
            except Exception:
                pass
        self.rgb_image = rgb

        # ✅ Layers 복원(마지막 커밋 선택이 있으면 그걸, 없으면 역추정 사용)
        sel = getattr(self, "_viewerband_last_committed_sel", None)
        if sel is None:
            sel = self._guess_view_bands_from_current()  # 이미 구현해두신 추정기
        if sel:
            try:
                self.layersDock.set_viewer_bands_from_selection(sel, fmt="{:.1f}", attach_to="image.rgb")
            except Exception:
                pass

        self.statusBar().showMessage("뷰어 대역 변경이 취소되어 원래 이미지/레이어로 복원했습니다.", 2000)


    def _guess_view_bands_from_current(self) -> dict | None:
        """
        현재 self.rgb_image와 self.cfg['data'](cube)를 이용해
        R/G/B에 가장 비슷한 밴드 index를 찾고, 연속 3밴드면 gray 모드로, 아니면 rgb 모드로 초기화값을 만든다.
        """
        try:
            cube = self.cfg["data"]            # (H,W,C)
            rgb  = getattr(self, "rgb_image", None)
            wl   = self.cfg.get("wavelength")
            if rgb is None or cube is None:
                return None
            H, W, C = cube.shape
            if rgb.shape[0] != H or rgb.shape[1] != W:
                return None

            def _norm(x):
                x = x.astype(np.float32, copy=False)
                mu = float(np.nanmean(x)); sd = float(np.nanstd(x)) + 1e-8
                return (x - mu) / sd

            R = _norm(rgb[:, :, 0])
            G = _norm(rgb[:, :, 1])
            B = _norm(rgb[:, :, 2])

            # 각 밴드와 R/G/B 상관계수 계산 (빠르게: 평균/곱 평균으로 근사)
            rs = []
            for k in range(C):
                ch = _norm(cube[:, :, k])
                # 피어슨 근사: mean(ch*R)
                rs.append([
                    float(np.nanmean(ch * R)),
                    float(np.nanmean(ch * G)),
                    float(np.nanmean(ch * B)),
                ])
            rs = np.asarray(rs)  # (C,3)

            kr = int(np.nanargmax(rs[:, 0]))
            kg = int(np.nanargmax(rs[:, 1]))
            kb = int(np.nanargmax(rs[:, 2]))

            # 연속 3밴드 패턴인지 검사
            trio = sorted([kr, kg, kb])
            is_consecutive = (trio[0] + 1 == trio[1]) and (trio[1] + 1 == trio[2])

            def _idx_to_w(i):
                if wl is None:
                    return i + 1
                try:
                    return float(wl[i])
                except Exception:
                    return i + 1

            if is_consecutive:
                # 가운데를 gray로
                center = trio[1]
                return {"mode": "gray", "gray": _idx_to_w(center)}
            else:
                return {"mode": "rgb", "r": _idx_to_w(kr), "g": _idx_to_w(kg), "b": _idx_to_w(kb)}
        except Exception:
            return None

    def _build_class_options_for_labeling(self) -> list[tuple[int, str]]:
        """
        픽셀 라벨링용 전체 클래스 옵션 생성:
        - 1순위: material_code_name_url / MATERIAL_API_BASE 에서 가져온 (cid, name)
        - 2순위: self.lib의 cid 들을 모두 포함 (이름 없으면 cid 문자열 표시)
        ※ self._last_id_to_name 은 사용하지 않는다.
        """
        id_to_name: dict[int, str] = {}

        # 1) API에서 전체 물질 코드/이름 가져오기 (가능하면)
        api_base = os.getenv("material_code_name_url") or os.getenv("MATERIAL_API_BASE")
        raw = self._fetch_class_options_safe(api_base) if api_base else []

        # raw 포맷 정규화
        if isinstance(raw, dict):
            raw_iter = raw.items()
            for k, v in raw_iter:
                try:
                    cid = int(k)
                except Exception:
                    continue
                name = v
                if isinstance(v, dict):
                    name = v.get("mtrl_nm") or v.get("name") or v.get("label") or v.get("desc")
                name_str = str(name).strip() if name not in (None, "") else f"Class {cid}"
                id_to_name[cid] = name_str
        elif isinstance(raw, (list, tuple)):
            for rec in raw:
                cid = name = None
                if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    cid, name = rec[0], rec[1]
                elif isinstance(rec, dict):
                    cid = rec.get("mtrl_cd") or rec.get("cid") or rec.get("class_id")
                    name = rec.get("mtrl_nm") or rec.get("name") or rec.get("label") or rec.get("desc")
                else:
                    continue

                try:
                    cid_int = int(str(cid).strip())
                except Exception:
                    continue
                name_str = str(name).strip() if name not in (None, "") else f"Class {cid_int}"
                id_to_name[cid_int] = name_str

        # 2) 라이브러리에 있는 CID 전체를 포함시키기 (이름 없으면 cid 자체를 이름으로 사용)
        lib = getattr(self, "lib", {}) or {}
        if lib:
            for cid in sorted(map(int, lib.keys())):
                if cid not in id_to_name:
                    id_to_name[cid] = str(cid)

        # 3) (cid, name) 리스트로 반환
        options = sorted(id_to_name.items(), key=lambda x: x[0])
        return options

    def _resolve_current_class_options(self) -> list[tuple[int, str]]:
        """user_type에 따라 personal(.info) 또는 server(API) 클래스 목록을 반환."""
        user_type = getattr(self, "user_type", "server")
        if user_type == "personal":
            class_options: list[tuple[int, str]] = []
            try:
                primary = self._cache_primary_path(self.cfg) if hasattr(self, "cfg") else None
                if not primary and hasattr(self, "_extract_src_path"):
                    primary = self._extract_src_path(self.cfg)

                if primary:
                    class_options = load_classes_from_info(primary) or []
                    logging.info(
                        "[PixelLabeling] personal: .info 파일에서 %d개 클래스 로드 완료",
                        len(class_options),
                    )
                else:
                    logging.warning("[PixelLabeling] personal: primary_path를 찾을 수 없습니다.")
            except Exception:
                logging.exception("[PixelLabeling] personal: .info 파일에서 클래스 정보 읽기 실패")
                class_options = []
            return class_options

        # server 사용자: API + 라이브러리 기반
        return self._build_class_options_for_labeling()

    def _build_unmixing_classmap_from_abundance(self, threshold: float, mapping: Optional[Dict[int, int]] = None):
        A = self.unmixing_abundance_map  # (H, W, K)
        if A is None:
            ...
        H, W, K = A.shape
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = np.clip(A, 0.0, 1.0)

        hit = (A >= float(threshold))
        cnt = hit.sum(axis=2)
        winner = A.argmax(axis=2).astype(np.int32)  # endmember index 0..K-1

        classmap = np.full((H, W), UNKNOWN, dtype=np.int32)

        # 매핑 함수: endmember index -> 실제 class id
        if mapping is None:
            mapping = {}

        # winner를 class id로 변환한 배열
        cid_map = np.vectorize(lambda em: mapping.get(int(em), int(em)))(winner)

        # 적용
        classmap[cnt == 1] = cid_map[cnt == 1]
        classmap[cnt >  1] = MULTIPLE

        used_cids = sorted(int(c) for c in np.unique(classmap) if c >= 0)

        self._augment_palette_for(used_cids, push_renderer=True, push_dock=True)

        # id_to_name는 기존 스펙트럼 라이브러리 이름 맵을 사용
        id_to_name = self._norm_id_to_name()
        self._register_map_semantics("unmixing", metric="abundance", distance=False)

        # 기존 unmixing 레이어를 지우고 다시 등록
        # (아예 Layer/Map에서 지우고 다시 저장하는 방식)
        try:
            self.layer_manager.remove("unmixing")
        except Exception:
            pass

        self.layer_manager.register_classmap(
            name="unmixing",
            classmap_i32=classmap,
            class_ids=used_cids,
            visible=True,
            id_to_name=id_to_name,
        )

        self.unmixing_classmap = classmap

            
    @QtCore.pyqtSlot(float)
    def _on_unmixing_threshold_changed(self, new_thr: float):
        """
        UnmixingDock에서 임계값을 바꿀 때 호출되는 슬롯.
        new_thr: 0~1 (또는 0~100, 둘 다 처리)
        """
        try:
            # 퍼센트 값도 허용 (예: 80 → 0.8)
            thr = float(new_thr)
            thr = thr / 100.0 if thr > 1.0 else thr
            thr = max(0.0, min(1.0, thr))

            if getattr(self, "unmixing_abundance_map", None) is None:
                self.statusBar().showMessage("Unmixing 결과가 없어 임계값 변경을 적용할 수 없습니다.", 2000)
                return

            self.unmixing_threshold = thr
            self._build_unmixing_classmap_from_abundance(thr)
        except Exception:
            logging.exception("[Unmixing] threshold changed handler failed")
            
    @QtCore.pyqtSlot(dict)
    def _on_unmixing_class_mapping_applied(self, mapping: dict):
        """
        UnmixingDock에서 엔드멤버→클래스 매핑이 넘어옴.
        mapping: {endmember_index: class_id}
        """
        self.unmixing_class_mapping = {int(k): int(v) for k, v in mapping.items()}
        thr = getattr(self, "unmixing_threshold", 0.8)
        self._build_unmixing_classmap_from_abundance(thr, mapping=self.unmixing_class_mapping)
# 단독 실행 테스트용(선택)
if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    win = MainWindow()
    win.resize(960, 640)
    win.show()
    sys.exit(app.exec_())
