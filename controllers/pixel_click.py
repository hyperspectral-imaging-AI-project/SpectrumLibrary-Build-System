# controllers/pixel_click.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional, Tuple, List, Dict
from collections import OrderedDict
import numpy as np

# on-demand KNN Top-10 계산 (SAD=SAM, SID, SCC)
from services.pixel_knn import topk_for_pixel


class ClickMode(Enum):
    """픽셀 클릭 동작 모드"""
    NONE = auto()            # 4) 픽셀 클릭 불필요
    LABEL = auto()           # 1) 라벨링 픽셀 지정
    DIFFUSION_SEED = auto()  # 2) 디퓨전 초기 시드 지정
    INSPECT = auto()         # 3) 분류 후보 Top-10 확인
    ANALYSIS = auto()        # 5) 지정 영역 분석(픽셀 누적/제거)


@dataclass
class PixelClickDeps:
    """
    외부 의존성 주입(필수/옵션 콜백들)
    - get_*: 현재 컨텍스트에서 필요한 데이터 접근용
    - *_callback: 라벨/디퓨전 선택 후 실제 도메인 로직을 호출
    """
    # 필수
    get_cube: Callable[[], np.ndarray]
    get_lib: Callable[[], Dict[int, np.ndarray]]
    get_metric_name: Callable[[], str]
    get_image_code: Callable[[], Optional[str]]

    # 다이얼로그는 컨트롤러가 띄우지 않음 — 메인에게 좌표만 통지
    present_topk_for_pixel: Callable[[int, int], None]
    analysis_callback: Optional[Callable[[int, int], None]] = None

    # 옵션
    get_lib_version: Optional[Callable[[], int]] = None
    get_meta_for_cid: Optional[Callable[[int], Dict]] = None
    label_callback: Optional[Callable[[int, int, Optional[int]], None]] = None
    seed_callback: Optional[Callable[[int, int, Optional[int]], None]] = None

class PixelClickController:
    """
    MapView의 픽셀 클릭 시그널을 받아 모드별로 동작한다.
    - LABEL: 클릭 좌표를 라벨링 콜백으로 전달
    - DIFFUSION_SEED: 클릭 좌표를 시드 콜백으로 전달
    - INSPECT: on-demand KNN Top-10 + LRU 캐시
    - NONE: 무시
    """

    def __init__(
        self,
        map_view,
        deps: PixelClickDeps,
        *,
        cache_cap: int = 10
    ):
        self.mv = map_view
        self.deps = deps

        # 상태
        self.mode: ClickMode = ClickMode.NONE
        self.focus_cid: Optional[int] = None  # Viewer에서 활성화(강조)된 클래스

        # LRU 캐시: key -> rows
        # rows: List[(rank, cid, score_str, name, desc)]
        self._cache: "OrderedDict[Tuple, List[Tuple[int, int, str, str, str]]]" = OrderedDict()
        self._cap = int(cache_cap)

        # MapView 시그널 연결 (존재할 때만)
        if hasattr(self.mv, "labelPixelPicked"):
            self.mv.labelPixelPicked.connect(self._on_label_pixel)
        if hasattr(self.mv, "diffusionSeedPicked"):
            self.mv.diffusionSeedPicked.connect(self._on_seed_pixel)
        if hasattr(self.mv, "inspectPixelPicked"):
            self.mv.inspectPixelPicked.connect(self._on_inspect_pixel)

    # ---------- 외부 제어 API ----------
    def set_mode(self, mode: ClickMode):
        self.mode = ClickMode(mode) if not isinstance(mode, ClickMode) else mode
        # MapView에도 커서/동작 반영
        if hasattr(self.mv, "set_click_mode"):
            try:
                self.mv.set_click_mode(self.mode)
            except Exception:
                pass

    def set_roi_active(self, active: bool):
        # ROI 드로잉 중이면 픽셀 픽업 차단(맵뷰가 지원할 때)
        if hasattr(self.mv, "set_roi_active"):
            try:
                self.mv.set_roi_active(bool(active))
            except Exception:
                pass

    def set_focus_cid(self, cid: Optional[int]):
        self.focus_cid = int(cid) if (cid is not None and cid >= 0) else None

    def clear_cache(self):
        self._cache.clear()

    # ---------- 내부 핸들러 ----------

    def _on_label_pixel(self, y: int, x: int):
        if self.mode != ClickMode.LABEL:
            return
        if self.deps.label_callback:
            try:
                self.deps.label_callback(y, x, self.focus_cid)
            except Exception:
                # 로깅은 상위에서 처리한다고 가정
                pass

    def _on_seed_pixel(self, y: int, x: int):
        if self.mode != ClickMode.DIFFUSION_SEED:
            return
        if self.deps.seed_callback:
            try:
                self.deps.seed_callback(y, x, self.focus_cid)
            except Exception:
                pass

    def _on_inspect_pixel(self, y: int, x: int):
        if self.mode not in (ClickMode.INSPECT, ClickMode.ANALYSIS):
            return

        if self.mode == ClickMode.ANALYSIS:
            if self.deps.analysis_callback:
                try:
                    self.deps.analysis_callback(int(y), int(x))
                except Exception:
                    pass
            return

        cube = self.deps.get_cube()
        H, W, C = cube.shape
        if not (0 <= y < H and 0 <= x < W):
            return

        metric = str(self.deps.get_metric_name()).strip().upper()
        if metric == "SAM":
            metric = "SAD"  # (프로젝트에서 SAM을 SAD 키로 통일 사용 중이면 유지)

        img_cd = self.deps.get_image_code()
        lib_ver = None
        if self.deps.get_lib_version:
            try:
                lib_ver = int(self.deps.get_lib_version())
            except Exception:
                lib_ver = None

        key = (
            int(y),
            int(x),
            metric,
            int(self.focus_cid) if self.focus_cid is not None else None,
            str(img_cd) if img_cd is not None else None,
            lib_ver,
        )

        # LRU 캐시 갱신만 하고, UI는 메인에 위임
        if key in self._cache:
            rows = self._cache.pop(key)
            self._cache[key] = rows
            try:
                self.deps.present_topk_for_pixel(y, x)  # ★ 메인에게 “좌표만” 통지
            except Exception:
                pass
            return

        # 미스면 계산 + 캐시에 넣기만 (표시는 메인)
        pix = cube[y, x, :].astype(np.float32, copy=False)
        lib = self.deps.get_lib()
        topk = topk_for_pixel(
            pix, lib,
            metric=metric,  # "SAD"|"SID"|"SCC"
            k=10,
            only_cid=self.focus_cid
        )

        rows: List[Tuple[int, int, str, str, str]] = []
        meta_fn = self.deps.get_meta_for_cid
        for rank, (cid, score) in enumerate(topk, 1):
            name, desc = str(cid), ""
            if meta_fn:
                try:
                    m = meta_fn(int(cid)) or {}
                    name = m.get("mtrl_nm", name)
                    desc = m.get("desc", desc)
                except Exception:
                    pass
            rows.append((rank, int(cid), f"{score:.4f}", name, desc))

        self._cache[key] = rows
        if len(self._cache) > self._cap:
            self._cache.popitem(last=False)

        try:
            self.deps.present_topk_for_pixel(y, x)  # ★ 메인에게 “좌표만” 통지
        except Exception:
            pass
