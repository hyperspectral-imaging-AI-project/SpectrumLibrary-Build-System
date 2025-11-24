# services/region_growing_service.py
from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple
import logging
from PyQt5.QtCore import QObject, pyqtSignal

from core.region_growing import (
    RegionGrowing, RegionGrowingConfig, RegionGrowingResult,
    create_region_growing_config, perform_region_growing
)


class RegionGrowingService(QObject):
    """
    Region Growing 서비스 - MainWindow와 Region Growing 알고리즘을 연결

    시그널:
        - regionGrown(RegionGrowingResult): 영역 성장 성공
        - regionGrowingFailed(str): 오류 메시지
    """

    regionGrown = pyqtSignal(object)
    regionGrowingFailed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hsi_data: Optional[np.ndarray] = None
        self._region_grower: Optional[RegionGrowing] = None
        self._current_config: Optional[RegionGrowingConfig] = None

        # (옵션) 좌표 변환용 View 주입 지점
        self._map_view = None        # QGraphicsView 예상
        self._image_item = None      # QGraphicsPixmapItem 예상
        self._pixel_scale = (1.0, 1.0)  # (sy, sx)

        # ROI(work area) 제한
        self._allowed_mask: Optional[np.ndarray] = None

        # 기본 설정
        self._default_config = RegionGrowingConfig(
            metric="SAM",
            threshold=0.10,
            margin=0.06,
            min_region_size=1,
            max_region_size=10000,
            connectivity=8,
        )

    # --- 좌표 변환 관련 주입 (필요 시) ---
    def set_mapview_refs(self, map_view, image_item=None, pixel_scale: Tuple[float, float] = (1.0, 1.0)) -> None:
        self._map_view = map_view
        self._image_item = image_item
        self._pixel_scale = pixel_scale

    # --- ROI(work area) 마스크 주입 ---
    def set_allowed_mask(self, mask: Optional[np.ndarray]) -> None:
        try:
            self._allowed_mask = (mask.astype(bool) if isinstance(mask, np.ndarray) else None)
        except Exception:
            self._allowed_mask = None

    # --- 데이터/설정 ---
    def set_hsi_data(self, hsi_data: np.ndarray) -> None:
        if not isinstance(hsi_data, np.ndarray) or hsi_data.ndim != 3:
            msg = "[RegionGrowing] HSI 데이터 형식이 (H,W,C) ndarray가 아닙니다."
            logging.error(msg)
            self.regionGrowingFailed.emit(msg)
            return
        self._hsi_data = np.ascontiguousarray(hsi_data)
        self._update_region_grower()
        logging.info(f"[RegionGrowing] HSI data set: {hsi_data.shape}")

    def set_config(
        self,
        metric: str = "SAM",
        threshold: float = 0.10,
        margin: float = 0.06,
        min_region_size: int = 1,
        max_region_size: int = 10000,
        connectivity: int = 8,
    ) -> None:
        if connectivity not in (4, 8):
            logging.warning(f"[RegionGrowing] connectivity={connectivity} (권장: 4 또는 8)")
        self._default_config = RegionGrowingConfig(
            metric=metric,
            threshold=threshold,
            margin=margin,
            min_region_size=min_region_size,
            max_region_size=max_region_size,
            connectivity=connectivity,
            allowed_mask=self._allowed_mask,
        )
        self._update_region_grower()
        logging.info(f"[RegionGrowing] Config updated: {metric}, threshold={threshold}, margin={margin}")

    def _update_region_grower(self) -> None:
        if self._hsi_data is not None:
            self._region_grower = RegionGrowing(self._hsi_data, self._default_config)
            self._current_config = self._default_config

    # --- 핵심 실행: 클릭/스펙트럼/다중 ---
    def grow_region_from_click(
        self,
        x: int,
        y: int,
        custom_spectrum: Optional[np.ndarray] = None,
    ) -> Optional[RegionGrowingResult]:
        """
        (x,y)는 이미지 좌표(=col,row)로 가정. 필요 시 화면좌표→이미지좌표 변환을 추가하세요.
        """
        if self._region_grower is None or self._hsi_data is None:
            msg = "HSI 데이터가 설정되지 않았습니다."
            logging.error(f"[RegionGrowing] {msg}")
            self.regionGrowingFailed.emit(msg)
            return None

        image_col, image_row = int(x), int(y)

        # ROI 밖 클릭은 즉시 실패 처리(조용히 무시하고 싶으면 return None)
        if self._allowed_mask is not None:
            try:
                if not bool(self._allowed_mask[image_row, image_col]):
                    msg = "Seed point is outside of work area (ROI)."
                    logging.info(f"[RegionGrowing] {msg}")
                    self.regionGrowingFailed.emit(msg)
                    return None
            except Exception:
                pass

        try:
            result = self._region_grower.grow_region(
                seed_x=image_col, seed_y=image_row,
                seed_spectrum=custom_spectrum,
                allowed_mask=self._allowed_mask
            )
            if result is None:
                msg = "RegionGrowing.grow_region() returned None."
                logging.error(f"[RegionGrowing] {msg}")
                self.regionGrowingFailed.emit(msg)
                return None

            logging.info(
                f"[RegionGrowing] Region grown from (row={image_row}, col={image_col}): "
                f"size={getattr(result, 'region_size', 'n/a')}, τ={self._default_config.threshold}, δ={self._default_config.margin}"
            )

            self.regionGrown.emit(result)
            return result

        except Exception as e:
            msg = f"Region Growing 실패: {str(e)}"
            logging.exception(f"[RegionGrowing] {msg}")
            self.regionGrowingFailed.emit(msg)
            return None

    def grow_region_from_spectrum(
        self,
        x: int,
        y: int,
        spectrum: np.ndarray,
    ) -> Optional[RegionGrowingResult]:
        return self.grow_region_from_click(x, y, custom_spectrum=spectrum)

    def grow_multiple_regions(
        self,
        coordinates: List[Tuple[int, int]],
        spectra: Optional[List[np.ndarray]] = None,
    ) -> List[RegionGrowingResult]:
        if self._region_grower is None or self._hsi_data is None:
            msg = "HSI 데이터가 설정되지 않았습니다."
            logging.error(f"[RegionGrowing] {msg}")
            self.regionGrowingFailed.emit(msg)
            return []

        results: List[RegionGrowingResult] = []
        for i, (x, y) in enumerate(coordinates):
            cs = None
            if spectra is not None and i < len(spectra):
                cs = spectra[i]
                if isinstance(cs, np.ndarray) and cs.ndim == 1 and cs.shape[0] != self._hsi_data.shape[2]:
                    logging.warning(f"[RegionGrowing] spectrum length mismatch at index {i} (skip)")
                    cs = None
            try:
                r = self._region_grower.grow_region(
                    seed_x=int(x), seed_y=int(y),
                    seed_spectrum=cs, allowed_mask=self._allowed_mask
                )
                if r is not None:
                    results.append(r)
            except Exception as e:
                logging.error(f"[RegionGrowing] Failed for coordinate ({x}, {y}): {e}")
        logging.info(f"[RegionGrowing] Multiple regions grown: {len(results)}/{len(coordinates)} successful")
        return results

    # --- 보조 ---
    def get_current_config(self) -> Optional[RegionGrowingConfig]:
        return self._current_config

    def get_hsi_data_shape(self) -> Optional[Tuple[int, int, int]]:
        return None if self._hsi_data is None else self._hsi_data.shape

    def clear_data(self) -> None:
        self._hsi_data = None
        self._region_grower = None
        self._current_config = None
        self._allowed_mask = None
        logging.info("[RegionGrowing] Data cleared")
