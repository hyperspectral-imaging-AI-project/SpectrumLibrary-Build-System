# core/region_growing.py
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from collections import deque
import logging

from core.metrics import sad, sid, scc_distance
# SAM이 core.metrics에 있으면 import, 없으면 간이 구현
try:
    from core.metrics import sam
except Exception:
    import math as _math
    def sam(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float32, copy=False); b = b.astype(np.float32, copy=False)
        na = float(np.linalg.norm(a)) + 1e-12
        nb = float(np.linalg.norm(b)) + 1e-12
        cos = float(np.clip(float(np.dot(a, b)) / (na * nb), -1.0, 1.0))
        return _math.acos(cos)  # 라디안


class RegionGrowingConfig:
    """Region Growing 설정 클래스 (τ=threshold, δ=margin)"""

    def __init__(
        self,
        metric: str = "SAM",                 # 기본 SAM 권장
        threshold: float = 0.10,             # τ: seed-현재 허용 (SAM 라디안 예시≈5.7°)
        margin: float = 0.06,                # δ: 현재/평균-이웃 허용 (SAM 라디안 예시≈3.4°)
        min_region_size: int = 10,
        max_region_size: int = 10000,
        connectivity: int = 8,               # 8-연결 권장
        use_spectral_weight: bool = True,
        spectral_weight_factor: float = 1.0,
        spatial_weight_factor: float = 0.1,
        allowed_mask: Optional[np.ndarray] = None,   # ROI 제한 (bool(H,W))
    ):
        self.metric = metric.upper()
        self.threshold = float(threshold)
        self.margin = float(margin)
        self.min_region_size = int(min_region_size)
        self.max_region_size = int(max_region_size)
        self.connectivity = int(connectivity)
        self.use_spectral_weight = bool(use_spectral_weight)
        self.spectral_weight_factor = float(spectral_weight_factor)
        self.spatial_weight_factor = float(spatial_weight_factor)
        self.allowed_mask = (allowed_mask.astype(bool) if isinstance(allowed_mask, np.ndarray) else None)


class RegionGrowingResult:
    """Region Growing 결과 클래스"""

    def __init__(
        self,
        region_mask: np.ndarray,                     # bool(H,W)
        region_pixels: List[Tuple[int, int]],        # [(y,x), ...]
        seed_coordinate: Tuple[int, int],            # (y,x)
        region_spectrum: np.ndarray,                 # (C,)
        config: RegionGrowingConfig,
        statistics: Dict[str, Any],
        seed_distance_map: Optional[np.ndarray] = None,   # ★ 추가
    ):
        self.region_mask = region_mask
        self.region_pixels = region_pixels
        self.seed_coordinate = seed_coordinate
        self.region_spectrum = region_spectrum
        self.config = config
        self.statistics = statistics
        self.seed_distance_map = seed_distance_map    # (H,W) float32, seed과의 거리 캐시

    @property
    def region_size(self) -> int:
        return len(self.region_pixels)

    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """(y_min, y_max, x_min, x_max)"""
        if not self.region_pixels:
            return (0, 0, 0, 0)
        ys, xs = zip(*self.region_pixels)
        return (min(ys), max(ys), min(xs), max(xs))


class RegionGrowing:
    """
    HSI 이미지에서 스펙트럼 유사성을 기반으로 Region Growing 수행

    Args:
        hsi_data: HSI 데이터 (H, W, C)
        config: Region Growing 설정
    """

    def __init__(self, hsi_data: np.ndarray, config: RegionGrowingConfig):
        assert isinstance(hsi_data, np.ndarray) and hsi_data.ndim == 3, "hsi_data must be (H,W,C)"
        self.hsi_data = hsi_data.astype(np.float32, copy=False)
        self.config = config
        self.H, self.W, self.C = self.hsi_data.shape

        # 거리 계산 함수
        self.distance_fn = self._get_distance_function()

        # 방향 벡터
        if self.config.connectivity == 4:
            self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:  # 8연결
            self.directions = [
                (-1, -1), (-1, 0), (-1, 1),
                ( 0, -1),          ( 0, 1),
                ( 1, -1), ( 1, 0), ( 1, 1),
            ]

    def _get_distance_function(self):
        metric_map = {"SAD": sad, "SID": sid, "SCC": scc_distance, "SAM": sam}
        if self.config.metric not in metric_map:
            logging.warning(f"[RegionGrowing] Unknown metric {self.config.metric}, fallback to SAM")
            return sam
        return metric_map[self.config.metric]

    def grow_region(
        self,
        seed_x: int,
        seed_y: int,
        seed_spectrum: Optional[np.ndarray] = None,
        allowed_mask: Optional[np.ndarray] = None,
    ) -> RegionGrowingResult:
        if not (0 <= seed_x < self.W and 0 <= seed_y < self.H):
            raise ValueError(f"Seed coordinates ({seed_x}, {seed_y}) out of bounds")

        # ★ 1) 모든 경로에서 존재하는 기본 stats 먼저 생성
        stats = {
            "region_size": 0,
            "average_distance": 0.0,
            "max_distance": float(self.config.threshold),
            "seed_coordinate": (int(seed_y), int(seed_x)),
            "bounding_box": (0, 0, 0, 0),
            "compactness": 0.0,
            "spectral_variance": 0.0,
        }

        # 허용 마스크 확정
        mask = allowed_mask if allowed_mask is not None else getattr(self.config, "allowed_mask", None)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (self.H, self.W):
                logging.warning("[RegionGrowing] allowed_mask shape mismatch, ignoring mask")
                mask = None

        # 시드가 ROI 밖이면 즉시 빈 결과 (기본 stats 사용)
        if mask is not None and not bool(mask[seed_y, seed_x]):
            empty = np.zeros((self.H, self.W), dtype=bool)
            return RegionGrowingResult(
                region_mask=empty,
                region_pixels=[],
                seed_coordinate=(int(seed_y), int(seed_x)),
                region_spectrum=self.hsi_data[seed_y, seed_x, :].astype(np.float32),
                config=self.config,
                statistics=stats,
            )

        # 시드 스펙트럼
        if seed_spectrum is None:
            seed_spectrum = self.hsi_data[seed_y, seed_x, :]
        else:
            seed_spectrum = np.asarray(seed_spectrum, dtype=np.float32)
        seed_spectrum = seed_spectrum.astype(np.float32, copy=False)

        # 상태
        region_mask = np.zeros((self.H, self.W), dtype=bool)
        visited = np.zeros((self.H, self.W), dtype=bool)
        region_pixels: List[Tuple[int, int]] = []
        q = deque([(seed_y, seed_x)])

        # ★ 2) seed-거리 캐시(옵션) — 항상 정의해둠
        seed_dist_map = np.full((self.H, self.W), np.inf, dtype=np.float32)
        seed_dist_map[seed_y, seed_x] = 0.0

        # 시드 편입
        region_mask[seed_y, seed_x] = True
        visited[seed_y, seed_x] = True
        region_pixels.append((seed_y, seed_x))

        region_sum = self.hsi_data[seed_y, seed_x, :].copy()

        total_distance = 0.0
        distance_count = 0

        τ = float(self.config.threshold)
        δ = float(self.config.margin)

        while q and len(region_pixels) < self.config.max_region_size:
            cy, cx = q.popleft()
            cur_spec = self.hsi_data[cy, cx, :]

            d_seed = float(self.distance_fn(cur_spec, seed_spectrum))
            seed_dist_map[cy, cx] = d_seed  # ★ 캐시에 기록
            total_distance += d_seed
            distance_count += 1

            if d_seed > τ:
                region_mask[cy, cx] = False
                try:
                    region_pixels.remove((cy, cx))
                except ValueError:
                    pass
                continue

            mean_spec = region_sum / float(len(region_pixels))

            for dy, dx in self.directions:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < self.H and 0 <= nx < self.W): continue
                if mask is not None and not bool(mask[ny, nx]):  continue
                if visited[ny, nx]:                              continue

                neigh = self.hsi_data[ny, nx, :]
                # (옵션) 이웃의 seed-거리도 기록(분석용)
                d_seed_neigh = float(self.distance_fn(neigh, seed_spectrum))
                if d_seed_neigh < seed_dist_map[ny, nx]:
                    seed_dist_map[ny, nx] = d_seed_neigh

                d_cur  = float(self.distance_fn(neigh, cur_spec))
                d_mean = float(self.distance_fn(neigh, mean_spec))

                visited[ny, nx] = True
                if (d_cur <= δ) and (d_mean <= δ):  # 필요 시 and (d_seed_neigh <= τ)
                    region_mask[ny, nx] = True
                    region_pixels.append((ny, nx))
                    q.append((ny, nx))
                    region_sum += neigh

        # 최소 크기 경고
        if len(region_pixels) < self.config.min_region_size:
            logging.warning(f"[RegionGrowing] Region size {len(region_pixels)} < min {self.config.min_region_size}")

        # 평균 스펙
        region_spectrum = (
            (region_sum / float(len(region_pixels))).astype(np.float32, copy=False)
            if region_pixels else seed_spectrum
        )

        # ★ 3) 끝에서 stats 안전 업데이트
        stats["region_size"] = len(region_pixels)
        stats["average_distance"] = (total_distance / max(distance_count, 1))
        stats["bounding_box"] = self._calculate_bounding_box(region_pixels)
        stats["compactness"] = self._calculate_compactness(region_pixels)
        stats["spectral_variance"] = self._calculate_spectral_variance(region_pixels)

        # ★ 4) seed_dist_map 요약은 영역이 있을 때만
        if region_pixels:
            ys, xs = zip(*region_pixels)  # 비어있지 않음 보장
            vals = seed_dist_map[list(ys), list(xs)]
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                stats["seed_dist_mean"] = float(np.mean(vals))
                stats["seed_dist_min"]  = float(np.min(vals))
                stats["seed_dist_max"]  = float(np.max(vals))

        return RegionGrowingResult(
            region_mask=region_mask,
            region_pixels=region_pixels,
            seed_coordinate=(int(seed_y), int(seed_x)),
            region_spectrum=region_spectrum,
            config=self.config,
            statistics=stats,
            seed_distance_map=seed_dist_map,  # 있으면 활용
        )


    def grow_multiple_regions(
        self,
        seeds: List[Tuple[int, int]],
        seed_spectra: Optional[List[np.ndarray]] = None,
        allowed_mask: Optional[np.ndarray] = None,
    ) -> List[RegionGrowingResult]:
        results: List[RegionGrowingResult] = []
        for i, (sx, sy) in enumerate(seeds):
            ss = None
            if seed_spectra and i < len(seed_spectra):
                ss = seed_spectra[i]
            try:
                r = self.grow_region(sx, sy, ss, allowed_mask=allowed_mask)
                results.append(r)
            except Exception as e:
                logging.error(f"[RegionGrowing] grow failed for seed ({sx},{sy}): {e}")
        return results

    # ---------------- internal utils ----------------
    def _calculate_bounding_box(self, pixels: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        if not pixels:
            return (0, 0, 0, 0)
        ys, xs = zip(*pixels)
        return (min(ys), max(ys), min(xs), max(xs))

    def _calculate_compactness(self, pixels: List[Tuple[int, int]]) -> float:
        if len(pixels) < 2:
            return 1.0
        y_min, y_max, x_min, x_max = self._calculate_bounding_box(pixels)
        area = len(pixels)
        perimeter = 2 * ((y_max - y_min + 1) + (x_max - x_min + 1))
        return (perimeter ** 2) / max(area, 1)

    def _calculate_spectral_variance(self, pixels: List[Tuple[int, int]]) -> float:
        if len(pixels) < 2:
            return 0.0
        spectra = np.array([self.hsi_data[y, x, :] for y, x in pixels], dtype=np.float32)
        mean_spec = np.mean(spectra, axis=0)
        distances = [float(self.distance_fn(s, mean_spec)) for s in spectra]
        return float(np.var(distances))


# ---- helpers (모듈 레벨이어야 import 가능) ----
def create_region_growing_config(
    metric: str = "SAM",
    threshold: float = 0.10,
    margin: float = 0.06,
    min_region_size: int = 10,
    max_region_size: int = 10000,
    connectivity: int = 8,
    allowed_mask: Optional[np.ndarray] = None,
) -> RegionGrowingConfig:
    return RegionGrowingConfig(
        metric=metric,
        threshold=threshold,
        margin=margin,
        min_region_size=min_region_size,
        max_region_size=max_region_size,
        connectivity=connectivity,
        allowed_mask=allowed_mask,
    )


def perform_region_growing(
    hsi_data: np.ndarray,
    seed_x: int,
    seed_y: int,
    seed_spectrum: Optional[np.ndarray] = None,
    metric: str = "SAM",
    threshold: float = 0.10,
    margin: float = 0.06,
    min_region_size: int = 10,
    max_region_size: int = 10000,
    connectivity: int = 8,
    allowed_mask: Optional[np.ndarray] = None,
) -> RegionGrowingResult:
    cfg = create_region_growing_config(
        metric=metric,
        threshold=threshold,
        margin=margin,
        min_region_size=min_region_size,
        max_region_size=max_region_size,
        connectivity=connectivity,
        allowed_mask=allowed_mask,
    )
    rg = RegionGrowing(hsi_data, cfg)
    return rg.grow_region(seed_x, seed_y, seed_spectrum, allowed_mask=allowed_mask)


# quick test
if __name__ == "__main__":
    H, W, C = 100, 120, 50
    test_hsi = np.random.rand(H, W, C).astype(np.float32)
    roi = np.zeros((H, W), bool); roi[30:80, 40:90] = True  # ROI 예시
    res = perform_region_growing(
        hsi_data=test_hsi,
        seed_x=60,
        seed_y=50,
        metric="SAM",
        threshold=0.10,   # τ
        margin=0.06,      # δ
        min_region_size=1,
        max_region_size=5000,
        connectivity=8,
        allowed_mask=roi
    )
    print(f"Region size: {res.region_size}")
    print(f"BBox: {res.bounding_box}")
    print(f"Stats: {res.statistics}")

