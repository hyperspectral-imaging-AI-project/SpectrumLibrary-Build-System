# core/vis.py
from __future__ import annotations
import numpy as np
from typing import Iterable, Optional, Tuple

def pick_bands_by_wavelength(wavelength: Iterable[float], targets=(650, 550, 450)) -> Tuple[int,int,int]:
    """타깃 파장(기본: R=650, G=550, B=450 nm)에 가장 가까운 밴드 인덱스를 반환."""
    wl = np.asarray(list(wavelength), dtype=float)
    idx = [int(np.argmin(np.abs(wl - t))) for t in targets]
    return idx[0], idx[1], idx[2]

# def percentile_stretch(img: np.ndarray, pmin=2.0, pmax=98.0) -> np.ndarray:
#     """채널별 퍼센트 스트레치(2–98%) 후 [0,1]로 스케일."""
#     x = img.astype(np.float32, copy=False)
#     if x.ndim == 2: x = x[..., None]
#     y = np.empty_like(x, dtype=np.float32)
#     for c in range(x.shape[2]):
#         band = x[..., c]
#         finite = np.isfinite(band)
#         if not finite.any():
#             y[..., c] = 0
#             continue
#         vmin, vmax = np.nanpercentile(band[finite], [pmin, pmax])
#         if vmax > vmin:
#             b = (band - vmin) / (vmax*1.2 - vmin)
#         else:
#             b = band
#         y[..., c] = np.clip(b, 0, 1)
#     return y if img.ndim == 3 else y[..., 0]

def to_uint8(img01: np.ndarray) -> np.ndarray:
    """[0,1] 범위 이미지를 uint8로 변환."""
    return (np.clip(img01, 0, 1) * 255.0).astype(np.uint8)

def make_rgb(arr_hwc: np.ndarray,
             wavelength: Optional[Iterable[float]] = None,
             rgb_indices: Optional[Tuple[int,int,int]] = None,
             stretch=True, pmin=2.0, pmax=98.0) -> np.ndarray:
    """
    HSI (H,W,C) → RGB (H,W,3) uint8
    - wavelength 있으면 650/550/450nm 근접 밴드 선택(혹은 rgb_indices 우선)
    - wavelength 없으면 C>=3은 앞 3밴드, C==1/2는 복제
    - stretch=True면 채널별 퍼센트 스트레치 적용
    """
    assert arr_hwc.ndim == 3, f"expected (H,W,C), got {arr_hwc.shape}"
    H, W, C = arr_hwc.shape
    x = arr_hwc.astype(np.float32, copy=False)

    # 밴드 선택
    if rgb_indices is not None:
        r,g,b = rgb_indices
    elif wavelength is not None and len(list(wavelength)) == C:
        r,g,b = pick_bands_by_wavelength(wavelength, (650, 550, 450))
    else:
        if C >= 3:
            r,g,b = 0,1,2
        elif C == 2:
            # 2채널이면 [G,G,B] 느낌으로
            return to_uint8(percentile_stretch(np.stack([x[...,0], x[...,0], x[...,1]], axis=-1), pmin, pmax) if stretch
                            else np.stack([x[...,0], x[...,0], x[...,1]], axis=-1))
        else:  # C==1
            g = x[...,0]
            return to_uint8(percentile_stretch(np.stack([g,g,g], axis=-1), pmin, pmax) if stretch
                            else np.stack([g,g,g], axis=-1))

    rgb = np.stack([x[..., r], x[..., g], x[..., b]], axis=-1)
    if stretch:
        # rgb = percentile_stretch(rgb, pmin, pmax)
        rgb = rgb**.4
        pass
    return to_uint8(rgb)

# (옵션) NIR false-color 프리셋
def make_false_color_nir(arr_hwc: np.ndarray, wavelength: Iterable[float]) -> np.ndarray:
    r,g,b = pick_bands_by_wavelength(wavelength, (800, 650, 550))
    return make_rgb(arr_hwc, wavelength=None, rgb_indices=(r,g,b), stretch=True)
