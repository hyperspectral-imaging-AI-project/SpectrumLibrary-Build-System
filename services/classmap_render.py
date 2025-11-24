# services/classmap_render.py
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional, Iterable

UNKNOWN  = -1
MULTIPLE = -2


class ClassmapRenderer:
    """
    ClassmapRenderer: 클래스맵을 RGBA로 렌더링하는 유틸.
    - 팔레트(SSOT): cid -> (R,G,B) 를 단일 소스로 보관/제공
    - 엄격(Strict) 모드: 팔레트에 없는 cid는 미표시(임의색 금지) → 지도/뷰어 색 일치 보장
    - 느슨(Loose) 모드: 팔레트에 없는 cid는 결정적 규칙색(HSV 해시) 생성하여 즉시 팔레트에 등록

    일반 사용 흐름:
      1) set_palette_rgb({...}) 한 번 호출 (프로그램 시작 또는 이미지 로드 직후)
      2) classmap_rgba(labels) 로 전체 렌더 or class_mask_rgba(...) 로 부분 렌더
      3) get_palette_ref_rgb() 로 동일 팔레트 참조를 뷰어(표)에 전달
    """

    def __init__(self, *, strict: bool = True) -> None:
        # 팔레트(SSOT): cid -> (R,G,B)
        self._palette_rgb: Dict[int, Tuple[int, int, int]] = {
            UNKNOWN:  (128, 128, 128),
            MULTIPLE: (0, 0, 0),
        }
        self.strict: bool = bool(strict)

    # ------------------------------------------------------------------
    # 팔레트 관리/조회
    # ------------------------------------------------------------------
    def set_palette_rgb(self, palette: Dict[int, Tuple[int, int, int]]) -> None:
        """팔레트를 통째로 교체합니다. 값은 0~255 정수 RGB tuple 이어야 합니다."""
        self._palette_rgb = {int(k): self._to_rgb_tuple(v) for k, v in palette.items()}

    def update_color(self, cid: int, color: Tuple[int, int, int]) -> None:
        """특정 클래스의 색을 갱신합니다."""
        self._palette_rgb[int(cid)] = self._to_rgb_tuple(color)

    def get_palette_ref_rgb(self) -> Dict[int, Tuple[int, int, int]]:
        """
        현재 팔레트 dict '참조'(복사 X)를 반환합니다.
        → MainWindow/Dock에서 이 참조를 그대로 사용하면 지도/표 색이 1:1로 일치합니다.
        """
        return self._palette_rgb

    def get_palette_dict_rgb(self) -> Dict[int, Tuple[int, int, int]]:
        """팔레트 사본을 반환합니다(외부에서 수정 못 하게 하려면 이걸 쓰세요)."""
        return dict(self._palette_rgb)

    def ensure_colors(self, class_ids: Iterable[int]) -> None:
        """
        느슨 모드(strict=False)에서만 의미 있음.
        전달된 class_ids 중 팔레트에 없는 항목은 결정적 규칙색(HSV 해시)으로 채워 넣습니다.
        """
        if self.strict:
            return
        for cid in class_ids:
            cid = int(cid)
            if cid not in self._palette_rgb:
                self._palette_rgb[cid] = self._fallback_rgb_for_cid(cid)

    def color_for(self, cid: int) -> Optional[Tuple[int, int, int]]:
        """
        현재 모드에 따라 색을 돌려줍니다.
        - strict=True: 팔레트에 없으면 None
        - strict=False: 팔레트에 없으면 결정적 규칙색을 생성하여 등록 후 반환
        """
        cid = int(cid)
        c = self._palette_rgb.get(cid)
        if c is not None:
            return c
        if self.strict:
            return None
        c = self._fallback_rgb_for_cid(cid)
        self._palette_rgb[cid] = c
        return c

    # ------------------------------------------------------------------
    # 렌더링
    # ------------------------------------------------------------------
    def class_mask_rgba(self, labels, target_class, color=None, alpha=180):
        lab = labels.astype(np.int32, copy=False)
        H, W = lab.shape
        out = np.zeros((H, W, 4), dtype=np.uint8)

        # ★ 추가: UNKNOWN(-1)은 항상 '투명'
        if int(target_class) == UNKNOWN:
            return out

        m = (lab == int(target_class))
        if not m.any():
            return out

        if color is None:
            color = self.color_for(int(target_class))
            if color is None:
                return out
        r, g, b = map(int, color)
        out[m, 0] = r
        out[m, 1] = g
        out[m, 2] = b
        out[m, 3] = np.uint8(alpha)
        return out


    def classmap_rgba(self, labels: np.ndarray, alpha: int = 180) -> np.ndarray:
        lab = labels.astype(np.int32, copy=False)
        H, W = lab.shape
        out = np.zeros((H, W, 4), dtype=np.uint8)

        ids = np.unique(lab)
        if not self.strict:
            self.ensure_colors(ids)

        for cid in ids:
            cid_i = int(cid)

            # ★ 변경: UNKNOWN(-1)은 '투명'(=그리지 않음)
            if cid_i == UNKNOWN:
                continue

            c = self._palette_rgb.get(cid_i) if self.strict else self.color_for(cid_i)
            if c is None:
                continue
            m = (lab == cid_i)
            if not m.any():
                continue
            r, g, b = c
            out[m, 0] = r
            out[m, 1] = g
            out[m, 2] = b
            out[m, 3] = np.uint8(alpha)
        return out

    def mask_rgba(self, mask: np.ndarray, color: Tuple[int, int, int, int] = (255, 0, 0, 120)) -> np.ndarray:
        """
        이진/불리언 mask를 단색 RGBA로 칠합니다.
        mask: (H,W) bool/0-1
        color: (R,G,B,A)
        """
        m = (mask > 0).astype(np.uint8)
        H, W = m.shape
        r, g, b, a = map(int, color)
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[..., 0] = r
        rgba[..., 1] = g
        rgba[..., 2] = b
        rgba[..., 3] = m * a
        return rgba

    def heatmap_rgba(self, arr: np.ndarray, cm: str = "viridis") -> np.ndarray:
        """
        간단한 colormap(viridis 유사)으로 2D 실수 배열을 RGBA로 변환합니다.
        """
        x = arr.astype(np.float32)
        finite = np.isfinite(x)
        vmin, vmax = (np.nanpercentile(x[finite], [2, 98]) if finite.any() else (0.0, 1.0))
        scale = max(1e-6, (vmax - vmin))
        x = np.clip((x - vmin) / scale, 0, 1)

        r = np.clip(4 * x - 1.5, 0, 1)
        g = np.clip(4 * x - 0.5, 0, 1)
        b = np.clip(1.5 - 4 * x, 0, 1)
        rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
        a = (x * 255).astype(np.uint8)[..., None]
        return np.concatenate([rgb, a], axis=-1)

    # ------------------------------------------------------------------
    # 내부 도우미
    # ------------------------------------------------------------------
    @staticmethod
    def _to_rgb_tuple(v) -> Tuple[int, int, int]:
        """(R,G,B)로 정규화합니다."""
        if isinstance(v, (tuple, list)) and len(v) >= 3:
            r, g, b = int(v[0]), int(v[1]), int(v[2])
            return (
                max(0, min(255, r)),
                max(0, min(255, g)),
                max(0, min(255, b)),
            )
        raise ValueError(f"Invalid RGB value: {v!r}")

    @staticmethod
    def _hsv_to_rgb255(h: float, s: float, v: float) -> Tuple[int, int, int]:
        """h,s,v in [0,1] → (R,G,B) in 0..255"""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(float(h), float(s), float(v))
        return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))

    def _fallback_rgb_for_cid(self, cid: int) -> Tuple[int, int, int]:
        """
        결정적 규칙색(HSV 해시) — 랜덤 금지.
        strict=False일 때만 사용하도록 권장.
        """
        h_deg = (int(cid) * 37) % 360
        h = h_deg / 360.0
        s = 0.78
        v = 0.90
        return self._hsv_to_rgb255(h, s, v)
