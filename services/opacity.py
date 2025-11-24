# services/opacity.py
from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple

def compute_alpha_from_gmin(
    gmin: Optional[np.ndarray],
    strict: float,
    base: float
) -> Optional[np.ndarray]:
    """
    g1(gmin) 기준으로 알파(0~255) 맵 생성.
      x < strict : 255  (불투명)
      strict <= x < base : 160
      x >= base : 80
    gmin NaN/미존재 위치는 0(완전 투명).
    """
    if gmin is None:
        return None
    g = np.asarray(gmin, dtype=np.float32)
    H, W = g.shape
    alpha = np.zeros((H, W), dtype=np.uint8)

    m0 = np.isfinite(g)
    if not np.any(m0):
        return alpha

    s = float(strict)
    b = float(base)

    m_strong = m0 & (g < s)
    m_mid    = m0 & (g >= s) & (g < b)
    m_weak   = m0 & (g >= b)

    alpha[m_strong] = 255
    alpha[m_mid]    = 160
    alpha[m_weak]   = 80
    return alpha
def compute_alpha_3way(
    scores: Optional[np.ndarray],   # (H,W) float, distance-like: 작을수록 유사
    strict: float,
    base: float,
    *,
    roi_mask: Optional[np.ndarray] = None,
    mid_alpha: int = 140            # strict~base 구간의 고정 투명도(원하면 조절)
) -> Optional[np.ndarray]:
    """
    3분할 알파:
      - score < strict              -> 255 (강한 유사)
      - strict <= score < base      -> mid_alpha (중간 유사)
      - score >= base               -> 0 (투명)
    ROI가 주어지면 ROI 밖은 무조건 0.
    NaN/미측정 위치는 0.
    """
    if scores is None:
        return None
    g = np.asarray(scores, dtype=np.float32)
    H, W = g.shape
    alpha = np.zeros((H, W), dtype=np.uint8)

    valid = np.isfinite(g)
    if roi_mask is not None:
        roi = np.asarray(roi_mask, dtype=bool)
        if roi.shape != (H, W):
            raise ValueError("roi_mask shape mismatch")
        valid &= roi  # ROI 밖은 자동으로 False → 알파 0

    if not np.any(valid):
        return alpha

    s = float(strict)
    b = float(base)

    # 3-way partition
    strong = valid & (g < s)
    mid    = valid & (g >= s) & (g < b)
    # weak  = valid & (g >= b)   # 0이 기본값이므로 별도 할당 불필요

    alpha[strong] = 255
    alpha[mid]    = np.uint8(np.clip(mid_alpha, 1, 254))
    # alpha[weak] stays 0
    return alpha

def build_opacity_overlay_rgba(
    classmap: np.ndarray,                   # (H,W) int32
    palette_rgb: Dict[int, Tuple[int,int,int]],  # cid -> (R,G,B)
    alpha: np.ndarray,                      # (H,W) uint8
    *,
    fallback_color_fn=None                  # Optional[cid -> (r,g,b)]
) -> np.ndarray:
    """
    classmap + palette + alpha → RGBA(uint8) 오버레이 생성.
    """
    H, W = classmap.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    # 색 채우기
    classes = np.unique(classmap)
    for cid in classes:
        mask = (classmap == int(cid))
        if not np.any(mask):
            continue
        rgb = palette_rgb.get(int(cid))
        if rgb is None and callable(fallback_color_fn):
            rgb = fallback_color_fn(int(cid))
        if rgb is None:
            rgb = (128, 128, 128)  # 최후의 보호색
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        rgba[mask, 0] = r
        rgba[mask, 1] = g
        rgba[mask, 2] = b

    # 알파 적용
    rgba[:, :, 3] = alpha
    return rgba

def build_selected_class_overlay_rgba(
    *,
    classmap: np.ndarray,                          # (H,W) int
    palette_rgb: Dict[int, Tuple[int,int,int]],
    topk_cids: np.ndarray,                         # (H,W,K) 또는 (h,w,K)
    topk_vals: np.ndarray,                         # (H,W,K) 또는 (h,w,K)
    target_cid: int,
    strict: float,
    base: float,
    metric: str = "SAD",
    roi_mask: Optional[np.ndarray] = None,
    assigned_only: bool = False,
    fallback_color_fn=None
) -> np.ndarray:
    """
    선택 클래스의 score로 3-way alpha를 만들고, alpha>0인 곳만 target 색으로 칠한 RGBA를 반환.
    ROI가 주어지면 ROI 밖은 항상 투명.
    """
    H, W = classmap.shape

    # --- 선택 클래스 점수 추출 (Top-K에서 target_cid의 점수만) ---
    sel = (topk_cids.astype(np.int32) == int(target_cid))     # (...,K)
    present = sel.any(axis=2)
    idx = sel.argmax(axis=2)
    scores_sub = np.take_along_axis(topk_vals, idx[..., None], axis=2).squeeze(-1).astype(np.float32)
    scores_sub[~present] = np.nan

    # --- ROI 서브 → 전역(H,W) 확장 ---
    if scores_sub.shape != (H, W):
        if roi_mask is None or roi_mask.shape != (H, W) or not np.any(roi_mask):
            raise ValueError("ROI-sized topk를 전역으로 확장하려면 roi_mask(H,W)가 필요합니다.")
        ys, xs = np.where(roi_mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        scores_full = np.full((H, W), np.nan, dtype=np.float32)
        roi_sub = roi_mask[y0:y1, x0:x1]
        if roi_sub.shape == scores_sub.shape:
            scores_full[y0:y1, x0:x1][roi_sub] = scores_sub[roi_sub]
        else:
            scores_full[y0:y1, x0:x1] = scores_sub
    else:
        scores_full = scores_sub

    # (옵션) 현재 분류가 target이 아닌 곳은 보지 않음
    if assigned_only:
        scores_full[classmap != int(target_cid)] = np.nan

    # --- 거리형이면 strict < base 강제 ---
    s, b = float(strict), float(base)
    if metric in ("SAD", "SID", "SCC") and s > b:
        s, b = b, s

    # --- ROI 기준 3-way alpha 생성 (base 이상은 완전 투명) ---
    alpha = compute_alpha_3way(scores_full, s, b, roi_mask=roi_mask, mid_alpha=140)

    # --- alpha>0 위치만 target 색으로 덮어 칠하도록 map 구성 ---
    cm_for_overlay = classmap.copy()
    cm_for_overlay[alpha > 0] = int(target_cid)

    rgba = build_opacity_overlay_rgba(
        cm_for_overlay, palette_rgb, alpha, fallback_color_fn=fallback_color_fn
    )
    return rgba

# services/opacity.py  (파일 맨 아래에 추가)
def compute_alpha_target_gradient(
    scores: np.ndarray,                 # (H,W) float, "작을수록 유사" 가정(SAD/SID/SCC, SAM)
    strict: float,
    base: float,
    *,
    roi_mask: Optional[np.ndarray] = None,   # (H,W) bool ; None이면 전체
    target_mask: Optional[np.ndarray] = None,# (H,W) bool ; None이면 무시
    gamma: float = 1.8,                      # >1이면 강한 쪽을 더 강조(대비 강화)
    min_alpha: int = 0,                      # base 경계에서의 최소 알파
    max_alpha: int = 255,                    # strict 경계에서의 최대 알파
    use_quantiles: bool = True,              # True면 엄청 큰/작은 이상치를 잘라서 스케일 안정화
    q_lo: float = 1.0,                       # strict 쪽 하위 q%를 하한으로
    q_hi: float = 99.0                       # base 쪽 상위 q%를 상한으로
) -> np.ndarray:
    """
    거리형 점수(작을수록 유사)에서 strict/base를 경계로 [max_alpha..min_alpha]까지
    비선형(γ) 그래디언트로 매핑한 알파맵을 생성. ROI/target 마스크 밖은 0.
    """
    H, W = scores.shape
    s = float(strict)
    b = float(base)
    if s > b:  # 거리형 보호: strict < base
        s, b = b, s

    a = np.asarray(scores, dtype=np.float32).copy()
    # ROI/Target 바깥은 투명 처리
    valid = np.isfinite(a)
    if roi_mask is not None:
        valid &= roi_mask.astype(bool)
    if target_mask is not None:
        valid &= target_mask.astype(bool)

    out = np.zeros((H, W), dtype=np.uint8)
    if not np.any(valid):
        return out

    v = a[valid]

    # 선택: 분위수로 극단값 클리핑 → 대비 안정화
    if use_quantiles and v.size > 32:
        lo = np.percentile(v, q_lo)
        hi = np.percentile(v, q_hi)
        s_eff = max(min(s, hi), lo)
        b_eff = max(min(b, hi), lo + 1e-6)
    else:
        s_eff, b_eff = s, b

    # 정규화: t = (b_eff - score) / (b_eff - s_eff), t<=0:0, t>=1:1
    denom = max(b_eff - s_eff, 1e-6)
    t = (b_eff - a) / denom
    t = np.clip(t, 0.0, 1.0)

    # 감마로 대비 강화 (γ>1이면 작은 score 부근을 더 밝게)
    t = np.power(t, gamma)

    # 알파 스케일링
    alpha_f = min_alpha + (max_alpha - min_alpha) * t
    out[valid] = np.uint8(np.clip(alpha_f[valid], 0, 255))

    # base 이상(정말 유사하지 않음)은 완전 투명 강제
    out[(a >= b) & valid] = 0
    # strict 미만(아주 유사)은 완전 불투명 강제
    out[(a < s) & valid] = np.uint8(max_alpha)

    return out

# def build_target_only_overlay_rgba(
#     *,
#     classmap: np.ndarray,
#     palette_rgb: Dict[int, Tuple[int,int,int]],
#     topk_cids: np.ndarray,
#     topk_vals: np.ndarray,
#     target_cid: int,
#     strict: float,
#     base: float,
#     metric: str = "SAD",
#     roi_mask: Optional[np.ndarray] = None,
#     fallback_color_fn=None,
#     # 새 파라미터(원하면 UI에서 노출):
#     gamma: float = 1.8,
#     mid_min_alpha: int = 30,      # 중간구간의 최저 알파(너무 옅어지지 않게 바닥 깔기)
#     use_quantiles: bool = True,
#     q_lo: float = 1.0,
#     q_hi: float = 99.0,
# ) -> np.ndarray:
#     H, W = classmap.shape

#     # Top-K에서 target 점수만 추출(없으면 NaN)
#     sel = (topk_cids.astype(np.int32) == int(target_cid))
#     present = sel.any(axis=2)
#     idx = sel.argmax(axis=2)
#     scores_sub = np.take_along_axis(topk_vals, idx[..., None], axis=2).squeeze(-1).astype(np.float32)
#     scores_sub[~present] = np.nan

#     # ROI 서브 크기라면 전역으로 확장
#     if scores_sub.shape != (H, W):
#         if roi_mask is None or roi_mask.shape != (H, W) or not np.any(roi_mask):
#             raise ValueError("ROI-sized topk를 전역으로 확장하려면 roi_mask(H,W)가 필요합니다.")
#         ys, xs = np.where(roi_mask)
#         y0, y1 = int(ys.min()), int(ys.max()) + 1
#         x0, x1 = int(xs.min()), int(xs.max()) + 1
#         scores = np.full((H, W), np.nan, dtype=np.float32)
#         roi_sub = roi_mask[y0:y1, x0:x1]
#         if roi_sub.shape == scores_sub.shape:
#             scores[y0:y1, x0:x1][roi_sub] = scores_sub[roi_sub]
#         else:
#             scores[y0:y1, x0:x1] = scores_sub
#     else:
#         scores = scores_sub

#     # 타깃/ROI 마스크
#     target_mask = (classmap == int(target_cid))
#     roi_bool = roi_mask.astype(bool) if roi_mask is not None else np.ones((H, W), dtype=bool)
#     valid_mask = target_mask & roi_bool

#     # 거리형이면 strict<base 강제
#     s, b = (base, strict) if (metric in ("SAD", "SID", "SCC") and strict > base) else (strict, base)

#     # 그래디언트 알파 생성 (대비 강화)
#     alpha = compute_alpha_target_gradient(
#         scores, s, b,
#         roi_mask=roi_bool,               # ROI 밖 0
#         target_mask=target_mask,         # 타깃 외 0
#         gamma=gamma,
#         min_alpha=mid_min_alpha,         # base 근처도 약간 보이게 바닥 깔기(원하면 0)
#         max_alpha=255,
#         use_quantiles=use_quantiles,
#         q_lo=q_lo, q_hi=q_hi
#     )
    
#     # 알파>0 인 곳만 타깃 색으로 칠함
#     cm_for_overlay = classmap.copy()
#     cm_for_overlay[alpha > 0] = int(target_cid)

#     return build_opacity_overlay_rgba(
#         cm_for_overlay, palette_rgb, alpha, fallback_color_fn=fallback_color_fn
#     )

from typing import Dict, Optional, Tuple
import numpy as np

def build_target_only_overlay_rgba(
    *,
    classmap: np.ndarray,
    palette_rgb: Dict[int, Tuple[int,int,int]],
    topk_cids: np.ndarray,
    topk_vals: np.ndarray,
    target_cid: int,
    strict: float,
    base: float,
    metric: str = "SAD",
    roi_mask: Optional[np.ndarray] = None,
    fallback_color_fn=None,
    # 대비/가시성 제어용
    gamma: float = 1.8,
    mid_min_alpha: int = 30,     # 중간 구간의 최소 알파(너무 연해지는 것 방지)
    use_quantiles: bool = True,
    q_lo: float = 1.0,
    q_hi: float = 99.0,
) -> np.ndarray:
    H, W = classmap.shape

    # --- Top-K에서 target_cid의 점수만 추출 (없으면 NaN) ---
    sel = (topk_cids.astype(np.int32) == int(target_cid))         # (...,K)
    present = sel.any(axis=2)
    idx = sel.argmax(axis=2)
    scores_sub = np.take_along_axis(topk_vals, idx[..., None], axis=2).squeeze(-1).astype(np.float32)
    scores_sub[~present] = np.nan

    # --- ROI 서브크기라면 전역(H,W)으로 확장 ---
    if scores_sub.shape != (H, W):
        if roi_mask is None or roi_mask.shape != (H, W) or not np.any(roi_mask):
            raise ValueError("ROI-sized topk를 전역으로 펼치려면 roi_mask(H,W)가 필요합니다.")
        ys, xs = np.where(roi_mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        scores = np.full((H, W), np.nan, dtype=np.float32)
        roi_sub = roi_mask[y0:y1, x0:x1]
        if roi_sub.shape == scores_sub.shape:
            scores[y0:y1, x0:x1][roi_sub] = scores_sub[roi_sub]
        else:
            scores[y0:y1, x0:x1] = scores_sub
    else:
        scores = scores_sub

    # --- 타깃/ROI 유효영역 마스크 ---
    target_mask = (classmap == int(target_cid))
    roi_bool = roi_mask.astype(bool) if roi_mask is not None else np.ones((H, W), dtype=bool)

    # --- 거리형이면 strict < base 보정 ---
    s, b = (base, strict) if (metric in ("SAD","SID","SCC") and strict > base) else (strict, base)

    # --- 타깃·ROI 영역에 대해서만 그래디언트 알파 생성 ---
    alpha = compute_alpha_target_gradient(
        scores, s, b,
        roi_mask=roi_bool,              # ROI 밖은 0
        target_mask=target_mask,        # 타깃 외 영역 0
        gamma=gamma,
        min_alpha=mid_min_alpha,
        max_alpha=255,
        use_quantiles=use_quantiles,
        q_lo=q_lo, q_hi=q_hi,
    )

    # === 여기부터가 핵심 변경점 ===
    # classmap을 재색칠하지 않고, 타깃 색을 쓰는 RGBA를 직접 구성한다.
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    rgb = palette_rgb.get(int(target_cid))
    if rgb is None and callable(fallback_color_fn):
        rgb = tuple(int(v) for v in fallback_color_fn(int(target_cid)))
    if rgb is None:
        rgb = (255, 0, 255)  # 최후의 보호색(예: 마젠타)

    r, g, b_ = int(rgb[0]), int(rgb[1]), int(rgb[2])

    # 타깃 ∧ ROI ∧ (alpha>0) 위치에만 타깃 색 채움
    vis = (alpha > 0)
    rgba[vis, 0] = r
    rgba[vis, 1] = g
    rgba[vis, 2] = b_
    rgba[..., 3] = alpha  # 알파는 전역 픽셀별로 이미 마스크 반영됨

    return rgba


def build_target_only_trafficlight_overlay_rgba(
    *,
    classmap: np.ndarray,                          # (H,W) int
    topk_cids: np.ndarray,                         # (H,W,K) 또는 (h,w,K)
    topk_vals: np.ndarray,                         # (H,W,K) 또는 (h,w,K)
    target_cid: int,
    strict: float,
    base: float,
    metric: str = "SAD",
    roi_mask: Optional[np.ndarray] = None,
    alpha: int = 180,                              # 모든 색 공통 알파(가시성)
    # 색상 커스터마이즈 가능
    color_yellow: Tuple[int,int,int] = (255, 255, 0),   # x < strict
    color_green:  Tuple[int,int,int] = (0, 200, 0),     # strict <= x < base
    color_red:    Tuple[int,int,int] = (230, 0, 0),     # x >= base
) -> np.ndarray:
    """
    ROI∩(class==target) 픽셀만 대상.
      - score < strict              → 노랑
      - strict <= score < base      → 초록
      - score >= base               → 빨강
    ROI 밖/타깃 외/점수없음은 완전 투명(알파=0).
    거리형(SAD/SID/SCC)은 값이 작을수록 유사 → strict<base 강제.
    """
    H, W = classmap.shape
    # 1) target 점수 2D 추출 (없으면 NaN)
    sel = (topk_cids.astype(np.int32) == int(target_cid))      # (...,K)
    present = sel.any(axis=2)
    idx = sel.argmax(axis=2)
    scores_sub = np.take_along_axis(topk_vals, idx[..., None], axis=2).squeeze(-1).astype(np.float32)
    scores_sub[~present] = np.nan

    # 2) ROI 서브 → 전역(H,W)로 확장
    if scores_sub.shape != (H, W):
        if roi_mask is None or roi_mask.shape != (H, W) or not np.any(roi_mask):
            raise ValueError("ROI-sized topk를 전역으로 확장하려면 roi_mask(H,W)가 필요합니다.")
        ys, xs = np.where(roi_mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        scores = np.full((H, W), np.nan, dtype=np.float32)
        roi_sub = roi_mask[y0:y1, x0:x1]
        if roi_sub.shape == scores_sub.shape:
            scores[y0:y1, x0:x1][roi_sub] = scores_sub[roi_sub]
        else:
            scores[y0:y1, x0:x1] = scores_sub
    else:
        scores = scores_sub

    # 3) 마스크들
    tgt = (classmap == int(target_cid))
    roi = roi_mask.astype(bool) if roi_mask is not None else np.ones((H, W), dtype=bool)
    valid = np.isfinite(scores) & tgt & roi

    # 4) 거리형이면 strict<base 보정
    s, b = float(strict), float(base)
    if metric in ("SAD", "SID", "SCC") and s > b:
        s, b = b, s

    # 5) 구간 마스크
    m_yellow = valid & (scores < s)          # 강한 유사
    m_green  = valid & (scores >= s) & (scores < b)
    m_red    = valid & (scores >= b)

    # 6) RGBA 생성 (나머지는 완전 투명)
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    if np.any(m_yellow):
        rgba[m_yellow, 0] = color_yellow[0]; rgba[m_yellow, 1] = color_yellow[1]; rgba[m_yellow, 2] = color_yellow[2]
        rgba[m_yellow, 3] = np.uint8(alpha)
    if np.any(m_green):
        rgba[m_green, 0] = color_green[0];   rgba[m_green, 1] = color_green[1];   rgba[m_green, 2] = color_green[2]
        rgba[m_green, 3] = np.uint8(alpha)
    if np.any(m_red):
        rgba[m_red, 0] = color_red[0];       rgba[m_red, 1] = color_red[1];       rgba[m_red, 2] = color_red[2]
        rgba[m_red, 3] = np.uint8(alpha)

    return rgba