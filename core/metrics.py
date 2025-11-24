# core/metrics.py
from __future__ import annotations
import numpy as np

_EPS = 1e-12

def _l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.maximum((x * x).sum(axis=axis), _EPS))

# def sad(x: np.ndarray, s: np.ndarray) -> np.ndarray:
#     """
#     Spectral Angle Distance (radian)
#     x: (..., C), s: (..., C) or (C,)
#     """
#     x = np.asarray(x, dtype=float)
#     s = np.asarray(s, dtype=float)
#     if s.ndim == 1:
#         s = np.broadcast_to(s, x.shape)
#     num = np.clip((x * s).sum(-1), -1.0, 1.0)
#     den = _l2norm(x, -1) * _l2norm(s, -1)
#     cos = np.clip(num / np.maximum(den, _EPS), -1.0, 1.0)
#     return np.arccos(cos)

def sad(x: np.ndarray, s: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    두 벡터(또는 벡터 집합) 간의 표준 스펙트럼 각 거리(SAD)를 계산합니다.
    SAD는 두 벡터 사이의 각도를 라디안 단위로 나타냅니다.

    수식:
    theta = arccos( (x . s) / (||x|| * ||s||) )

    Args:
        x (np.ndarray): 첫 번째 벡터 또는 벡터의 배열. (..., C) 형태.
        s (np.ndarray): 두 번째 벡터 또는 벡터의 배열. (..., C) 또는 (C,) 형태.
        axis (int): 벡터의 채널(특징)이 있는 축. 기본값은 -1 (마지막 축).

    Returns:
        np.ndarray: 계산된 SAD 값(라디안).
    """
    # 0으로 나누기 방지를 위한 매우 작은 값
    _EPS = np.finfo(float).eps  # 기계 엡실론 (예: 약 2.22e-16)

    # 1. 분자: 내적 (Dot Product)
    # (x * s).sum(axis)와 동일
    dot_product = np.einsum('...i,...i->...', x, s)

    # 2. 분모: 각 벡터의 L2 노름(크기)의 곱
    norm_x = np.linalg.norm(x, axis=axis)
    norm_s = np.linalg.norm(s, axis=axis)
    denominator = norm_x * norm_s

    # 3. 코사인 유사도 계산 (수치적 안정성 확보)
    # 분모가 0에 가까울 경우 0으로 나누는 것을 방지하기 위해 _EPS 사용
    # (np.maximum을 사용하면 0인 경우에도 안전하게 0 / _EPS = 0이 됨)
    cosine_similarity = dot_product / np.maximum(denominator, _EPS)
    
    # 4. 아크코사인 계산 (수치적 안정성 확보)
    # 부동 소수점 오류로 인해 값이 -1.0 ~ 1.0 범위를 미세하게 벗어날 수 있으므로
    # arccos에 넣기 전에 값을 강제로 [-1.0, 1.0] 범위로 클리핑(clipping)합니다.
    clipped_cosine = np.clip(cosine_similarity, -1.0, 1.0)

    # 5. 최종 각도(라디안) 반환
    angle = np.arccos(clipped_cosine)
    
    return angle

def sid(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Spectral Information Divergence
    x, s >= 0 권장. 각 벡터를 확률분포로 정규화 후
    SID(x,s) = Σ x*log(x/s) + Σ s*log(s/x)
    """
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = np.broadcast_to(s, x.shape)
    x = np.maximum(x, _EPS); s = np.maximum(s, _EPS)
    px = x / np.maximum(x.sum(-1, keepdims=True), _EPS)
    ps = s / np.maximum(s.sum(-1, keepdims=True), _EPS)
    return (px * np.log(px / ps)).sum(-1) + (ps * np.log(ps / px)).sum(-1)

def scc_distance(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    1 - Pearson correlation (거리화)
    범위: [0, 2], 작을수록 유사
    """
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = np.broadcast_to(s, x.shape)
    x_m = x - x.mean(-1, keepdims=True)
    s_m = s - s.mean(-1, keepdims=True)
    num = (x_m * s_m).sum(-1)
    den = _l2norm(x_m, -1) * _l2norm(s_m, -1)
    corr = num / np.maximum(den, _EPS)
    return 1.0 - corr

def get_metric(name: str):
    key = name.strip().lower()
    if key == "sad": return sad
    if key == "sid": return sid
    if key == "scc": return scc_distance
    raise ValueError(f"unknown metric: {name}")

sam = sad