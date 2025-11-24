# services/pixel_knn.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from core.metrics import get_metric  # ← 네 모듈 사용(SAD=SAM, SID, SCC)

def topk_for_pixel(
    pix: np.ndarray,                      # (C,)
    lib: Dict[int, np.ndarray],           # cid -> (P_i, C) prototypes
    *,
    metric: str = "SAD",                  # "SAD" | "SID" | "SCC"
    k: int = 10,
    only_cid: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """
    클릭한 1픽셀의 스펙트럼 pix와 라이브러리(prototypes)를 비교해
    '작을수록 유사'인 거리/각도 점수로 Top-K (클래스 단위) 반환.
    반환: [(cid, score), ...] 오름차순 정렬, 최대 k개

    참고: metrics.sad(x,s)는 SAM과 동일(라디안).
    """
    # --- 입력 정규화 ---
    b = np.asarray(pix, dtype=float).astype(np.float32, copy=False)
    if b.ndim != 1:
        raise ValueError(f"pix must be (C,), got {b.shape}")
    C = b.shape[0]

    # --- 메트릭 선택 (SAD=SAM) ---
    fn = get_metric(metric)  # sad | sid | scc_distance

    # --- 검색 대상 클래스 ---
    cids = [int(only_cid)] if (only_cid is not None) else [int(c) for c in lib.keys()]
    
    cand: List[Tuple[int, float]] = []
    for cid in cids:
        A = lib.get(int(cid))
        if A is None:
            continue
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[1] != C or A.size == 0:
            # 형상이 안 맞으면 스킵
            continue

        A = A.astype(np.float32, copy=False)  # (P_i, C)
        # core.metrics 함수들은 (..,C) vs (C,) 브로드캐스트 지원
        d = fn(A, b)                          # (P_i,)
        best = float(np.min(d))               # 클래스 내 최인접 프로토타입 점수
        cand.append((int(cid), best))

        # (옵션) 프로토타입 단위 Top-K를 원하면 아래처럼 확장 가능:
        # for val in np.partition(d, min(k-1, len(d)-1))[:k]:
        #     cand.append((int(cid), float(val)))

    cand.sort(key=lambda t: t[1])
    return cand[:k]
