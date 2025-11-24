# core/autoclass.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
from core.metrics import get_metric  # sad/sid/scc/sam… 등 원시 '거리' 반환 함수 (작을수록 유사)

UNKNOWN  = -1
MULTIPLE = -2
HOLD     = -3  # prev_map 재평가용

def classify_cube(
    cube: np.ndarray,                       # (H,W,C)
    library: Dict[int, np.ndarray],         # {class_id: (P_i, C)} — 프로토타입(레퍼런스 스펙)
    *,
    metric: str = "sad",                    # "sad"(=SAM 라디안) | "sid" | "scc" (1-corr) | "sam" 등
    tau: float = 0.05,                      # 임계값(작을수록 유사) — 원시 스케일 기준
    delta: float = 0.03,                    # 1등-2등 마진(원시 스케일)
    prev_map: Optional[np.ndarray] = None,  # (H,W) — Unknown/HOLD만 재평가
    unknown_code: int = UNKNOWN,
    multiple_code: int = MULTIPLE,
    hold_code: int = HOLD,
    chunk_size: int = 100_000,              # 픽셀 청킹
    save_topk: bool = True,                 # 픽셀별 Top-K 캐시 저장
    topk_k: int = 10,                       # Top-K 개수
    ref_ids: Optional[Dict[int, List[Any]]] = None,  # ★ {cid: [label_id0, ...]} — library 행 순서와 동일
    save_best_proto: bool = True            # ★ Top-1에서 선택된 프로토 행 인덱스/ID를 extras에 포함
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    전 과정(판정/캐시)이 '원시 거리'로 동작합니다.
      - metric='sam' 또는 'sad' 구현체가 SAM 라디안 등 '작을수록 유사'인 값을 반환한다고 가정
      - metric='sid' → SID
      - metric='scc' → 1 - 상관계수

    반환:
      classmap(H,W), extras(dict):
        - gmin(H,W)       : Top-1 원시 거리
        - margin(H,W)     : Top-2 - Top-1
        - best_class(H,W) : Top-1 class id
        - second_class(H,W): Top-2 class id
        - (옵션) topk_cids(H,W,K), topk_vals(H,W,K)  # 모두 원시 스케일
        - (옵션) best_proto_idx_top1(H,W)            # Top-1 클래스에서 선택된 library 행 인덱스
        - (옵션) best_proto_id_top1(H,W)             # ref_ids가 제공된 경우 Top-1의 라벨 ID(object)
        - lib_row_counts(dict[class_id->int])         # 각 클래스 프로토 개수
    """
    H, W, C = cube.shape
    N = H * W

    # ---------- 입력/라이브러리 정리 ----------
    X = cube.reshape(-1, C).astype(np.float32, copy=False)  # (N,C)
    class_ids = sorted(int(c) for c in library.keys())
    if not class_ids:
        raise ValueError("library가 비어 있습니다.")
    class_ids_arr = np.asarray(class_ids, dtype=np.int32)

    protos: List[np.ndarray] = []
    lib_row_counts: Dict[int, int] = {}
    for cid in class_ids:
        P = np.asarray(library[cid], dtype=np.float32, order="C")
        if P.ndim != 2 or P.shape[1] != C:
            raise ValueError(f"prototype shape must be (P_i, C); got {P.shape} for class {cid}")
        protos.append(P)
        lib_row_counts[cid] = int(P.shape[0])

    metric_fn = get_metric(metric)  # core/metrics.py 의 거리 함수: (B,1,C) vs (1,Pi,C) → (B,Pi)

    # ---------- 재평가 마스크 ----------
    if prev_map is None:
        re_mask = np.ones(N, dtype=bool)
        out = np.empty(N, dtype=np.int32)
    else:
        prev_flat = prev_map.reshape(-1)
        re_mask = (prev_flat == unknown_code) | (prev_flat == hold_code)
        out = prev_flat.astype(np.int32, copy=True)

    # ---------- 보조 출력 버퍼 ----------
    gmin   = np.full(N, np.nan, dtype=np.float32)
    margin = np.full(N, np.nan, dtype=np.float32)
    best_class   = np.full(N, -999, dtype=np.int32)
    second_class = np.full(N, -999, dtype=np.int32)

    # ---------- Top-K 캐시 버퍼 ----------
    K_class = len(class_ids)
    K_top = min(int(topk_k), K_class) if save_topk else 0
    topk_cids_flat = None
    topk_vals_flat = None
    if save_topk and K_top > 0:
        topk_cids_flat = np.full((N, K_top), -999, dtype=np.int32)
        topk_vals_flat = np.full((N, K_top), np.nan, dtype=np.float32)

    # ---------- Top-1 참조 프로토 인덱스/ID 버퍼 ----------
    best_proto_idx_flat = None
    best_proto_id_flat  = None
    if save_best_proto:
        best_proto_idx_flat = np.full(N, -1, dtype=np.int32)
        if ref_ids is not None:
            best_proto_id_flat = np.empty(N, dtype=object)

    idxs = np.nonzero(re_mask)[0]
    if idxs.size == 0:
        classmap = out.reshape(H, W)
        extras = {
            "gmin": gmin.reshape(H, W),
            "margin": margin.reshape(H, W),
            "best_class": best_class.reshape(H, W),
            "second_class": second_class.reshape(H, W),
            "lib_row_counts": lib_row_counts,
        }
        if save_topk and K_top > 0:
            extras["topk_cids"] = topk_cids_flat.reshape(H, W, K_top)
            extras["topk_vals"] = topk_vals_flat.reshape(H, W, K_top)
        if save_best_proto:
            extras["best_proto_idx_top1"] = best_proto_idx_flat.reshape(H, W)
            if best_proto_id_flat is not None:
                extras["best_proto_id_top1"] = best_proto_id_flat.reshape(H, W)
        return classmap, extras

    # ---------- 청크 처리 ----------
    for start in range(0, idxs.size, chunk_size):
        sel = idxs[start:start + chunk_size]
        B = sel.size
        Xb = X[sel]  # (B,C)

        # 클래스별 최소 거리행렬 G: (B, K_class)
        G = np.empty((B, K_class), dtype=np.float32)

        # ★ 클래스별 '그 클래스 내부 argmin 행 인덱스' 버퍼
        argmin_in_class = np.empty((B, K_class), dtype=np.int32)

        # 각 클래스에 대해 거리 계산
        for k, P in enumerate(protos):
            # (B,Pi) — 거리 행렬
            dists = metric_fn(Xb[:, None, :], P[None, :, :]).astype(np.float32, copy=False)
            if not np.isfinite(dists).all():
                np.nan_to_num(dists, copy=False, nan=np.inf, posinf=np.inf, neginf=np.inf)

            j_min = np.argmin(dists, axis=1)              # (B,)
            G[np.arange(B), k] = dists[np.arange(B), j_min]
            argmin_in_class[:, k] = j_min

        # Top-2 및 판정
        idx1 = np.argmin(G, axis=1)                       # (B,)
        g1   = G[np.arange(B), idx1]                      # (B,)
        G2 = G.copy()
        G2[np.arange(B), idx1] = np.inf
        idx2 = np.argmin(G2, axis=1)
        g2   = G2[np.arange(B), idx2]
        c1   = class_ids_arr[idx1]
        c2   = class_ids_arr[idx2]

        cond_unknown = (g1 > tau)
        cond_single  = (~cond_unknown) & ((g2 - g1) >= delta)

        out_sel = np.full(B, multiple_code, dtype=np.int32)
        out_sel[cond_unknown] = unknown_code
        out_sel[cond_single]  = c1[cond_single]

        out[sel] = out_sel
        gmin[sel] = g1
        mg = (g2 - g1)
        mg[~np.isfinite(mg)] = np.nan
        margin[sel] = mg
        best_class[sel] = c1
        second_class[sel] = c2

        # Top-K 저장(원시 G)
        if save_topk and K_top > 0:
            part = np.argpartition(G, K_top - 1, axis=1)[:, :K_top]  # (B,K_top) — 작은 값 K개
            vals = np.take_along_axis(G, part, axis=1)                # (B,K_top)
            order = np.argsort(vals, axis=1)
            part = np.take_along_axis(part, order, axis=1)            # (B,K_top)
            vals = np.take_along_axis(vals, order, axis=1)            # (B,K_top)
            topk_cids_flat[sel, :] = class_ids_arr[part]
            topk_vals_flat[sel, :] = vals

        # ★ Top-1 클래스에서 사용된 라이브러리 '행 인덱스' 기록
        if save_best_proto:
            used_row_idx = argmin_in_class[np.arange(B), idx1]        # (B,)
            best_proto_idx_flat[sel] = used_row_idx.astype(np.int32)

            # ★ 라벨 ID 매핑(있으면)
            if ref_ids is not None:
                if best_proto_id_flat is None:
                    best_proto_id_flat = np.empty(N, dtype=object)
                cids_block = class_ids_arr[idx1]                      # (B,)
                # 픽셀별로 cid/row_idx로 ref_ids[cid][row_idx] 조회
                for bi in range(B):
                    cid_b  = int(cids_block[bi])
                    ridx_b = int(used_row_idx[bi])
                    rid_list = ref_ids.get(cid_b)
                    best_proto_id_flat[sel[bi]] = (rid_list[ridx_b]
                        if rid_list and 0 <= ridx_b < len(rid_list) else None)

    # ---------- 마무리 ----------
    classmap = out.reshape(H, W)
    extras = {
        "gmin": gmin.reshape(H, W),
        "margin": margin.reshape(H, W),
        "best_class": best_class.reshape(H, W),
        "second_class": second_class.reshape(H, W),
        "lib_row_counts": lib_row_counts,
    }
    if save_topk and K_top > 0:
        extras["topk_cids"] = topk_cids_flat.reshape(H, W, K_top)
        extras["topk_vals"] = topk_vals_flat.reshape(H, W, K_top)  # 원시 스케일
    if save_best_proto:
        extras["best_proto_idx_top1"] = best_proto_idx_flat.reshape(H, W)
        if best_proto_id_flat is not None:
            extras["best_proto_id_top1"] = best_proto_id_flat.reshape(H, W)
    return classmap, extras