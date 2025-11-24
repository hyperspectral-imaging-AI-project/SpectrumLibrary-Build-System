# services/spec_library.py
from typing import List, Dict, Tuple, Union
import numpy as np
from itertools import chain

def build_library_from_spec_libs(
    spec_lib_dicts: List[Union[dict, list, tuple, np.ndarray]],
    expect_c: int
) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
    items_blocks = []
    for pack in (spec_lib_dicts or []):
        # 1) pack → items 추출
        if isinstance(pack, dict):
            items = pack.get("spec_lib", None)
        elif isinstance(pack, (list, tuple, np.ndarray)):
            items = pack
        else:
            continue

        # 2) ndarray → python list 로 정규화 (object 배열 대비)
        if isinstance(items, np.ndarray):
            items = items.tolist()

        # 3) 리스트/튜플만 허용 + 길이 체크 (진릿값 비교 금지)
        if items is None:
            continue
        if not isinstance(items, (list, tuple)):
            continue
        if len(items) == 0:
            continue

        items_blocks.append(items)

    # 모두 평탄화
    items = list(chain.from_iterable(items_blocks))  # [{mtrl_cd, ref}, ...]

    # 키/레퍼런스 추출
    keys = [it.get("mtrl_cd") for it in items if isinstance(it, dict)]
    refs = [it.get("ref")     for it in items if isinstance(it, dict)]

    # 문자열 라벨 → id 부여
    only_strs = [k for k in keys if isinstance(k, str)]
    uniq_strs = list(dict.fromkeys(only_strs).keys())
    name_to_id: Dict[str, int] = {name: i + 1 for i, name in enumerate(uniq_strs)}

    def to_cid(k) -> int:
        if isinstance(k, str):
            if k not in name_to_id:
                name_to_id[k] = len(name_to_id) + 1
            return name_to_id[k]
        try:
            return int(k)
        except Exception:
            return -1  # 스킵 대상

    cids = [to_cid(k) for k in keys]

    def rows_from_ref(r):
        if r is None:
            return []
        a = np.asarray(r, dtype=float)
        if a.size == 0 or a.ndim == 0:
            return []
        if a.ndim == 1:
            return [a] if a.size == expect_c else []
        if a.ndim == 2:
            if a.shape[1] == expect_c:
                return [a[i, :] for i in range(a.shape[0])]
            if a.shape[0] == expect_c:
                return [a[:, i] for i in range(a.shape[1])]
            return []
        return []

    rows_by_cid: Dict[int, list] = {}
    for cid, ref in zip(cids, refs):
        if cid is None or cid < 0:
            continue
        rows = rows_from_ref(ref)
        if len(rows) == 0:
            continue
        rows_by_cid.setdefault(int(cid), []).extend(rows)

    lib: Dict[int, np.ndarray] = {}
    for cid, vecs in rows_by_cid.items():
        if len(vecs) == 0:
            continue
        arr = np.vstack(vecs).astype(np.float32, copy=False)
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = 0.0
        lib[cid] = arr

    id_to_name: Dict[int, str] = {cid: str(cid) for cid in lib.keys()}
    for name, cid in name_to_id.items():
        if cid in lib:
            id_to_name[cid] = name

    return lib, id_to_name

# ----- format_normalizer.py (모듈로 빼도 좋고, MainWindow 안에 넣어도 됨) -----
import numpy as np
from typing import Any, Dict

class FormatError(ValueError): ...
def _to_2d_f32(a: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float32)
    if arr.ndim == 1: arr = arr[None, :]
    if arr.ndim != 2 or arr.size == 0:
        raise FormatError(f"[{name}] 2D float32 필요, got shape={arr.shape}")
    return arr

def _interp_to_c(mat: np.ndarray, C: int) -> np.ndarray:
    if mat.shape[1] == C:
        return mat.astype(np.float32, copy=False)
    x_old = np.linspace(0,1,mat.shape[1], dtype=np.float32)
    x_new = np.linspace(0,1,C,           dtype=np.float32)
    out = np.empty((mat.shape[0], C), np.float32)
    for i in range(mat.shape[0]):
        out[i] = np.interp(x_new, x_old, mat[i])
    return out

def detect_shape_info(cube) -> int:
    a = np.asarray(cube)
    if a.ndim != 3: raise FormatError(f"[detect] cube (H,W,C) 필요, nd={a.ndim}")
    return int(a.shape[2])

def norm_any_to_dict(
    x: Any, *,  # 저장/메모리 원본
    expect_c: int,            # 목표 C
    cube=None,                # (H,W,C). rows에서 좌표 추출 시 필요
    prefer_key: str = "rfl",  # 라벨은 'rfl', 스펙 라이브러리는 'ref' 추천
    src_name: str = "unknown" # 로깅 구분용
) -> Dict[int, np.ndarray]:
    """
    지원 입력:
      1) dict[int-> (N,C)] 또는 dict[int-> dict{prefer_key|ref|rfl: (N,C)}]
      2) np.ndarray(dtype=object) 0-D / (1,) / (N,) : 내부에 dict 들
      3) list of dict rows: [{"mtrl_cd"/"cid", "rfl" or ("img_x","img_y")}...]
      4) list of (cid, vec) 또는 (cid, (N,C))
    반환: dict[int -> (N,C) float32]
    """
    # 0) cube 사전검증: 좌표 케이스에서만 필요
    H=W=None
    if cube is not None:
        a = np.asarray(cube)
        if a.ndim != 3: raise FormatError(f"[{src_name}] cube (H,W,C) 필요")
        H, W = int(a.shape[0]), int(a.shape[1])

    def _from_dict(d: dict) -> Dict[int, np.ndarray]:
        out = {}
        for k, v in d.items():
            # cid
            try: cid = int(k)
            except Exception:
                # 값이 dict인 경우 안에서 cid를 구할 수 있으면 시도
                if isinstance(v, dict):
                    cid2 = v.get("mtrl_cd") or v.get("cid")
                    try: cid = int(cid2)
                    except Exception: 
                        continue
                else:
                    continue
            # value
            val = v
            if isinstance(v, dict):
                val = (v.get(prefer_key) or v.get("ref") or v.get("rfl"))
                if val is None: 
                    # 좌표만 저장된 경우는 여기서 건너뛰고 rows 경로에서 처리
                    continue
            m = _to_2d_f32(val, name=f"{src_name}[{cid}]")
            m = _interp_to_c(m, expect_c)
            out.setdefault(cid, []).append(m)
        return {cid: np.vstack(v).astype(np.float32) for cid, v in out.items()}

    # 1) dict 직접
    if isinstance(x, dict):
        return _from_dict(x)

    # ... norm_any_to_dict 내부 ...
    def _merge_dicts_vstack(a: dict, b: dict) -> dict:
        if not a: return dict(b)
        if not b: return dict(a)
        out = dict(a)
        for k, v in b.items():
            if k in out:
                A = np.asarray(out[k], dtype=np.float32)
                B = np.asarray(v,      dtype=np.float32)
                if A.ndim == 1: A = A[None, :]
                if B.ndim == 1: B = B[None, :]
                if A.shape[1] != B.shape[1]:
                    # 이 단계까지 오면 둘 다 expect_c로 보정된 상태여야 함
                    raise FormatError(f"[{src_name}] 병합 채널 불일치 cid={k}: {A.shape[1]} vs {B.shape[1]}")
                out[k] = np.vstack([A, B]).astype(np.float32)
            else:
                out[k] = v
        return out

    # 2) object ndarray
    if isinstance(x, np.ndarray) and x.dtype == object:
        # 0-D
        if x.ndim == 0:
            v = x.item()
            if isinstance(v, dict):
                return _from_dict(v)
            if isinstance(v, list):
                # 0-D 내부가 list면 list 분기로 재사용 (rows 또는 (cid, vec) 쌍)
                return norm_any_to_dict(
                    v, expect_c=expect_c, cube=cube, prefer_key=prefer_key,
                    src_name=f"{src_name}.obj0-list"
                )
            # 그 외는 포맷 오류
            raise FormatError(f"[{src_name}] 0-D object 내부가 dict/list 아님: {type(v)}")

        # (1,)
        if x.ndim == 1 and x.size == 1:
            v = x.ravel()[0]
            if isinstance(v, dict):
                return _from_dict(v)
            if isinstance(v, list):
                return norm_any_to_dict(
                    v, expect_c=expect_c, cube=cube, prefer_key=prefer_key,
                    src_name=f"{src_name}.obj1-list"
                )
            raise FormatError(f"[{src_name}] (1,) object 내부가 dict/list 아님: {type(v)}")

        # (N,) → 요소별 dict 또는 list까지 허용하여 병합
        if x.ndim == 1:
            out: dict[int, np.ndarray] = {}
            for i, v in enumerate(x.ravel()):
                if isinstance(v, dict):
                    part = _from_dict(v)
                elif isinstance(v, list):
                    part = norm_any_to_dict(
                        v, expect_c=expect_c, cube=cube, prefer_key=prefer_key,
                        src_name=f"{src_name}.objN-list[{i}]"
                    )
                else:
                    raise FormatError(f"[{src_name}] (N,) object[{i}] dict/list 아님: {type(v)}")
                out = _merge_dicts_vstack(out, part)
            return out

        raise FormatError(f"[{src_name}] object array 지원안함: ndim={x.ndim}")
    # 3) list of dict rows
    # ... norm_any_to_dict 내부, rows 처리 분기 (교체본)
    if isinstance(x, list) and x and all(isinstance(e, dict) for e in x):
        if cube is None:
            raise FormatError(f"[{src_name}] rows 포맷인데 cube가 필요합니다.")
        out = {}
        H, W, _ = np.asarray(cube).shape
        for idx, r in enumerate(x):
            # 1) cid
            cid = r.get("mtrl_cd", r.get("cid"))
            try:
                cid = int(cid)
            except Exception:
                raise FormatError(f"[{src_name}] rows[{idx}] cid 변환 실패")

            # 2) 좌표 우선 (있으면)
            spec = None
            if ("img_y" in r) and ("img_x" in r):
                y, x2 = int(r["img_y"]), int(r["img_x"])
                if not (0 <= y < H and 0 <= x2 < W):
                    raise FormatError(f"[{src_name}] rows[{idx}] 좌표 범위 오류")
                s = np.asarray(cube)[y, x2, :].astype(np.float32)
                spec = _interp_to_c(s[None, :], expect_c)

            # 3) 스펙트럼 키 탐색: prefer_key → ref → rfl → spec → spectrum
            if spec is None:
                # 불리언 평가 없이 키 탐색
                cand, used_key = _pick_first_present(
                    r, (prefer_key, "ref", "rfl", "spec", "spectrum")
                )
                if cand is None:
                    raise FormatError(
                        f"[{src_name}] rows[{idx}] 스펙트럼 키 없음 "
                        f"(keys={list(r.keys())}, prefer_key='{prefer_key}')"
                    )

                m = _to_2d_f32(cand, name=f"{src_name}.rows[{idx}].{used_key}")
                spec = _interp_to_c(m, expect_c)

            out.setdefault(cid, []).append(spec)

        return {cid: np.vstack(v).astype(np.float32) for cid, v in out.items()}


    # 4) list of (cid, vec/mat)
    if isinstance(x, list) and x and all(isinstance(e, (list, tuple)) and len(e)==2 for e in x):
        out = {}
        for idx, (cid, vec) in enumerate(x):
            try: cid = int(cid)
            except Exception:
                raise FormatError(f"[{src_name}] pair[{idx}] cid 변환 실패")
            m = _to_2d_f32(vec, name=f"{src_name}.pair[{idx}]")
            m = _interp_to_c(m, expect_c)
            out.setdefault(cid, []).append(m)
        return {cid: np.vstack(v).astype(np.float32) for cid, v in out.items()}

    # → 여기로 오면 ‘우리가 모르는’ 포맷
    raise FormatError(f"[{src_name}] 지원하지 않는 형식: {type(x)}")

def _pick_first_present(d: dict, keys: tuple[str, ...]):
    # 불리언 평가(or) 금지: 존재/None 여부만 본다
    for k in keys:
        if k in d:
            v = d[k]
            if v is not None:
                return v, k
    return None, None
