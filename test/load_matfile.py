from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def _clean_mat_dict(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("__")}


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", errors="replace")
        except Exception:
            return str(x)
    if isinstance(x, np.ndarray) and x.dtype.kind in ("U", "S") and x.size == 1:
        return str(x.item())
    if isinstance(x, np.ndarray) and x.size == 1:
        try:
            return str(x.item())
        except Exception:
            return str(x)
    return str(x)


def _as_int(x: Any, default: int = -1) -> int:
    try:
        if isinstance(x, np.ndarray) and x.size == 1:
            x = x.item()
        return int(x)
    except Exception:
        return default


def _decode_matlab_char_matrix(a: np.ndarray) -> list[str] | None:
    """
    scipy.io.loadmat에서 MATLAB char matrix가 numpy로 로드된 케이스를
    ["row1", "row2", ...] 형태로 복원.
    """
    if not isinstance(a, np.ndarray):
        return None
    if a.ndim != 2:
        return None
    if a.dtype.kind not in ("U", "S"):
        return None
    rows: list[str] = []
    for r in a:
        # r: 1D array of characters/bytes
        if a.dtype.kind == "S":
            try:
                s = b"".join([bytes([x]) if isinstance(x, (int, np.integer)) else x for x in r]).decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                s = "".join([str(x) for x in r])
        else:
            s = "".join([str(x) for x in r])
        s = s.rstrip()
        if s:
            rows.append(s)
    return rows


def _decode_cellstr(a: np.ndarray) -> list[str] | None:
    """
    MATLAB cellstr(문자열 cell 배열) → python list[str]
    """
    if not isinstance(a, np.ndarray):
        return None
    if a.dtype != object:
        return None
    out: list[str] = []
    for x in a.ravel().tolist():
        if isinstance(x, np.ndarray):
            # char matrix(1, N) 같은 형태일 수 있음
            cm = _decode_matlab_char_matrix(x)
            if cm:
                out.extend(cm)
                continue
            if x.size == 1:
                out.append(_as_str(x))
                continue
            out.append(_as_str(x))
        else:
            out.append(_as_str(x))
    return out


def _decode_class_map_value(x: Any) -> list[tuple[str, str]] | None:
    """
    class_map을 [(cid, name), ...] 형태로 정규화.
    지원:
    - JSON 문자열
    - MATLAB Nx2 cell(object ndarray)
    """
    if x is None:
        return None

    # 1) JSON string
    if isinstance(x, str):
        try:
            parsed = json.loads(x)
            if isinstance(parsed, list):
                out: list[tuple[str, str]] = []
                for item in parsed:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        out.append((str(item[0]), str(item[1])))
                return out
        except Exception:
            return None

    # 2) ndarray (cell)
    if isinstance(x, np.ndarray) and x.dtype == object:
        arr = x
        # Nx1 cell, each cell contains a tuple/1x2-like object
        if arr.ndim == 2 and arr.shape[1] == 1:
            out1: list[tuple[str, str]] = []
            for cell in arr[:, 0].tolist():
                if isinstance(cell, (list, tuple)) and len(cell) >= 2:
                    out1.append((_as_str(cell[0]), _as_str(cell[1])))
                elif isinstance(cell, np.ndarray) and cell.size >= 2:
                    flat = cell.ravel().tolist()
                    out1.append((_as_str(flat[0]), _as_str(flat[1])))
                else:
                    out1.append((_as_str(cell), ""))
            return out1
        # Nx2 or (N,2,...) 형태를 최대한 (N,2)로 해석
        if arr.ndim >= 2 and arr.shape[-1] == 2:
            arr2 = arr.reshape(-1, 2)
            out2: list[tuple[str, str]] = []
            for cid_v, name_v in arr2.tolist():
                out2.append((_as_str(cid_v), _as_str(name_v)))
            return out2

        # fallback: list of rows
        out3: list[tuple[str, str]] = []
        for r in arr.tolist():
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                out3.append((_as_str(r[0]), _as_str(r[1])))
        return out3 or None

    return None


def _print_summary(mat_path: str) -> None:
    p = Path(mat_path)
    m = loadmat(str(p))
    d = _clean_mat_dict(m)

    print("[file]", str(p))
    print("[keys]", sorted(d.keys()))

    # our saved schema prefers these keys
    kind = _as_str(d.get("kind"))
    image_code = _as_str(d.get("image_code"))
    height = _as_int(d.get("height"))
    width = _as_int(d.get("width"))
    channels = _as_int(d.get("channels"))

    print("[meta] kind =", kind)
    print("[meta] image_code =", image_code)
    print("[meta] height/width/channels =", height, width, channels)

    arr = d.get("data")
    if isinstance(arr, np.ndarray):
        print("[data] dtype/shape =", arr.dtype, arr.shape)
        # show a tiny sample (avoid huge print)
        flat = arr.ravel()
        print("[data] head =", flat[:10])
        if np.issubdtype(arr.dtype, np.integer):
            u = np.unique(arr)
            print("[data] unique(head) =", u[:20], "(n=", u.size, ")")
    else:
        print("[data] missing or not ndarray:", type(arr))

    # explicit class fields (preferred)
    class_ids = d.get("class_ids")
    class_names = d.get("class_names")
    if isinstance(class_ids, np.ndarray) and isinstance(class_names, np.ndarray):
        ids = [int(x) for x in np.asarray(class_ids).ravel().tolist()]
        names: list[str] = []
        # 1) cellstr 형태
        cell = _decode_cellstr(class_names)
        if cell:
            names = cell
        else:
            # 2) char matrix 형태
            cm = _decode_matlab_char_matrix(class_names)
            if cm:
                names = cm
            else:
                names = [str(x) for x in np.asarray(class_names).ravel().tolist()]
        print("[classes] class_ids =", ids[:30])
        print("[classes] class_names =", names[:30])
        print("[classes] pairs(head) =", list(zip(ids, names))[:30])
    else:
        print("[classes] class_ids/class_names not found")

    # class_map: [('8','name'), ...] 형태 (npz는 json str, mat은 Nx2 cell 가능)
    if "class_map" in d:
        cm_raw = d.get("class_map")
        
        print("class_map", cm_raw)
        # decoded = _decode_class_map_value(cm_raw)
        # if decoded:
        #     print("[class_map] head =", decoded[:30])
        # else:
        #     print("[class_map] present but could not decode. type =", type(cm_raw))

    # fallback: id_to_name_json
    if "id_to_name_json" in d:
        try:
            s = _as_str(d["id_to_name_json"])
            parsed = json.loads(s) if s else None
            if isinstance(parsed, dict):
                # show first few
                items = list(parsed.items())[:30]
                print("[id_to_name_json] keys(head) =", [k for k, _ in items])
                print("[id_to_name_json] items(head) =", items)
            else:
                print("[id_to_name_json] not a dict:", type(parsed))
        except Exception as e:
            print("[id_to_name_json] parse failed:", e)


if __name__ == "__main__":
    # 사용법:
    #   python test/load_matfile.py "C:\\path\\to\\your.mat"
    import sys

    if len(sys.argv) >= 2:
        mat_path = sys.argv[1]
    else:
        # 기본값: 여기 경로를 원하는 파일로 바꿔서 사용하세요.
        mat_path = r"E:\3dlabs_data\sample\classification.mat"

    _print_summary(mat_path)
