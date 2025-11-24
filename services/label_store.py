# services/label_store.py
from __future__ import annotations

"""
LabelStore: 라벨(행 기반) 저장/조회/집계 서비스

- 기본 저장: Parquet (pyarrow/fastparquet 엔진 필요)
- 폴백 저장: CSV (엔진 미설치 시 자동 폴백)
- 스펙트럼은 bytes(BLOB 유사)로 저장하며, 로드시 np.frombuffer로 복원

Schema (열):
    row_id        : str (UUID 등 고유값)
    label_code    : str (조회 표준키; 고정식 생성 권장)
    mtrl_cd       : int (클래스 ID)
    image_cd      : int (원본 이미지 코드)
    y, x          : int (픽셀 좌표)
    patch_size    : int | None
    bbox_x,y,w,h  : int | None (패치 박스)
    has_raw       : bool
    has_cr        : bool
    rfl_raw       : bytes | None   (float32 1D)
    rfl_cr        : bytes | None   (float32 1D)
    wl_ref        : str | None
    meta_json     : str(JSON) | None
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Iterable, Union

import numpy as np
import pandas as pd


# -------- 내부 유틸 --------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _resample_1d(arr: np.ndarray, dst_c: int) -> np.ndarray:
    """밴드 수가 다를 때 균등축 선형보간으로 (C_old,) -> (dst_c,)."""
    if arr.size == dst_c:
        return arr.astype(np.float32, copy=False)
    x_old = np.linspace(0.0, 1.0, num=arr.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=dst_c, dtype=np.float32)
    return np.interp(x_new, x_old, arr.astype(np.float32)).astype(np.float32)


def _to_bytes_1d(a: Optional[np.ndarray]) -> Optional[bytes]:
    if a is None:
        return None
    arr = np.asarray(a, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32).tobytes()


def _from_bytes_1d(b: Optional[bytes]) -> Optional[np.ndarray]:
    if b is None or (isinstance(b, float) and np.isnan(b)):  # CSV NaN 방어
        return None
    try:
        return np.frombuffer(b, dtype=np.float32)
    except Exception:
        # CSV 백엔드에서 base64/hex 등으로 저장했다면 여기서 파싱하도록 확장
        return None


class LabelStore:
    """
    라벨(행 저장) 파일을 관리하고, (cid -> (N,C)) 집계 뷰를 생성하는 서비스.
    기본은 Parquet를 사용하되, 엔진이 없으면 CSV로 폴백한다.
    """

    def __init__(self, root_dir: str, filename: str = "labels.parquet"):
        self.root = os.path.abspath(root_dir)
        _ensure_dir(self.root)

        # 백엔드 자동 결정
        self._backend = "parquet"
        self._parquet_path = os.path.join(self.root, filename)
        self._csv_path = os.path.join(self.root, "labels.csv")

        if not self._parquet_supported():
            self._backend = "csv"

        self._ensure_schema()

    # ---------------- 백엔드/스키마 ----------------
    def _parquet_supported(self) -> bool:
        try:
            # pyarrow or fastparquet 둘 중 하나만 있어도 OK
            # 실제 쓰기 시점에 엔진을 자동 선택하도록 맡김
            pd.DataFrame({"a": [1]}).to_parquet(os.path.join(self.root, "__probe__.parquet"))
            os.remove(os.path.join(self.root, "__probe__.parquet"))
            return True
        except Exception:
            return False

    @property
    def path(self) -> str:
        return self._parquet_path if self._backend == "parquet" else self._csv_path

    def _empty_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "row_id", "label_code",
                "mtrl_cd", "image_cd", "y", "x",
                "patch_size", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                "has_raw", "has_cr", "rfl_raw", "rfl_cr",
                "wl_ref", "meta_json",
            ]
        )

    def _ensure_schema(self) -> None:
        if os.path.isfile(self.path):
            return
        df = self._empty_df()
        self._write_df(df)

    def _read_df(self) -> pd.DataFrame:
        if not os.path.isfile(self.path):
            return self._empty_df()
        if self._backend == "parquet":
            return pd.read_parquet(self.path)
        # CSV 백엔드
        df = pd.read_csv(self.path)
        # bytes 칼럼이 문자열로 저장되어 있을 수 있음 → 여기선 그대로 두고,
        # _from_bytes_1d에서 NaN/str은 None 처리.
        return df

    def _write_df(self, df: pd.DataFrame) -> None:
        if self._backend == "parquet":
            # 인덱스 불필요
            df.to_parquet(self.path, index=False)
        else:
            df.to_csv(self.path, index=False)

    # ---------------- 공개 API ----------------
    def append_rows(self, rows: List["LabelRow"] | List[Dict]) -> None:
        """
        행 단위 라벨을 append.
        - label_code 중복이 있으면 기존 행을 제거 후 새 행으로 대체(덮어쓰기 정책).
        - rfl_raw / rfl_cr 은 np.ndarray(1D) → bytes로 직렬화하여 저장.
        """
        if not rows:
            return

        # 입력 정규화
        recs: List[Dict] = []
        for r in rows:
            # 객체/딕셔너리 모두 허용
            if hasattr(r, "__dict__"):
                obj = r
                row_id = getattr(obj, "row_id", None)
                label_code = getattr(obj, "label_code", None)
                mtrl_cd = getattr(obj, "mtrl_cd", None)
                image_cd = getattr(obj, "image_cd", None)
                y = getattr(obj, "y", None)
                x = getattr(obj, "x", None)
                patch_size = getattr(obj, "patch_size", None)
                patch_bbox = getattr(obj, "patch_bbox", None)
                rfl_raw = getattr(obj, "rfl_raw", None)
                rfl_cr = getattr(obj, "rfl_cr", None)
                wl_ref = getattr(obj, "wl_ref", None)
                meta = getattr(obj, "meta", None)
            else:
                row_id = r.get("row_id")
                label_code = r.get("label_code")
                mtrl_cd = r.get("mtrl_cd")
                image_cd = r.get("image_cd")
                y = r.get("y"); x = r.get("x")
                patch_size = r.get("patch_size")
                patch_bbox = r.get("patch_bbox")
                rfl_raw = r.get("rfl_raw")
                rfl_cr = r.get("rfl_cr")
                wl_ref = r.get("wl_ref")
                meta = r.get("meta")

            # bbox tuple → 4열로 분해
            bx = by = bw = bh = None
            if patch_bbox and isinstance(patch_bbox, (tuple, list)) and len(patch_bbox) == 4:
                bx, by, bw, bh = patch_bbox

            recs.append({
                "row_id":    row_id or "",
                "label_code": label_code or "",
                "mtrl_cd":   int(mtrl_cd) if mtrl_cd is not None else None,
                "image_cd":  int(image_cd) if image_cd is not None else None,
                "y":         int(y) if y is not None else None,
                "x":         int(x) if x is not None else None,
                "patch_size": int(patch_size) if patch_size is not None else None,
                "bbox_x":    int(bx) if bx is not None else None,
                "bbox_y":    int(by) if by is not None else None,
                "bbox_w":    int(bw) if bw is not None else None,
                "bbox_h":    int(bh) if bh is not None else None,
                "has_raw":   bool(rfl_raw is not None),
                "has_cr":    bool(rfl_cr  is not None),
                "rfl_raw":   _to_bytes_1d(rfl_raw),
                "rfl_cr":    _to_bytes_1d(rfl_cr),
                "wl_ref":    wl_ref if wl_ref is not None else None,
                "meta_json": json.dumps(meta or {}, ensure_ascii=False),
            })

        df_cur = self._read_df()
        df_new = pd.DataFrame.from_records(recs)

        # label_code 기준 덮어쓰기 정책
        if "label_code" in df_cur.columns and "label_code" in df_new.columns:
            to_drop = set(df_new["label_code"].astype(str))
            if len(to_drop) > 0:
                df_cur = df_cur[~df_cur["label_code"].astype(str).isin(to_drop)]

        df_out = pd.concat([df_cur, df_new], ignore_index=True)
        self._write_df(df_out)

    def load_rows(self,
                  image_cd: Optional[int] = None,
                  mtrl_cd: Optional[int] = None) -> pd.DataFrame:
        """
        행 전체(또는 필터) DataFrame 반환.
        """
        df = self._read_df()
        if image_cd is not None:
            df = df[df["image_cd"] == int(image_cd)]
        if mtrl_cd is not None:
            df = df[df["mtrl_cd"] == int(mtrl_cd)]
        return df.reset_index(drop=True)

    def get_row_by_code(self, label_code: str) -> Optional[pd.Series]:
        """
        label_code로 단일 행 반환.
        """
        df = self._read_df()
        hit = df[df["label_code"] == str(label_code)]
        if len(hit) == 0:
            return None
        return hit.iloc[0]

    def get_spectrum_by_code(self, label_code: str, use: str = "raw") -> Optional[np.ndarray]:
        """
        label_code → 스펙트럼(1D float32) 반환.
        use: 'raw' | 'cr'
        """
        row = self.get_row_by_code(label_code)
        if row is None:
            return None
        col = "rfl_raw" if use == "raw" else "rfl_cr"
        return _from_bytes_1d(row[col])

    def get_coords_by_code(self, label_code: str) -> Optional[Tuple[int, int, int]]:
        """
        label_code → (image_cd, y, x)
        """
        row = self.get_row_by_code(label_code)
        if row is None:
            return None
        return int(row["image_cd"]), int(row["y"]), int(row["x"])

    def aggregate_matrix(self, expect_c: int, use: str = "raw") -> Dict[int, np.ndarray]:
        """
        행 저장 DB를 클래스별 (N,C) float32 매트릭스로 집계.
        use: 'raw' | 'cr'
        """
        col = "rfl_raw" if use == "raw" else "rfl_cr"
        df = self._read_df()
        if df.empty:
            return {}

        out: Dict[int, List[np.ndarray]] = {}
        for idx, row in df.iterrows():
            b = row[col]
            arr = _from_bytes_1d(b)
            if arr is None:
                continue
            if arr.size != expect_c:
                arr = _resample_1d(arr, expect_c)
            cid = row.get("mtrl_cd")
            if pd.isna(cid):
                continue
            cid = int(cid)
            out.setdefault(cid, []).append(arr.astype(np.float32, copy=False))

        return {cid: np.vstack(v).astype(np.float32) for cid, v in out.items() if v}

    # ---------------- 유지보수 유틸 ----------------
    def delete_by_codes(self, label_codes: Iterable[str]) -> int:
        """
        label_code 목록으로 행 삭제. 삭제된 건수 반환.
        """
        codes = set(map(str, label_codes or []))
        if not codes:
            return 0
        df = self._read_df()
        before = len(df)
        df = df[~df["label_code"].astype(str).isin(codes)]
        self._write_df(df)
        return before - len(df)

    def upsert_row(self, record: Dict) -> None:
        """
        단일 행 upsert (label_code 기준 덮어쓰기).
        """
        self.append_rows([record])

    # CSV 백엔드에서 bytes를 안전하게 다루고 싶다면,
    # 아래 두 메서드를 base64 직렬화/역직렬화로 확장할 수 있다.
    # 현재 구현은 Parquet 권장을 전제로 bytes 직렬화 저장을 유지한다.
