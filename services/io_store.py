# services/io_store.py
from __future__ import annotations
import os
import numpy as np
from typing import Optional, Tuple

class IOStore:
    """파일 포맷(.npz/.npy) 및 image_code 검증/복원 담당."""
    def __init__(self, get_current_image_code: callable, get_current_shape: callable):
        """
        get_current_image_code(): () -> Optional[str]
        get_current_shape(): () -> Tuple[int,int]  # (H,W)
        """
        self._get_code = get_current_image_code
        self._get_shape = get_current_shape

    def save_npz_with_meta(self, path: str, *, data: np.ndarray, kind: str) -> None:
        code = self._get_code()
        if not code:
            raise ValueError("현재 image_code가 없습니다.")
        h = int(data.shape[0]) if data.ndim >= 2 else None
        w = int(data.shape[1]) if data.ndim >= 2 else None
        c = int(data.shape[2]) if data.ndim == 3 else None
        np.savez(path, data=data, image_code=code, kind=kind, height=h, width=w, channels=c)

    def load_npz_with_check(self, path: str) -> tuple[str, np.ndarray]:
        z = np.load(path, allow_pickle=False)
        if "data" not in z or "image_code" not in z or "kind" not in z:
            raise ValueError("필수 키(data,image_code,kind) 누락")
        arr  = np.asarray(z["data"])
        code = str(z["image_code"])
        kind = str(z["kind"])
        cur  = self._get_code()
        if not cur:
            raise ValueError("현재 image_code가 없습니다.")
        if code != cur:
            raise ValueError(f"image_code 불일치: 파일={code}, 현재={cur}")
        # 1D → 2D 복원
        H, W = self._get_shape()
        if arr.ndim == 1 and arr.size == H*W:
            arr = arr.reshape(H, W)
        return kind, arr
