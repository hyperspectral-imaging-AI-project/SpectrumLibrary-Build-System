# models/labels.py
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import uuid, numpy as np

@dataclass
class LabelRow:
    row_id: str
    label_code: str                # ★ 추가: 조회/그래프 표준 키
    mtrl_cd: int
    image_cd: int
    y: int
    x: int
    patch_size: Optional[int] = None
    patch_bbox: Optional[Tuple[int,int,int,int]] = None
    rfl_raw: Optional[np.ndarray] = None
    rfl_cr:  Optional[np.ndarray] = None
    wl_ref:  Optional[str] = "image"
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex
