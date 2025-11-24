# dataio/sidecar.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import json, os

SCHEMA_VERSION = 1

def info_path(data_path: Path) -> Path:
    # sample.mat → sample.mat.info / sample.hdr → sample.hdr.info
    return data_path.with_suffix(data_path.suffix + ".info")

def read_info(data_path: Path) -> Optional[Dict[str, Any]]:
    p = info_path(data_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def write_info(data_path: Path, payload: Dict[str, Any]) -> None:
    p = info_path(data_path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    p.parent.mkdir(parents=True, exist_ok=True)
    # 스키마 버전 부여
    if "schema_version" not in payload:
        payload = { "schema_version": SCHEMA_VERSION, **payload }
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
