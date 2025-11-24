# services/resampling_cache.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np


__all__ = [
    "make_cache_key",
    "try_load_cache",
    "save_cache",
    "resample_cache_paths",
    "CACHE_INFO_SUFFIX",
    "CACHE_NPZ_SUFFIX",
    "load_classes_from_info",
    "update_class_in_info",
]

# 캐시 파일 접미사 (원천파일 옆에 생성)
CACHE_INFO_SUFFIX = ".resample.info"   # JSON 메타
CACHE_NPZ_SUFFIX  = ".resample.npz"    # 배열/리스트 묶음

# ------------------------------------------------------------
# 내부 유틸
# ------------------------------------------------------------
def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _hash_array(a: Optional[np.ndarray]) -> str:
    """가변 길이 배열을 해시(내용 기반). None/빈배열은 빈 문자열."""
    if a is None:
        return ""
    a = np.asarray(a)
    if a.size == 0:
        return ""
    # dtype/shape까지 반영
    blob = b"|".join(
        [
            str(a.dtype).encode("utf-8"),
            str(a.shape).encode("utf-8"),
            a.tobytes(),
        ]
    )
    return _sha256_bytes(blob)


def _norm_path(p: Optional[str | os.PathLike]) -> Optional[str]:
    if not p:
        return None
    try:
        return os.path.abspath(os.fspath(p))
    except Exception:
        return None


def _to_native(obj: Any) -> Any:
    """np.load 로 복원한 object ndarray를 Python 기본 객체로 변환."""
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            if obj.ndim == 0:
                return _to_native(obj.item())
            return [_to_native(x) for x in obj.tolist()]
        return obj
    return obj


# ------------------------------------------------------------
# 공개 API
# ------------------------------------------------------------
def resample_cache_paths(primary_path: str) -> Tuple[str, str]:
    """
    원천 파일(primary_path) 옆에 두 개의 캐시 파일 경로를 만든다.
      - <stem><suffix>.resample.info
      - <stem><suffix>.resample.npz
    """
    p = Path(primary_path)
    info = str(p.with_suffix(p.suffix + CACHE_INFO_SUFFIX))
    npz  = str(p.with_suffix(p.suffix + CACHE_NPZ_SUFFIX))
    return info, npz


def make_cache_key(
    cfg: Dict[str, Any],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    캐시 유효성 판단을 위한 키 생성.
    기본 구성:
      - camera_code
      - wavelength 해시
      - fwhm 해시
      - nbands
    필요 시 extra로 버전/옵션 등을 추가하세요.
    """
    # wavelength / fwhm 은 list/ndarray 등 다양한 형식일 수 있음
    wv = np.asarray(cfg.get("wavelength", []), dtype=np.float64)
    fw = (
        np.asarray(cfg.get("fwhm", []), dtype=np.float64)
        if cfg.get("fwhm") is not None
        else np.array([], dtype=np.float64)
    )

    key = {
        "camera_code": str(cfg.get("camera_code", "")),
        "wv_sha": _hash_array(wv),
        "fwhm_sha": _hash_array(fw),
        "nbands": int(wv.size),
    }
    if extra:
        # 단순한 키/값만(직렬화 가능한 값) 병합
        key.update(
            {
                str(k): (
                    v if isinstance(v, (str, int, float, bool)) else str(v)
                )
                for k, v in extra.items()
            }
        )
    return key


def try_load_cache(
    primary_path: Optional[str],
    cfg: Dict[str, Any],
) -> Optional[Tuple[Any, Any, Any, Any, Any]]:
    """
    캐시 파일이 있고, make_cache_key(cfg)가 일치하면
    (splib_raw, splib_cr, label_raw, label_cr, label_coords) 튜플을 반환.
    없거나 키가 다르면 None.
    """
    primary = _norm_path(primary_path)
    if not primary:
        return None

    info_path, npz_path = resample_cache_paths(primary)
    if not (os.path.isfile(info_path) and os.path.isfile(npz_path)):
        return None

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        want = make_cache_key(cfg)
        have = meta.get("cache_key", {})

        # 필수 필드 비교(부족하면 None 처리)
        if not (
            str(have.get("camera_code", "")) == want["camera_code"]
            and str(have.get("wv_sha", "")) == want["wv_sha"]
            and str(have.get("fwhm_sha", "")) == want["fwhm_sha"]
            and int(meta.get("nbands", -1)) == want["nbands"]
        ):
            return None

        z = np.load(npz_path, allow_pickle=True)
        # 저장 시 dtype=object로 저장했을 수 있음. 그대로 반환(호출측에서 형 맞춰 사용)
        splib_raw = _to_native(z["splib_raw"])
        splib_cr  = _to_native(z["splib_cr"])
        label_raw = _to_native(z["label_raw"])
        label_cr  = _to_native(z["label_cr"])
        label_coords = _to_native(z["label_coords"]) if "label_coords" in z else {}

        return (splib_raw, splib_cr, label_raw, label_cr, label_coords)
    except Exception:
        # 캐시가 손상됐을 수 있으므로 조용히 실패 처리
        logging.exception("[ResampleCache] load failed — ignore and recompute")
        return None


def _pack_obj(obj):
    arr = np.empty((), dtype=object)  # shape=()
    arr[()] = obj                     # dict를 통째로 저장
    return arr


def save_cache(
    primary_path: Optional[str],
    cfg: Dict[str, Any],
    splib_raw: Any,
    splib_cr: Any,
    label_raw: Any,
    label_cr: Any,
    label_coords: Optional[Dict[int, Any]] = None,
    *,
    meta_extra: Optional[Dict[str, Any]] = None,
    classes_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
    write_info: bool = True,
) -> None:
    """
    Args:
        classes_metadata: {cid: {"cid": int, "mtrl_nm": str, "desc": str}}
            None이면 자동 수집하지 않음. main_window.py에서 수집하여 전달 권장.
    """
    primary = _norm_path(primary_path)
    if not primary:
        return

    info_path, npz_path = resample_cache_paths(primary)
    try:
        # 저장 전 상태 로그
        label_raw_count = len(label_raw) if isinstance(label_raw, dict) else 0
        label_cr_count = len(label_cr) if isinstance(label_cr, dict) else 0
        logging.info("[ResampleCache] 저장 시도: npz_path=%s, label_raw=%d classes, label_cr=%d classes", 
                    npz_path, label_raw_count, label_cr_count)
        
        # ★ dict를 0-D object로 저장
        np.savez(
            npz_path,
            splib_raw=_pack_obj(splib_raw),
            splib_cr=_pack_obj(splib_cr),
            label_raw=_pack_obj(label_raw),
            label_cr=_pack_obj(label_cr),
            label_coords=_pack_obj(label_coords or {}),
        )
        
        # 저장 후 검증
        if os.path.isfile(npz_path):
            file_size = os.path.getsize(npz_path)
            logging.info("[ResampleCache] .npz 파일 저장 완료: %s (크기: %d bytes)", npz_path, file_size)
        else:
            logging.error("[ResampleCache] .npz 파일 저장 실패: 파일이 생성되지 않았습니다. %s", npz_path)

        if not write_info:
            return

        # 기존 .info 파일 읽기 (classes 정보 보존을 위해)
        existing_meta = {}
        if os.path.isfile(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    existing_meta = json.load(f)
            except Exception:
                pass  # 읽기 실패 시 빈 딕셔너리 사용
        
        # 새로운 meta 생성 (기존 정보 보존)
        meta = {
            "version": 1,
            "primary_path": primary,
            "nbands": int(np.asarray(cfg.get("wavelength", [])).size),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cache_key": make_cache_key(cfg),
        }
        
        # ★ 기존 classes 정보를 항상 먼저 보존 (classes_metadata와 관계없이)
        if "classes" in existing_meta:
            meta["classes"] = dict(existing_meta["classes"])  # 복사본 생성
            logging.info("[ResampleCache] 기존 classes 정보 보존: %d개 클래스", len(meta["classes"]))
        else:
            meta["classes"] = {}
        
        # 클래스 메타데이터 추가/업데이트 (문자열 키로 변환하여 JSON 호환)
        if classes_metadata:
            # classes_metadata로 업데이트 (기존 항목은 유지, 새로운 항목은 추가)
            for cid, info in classes_metadata.items():
                meta["classes"][str(cid)] = info
            logging.info("[ResampleCache] classes_metadata로 업데이트: %d개 클래스 추가/업데이트", len(classes_metadata))
        
        # 기존 extra 정보 보존
        if "extra" in existing_meta and not meta_extra:
            meta["extra"] = existing_meta["extra"]
        
        if meta_extra:
            meta["extra"] = meta_extra

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        classes_count = len(meta.get("classes", {}))
        logging.info("[ResampleCache] .info 파일 저장 완료: %s (classes: %d개)", info_path, classes_count)
    except Exception as e:
        logging.exception("[ResampleCache] save failed: %s", e)
        raise  # 예외를 다시 발생시켜 호출자에게 알림


# --- NEW: 사용자 클래스 스펙 누적 저장 -------------------------------
def append_classes(
    primary_path: Optional[str],
    cfg: Dict[str, Any],
    class_raw: Optional[Dict[int, np.ndarray]] = None,
    class_cr: Optional[Dict[int, np.ndarray]] = None,
) -> bool:
    primary = _norm_path(primary_path)
    if not primary:
        return False

    info_path, npz_path = resample_cache_paths(primary)
    nbands = int(np.asarray(cfg.get("wavelength", [])).size)

    def _to2d(a):
        if a is None:
            return None
        arr = np.asarray(a, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr if arr.ndim == 2 else None

    def _resample2d(mat: np.ndarray, dst_c: int) -> np.ndarray:
        if mat.shape[1] == dst_c:
            return mat.astype(np.float32, copy=False)
        x_old = np.linspace(0.0, 1.0, num=mat.shape[1], dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=dst_c,        dtype=np.float32)
        out = np.empty((mat.shape[0], dst_c), dtype=np.float32)
        for i in range(mat.shape[0]):
            out[i] = np.interp(x_new, x_old, mat[i])
        return out

    def _merge(dst: dict, add: Optional[dict]):
        if not isinstance(add, dict):
            return
        for k, v in add.items():
            try:
                cid = int(k)
            except Exception:
                continue
            m = _to2d(v)
            if m is None or m.size == 0:
                continue
            if nbands > 0 and m.shape[1] != nbands:
                m = _resample2d(m, nbands)

            prev = dst.get(cid)
            if prev is None:
                dst[cid] = m.astype(np.float32, copy=False)
            else:
                if nbands > 0 and prev.shape[1] != nbands:
                    prev = _resample2d(prev, nbands)
                dst[cid] = np.vstack([prev, m]).astype(np.float32)

    # 기존 label_raw / label_cr 읽기
    old_raw, old_cr = {}, {}
    if os.path.isfile(npz_path):
        z = np.load(npz_path, allow_pickle=True)
        old_raw = _to_native(z["label_raw"]) if "label_raw" in z else {}
        old_cr  = _to_native(z["label_cr"])  if "label_cr"  in z else {}
        if not isinstance(old_raw, dict):
            old_raw = {}
        if not isinstance(old_cr, dict):
            old_cr = {}

    new_raw = dict(old_raw)
    new_cr  = dict(old_cr)
    _merge(new_raw, class_raw)
    _merge(new_cr,  class_cr)

    # 기존 npz의 다른 키(splib_* 등)는 그대로 유지
    payload: Dict[str, Any] = {}
    if os.path.isfile(npz_path):
        z = np.load(npz_path, allow_pickle=True)
        for key in z.files:
            if key not in ("label_raw", "label_cr"):
                payload[key] = z[key]

    payload["label_raw"] = np.asarray(new_raw, dtype=object)
    payload["label_cr"]  = np.asarray(new_cr,  dtype=object)
    np.savez(npz_path, **payload)

    # 메타 정보 갱신
    meta = {
        "version": 1,
        "primary_path": primary,
        "nbands": nbands,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cache_key": make_cache_key(cfg),
    }
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            old_meta = json.load(f)
        old_meta.update(meta)
        meta = old_meta
    except Exception:
        pass

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info(
        "[append_classes] wrote: %s (label_raw=%d classes, label_cr=%d classes)",
        os.path.basename(npz_path),
        len(new_raw),
        len(new_cr),
    )
    return True


def load_classes_from_info(
    primary_path: Optional[str],
) -> List[Tuple[int, str]]:
    """
    .info 파일에서 클래스 정보를 읽어서 (mtrl_cd, mtrl_nm) 리스트를 반환.
    
    Args:
        primary_path: 원본 이미지 파일 경로
    
    Returns:
        [(mtrl_cd, mtrl_nm), ...] 형태의 리스트
    """
    primary = _norm_path(primary_path)
    if not primary:
        return []
    
    info_path, _ = resample_cache_paths(primary)
    if not os.path.isfile(info_path):
        return []
    
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        classes_info = meta.get("classes", {})
        if not isinstance(classes_info, dict):
            return []
        
        result = []
        for cid_str, info in classes_info.items():
            try:
                cid_int = int(cid_str)
                mtrl_nm = str(info.get("mtrl_nm", f"Class {cid_int}"))
                desc = info.get('desc')
                result.append((cid_int, mtrl_nm, desc))
            except (TypeError, ValueError):
                continue
        
        # CID 기준 정렬
        result.sort(key=lambda x: x[0])
        return result
    except Exception as e:
        logging.exception(f"[load_classes_from_info] .info 파일 읽기 실패: {e}")
        return []


def update_class_in_info(
    primary_path: Optional[str],
    mtrl_cd: int,
    mtrl_nm: str,
    description: Optional[str] = None,
) -> bool:
    """
    .info 파일의 classes 필드에 신규 클래스 정보를 추가/업데이트.
    
    Args:
        primary_path: 원본 이미지 파일 경로
        mtrl_cd: 클래스 ID (정수)
        mtrl_nm: 물질 이름 (문자열)
        description: 설명 (선택적, 문자열)
    
    Returns:
        성공 여부 (bool)
    """
    primary = _norm_path(primary_path)
    if not primary:
        logging.warning("[update_class_in_info] primary_path가 없습니다. primary_path=%s", primary_path)
        return False
    
    info_path, _ = resample_cache_paths(primary)
    logging.info("[update_class_in_info] 저장 시도: primary=%s, info_path=%s, mtrl_cd=%d, mtrl_nm=%s", 
                primary, info_path, mtrl_cd, mtrl_nm)
    
    try:
        # 기존 .info 파일 읽기
        meta = {}
        if os.path.isfile(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                logging.info("[update_class_in_info] 기존 .info 파일 읽기 완료: classes=%d개", 
                           len(meta.get("classes", {})))
            except Exception as e:
                logging.warning(f"[update_class_in_info] 기존 .info 파일 읽기 실패: {e}")
                meta = {}
        else:
            logging.info("[update_class_in_info] .info 파일이 없어 새로 생성합니다: %s", info_path)
        
        # classes 필드 초기화 (없으면 생성)
        if "classes" not in meta:
            meta["classes"] = {}
            logging.info("[update_class_in_info] classes 필드가 없어 새로 생성합니다.")
        
        # 신규 클래스 정보 추가/업데이트
        cid_str = str(int(mtrl_cd))
        old_class = meta["classes"].get(cid_str)
        meta["classes"][cid_str] = {
            "mtrl_cd": int(mtrl_cd),
            "mtrl_nm": str(mtrl_nm),
        }
        if description:
            meta["classes"][cid_str]["dsc"] = str(description)
        
        if old_class:
            logging.info("[update_class_in_info] 기존 클래스 업데이트: cid=%s, old=%s, new=%s", 
                       cid_str, old_class.get("mtrl_nm"), mtrl_nm)
        else:
            logging.info("[update_class_in_info] 신규 클래스 추가: cid=%s, mtrl_nm=%s", cid_str, mtrl_nm)
        
        # 디렉토리 생성 (필요한 경우)
        info_path_obj = Path(info_path)
        info_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # .info 파일 저장
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        # 저장 후 검증
        if os.path.isfile(info_path):
            file_size = os.path.getsize(info_path)
            logging.info("[update_class_in_info] .info 파일 저장 완료: %s (크기: %d bytes, classes: %d개)", 
                       info_path, file_size, len(meta.get("classes", {})))
            
            # 저장된 내용 확인
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    saved_meta = json.load(f)
                saved_class = saved_meta.get("classes", {}).get(cid_str)
                if saved_class:
                    logging.info("[update_class_in_info] 저장 확인: cid=%s, mtrl_nm=%s", 
                               cid_str, saved_class.get("mtrl_nm"))
                else:
                    logging.error("[update_class_in_info] 저장 확인 실패: cid=%s가 저장되지 않았습니다.", cid_str)
            except Exception as e:
                logging.warning("[update_class_in_info] 저장 확인 중 오류: %s", e)
        else:
            logging.error("[update_class_in_info] .info 파일 저장 실패: 파일이 생성되지 않았습니다. %s", info_path)
            return False
        
        return True
        
    except Exception as e:
        logging.exception(f"[update_class_in_info] 클래스 정보 저장 실패: {e}")
        return False
