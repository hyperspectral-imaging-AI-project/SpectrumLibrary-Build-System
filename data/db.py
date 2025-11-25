from __future__ import annotations
import os
import json
import requests
import re
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from scipy.io import savemat   # ✅ 추가
from PIL import Image   # ✅ 추가: Pillow
import numpy as np

import os
from pathlib import Path
import numpy as np
from scipy.io import savemat

try:
    import hdf5storage  # v7.3 .mat (HDF5) 저장용
except Exception:
    hdf5storage = None


def _downcast_for_mat(data: dict) -> dict:
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            a = v
            if a.dtype == np.float64:
                a = a.astype(np.float32, copy=False)
            elif a.dtype == np.int64:
                a = a.astype(np.int32, copy=False)
            out[k] = a
        else:
            out[k] = v
    return out


def _total_nbytes(data: dict) -> int:
    s = 0
    for v in data.values():
        if isinstance(v, np.ndarray):
            s += int(v.nbytes)
        elif isinstance(v, (list, tuple)):
            for e in v:
                if isinstance(e, np.ndarray):
                    s += int(e.nbytes)
    return s


def safe_save_mat(mat_path_str: str, data: dict) -> str:
    """
    1) dtype 다운캐스트(float64→float32, int64→int32)
    2) 총 바이트 < 2GB : SciPy v5 .mat 저장
    3) 총 바이트 ≥ 2GB : hdf5storage로 v7.3 .mat(HDF5) 저장 (확장자 .mat 유지)
    """
    mat_path = Path(mat_path_str)
    mat_path.parent.mkdir(parents=True, exist_ok=True)

    data_dc = _downcast_for_mat(data)
    total = _total_nbytes(data_dc)

    limit = 1_900_000_000  # 여유치 포함 1.9GB
    if total < limit:
        savemat(str(mat_path), data_dc, do_compression=True)
        return str(mat_path)

    # 2GB 이상이면 v7.3 .mat (HDF5)
    if hdf5storage is None:
        raise RuntimeError(
            "대용량 .mat 저장을 위해 'hdf5storage'가 필요합니다. "
            "pip install hdf5storage 후 다시 실행하세요."
        )

    # MATLAB 호환 + 압축 옵션
    opts = hdf5storage.Options(
        store_python_metadata=False,  # MATLAB 호환성 ↑
        matlab_compatible=True,
        compress=True
    )
    hdf5storage.savemat(str(mat_path), data_dc, options=opts)
    return str(mat_path)


def _parse_number_list(text: str) -> List[float]:
    s = text.strip()
    # 1) 안전한 파서: Python 리스트 리터럴 그대로 평가
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [float(x) for x in val]
    except Exception:
        pass

    # 2) 폴백: 숫자만 뽑기 (부호/소수/지수표기 지원)
    nums = re.findall(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', s)
    return [float(x) for x in nums]


### API로 진행행
def get_camera_list(
    st_wv: float,
    ed_wv: float,
    base_url: str,
    timeout: int = 10,
) -> List[Dict[str, Any]]:
    """
    POST {base_url}/program-service/camera-list
    Headers:
      Content-Type: application/json
      Authorization: Bearer {token}
    Body:
      { "st_wv": float, "ed_wv": float }
    Response:
      { "code": number, "message": string, "result": [ { "cmr_cd","cmr_nm","wv","fwhm" } ] }
    반환 rows 스키마(앱 통일):
      { "CMR_CD": int, "CMR_NM": str, "WV": List[float], "FWHM": List[float] }
    """
    headers = {
        "Content-Type": "application/json",
    }
    payload = {"st_wv": float(st_wv), "ed_wv": float(ed_wv)}

    resp = requests.post(base_url, headers=headers, data =json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    
    if int(data.get("status_code", 0)) != 200:
        raise RuntimeError(f"API error: code={data.get('code')} message={data.get('message')}")

    result = data.get("result") or []
    rows: List[Dict[str, Any]] = []
    for r in result:
        rows.append({
            "CMR_CD": int(r.get("cmr_cd")) if r.get("cmr_cd") is not None else -1,
            "CMR_NM": str(r.get("cmr_nm") or ""),
            "WV"    : _parse_number_list(r.get("wv")),
            "FWHM"  : _parse_number_list(r.get("fwhm")),
        })
    return rows


def _stringify_number_list(xs: List[float]) -> str:
    """API 스펙대로 숫자리스트를 문자열 JSON 배열로 직렬화"""
    return json.dumps([float(x) for x in xs], ensure_ascii=False)

def add_camera(
    cmr_nm: str,
    wv: List[float],
    fwhm: List[float],
    base_url: str,
    timeout: int = 10,
) -> int:
    """
    POST {base_url}/program-service/camera-add
    Headers:
      Content-Type: application/json
      Authorization: Bearer {token}
    Body:
      {
        "cmr_nm": string,
        "wv":    string,  # "[390.135, 392.638, ...]"
        "fwhm":  string,  # "[8.1, 8.1, ...]"
        "st_wv": float,
        "ed_wv": float
      }
    Response:
      { "code": 200, "message": "Success", "result": [{"cmr_cd": 3}] }
    반환: 생성된 cmr_cd (int)
    """
    if not cmr_nm.strip():
        raise ValueError("cmr_nm(카메라 이름)이 비어 있습니다.")
    if not wv:
        raise ValueError("wv(파장 리스트)가 비어 있습니다.")
    if len(fwhm) and len(fwhm) != len(wv):
        raise ValueError(f"FWHM 길이({len(fwhm)})가 WV 길이({len(wv)})와 다릅니다.")

    headers = {
        "Content-Type": "application/json",
    }
    body = {
        "cmr_nm": cmr_nm,
        "wv": _stringify_number_list(wv),
        "fwhm": _stringify_number_list(fwhm) if fwhm else "[]",
        "st_wv": float(min(wv)),
        "ed_wv": float(max(wv)),
    }

    resp = requests.post(base_url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    
    if int(data.get("status_code", 0)) != 200:
        raise RuntimeError(f"API error: code={data.get('code')} message={data.get('message')}")

    result = data.get("result") or []
    if not result or "cmr_cd" not in result[0]:
        raise RuntimeError("API 응답에 cmr_cd가 없습니다.")
    return int(result[0]["cmr_cd"])


def search_spectrum_library(
    # cmr_nm: str,
    st_wv: float,
    ed_wv: float,
    base_url: str,
    timeout: int = 10,
) -> int:
    """
    POST {base_url}/program-service/camera-add
    Headers:
      Content-Type: application/json
      Authorization: Bearer {token}
    Body:
      {
        "cmr_nm": string,
        "wv":    string,  # "[390.135, 392.638, ...]"
        "fwhm":  string,  # "[8.1, 8.1, ...]"
        "st_wv": float,
        "ed_wv": float
      }
    Response:
      { "code": 200, "message": "Success", "result": [{"cmr_cd": 3}] }
    반환: 생성된 cmr_cd (int)
    """
    if not st_wv:
        raise ValueError("Start wavelength 가 비어 있습니다.")
    if not ed_wv:
        raise ValueError(f"End wavelength 가 비어 있습니다.")

    headers = {
        "Content-Type": "application/json",
    }
    body = {
        "st_wv": st_wv,
        "ed_wv": ed_wv,
    }

    resp = requests.post(base_url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if int(data.get("status_code", 0)) != 200:
        raise RuntimeError(f"API error: code={data.get('code')} message={data.get('message')}")

    result = data.get("result") or []
    return result

def search_labeling_data(
    st_wv: float,
    ed_wv: float,
    base_url: str,
    timeout: int = 10,):
    
    if not st_wv:
        raise ValueError("Start wavelength 가 비어 있습니다.")
    if not ed_wv:
        raise ValueError(f"End wavelength 가 비어 있습니다.")

    headers = {
        "Content-Type": "application/json",
    }
    body = {
        "st_wv": st_wv,
        "ed_wv": ed_wv,
    }

    resp = requests.post(base_url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    print(data)
    if int(data.get("status_code", 0)) != 200:
        raise RuntimeError(f"API error: code={data.get('code')} message={data.get('message')}")

    result = data.get("result") or []
    return result

def search_material_filtering_list(
    base_url:str,
    mtrl_ids:List[int],
    timeout:float = 10.0,
):
    """
    POST {base_url}  
    Body: {"mtrl_cd": [0, 5, 7, ...]}
    Return: [{"mtrl_cd":int, "mtrl_nm":str, "desc":str}, ...]
    """
    if not isinstance(mtrl_ids, list) or not all(isinstance(x, int) for x in mtrl_ids):
        raise ValueError("mtrl_ids must be a List[int]")

    url = base_url.rstrip("/")
    payload = {"mtrl_cd": mtrl_ids}
    _headers = {
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, json=payload, headers=_headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"HTTP error: {e}") from e
    except ValueError as e:
        raise RuntimeError(f"Invalid JSON response: {e}") from e

    # 기본 스키마 검증/정규화
    code = data.get("status_code")
    if code != 200:
        raise RuntimeError(f"API error: code={code}, message={data.get('message')}")
    result = data.get("result") or []
    if not isinstance(result, list):
        raise RuntimeError("API response 'result' is not a list")

    out: List[Dict[str, Any]] = []
    for r in result:
        if not isinstance(r, dict):
            continue
        out.append({
            "mtrl_cd": int(r.get("mtrl_cd")) if r.get("mtrl_cd") is not None else None,
            "mtrl_nm": str(r.get("mtrl_nm") or ""),
            "desc":    str(r.get("dsc") or ""),
        })
    return out

def image_check_and_save(
    check_url:str,
    save_url:str,
    raw_image:np.array,
    rgb_image:np.array,
    save_path : str,
    cmr_cd:int,
    timeout:float = 10.0,
):
    """
    check_url : 영상 등록 여부 확인
    save_url : Save를 진행한 후 이미지 코드를 갖게 됨
    raw_image : 초분광 이미지
    rgb_image : RGB 이미지
    """
    def hash_array(arr: np.ndarray) -> str:
        return hashlib.md5(arr.tobytes()).hexdigest()
    
    url = check_url.rstrip("/")
    hash_value = hash_array(rgb_image)
    
    payload = {"img_hash": hash_value}
    _headers = {
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, json=payload, headers=_headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"HTTP error: {e}") from e
    except ValueError as e:
        raise RuntimeError(f"Invalid JSON response: {e}") from e

    # 기본 스키마 검증/정규화
    code = data.get("status_code")
    if code != 200:
        raise RuntimeError(f"API error: code={code}, message={data.get('message')}")
    result = data.get("result") or []
    if not isinstance(result, list):
        raise RuntimeError("API response 'result' is not a list")
    
    if result[0]['exist_yn'] == 1:
        return result[0]['img_cd']
    elif result[0]['exist_yn'] == 0:
        name = datetime.now().strftime('%Y%m%d-%H%M%S')
        img_pth = os.path.join(save_path, f'{name}.mat')
        rgb_pth = os.path.join(save_path, f'{name}.png')
        
        # # raw_image 저장
        # data = {'data':raw_image}
        # safe_save_mat(img_pth, data)
        
        Image.fromarray(rgb_image, mode = 'RGB').save(rgb_pth)
        
        h, w, b = raw_image.shape
        
        url = save_url
        
        payload = {"cmr_cd": cmr_cd,
                   "img_pth":img_pth,
                   'rgb_pth':rgb_pth,
                   'img_x_sz':h,
                   'img_y_sz':w,
                   'img_z_sz':b,
                   'img_hash':hash_value}
        
        _headers = {
            "Content-Type": "application/json",
        }
        
        try:
            resp = requests.post(url, json=payload, headers=_headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"HTTP error: {e}") from e
        except ValueError as e:
            raise RuntimeError(f"Invalid JSON response: {e}") from e        
        
        return data[0]['img_cd']
    
def send_label_add(api_base: str, targets: Dict, timeout: float = 10.0) -> Dict[str, Any]:
    
    """
    프로그램 라벨_등록 API 호출
    - api_base: 예) "http://183.98.149.222:18000/program-service/label-add"
    - targets : [{"img_cd":int, "mtrl_cd":int, "img_x":int, "img_y":int, "rfl": List[float]}, ...]
    - return  : 서버의 JSON 응답(dict)
    """
    url = api_base  # 스펙상 완전 경로
    print(url)
    headers = {"Content-Type": "application/json"}
    payload = {'target':targets}

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    print(resp)
    # 실패 코드면 예외 발생시켜 상위에서 처리하거나 여기서 메시지 반환하도록 선택
    resp.raise_for_status()
    return resp.json()


def material_add(api_base:str, mtrl_nm:str, dsc : str, timeout :float = 10.0):
    url = api_base
    headers = {"Content-Type": "application/json"}
    payload = {'target':[{"mtrl_nm":mtrl_nm, "dsc":dsc}]}

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    # 실패 코드면 예외 발생시켜 상위에서 처리하거나 여기서 메시지 반환하도록 선택
    resp.raise_for_status()
    mtrl_cd = resp.json()['result'][0]['mtrl_cd']
    return mtrl_cd

def material_code_name(api_base:str, timeout :float = 10.0):
    url = api_base
    headers = {"Content-Type": "application/json"}
    payload = {}

    resp = requests.post(url,json = payload, headers=headers, timeout=timeout)
    # 실패 코드면 예외 발생시켜 상위에서 처리하거나 여기서 메시지 반환하도록 선택
    resp.raise_for_status()
    result = resp.json()['result']
    return result


# print(search_labeling_data(st_wv = 399.109985, ed_wv = 991.539978, base_url = 'http://183.98.149.222:18000/program-service/label-list'))
# print(material_code_name(api_base = "http://183.98.149.222:18000/program-service/material-list/name"))