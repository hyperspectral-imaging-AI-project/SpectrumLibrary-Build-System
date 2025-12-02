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
import logging
import paramiko
from pathlib import Path
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image

# === 환경 설정 ===
SFTP_HOST = "gnew-office.tplinkdns.com"
SFTP_PORT = 22
SFTP_USER = "shjung"
SFTP_PASSWORD = "!gnew007"

REMOTE_TMP_DIR   = "/disk1/explainSystem/tmp"  # 임시
REMOTE_FINAL_DIR = "/disk1/explainSystem/img"  # 최종

def _ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    """
    SFTP 상의 지정 디렉터리가 없으면 중첩으로 생성.
    예: /data/incoming 와 같이 여러 단계 경로를 안전하게 만든다.
    """
    remote_dir = remote_dir.rstrip("/")
    if not remote_dir:
        return

    parts = remote_dir.split("/")
    path = ""
    for part in parts:
        if part == "":
            # 루트(/) 처리
            path = "/"
            continue

        if path == "/":
            path = f"/{part}"
        else:
            path = f"{path}/{part}"

        try:
            sftp.listdir(path)
        except IOError:
            try:
                sftp.mkdir(path)
            except Exception as e:
                logging.exception(f"원격 디렉터리 생성 실패: {path} ({e})")
                raise


def sftp_upload_then_move(
    local_path: str,
    remote_tmp_dir: str = REMOTE_TMP_DIR,
    remote_final_dir: str = REMOTE_FINAL_DIR,
    host: str = SFTP_HOST,
    port: int = SFTP_PORT,
    username: str = SFTP_USER,
    password: str = SFTP_PASSWORD,
    target_filename: Optional[str] = None,
    delete_local: bool = False,
) -> str:
    """
    1) 로컬 파일을 SFTP 임시 경로(remote_tmp_dir)에 업로드
    2) 업로드 성공 시 최종 경로(remote_final_dir)로 rename
    3) (옵션) 업로드/이동이 모두 성공하면 로컬 파일 삭제(delete_local=True)
    4) 최종 원격 경로를 문자열로 반환

    예:
        /data/local/abc.mat  ->  /data/incoming/abc.mat  (put)
                              ->  /data/final/abc.mat    (rename)
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"로컬 파일 없음: {local_path}")

    if target_filename is None:
        target_filename = local_path.name

    remote_tmp_path = f"{remote_tmp_dir.rstrip('/')}/{target_filename}"
    remote_final_path = f"{remote_final_dir.rstrip('/')}/{target_filename}"

    transport = paramiko.Transport((host, port))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    try:
        # 0. 임시/최종 디렉터리 존재 보장
        _ensure_remote_dir(sftp, remote_tmp_dir)
        _ensure_remote_dir(sftp, remote_final_dir)

        # 1. 임시 경로 업로드
        sftp.put(str(local_path), remote_tmp_path)

        # 2. 최종 경로로 rename (move)
        sftp.rename(remote_tmp_path, remote_final_path)

        # 3. (선택) 로컬 파일 삭제
        if delete_local:
            try:
                local_path.unlink()
            except Exception as e:
                # 삭제 실패해도 업로드는 이미 완료된 상태이므로 경고만 남김
                logging.warning(f"로컬 파일 삭제 실패: {local_path} ({e})")

        return remote_final_path

    finally:
        try:
            sftp.close()
        except Exception:
            pass
        try:
            transport.close()
        except Exception:
            pass

def image_check_and_save(
    check_url: str,
    save_url: str,
    raw_image: np.ndarray,
    rgb_image: np.ndarray,
    save_path: str,
    cmr_cd: int,
    timeout: float = 10.0,
) -> str:
    
    """
    check_url : 영상 등록 여부 확인용 API
    save_url  : 이미지 정보 저장 API (img_cd 반환)
    raw_image : 초분광 이미지 (H, W, B)
    rgb_image : RGB 이미지 (H, W, 3)

    동작:
      1) rgb_image의 hash로 중복 여부 확인
      2) 이미 존재하면 img_cd 바로 반환
      3) 없으면 .mat, .png 로컬 저장 후
         - SFTP에 업로드(임시 디렉터리 → 최종 디렉터리로 이동)
         - 최종 SFTP 경로를 payload에 넣어 save_url 호출
         - save_url 결과에서 img_cd 반환
    """
    import os  # 필요한 경우

    def hash_array(arr: np.ndarray) -> str:
        return hashlib.md5(arr.tobytes()).hexdigest()

    # ---------- 1. 중복 체크 ----------
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
        raise RuntimeError(f"HTTP error (check_url): {e}") from e
    except ValueError as e:
        raise RuntimeError(f"Invalid JSON response (check_url): {e}") from e

    # 기본 스키마 검증/정규화
    code = data.get("status_code")
    if code != 200:
        raise RuntimeError(
            f"API error (check_url): code={code}, message={data.get('message')}"
        )

    result = data.get("result") or []
    if not isinstance(result, list):
        raise RuntimeError("API response 'result' is not a list (check_url)")
    if not result:
        raise RuntimeError("API response 'result' is empty list (check_url)")

    first = result[0]
    exist_yn = first.get("exist_yn")

    # ---------- 2. 이미 존재하는 경우: img_cd 반환 ----------
    if exist_yn == 1:
        img_cd = first.get("img_cd")
        if img_cd is None:
            raise RuntimeError("API response missing 'img_cd' while exist_yn == 1")
        return img_cd

    # ---------- 3. 존재하지 않는 경우: 저장 로직 ----------
    if exist_yn != 0:
        raise RuntimeError(f"Unexpected exist_yn value: {exist_yn}")

    # 로컬 저장 경로 준비
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    name = datetime.now().strftime("%Y%m%d-%H%M%S")
    local_mat_path = save_dir / f"{name}.mat"
    local_rgb_path = save_dir / f"{name}.png"

    # 3-1) raw_image .mat 로컬 저장
    mat_data = {"data": raw_image}
    safe_save_mat(str(local_mat_path), mat_data)

    # 3-2) rgb_image .png 로컬 저장
    Image.fromarray(rgb_image.astype(np.uint8), mode="RGB").save(str(local_rgb_path))

    # 3-3) SFTP로 업로드 후 최종 경로 받기
    #      (필요 없으면 이 부분 주석 처리하고, 바로 local_mat_path / local_rgb_path 사용해도 됨)
    remote_mat_path = sftp_upload_then_move(str(local_mat_path))
    remote_rgb_path = sftp_upload_then_move(str(local_rgb_path))

    # 이미지 사이즈
    h, w, b = raw_image.shape  # H, W, Bands

    # ---------- 4. save_url 호출 ----------
    url = save_url.rstrip("/")
    payload = {
        "cmr_cd": cmr_cd,
        # 여기서 img_pth, rgb_pth 를 SFTP 최종 경로 기준으로 보냄
        "img_pth": remote_mat_path,   # 또는 str(local_mat_path) 로 바꾸면 로컬 경로 사용
        "rgb_pth": remote_rgb_path,   # 마찬가지
        "img_x_sz": h,
        "img_y_sz": w,
        "img_z_sz": b,
        "img_hash": hash_value,
    }

    try:
        resp = requests.post(url, json=payload, headers=_headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"HTTP error (save_url): {e}") from e
    except ValueError as e:
        raise RuntimeError(f"Invalid JSON response (save_url): {e}") from e

    # save_url 의 응답 구조는 실제 API 스펙에 따라 다를 수 있음
    # 예: {"status_code":200, "result": {"img_cd":123}} 형태라면 아래처럼 조정 필요
    # 여기서는 원래 코드에 맞춰 data[0]['img_cd'] 를 그대로 사용하되, 방어 코드 추가

    if isinstance(data, list):
        if not data:
            raise RuntimeError("save_url response is empty list")
        img_cd = data[0].get("img_cd")
        if img_cd is None:
            raise RuntimeError("save_url response missing 'img_cd'")
        return img_cd
    elif isinstance(data, dict):
        # 만약 dict 형태라면 이런 식으로 처리 (필요에 따라 수정)
        if data.get("status_code") != 200:
            raise RuntimeError(
                f"API error (save_url): code={data.get('status_code')}, "
                f"message={data.get('message')}"
            )
        
        result = data.get("result")
        img_cd = result[0].get("img_cd")
        if img_cd is None:
            raise RuntimeError("save_url response missing 'img_cd' in 'result'")
        return img_cd
    else:
        raise RuntimeError(f"Unexpected response type from save_url: {type(data)}")
    
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


def call_unmixing(
    base_url : str,
    img_cd: int,
    st_x: int,
    st_y: int,
    ed_x: int,
    ed_y: int,
    num_endmembers: int,
    # timeout: int = 30,
) -> Dict[str, Any]:
    """
    프로그램 주분 분광혼합분석 API 호출 함수

    http://{APIIP}:{port}/program-service/inference/unmixing
    """
    
    url = base_url

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "img_cd": img_cd,
        "st_h": st_y,
        "st_w": st_x,
        "ed_h": ed_y,
        "ed_w": ed_x,
        "num_endmembers": num_endmembers,
    }
    print(payload)

    try:
        resp = requests.post(url, json=payload, headers=headers)
    except requests.RequestException as e:
        raise RuntimeError(f"API 요청 실패: {e}")

    # HTTP 에러 코드(4xx, 5xx) 체크
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code} 에러: {resp.text}")

    data = resp.json()

    # 명세서 기준 응답 구조:
    # {
    #   "code": 200,
    #   "message": "정상",
    #   "result": [
    #       {
    #           "endmember": [...],
    #           "abondance_map": [...]
    #       }
    #   ]
    # }
    

    if data.get("status_code") != 200:
        raise RuntimeError(f"API 처리 실패(code={data.get('code')}): {data.get('message')}")
    
    elif data.get("status_code") == 200:
        return data['result'][0]['endmember'], data['result'][0]['abondance_map']
        # return data['result']['endmember'], data['result']['abondance_map'] 

# print(search_labeling_data(st_wv = 399.109985, ed_wv = 991.539978, base_url = 'http://183.98.149.222:18000/program-service/label-list'))
# print(material_code_name(api_base = "http://183.98.149.222:18000/program-service/material-list/name"))
# unmixing_inference_url = 'http://183.98.149.222:18000/program-service/inference/unmixing'
# print(call_unmixing(img_cd = 16,base_url = unmixing_inference_url, st_x = 200, st_y = 200, ed_x = 300, ed_y = 300, num_endmembers = 5))