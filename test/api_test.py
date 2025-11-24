import os
import requests
import numpy as np
import hashlib
from data.db import image_check_and_save, search_material_filtering_list

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
        
        # raw_image 저장
        data = {'data':raw_image}
        safe_save_mat(img_pth, data)
        
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

a = image_check_and_save('http://183.98.149.222:18000/program-service/image-exist', 'abc')
print(a)
