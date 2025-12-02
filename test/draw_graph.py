import requests
import random
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import Any, Dict, List, Optional
import pandas as pd

def search_material_filtering_list(
    mtrl_ids:List[int],
    timeout:float = 10.0,
    base_url:str = 'http://183.98.149.222:18000/program-service/material-filtering-list',    
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


# def search_spectrum_library(
#     # cmr_nm: str,
#     st_wv: float,
#     ed_wv: float,
#     base_url: str =f'http://183.98.149.222:18000/program-service/spectrum-library-list',
#     timeout: int = 10,
# ) -> int:
#     """
#     POST {base_url}/program-service/camera-add
#     Headers:
#       Content-Type: application/json
#       Authorization: Bearer {token}
#     Body:
#       {
#         "cmr_nm": string,
#         "wv":    string,  # "[390.135, 392.638, ...]"
#         "fwhm":  string,  # "[8.1, 8.1, ...]"
#         "st_wv": float,
#         "ed_wv": float
#       }
#     Response:
#       { "code": 200, "message": "Success", "result": [{"cmr_cd": 3}] }
#     반환: 생성된 cmr_cd (int)
#     """
#     if not st_wv:
#         raise ValueError("Start wavelength 가 비어 있습니다.")
#     if not ed_wv:
#         raise ValueError(f"End wavelength 가 비어 있습니다.")

#     headers = {
#         "Content-Type": "application/json",
#     }
#     body = {
#         "st_wv": st_wv,
#         "ed_wv": ed_wv,
#     }

#     resp = requests.post(base_url, headers=headers, json=body, timeout=timeout)
#     resp.raise_for_status()
#     data = resp.json()
#     if int(data.get("status_code", 0)) != 200:
#         raise RuntimeError(f"API error: code={data.get('code')} message={data.get('message')}")

#     result = data.get("result") or []
#     return result

# a = search_spectrum_library(st_wv = 400, ed_wv = 900)

# sample_result = random.sample(a, 10)
# print(sample_result)
# with open('sample.json', 'w') as f:
#     json.dump(sample_result, f, indent = 4)
    
    
# with open('sample.json', 'r') as f:
#     data = json.load(f)
    
# modified_data = []


# for i in data:
    
#     i['rfl'] = eval(i['rfl'])
#     i['wv'] = eval(i['wv'])
    
#     wv_filtered = []
#     spec_filtered = []
#     for w, s in zip(i['wv'], i['rfl']):
#         if s > 0:
#             wv_filtered.append(w)
#             spec_filtered.append(s)
#     # mask = i['rfl'] > 0
#     # i['wv'] = i['wv'][mask]
#     # i['rfl'] = i['rfl'][mask]
#     sns.lineplot(x = wv_filtered, y = spec_filtered)
#     plt.tight_layout()
#     plt.savefig(os.path.join(r'C:\Users\pde15\Desktop\hsi_crop_preprcoessing\pyqt_code\third_pixel_classification_tool\images',f"{i['mtrl_cd']}.png"), dpi=300)  # png, jpg, pdf, svg 등 가능
#     plt.close()
    
#     i['rfl'] =spec_filtered
#     i['wv'] = wv_filtered
    
#     modified_data.append(i)
    
# with open('modified_json.json', 'w') as f:
#     json.dump(modified_data, f, indent = 4)

# with open('modified_json.json', 'r') as f:
#     data = json.load(f)

# modified_data = []

# for i in data:    
#     i['label'] = search_material_filtering_list(mtrl_ids = [i['mtrl_cd']])[0]['mtrl_nm']
#     modified_data.append(i)
    
# with open('mod_modified_json.json', 'w', encoding= 'utf-16') as f:
#     json.dump(modified_data, f, indent = 4, ensure_ascii = False)

with open('mod_modified_json.json', 'r', encoding = 'utf-16') as f:
    data = json.load(f)
    
final_data = {'wavelength':[], 'reflectance':[], 'mtrl_cd':[], 'mtrl_nm':[]}
for i in data:
    final_data['wavelength'].append(i['wv'])
    final_data['reflectance'].append(i['rfl'])
    final_data['mtrl_cd'].append(i['mtrl_cd'])
    final_data['mtrl_nm'].append(i['label'])
    
pd.DataFrame(final_data).to_csv('final_data.csv', index = False, encoding = 'cp949')
     
    