import json
import numpy as np
from pathlib import Path

PATH = r'E:\시연\BG\spec_data_filler_BG.json'

# ===== 1) JSON 로드 =====
json_path = Path(PATH)  # 파일 경로 수정해서 사용
with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

# meta.band_centers → wavelength 배열 (float32)
wavelength = np.asarray(data["meta"]["band_centers"], dtype=np.float32)

# classes: class_id("CLS_0000" 등) -> 정수 cid 매핑
# 우선순위: order 필드 > class_id 내부 번호
class_id_to_cid = {}
for cls in data["classes"]:
    class_id = cls["class_id"]          # 예: "CLS_0000"
    order = cls.get("order", None)      # 예: 0, 1 ...
    if order is not None:
        cid = int(order)
    else:
        # "CLS_0000" → 0
        cid = int(class_id.split("_")[1])
    class_id_to_cid[class_id] = cid

# ===== 2) 스펙트럼을 클래스별로 모아서 dict[int -> (N,C) ndarray] 만들기 =====
label_raw_dict = {}  # {cid: (N, C)}

for spec in data["spectra"]:
    class_id = spec["class_id"]         # 예: "CLS_0000", "CLS_0001"
    cid = class_id_to_cid[class_id]     # 0, 1, ...

    values = np.asarray(spec["values"], dtype=np.float32)  # (C,)
    if cid not in label_raw_dict:
        label_raw_dict[cid] = []
    label_raw_dict[cid].append(values)

# 리스트 → (N, C) array 로 변환
for cid, specs in label_raw_dict.items():
    label_raw_dict[cid] = np.stack(specs, axis=0)  # (N, C)

# object 배열로 감싸기 (예시 resample.npz 가 object 배열을 쓰고 있어서 맞춰줌)
splib_raw = np.array([], dtype=object)      # 여기서는 비움
splib_cr  = np.array([], dtype=object)      # 여기서도 비움
label_raw_dict
label_cr  = np.array([], dtype=object)
label_coords = np.array([], dtype=object)
# ===== 3) npz로 저장 =====
# 이대로 저장하면 np.load(..., allow_pickle=True)["splib_raw"].item() 으로 dict 꺼낼 수 있는 구조가 됨
out_path = 'E:\시연\BG\C3'

# 4) npz 저장
np.savez(
    out_path,
    splib_raw=splib_raw,
    splib_cr=splib_cr,
    label_raw=label_raw_dict,
    label_cr=label_cr,
    label_coords=label_coords,
)
# np.savez(
#     out_path,
#     wavelength=wavelength,       # (C,)
#     splib_raw=splib_raw_dict,    # {cid: (N,C)}
# )

print("saved to:", out_path)