import numpy as np
from scipy.io import loadmat, savemat
from typing import Optional, Tuple, Dict, Any

# mat_path = r'E:\Indian_pines\Indian_pines.mat'
# data = loadmat(mat_path, struct_as_record = False, squeeze_me = True)
# data = {k:v for k,v in data.items() if not k.startswith("__")}
# data['indian_pines'] = data['indian_pines']/10000.0

# savemat(mat_path, data, do_compression = True)

mat_path = r"E:\Indian_pines\Indian_pines_gt.mat"
# npz_path = r"C:\Users\pde15\Desktop\hsi_crop_preprcoessing\pyqt_code\third_pixel_classification_tool\map_sample\sample.npz"
npz_path = r"C:\Users\pde15\Desktop\hsi_crop_preprcoessing\pyqt_code\third_pixel_classification_tool\map_sample\Indian_pines_gt.npz"
data = loadmat(mat_path)

def convert_indian_pines_gt_to_npz(mat_path: str, npz_path: str):
    m = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    gt = m["indian_pines_gt"].astype(np.int16)  # 원하는 dtype으로
    H, W = gt.shape
    # 예시 스키마 구성
    out = {
        "data": gt,                    # 실제 데이터(여기서는 classmap)
        "image_code": 9,
        "kind": np.array("classmap"),  # 'classmap' | 'roi' | 'mask' 등
        "height": np.array(H),
        "width": np.array(W),
        "channels": None        # GT는 1채널
    }
    np.savez_compressed(npz_path, **out)
convert_indian_pines_gt_to_npz(mat_path, npz_path)


# def unique_counts(arr: np.ndarray, *, sort_by_count: bool = True) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     1D/ND 배열의 고유값과 개수 반환.
#     """
#     a = np.asarray(arr)
#     u, idx, inv, cnt = np.unique(a[~np.isnan(a)] if a.dtype.kind == "f" else a,
#                                  return_index=True, return_inverse=True, return_counts=True)
#     if sort_by_count:
#         order = np.argsort(-cnt)
#         u, cnt = u[order], cnt[order]
#     return u, cnt


# print(unique_counts(arr = data['indian_pines_gt']))
