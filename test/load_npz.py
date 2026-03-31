import numpy as np
import json

# path = r"C:\Users\pde15\Desktop\hsi_crop_preprcoessing\pyqt_code\third_pixel_classification_tool\map_sample\classification1.npz"
# path = r'E:\시연\BG\C3\C3-1.hdr.resample.npz'
# path = r'E:\시연\BG\C3\C3-1.hdr.resample.npz'
# path = r'E:\\3dlabs_data\\sample\\sample.mat.resample.npz'
path = r'E:\3dlabs_data\sample\classification.npz'
with np.load(path, allow_pickle=False) as z:  # 권장: allow_pickle=False
    print(z.files)                  # ['data', 'image_code', 'kind', 'height', 'width', ...]
    
    for i in z.files:        
        try:
            print(z[i])
        except ValueError as e:
            # 기존 파일에 object array가 섞여 있으면 allow_pickle=False로는 로드 불가
            print(f"[skip:{i}] {e}")

    # classmap 라벨명(id_to_name) 출력
    if "id_to_name_json" in z.files:
        try:
            raw = z["id_to_name_json"]
            s = str(raw.item() if hasattr(raw, "item") else raw)
            id_to_name = json.loads(s)
            print("[id_to_name]", id_to_name)
        except Exception as e:
            print("[id_to_name] parse failed:", e)

    # 명시적 class 명 필드 확인
    if "class_ids" in z.files and "class_names" in z.files:
        print("[class_ids]", z["class_ids"])
        print("[class_names]", z["class_names"])

    # print(z['label_coords'].item())
    
    # print(z['label_cr'].item())
    # print(z['splib_raw'].item())
    # print(z['label_raw'].item())
    # data = z["data"]                # ndarray
    # image_code = str(z["image_code"])
    # kind = str(z["kind"])           # 'classmap' | 'roi' | 'mask' 등
    
    # print(z['splib_raw'])
    # print(type(z['splib_cr']))
    # print(z['label_raw'])
    # print(z['label_cr'])
    # 
    # print(len(z['label_raw'].item()[0]))
    

    
    # print(z['label_raw'])
# print(data)
# print(image_code)
# print(kind)