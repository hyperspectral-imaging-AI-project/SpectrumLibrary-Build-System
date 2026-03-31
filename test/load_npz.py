import numpy as np

# path = r"C:\Users\pde15\Desktop\hsi_crop_preprcoessing\pyqt_code\third_pixel_classification_tool\map_sample\classification1.npz"
# path = r'E:\시연\BG\C3\C3-1.hdr.resample.npz'
# path = r'E:\시연\BG\C3\C3-1.hdr.resample.npz'
path = r'E:\\3dlabs_data\\sample\\sample.mat.resample.npz'
with np.load(path, allow_pickle=True) as z:  # 권장: context manager
    print(z.files)                  # ['data', 'image_code', 'kind', 'height', 'width', ...]
    
    for i in z.files:
        
        if i == 'label_raw':
            print(z[i])

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