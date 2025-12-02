import numpy as np
import scipy.io as sio
from pathlib import Path

def mat_gt_to_npz(
    mat_path: str,
    npz_path: str = None,
    var_name: str = None,
    image_code: int = 9,
):
    """
    .mat 파일에 들어있는 ground truth(class map)를
    Indian_pines_gt.npz 와 같은 스키마로 변환해 저장하는 함수.

    Parameters
    ----------
    mat_path : str
        입력 .mat 파일 경로 (예: "Botswana_gt.mat")
    npz_path : str, optional
        출력 .npz 파일 경로 (기본: mat_path 와 같은 이름에 확장자만 .npz)
    var_name : str, optional
        .mat 파일 안에서 GT가 들어있는 변수명 (None이면 자동 추정)
    image_code : int, optional
        classmap를 나타내는 코드 (Indian_pines/우리가 쓰던 값: 9)
    """

    mat_path = Path(mat_path)
    if npz_path is None:
        npz_path = mat_path.with_suffix(".npz")

    # 1) .mat 로드
    data_mat = sio.loadmat(mat_path)

    # 2) GT 변수명 결정
    if var_name is None:
        # "__" 로 시작하는 메타 키 제외하고 남는 것들 중 첫 번째를 GT로 가정
        keys = [k for k in data_mat.keys() if not k.startswith("__")]
        if len(keys) != 1:
            raise ValueError(
                f"var_name을 지정하지 않았고, .mat 안에 후보 변수가 {keys}처럼 여러 개입니다. "
                f"GT 변수명을 var_name 인자로 명시해 주세요."
            )
        var_name = keys[0]

    gt = data_mat[var_name]
    if gt.ndim != 2:
        raise ValueError(f"GT 배열은 2차원이어야 합니다. 현재 shape={gt.shape}")

    # 3) 스키마 필드 준비
    height, width = gt.shape

    data = gt.astype(np.int16)              # data: (H, W), int16
    image_code_arr = np.array(image_code, dtype=np.int32)  # scalar
    kind_arr = np.array("classmap")         # scalar string
    height_arr = np.array(height, dtype=np.int32)
    width_arr = np.array(width, dtype=np.int32)
    channels_arr = np.array(None, dtype=object)  # classmap이므로 채널 정보 없음

    # 4) .npz 저장
    np.savez(
        npz_path,
        data=data,
        image_code=image_code_arr,
        kind=kind_arr,
        height=height_arr,
        width=width_arr,
        channels=channels_arr,
    )

    print(f"Saved GT npz: {npz_path}")
    return str(npz_path)

mat_gt_to_npz(r"E:\pavia_university\PaviaU_gt.mat", image_code = 18)  # -> Botswana_gt.npz 생성
