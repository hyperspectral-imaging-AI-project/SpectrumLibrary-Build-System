import json
import numpy as np
from pathlib import Path

# ===============================
# 0) 경로 설정
# ===============================
# JSON_PATH = Path(r"E:\시연\BG\spec_data_filler_BG.json")   # wavelength용
# HDR_PATH  = Path(r"E:\시연\BG\C3\C3-1.hdr")               # ENVI hdr
# RAW_PATH  = Path(r"E:\시연\BG\C3\C3-1.raw")               # ENVI raw
# OUT_PATH  = Path(r"E:\시연\BG\C3\C3-1_mean_spec.npz")     # 출력 npz 경로
HDR_PATH = 'E:\시연\FM_vec\C1-3\C1-3_vec.hdr'
OUT_PATH = 'E:\시연\FM_vec\C1-3_vec.hdr.resample.npz'
RAW_PATH = 'E:\시연\FM_vec\C1-3\C1-3_vec.raw'

# ===============================
# 1) JSON 로드 → wavelength 가져오기
# ===============================
# with JSON_PATH.open("r", encoding="utf-8") as f:
#     data = json.load(f)

# meta.band_centers → wavelength 배열 (float32)
# wavelength = np.asarray(data["meta"]["band_centers"], dtype=np.float32)  # (C,)

# ===============================
# 2) ENVI .hdr / .raw 로 큐브 읽기
# ===============================
def read_envi_hdr(hdr_path: str | Path) -> dict:
    """
    ENVI .hdr 파일을 단순 파싱해서 dict로 반환
    """
    hdr_path = Path(hdr_path)
    header: dict = {}

    with hdr_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.lower().startswith("envi"):
            continue
        if "=" not in line:
            continue

        k, v = line.split("=", 1)
        k = k.strip().lower()
        v = v.strip()

        # { ... } 감싸진 값 제거
        if v.startswith("{") and v.endswith("}"):
            v = v[1:-1].strip()

        header[k] = v

    # 숫자 필드 형변환
    for key in ["samples", "lines", "bands", "header offset", "data type", "byte order"]:
        if key in header:
            try:
                header[key] = int(header[key])
            except ValueError:
                pass

    # interleave 기본값
    if "interleave" in header:
        header["interleave"] = header["interleave"].lower()
    else:
        header["interleave"] = "bsq"

    return header


def _envi_dtype(data_type: int):
    """
    ENVI data type → numpy dtype 매핑
    """
    mapping = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        12: np.uint16,
        13: np.uint32,
        14: np.int64,
        15: np.uint64,
    }
    if data_type not in mapping:
        raise ValueError(f"지원하지 않는 ENVI data type: {data_type}")
    return mapping[data_type]


def load_envi_raw(hdr_path: str | Path, raw_path: str | Path) -> np.ndarray:
    """
    ENVI .hdr / .raw → (lines, samples, bands) 형태 cube 로드
    """
    hdr = read_envi_hdr(hdr_path)
    samples = int(hdr["samples"])
    lines   = int(hdr["lines"])
    bands   = int(hdr["bands"])
    interleave    = hdr.get("interleave", "bsq").lower()
    data_type     = int(hdr["data type"])
    header_offset = int(hdr.get("header offset", 0))

    dtype = _envi_dtype(data_type)

    raw_path = Path(raw_path)
    with raw_path.open("rb") as f:
        if header_offset > 0:
            f.seek(header_offset)
        arr = np.fromfile(f, dtype=dtype)

    expected = samples * lines * bands
    if arr.size != expected:
        raise ValueError(
            f"raw 데이터 크기가 헤더와 다름: "
            f"expected={expected}, actual={arr.size}, "
            f"samples={samples}, lines={lines}, bands={bands}"
        )

    if interleave == "bsq":
        cube = arr.reshape((bands, lines, samples)).transpose(1, 2, 0)
    elif interleave == "bil":
        cube = arr.reshape((lines, bands, samples)).transpose(0, 2, 1)
    elif interleave == "bip":
        cube = arr.reshape((lines, samples, bands))
    else:
        raise ValueError(f"지원하지 않는 interleave 형식: {interleave}")

    return cube.astype(np.float32, copy=False)


# ===============================
# 3) 전체 픽셀 기준 평균 스펙트럼 계산
# ===============================
def compute_mean_spectrum(cube: np.ndarray) -> np.ndarray:
    """
    cube: (lines, samples, bands)
    → 전체 픽셀 평균 스펙트럼 (bands,)
    """
    if cube.ndim != 3:
        raise ValueError(f"cube는 3D여야 함: got {cube.shape}")
    lines, samples, bands = cube.shape
    pixels = cube.reshape(-1, bands)  # (N_pixels, bands)
    mean_spec = pixels.mean(axis=0)
    return mean_spec.astype(np.float32)


# 큐브 로드 & 평균 스펙트럼 계산
cube      = load_envi_raw(HDR_PATH, RAW_PATH)
mean_spec = compute_mean_spectrum(cube)  # (C,)

# ===============================
# 4) splib_raw dict 구성 (cid=0에 평균 1개)
# ===============================
# 네가 쓰던 포맷과 동일하게: {cid: (N, C)}
label_raw_dict: dict[int, np.ndarray] = {}
splib_raw_dict: dict[int, np.ndarray] = {}
splib_cr_dict: dict[int, np.ndarray] = {}
label_cr_dict: dict[int, np.ndarray] = {}
cid = 0
label_raw_dict[cid] = mean_spec[np.newaxis, :]  # (1, C)

# ===============================
# 5) npz 저장 (포맷 동일)
# ===============================
np.savez(
    OUT_PATH,
    splib_raw = splib_raw_dict,
    splib_cr = splib_cr_dict,    
    # wavelength=wavelength,    # (C,)
    label_raw=label_raw_dict, # {cid: (N, C)}
    label_cr = label_cr_dict
)

print("saved to:", OUT_PATH)
# print("wavelength shape:", wavelength.shape)
print("mean_spec shape:", mean_spec.shape)
