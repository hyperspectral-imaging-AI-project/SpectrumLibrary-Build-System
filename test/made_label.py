import json
import numpy as np
from pathlib import Path

# ENVI header(.hdr) 읽기
def read_envi_hdr(hdr_path: str | Path) -> dict:
    """
    ENVI .hdr 파일을 파싱해서 dict로 반환
    예) {'samples': 512, 'lines': 512, 'bands': 224, 'interleave': 'bsq', 'data type': 4, ...}
    """
    hdr_path = Path(hdr_path)
    header = {}

    with hdr_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # "ENVI" 첫 줄은 버리고, key = value 형태 파싱
    for line in lines:
        line = line.strip()
        if not line or line.lower().startswith("envi"):
            continue
        if "=" not in line:
            continue

        k, v = line.split("=", 1)
        k = k.strip().lower()      # key는 소문자로 정규화
        v = v.strip()

        # { } 로 감싼 값 제거
        if v.startswith("{") and v.endswith("}"):
            v = v[1:-1].strip()

        header[k] = v

    # 숫자 필드는 바로 int로 변환
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
    ENVI data type 번호 → numpy dtype 매핑
    (가장 자주 쓰이는 것만)
      1: uint8
      2: int16
      3: int32
      4: float32
      5: float64
     12: uint16
     13: uint32
     14: int64
     15: uint64
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
    ENVI .hdr / .raw 로부터 초분광 큐브를 읽어서
    (lines, samples, bands) 형태의 numpy 배열로 반환.
    """
    hdr = read_envi_hdr(hdr_path)
    samples = int(hdr["samples"])
    lines   = int(hdr["lines"])
    bands   = int(hdr["bands"])
    interleave = hdr.get("interleave", "bsq").lower()
    data_type  = int(hdr["data type"])
    header_offset = int(hdr.get("header offset", 0))

    dtype = _envi_dtype(data_type)

    raw_path = Path(raw_path)
    with raw_path.open("rb") as f:
        if header_offset > 0:
            f.seek(header_offset)
        data = np.fromfile(f, dtype=dtype)

    # 총 픽셀 수 검증
    expected = samples * lines * bands
    if data.size != expected:
        raise ValueError(
            f"raw 데이터 크기가 헤더 정보와 맞지 않습니다. "
            f"expected={expected}, actual={data.size}"
        )

    # interleave 에 따라 reshape & transpose
    if interleave == "bsq":
        # (bands, lines, samples) → (lines, samples, bands)
        cube = data.reshape((bands, lines, samples)).transpose(1, 2, 0)
    elif interleave == "bil":
        # (lines, bands, samples) → (lines, samples, bands)
        cube = data.reshape((lines, bands, samples)).transpose(0, 2, 1)
    elif interleave == "bip":
        # (lines, samples, bands)
        cube = data.reshape((lines, samples, bands))
    else:
        raise ValueError(f"지원하지 않는 interleave 형식: {interleave}")

    # float32 로 통일
    cube = cube.astype(np.float32, copy=False)
    return cube

def compute_mean_spectrum(cube: np.ndarray) -> np.ndarray:
    """
    cube: (lines, samples, bands)
    전체 픽셀 기준 평균 스펙트럼 → (bands,) 반환
    """
    if cube.ndim != 3:
        raise ValueError(f"cube는 (lines, samples, bands) 3D 배열이어야 합니다. got shape={cube.shape}")
    lines, samples, bands = cube.shape

    # (N_pixels, bands) 형태로 변환 후 평균
    pixels = cube.reshape(-1, bands)  # (lines * samples, bands)
    mean_spec = pixels.mean(axis=0)
    return mean_spec.astype(np.float32)

def save_mean_spectrum_npz(
    hdr_path: str | Path,
    raw_path: str | Path,
    out_npz_path: str | Path,
    cid: int = 0,
    class_name: str = "mean_all_pixels",
):
    """
    1) .hdr / .raw 를 읽어서 초분광 큐브 생성
    2) 전체 픽셀 기준 평균 스펙트럼 계산
    3) cid=0 (기본값) 으로 라벨링 데이터에 저장
    4) 저장 형식은 예시 resample.npz 의 key 이름(splib_raw, label_raw 등)에 맞추되
       내부 object 구조는 '추측' 기반 최소 형태로 구성
    """
    hdr_path = Path(hdr_path)
    raw_path = Path(raw_path)
    out_npz_path = Path(out_npz_path)

    # 1) 큐브 로드
    cube = load_envi_raw(hdr_path, raw_path)

    # 2) 평균 스펙트럼
    mean_spec = compute_mean_spectrum(cube)  # (bands,)

    # 3) 라벨 object 구성 (추측 포맷)
    label_entry = {
        "cid": int(cid),
        "name": class_name,
        "spectrum": mean_spec,
        "meta": {
            "source": str(hdr_path.name),
            "note": "전체 픽셀 평균 스펙트럼",
        },
    }

    # object 배열로 감싸기 (예시 resample.npz 가 object 배열을 쓰고 있어서 맞춰줌)
    splib_raw = np.array([], dtype=object)      # 여기서는 비움
    splib_cr  = np.array([], dtype=object)      # 여기서도 비움
    label_raw = np.array([label_entry], dtype=object)
    label_cr  = np.array([], dtype=object)
    label_coords = np.array([], dtype=object)

    # 4) npz 저장
    np.savez(
        out_npz_path,
        splib_raw=splib_raw,
        splib_cr=splib_cr,
        label_raw=label_raw,
        label_cr=label_cr,
        label_coords=label_coords,
    )

    return out_npz_path, mean_spec


hdr_path = 'E:\시연\FM_vec\C1-3\C1-3_vec.hdr'
out_npz = 'E:\시연\FM_vec\C1-3_vec.hdr.resample.npz'
raw_path = 'E:\시연\FM_vec\C1-3\C1-3_vec.raw'

save_mean_spectrum_npz()