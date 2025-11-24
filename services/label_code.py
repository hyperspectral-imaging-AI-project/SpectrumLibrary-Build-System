# services/label_code.py
import hashlib, datetime
import numpy as np

# 크록포드 Base32 (OILU 제거) – 짧고 안정적인 표기
_ALPH = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

def _base32_crockford(b: bytes, length: int) -> str:
    n = int.from_bytes(b, "big", signed=False)
    out = []
    while n and len(out) < length:
        out.append(_ALPH[n & 31])
        n >>= 5
    while len(out) < length:
        out.append('0')
    return ''.join(reversed(out))

def make_label_code(*, image_cd: int, y: int, x: int, mtrl_cd: int,
                    rfl_raw: np.ndarray | None, decimals: int = 6,
                    when: datetime.datetime | None = None) -> str:
    """
    동일 (image_cd, y, x, rfl_raw) → 항상 같은 코드.
    rfl_raw가 None이면 좌표+imgcd만으로 생성(권장X).
    """
    when = when or datetime.datetime.utcnow()
    yy = when.year % 100
    jjj = int(when.strftime("%j"))  # 001~366

    h = hashlib.sha1()
    h.update(str(int(image_cd)).encode())
    h.update(b":"); h.update(str(int(y)).encode())
    h.update(b":"); h.update(str(int(x)).encode())
    h.update(b":"); h.update(str(int(mtrl_cd)).encode())
    if rfl_raw is not None:
        v = np.asarray(rfl_raw, dtype=np.float32)
        if v.ndim != 1: v = v.reshape(-1)
        vq = np.round(v, decimals=decimals)   # 소수점 고정(재현성)
        h.update(vq.tobytes())
    digest = h.digest()
    k8 = _base32_crockford(digest[:5], 8)     # 5바이트 → 8문자

    return f"LBL-{int(image_cd):06d}-{yy:02d}{jjj:03d}-{int(mtrl_cd)}-{k8}"
