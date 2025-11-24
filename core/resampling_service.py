import numpy as np
import spectral
import ast
from data.db import search_spectrum_library, search_labeling_data
def perform_continuum_removal(spectrum, wavelength, *, 
                    method='convex_hull', mode='reflectance', eps=1e-12):
    """
    Continuum Removal (CR)
    - continuum: 스펙트럼 상단의 볼록껍질(upper convex hull) 조각보간(선형)으로 구성
    - CR 결과: reflectance 기준은 S / C (권장), absorbance 기준은 S - C
    
    Parameters
    ----------
    wavelength : (N,) array-like
        파장(단조 증가 권장; 단조가 아니면 내부에서 정렬)
    spectrum : (N,) array-like
        스펙트럼 (반사율 또는 흡광도 등)
    method : {'convex_hull'}
        현재는 볼록껍질 기반만 구현(ENVI/Clark 방식)
    mode : {'reflectance', 'absorbance'}
        'reflectance'  →  CR = S / C  (권장; 반사/복사 기준)
        'absorbance'   →  CR = S - C  (흡수계수/흡광도 기준)
    eps : float
        0 나눗셈 방지용 최소값
    """
    wl = np.asarray(wavelength).astype(float)
    y  = np.asarray(spectrum).astype(float)

    # 유효/정렬/중복 제거
    m = np.isfinite(wl) & np.isfinite(y)
    wl, y = wl[m], y[m]
    order = np.argsort(wl)
    wl, y = wl[order], y[order]
    # 중복 파장 제거(마지막 값 우선)
    uniq, idx = np.unique(wl, return_index=True)
    wl, y = wl[idx], y[idx]

    if wl.size < 3:
        raise ValueError("파장 샘플이 3개 미만이면 컨티늄을 정의하기 어렵습니다.")

    # --- Upper Convex Hull (Andrew’s monotone chain 변형, 상단 껍질만) ---
    def cross(o, a, b):
        # 2D 외적 (o->a) x (o->b)
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    pts = np.column_stack([wl, y])
    # 상단 껍질을 왼→오 순으로 구축
    upper = []
    for p in pts:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) >= 0:
            upper.pop()
        upper.append(tuple(p))
    upper = np.array(upper)  # (K,2), K>=2

    # 첫/끝 파장이 껍질에 반드시 포함되도록 보정
    if upper[0,0] != wl[0]:
        upper = np.vstack(([wl[0], y[0]], upper))
    if upper[-1,0] != wl[-1]:
        upper = np.vstack((upper, [wl[-1], y[-1]]))

    # --- 선형 보간으로 continuum 계산 ---
    cw = upper[:,0]
    cy = upper[:,1]
    continuum = np.interp(wl, cw, cy)

    # --- Continuum Removal ---
    if mode == 'reflectance':
        cr = y / np.maximum(continuum, eps)  # S/C (권장)
    elif mode == 'absorbance':
        cr = y - continuum                   # S - C (흡광도 계열)
    else:
        raise ValueError("mode는 {'reflectance','absorbance'} 중 하나여야 합니다.")

    # 원래 순서 복원: 입력이 비단조였을 수 있으므로
    inv = np.argsort(order[idx])  # idx로 고유화한 후의 역정렬 인덱스
    cr = cr[inv]
    continuum = continuum[inv]
    return cr, continuum

def resampling(row, spectrum_url, labeling_url):
    
    st_wv = row.get('ST_WV') or [] 
    ed_wv = row.get('ED_WV') or []
    fwhm = row.get('FWHM') or []
    wavelength = row.get('WV') or [] 
    
    spectrum_result = search_spectrum_library(
        st_wv = st_wv, ed_wv = ed_wv, base_url = spectrum_url
    )
    
    labeling_result = search_labeling_data(
        st_wv = st_wv, ed_wv = ed_wv, base_url = labeling_url
    )
    print('라벨링 데이터 존재 여부 확인:',labeling_result)
    spectrum_library_raw = []
    spectrum_library_cr = []
    labeling_result_raw = []
    labeling_result_cr = []
    
    
    print(labeling_result)
        
    for k in labeling_result:
        lib_name = k.get('mtrl_cd')
        lib_wvs = ast.literal_eval(k.get('wv'))
        lib_fwhm = ast.literal_eval(k.get('fwhm'))
        lib_ref = ast.literal_eval(k.get('rfl'))
        
        BandResampler = spectral.BandResampler(centers1 = lib_wvs, centers2 = wavelength, fwhm1 = lib_fwhm, fwhm2 = fwhm)
        resampled_ref = BandResampler(lib_ref)        
        labeling_result_raw.append(
            {
                'mtrl_cd':lib_name,
                'wavelength':wavelength,
                'fwhm':fwhm,
                'ref':resampled_ref
            }
        )
                
        ## CR 버전
        resampled_cr_ref, _ = perform_continuum_removal(resampled_ref, wavelength)
        labeling_result_cr.append(
            {
                'mtrl_cd':lib_name,
                'wavelength':wavelength,
                'fwhm':fwhm,
                'ref':resampled_cr_ref
            }
        )
        
    for i in spectrum_result:
        lib_name = i.get('mtrl_cd')
        lib_wvs = ast.literal_eval(i.get('wv'))
        lib_fwhm = ast.literal_eval(i.get('fwhm'))
        lib_ref = ast.literal_eval(i.get('rfl'))
        
        
        ## Raw 버전
        BandResampler = spectral.BandResampler(centers1 = lib_wvs, centers2 = wavelength, fwhm1 = lib_fwhm, fwhm2 = fwhm)
        resampled_ref = BandResampler(lib_ref)
        spectrum_library_raw.append({
            "mtrl_cd":lib_name,
            "wavelength":wavelength,
            "fwhm":fwhm,
            "ref":resampled_ref
        })
        
        ## Cr 버전
        resampled_cr_ref ,_ = perform_continuum_removal(resampled_ref, wavelength)
        spectrum_library_cr.append({
            "mtrl_cd": lib_name,
            "wavelength": wavelength,
            "fwhm": fwhm,
            "ref": np.asarray(resampled_cr_ref, dtype=float)
        })
    return spectrum_library_raw, spectrum_library_cr, labeling_result_raw, labeling_result_cr