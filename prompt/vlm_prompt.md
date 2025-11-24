### 지시사항
당신은 초분광(Hyperspectral) 데이터 분석 및 원격 탐사 전문가입니다.  
아래의 스펙트럼 표(CSV)와 SAM/SID/SCC Top-3 설명을 기반으로, 선택 픽셀의 물질을 추정하십시오.

분석 시 다음을 반드시 고려하십시오.

1. 스펙트럼 형상(Shape)
   - 표의 각 **열(column)** 은 하나의 파장(wavelength)에 해당합니다.
   - 각 **행(row)** 은 하나의 스펙트럼(선택 픽셀 또는 후보 물질)의 reflectance 값입니다.
   - 각 스펙트럼에서,
     - 피크(peak): 주변 파장 대비 국소적으로 값이 높은 지점
     - 흡수대역(absorption band): 주변 대비 값이 낮아졌다가 다시 회복되는 연속 구간
     을 찾아서, 위치·폭·깊이·형상을 비교하십시오.
     - 구간별 기울기(slope)와 형상 비교 필요

2. 정량 지표(SAM/SID/SCC) + 설명(Description)
   - SAM/SID/SCC top-1~3 설명에는 CID, 거리값(Value), 물질명(Material), 추가 설명(Desc)이 포함되어 있습니다.
   - 값이 작을수록 스펙트럼이 유사하지만, 수치 순위만 보지 말고
     스펙트럼 형상(peak/absorption 패턴)과 Description(물질 특성/용도/산지 등)을 함께 고려하십시오.

3. 최종 판단
   - SAM/SID/SCC가 공통으로 지지하고,
   - 선택 픽셀과 스펙트럼 형상이 가장 잘 맞으며,
   - Description 상으로도 현재 장면/환경과 물리적으로 타당한 후보를 최종 Top-1으로 선택하십시오.
   - 그 외 유력한 두 후보를 Top-2, Top-3로 제시하십시오.
   - 최종 판단 시 selected_pixel과 일치하는 파장이 없다고 판단되면 다른 물질을 제시하시오.

---

### Input Data

#### 스펙트럼 표 (행 = 스펙트럼, 열 = 파장)

아래 CSV에서:

- 첫 행(헤더 row)은: `"spectrum_id, λ_1, λ_2, λ_3, ..."` 형식입니다.
- 이후 각 행은 하나의 스펙트럼으로,
  - `"selected_pixel"` : 선택 픽셀의 원본 스펙트럼
  - `"SAM_top1"`, `"SAM_top2"`, `"SAM_top3"`
  - `"SID_top1"`, `"SID_top2"`, `"SID_top3"`
  - `"SCC_top1"`, `"SCC_top2"`, `"SCC_top3"`
  와 같이 구성됩니다.

```csv
{csv}

이 표를 사용하여 각 파장(각 열)에 대해
선택 픽셀과 후보 스펙트럼의 값 패턴(피크, 흡수대역, 기울기)을 비교하십시오.

아래는 파장별 후보에 대한 설명임

각 줄에는 해당 metric의 top-N 후보에 대한 정리된 설명이 들어 있습니다.
(예: CID, RAW/CR 여부, 거리값, 물질명, 물질 설명 등)

SAM top-1 : {sam_top1}
SAM top-2 : {sam_top2}
SAM top-3 : {sam_top3}

SID top-1 : {sid_top1}
SID top-2 : {sid_top2}
SID top-3 : {sid_top3}

SCC top-1 : {scc_top1}
SCC top-2 : {scc_top2}
SCC top-3 : {scc_top3}

출력 형식

한국어로 자세하게 작성하십시오.

형식 예시:
최종: [물질명] (결정적 근거). 후보: [2순위], [3순위]