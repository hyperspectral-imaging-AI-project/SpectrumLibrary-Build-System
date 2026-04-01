# Third Pixel Classification Tool

PyQt5 기반의 초분광 이미지(HSI) 처리 및 픽셀 분류 데스크톱 애플리케이션입니다.

## 📋 목차

- [주요 기능](#주요-기능)
- [시스템 요구사항](#시스템-요구사항)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [환경 설정](#환경-설정)
- [프로젝트 구조](#프로젝트-구조)
- [개발 가이드](#개발-가이드)
- [문제 해결](#문제-해결)

---

## 🎯 주요 기능

### 이미지 처리
- **ENVI HDR/RAW 파일 로드**: 초분광 이미지 데이터 로드 및 표시
- **RGB 가시화**: 초분광 데이터를 RGB 이미지로 변환하여 표시
- **밴드 뷰어**: 개별 밴드 이미지 확인 및 분석
- **카메라 관리**: 카메라 정보 등록 및 조회

### 픽셀 분류 및 라벨링
- **픽셀 라벨링**: 개별 픽셀에 클래스 라벨 할당
- **사용자 지정 라벨링**: 사용자가 직접 픽셀을 선택하여 라벨링
- **클래스맵 라벨링**: 분류 결과를 기반으로 한 라벨링
- **라벨링 데이터베이스 검색**: 서버에서 라벨링 데이터 검색
- **AI 기반 라벨 추천**: Vision Language Model을 활용한 라벨 추천

### 분류 및 분석
- **Top-K 유사도 분석**: 특정 픽셀의 상위 K개 유사 클래스 표시
- **자동 분류**: 스펙트럼 라이브러리를 이용한 자동 분류
- **임계값 설정**: 분류 결과 필터링을 위한 임계값 조정
- **분류맵 생성**: 전체 이미지에 대한 분류 결과 시각화
- **분석 선택**: 다양한 분석 모드 선택 및 설정

### ROI(Region of Interest) 관리
- **작업 영역 설정**: 분석할 영역을 사각형 또는 자유곡선으로 지정
- **클래스 ROI**: 클래스별 ROI 영역 관리
- **레이어 관리**: 다중 레이어 오버레이 및 가시성 제어

### 스펙트럼 분석
- **스펙트럼 라이브러리**: 참조 스펙트럼 관리 및 활용
- **Continuum Removal**: 스펙트럼 전처리 기능
- **스펙트럼 비교**: 다양한 메트릭(SAD, SID, SAM 등)을 이용한 유사도 계산
- **언믹싱(Unmixing)**: 혼합 픽셀 분해 분석

### 고급 기능
- **영역 확장(Region Growing)**: 시드 픽셀에서 유사 픽셀로 영역 확장
- **확산 기반 선택**: 확산 알고리즘을 이용한 영역 선택
- **투명도 오버레이**: 투명도 기반 시각화
- **워크스페이스 관리**: 작업 환경 저장 및 불러오기

### 데이터 관리
- **캐시 시스템**: 처리된 데이터를 `.npz` 및 `.info` 파일로 캐싱
- **라벨 저장**: 라벨링 결과를 로컬 또는 서버에 저장
- **Personal/Server 모드**: 로컬 저장 또는 서버 API 연동 선택
- **SFTP 업로드**: 원격 서버로 파일 업로드 지원

---

## 💻 시스템 요구사항

- **Python**: 3.8 이상
- **운영체제**: Windows 10/11, Linux, macOS
- **메모리**: 최소 4GB RAM (대용량 이미지 처리 시 8GB 이상 권장)
- **디스크**: 최소 500MB 여유 공간

---

## 📦 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd third_pixel_classification_tool
```

### 2. 가상환경 생성 및 활성화 (권장)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install PyQt5>=5.15.0
pip install numpy>=1.20.0
pip install scipy>=1.7.0
pip install matplotlib>=3.4.0
pip install requests>=2.25.0
pip install Pillow>=8.0.0
pip install python-dotenv>=0.19.0
pip install paramiko>=2.7.0
pip install openai>=1.0.0  # VLM 기능 사용 시
pip install hdf5storage>=0.1.18  # HDF5 .mat 파일 지원 (선택사항)
```

또는 `requirements.txt` 파일이 있다면:

```bash
pip install -r requirements.txt
```

### 4. 실행

```bash
python app.py
```

---

## 🚀 사용 방법

### 이미지 로드

1. **File > Image Load** 메뉴 선택
2. HDR 파일과 RAW 파일 경로 지정
3. 이미지 축 설정 (Height, Width, Bands)
4. 파장 범위 입력 (Start/End Wavelength)
5. 카메라 선택 (선택사항)
6. **Check** 버튼으로 이미지 정보 확인
7. **OK** 버튼으로 이미지 로드

### 픽셀 라벨링

1. **Tools > Pixel Labeling** 메뉴 선택
2. 라벨링 모드 선택:
   - **사용자 지정 라벨링**: 직접 픽셀 선택
   - **클래스맵 라벨링**: 분류 결과 기반
   - **추천 라벨링**: AI 기반 추천
3. 이미지에서 픽셀 클릭하여 라벨 할당
4. **등록** 버튼으로 라벨 저장

### 분류 수행

1. **Tools > Pixel Classification** 메뉴 선택
2. 스펙트럼 라이브러리 로드 확인
3. 임계값 설정 (Strict, Base)
4. **적용** 버튼으로 분류 실행
5. 분류 결과 확인 및 조정

### ROI 설정

1. **ROI 툴바**에서 모드 선택:
   - **Rect**: 사각형 영역
   - **Poly**: 자유곡선 영역
   - **Clear**: 영역 초기화
2. 이미지에서 드래그하여 영역 지정
3. 작업 영역으로 설정된 영역만 분석 대상

### 카메라 관리

1. **File > Camera Add** 메뉴 선택
2. 카메라 정보 입력:
   - 카메라 이름
   - 파장대역 (예: 400,500,600)
   - FWHM (예: 8.1,8.1,8.1)
3. **카테고리 가져오기** 버튼으로 기존 카메라 목록 조회
4. **저장** 버튼으로 카메라 등록

---

## ⚙️ 환경 설정

### Personal 모드 (로컬 저장)

Personal 모드에서는 모든 데이터가 로컬에 저장됩니다.

```python
# 코드에서 설정
self.user_type = "personal"
```

### Server 모드 (API 연동)

Server 모드에서는 외부 API를 통해 데이터를 관리합니다.

`.env` 파일 생성 (프로젝트 루트):

```env
# API 엔드포인트 설정
camera_list_url=http://your-api-server/program-service/camera-list
camera_add_url=http://your-api-server/program-service/camera-add
material_code_name_url=http://your-api-server/program-service/material-list/name
material_add_url=http://your-api-server/program-service/material-add
image_check_url=http://your-api-server/program-service/image-exist
image_save_url=http://your-api-server/program-service/image-save
label_add_url=http://your-api-server/program-service/label-add
spectrum_library_url=http://your-api-server/program-service/spectrum-library-list
labeling_data_url=http://your-api-server/program-service/label-list
unmixing_inference_url=http://your-api-server/program-service/inference/unmixing

# OpenAI API (VLM 기능 사용 시)
OPENAI_API_KEY=your-openai-api-key

# SFTP 설정 (선택사항)
SFTP_HOST=gnew-office.tplinkdns.com
SFTP_PORT=22
SFTP_USER=your-username
SFTP_PASSWORD=your-password
```

또는 환경변수로 직접 설정:

```bash
# Windows
set camera_list_url=http://your-api-server/program-service/camera-list

# Linux/macOS
export camera_list_url=http://your-api-server/program-service/camera-list
```

---

## 📁 프로젝트 구조

```
third_pixel_classification_tool/
├── app.py                      # 애플리케이션 엔트리 포인트
├── README.md                   # 프로젝트 문서
│
├── views/                      # UI 뷰 컴포넌트
│   ├── main_window.py         # 메인 윈도우
│   ├── mapview.py             # 지도 뷰 (이미지 표시)
│   ├── dialogs/               # 다이얼로그
│   │   ├── image_load.py      # 이미지 로드 다이얼로그
│   │   ├── pixel_labeling_dialog.py
│   │   ├── user_labeling_dialog.py
│   │   ├── classmap_labeling_dialog.py
│   │   ├── pixel_classification.py
│   │   ├── analysis_selection_dialog.py
│   │   ├── camera_add.py
│   │   ├── material_add_dialog.py
│   │   ├── unmixing_dialog.py
│   │   ├── diffusion_dialog.py
│   │   ├── search_labeling_database.py
│   │   ├── recommend_label_wizard.py
│   │   ├── workspace_dialog.py
│   │   ├── viewer_band_dialog.py
│   │   ├── endmember_detail_dialog.py
│   │   └── labeling_candidate_review.py
│   └── docks/                 # 도크 위젯
│       ├── layer_dock.py      # 레이어 관리 도크
│       ├── pixel_classification_dock.py
│       └── unmixing_dock.py
│
├── controllers/               # 컨트롤러 (비즈니스 로직)
│   ├── pixel_click.py        # 픽셀 클릭 처리
│   ├── roi_controller.py     # ROI 관리
│   ├── class_roi_controller.py
│   └── analysis_selection_controller.py
│
├── services/                  # 서비스 레이어
│   ├── layer_manager.py      # 레이어 관리
│   ├── resampling_cache.py   # 캐시 관리
│   ├── pixel_knn.py          # KNN 분류
│   ├── classmap_render.py    # 클래스맵 렌더링
│   ├── paletteService.py     # 팔레트 관리
│   ├── transparency_service.py
│   ├── region_growing_service.py
│   ├── opacity.py
│   ├── spec_library.py
│   ├── label_store.py
│   ├── label_code.py
│   ├── io_store.py
│   └── overlay_temp.py
│
├── core/                     # 핵심 알고리즘
│   ├── autoclass.py         # 자동 분류
│   ├── metrics.py           # 유사도 메트릭
│   ├── resampling_service.py # 리샘플링
│   ├── vis.py               # 시각화 유틸리티
│   ├── region_growing.py    # 영역 확장
│   └── vlm_generation.py     # VLM 생성
│
├── data/                     # 데이터 접근
│   └── db.py                # API 연동
│
├── dataio/                   # 데이터 입출력
│   └── sidecar.py           # 사이드카 파일 처리
│
├── models/                   # 데이터 모델
│   └── labels.py            # 라벨 모델
│
├── ui/                       # UI 파일 (.ui)
│   ├── main_window.ui
│   ├── image_load.ui
│   ├── pixel_labeling_dialog.ui
│   ├── user_labeling_dialog.ui
│   ├── classmap_labeling_dialog.ui
│   ├── layer_dock.ui
│   ├── pixel_classification_dock.ui
│   ├── unmixing_dock.ui
│   └── ... (기타 UI 파일들)
│
├── icons/                    # 아이콘 리소스
│   ├── gnew_icon.png
│   └── roi_plus.png
│
├── test/                     # 테스트 스크립트
│   ├── api_test.py
│   ├── test_region_growing.py
│   ├── load_npz.py
│   └── ... (기타 테스트 파일들)
│
├── examples/                 # 예제 코드
│   └── region_growing_example.py
│
├── prompt/                   # 프롬프트 문서
│   └── vlm_prompt.md
│
├── logs/                     # 로그 파일
├── map_sample/               # 샘플 맵 데이터
└── label_local/              # 로컬 라벨 데이터
```

### 아키텍처

이 프로젝트는 **MVC(Model-View-Controller) 패턴**을 기반으로 합니다:

- **View Layer**: UI 컴포넌트 (`views/`)
- **Controller Layer**: 이벤트 처리 (`controllers/`)
- **Service Layer**: 비즈니스 로직 (`services/`)
- **Core Layer**: 핵심 알고리즘 (`core/`)
- **Data Layer**: 데이터 접근 (`data/`, `dataio/`, `models/`)

---

## 🔧 개발 가이드

### 코드 스타일

- **PEP 8** 준수
- **타입 힌트** 사용 권장
- **독스트링** 작성 (클래스 및 주요 함수)

### 주요 아키텍처 패턴

- **MVC 패턴**: Model-View-Controller 구조
- **서비스 레이어**: 비즈니스 로직 분리
- **의존성 주입**: 느슨한 결합 유지
- **시그널/슬롯**: PyQt5 시그널을 통한 이벤트 처리

### 로깅

애플리케이션 로그는 `logs/app.log`에 저장됩니다.

```python
import logging
logging.info("정보 메시지")
logging.warning("경고 메시지")
logging.error("오류 메시지")
```

### 캐시 시스템

처리된 데이터는 다음 형식으로 캐싱됩니다:

- **`.npz` 파일**: NumPy 배열 데이터 (스펙트럼 라이브러리, 라벨 등)
- **`.info` 파일**: JSON 형식 메타데이터 (클래스 정보, 설정 등)

캐시 파일 위치: `{원본파일경로}.resample.npz`, `{원본파일경로}.resample.info`

### 신규 기능 추가 가이드

1. **UI 추가**: `ui/` 디렉토리에 `.ui` 파일 생성 (Qt Designer 사용)
2. **다이얼로그 추가**: `views/dialogs/` 디렉토리에 Python 파일 생성
3. **서비스 추가**: `services/` 디렉토리에 서비스 클래스 생성
4. **컨트롤러 추가**: `controllers/` 디렉토리에 컨트롤러 클래스 생성

---

## 🐛 문제 해결

### 이미지가 로드되지 않을 때

- HDR 파일과 RAW 파일 경로 확인
- 이미지 축 설정이 올바른지 확인
- 파일 형식이 ENVI 표준인지 확인
- 파일 크기가 메모리 용량을 초과하지 않는지 확인

### 분류가 작동하지 않을 때

- 스펙트럼 라이브러리가 로드되었는지 확인
- 파장 범위가 일치하는지 확인
- 캐시 파일이 손상되었을 수 있으므로 삭제 후 재시도
- 로그 파일(`logs/app.log`) 확인

### API 연결 오류

- `.env` 파일의 URL 설정 확인
- 네트워크 연결 상태 확인
- API 서버 상태 확인
- 환경변수가 올바르게 로드되었는지 확인

### 성능 문제

- 대용량 이미지의 경우 ROI를 설정하여 작업 영역 제한
- 캐시 파일 활용
- 메모리 사용량 모니터링
- 필요시 이미지 크기 축소

---

## 📝 주요 변경 이력

### Version 0.1.0
- 초기 릴리스
- 기본 이미지 로드 및 표시 기능
- 픽셀 라벨링 기능
- 분류 기능
- ROI 관리 기능
- 레이어 시스템
- Personal/Server 모드 지원

---

## 📄 라이선스

Copyright (c) 2024 GNEWSOFT. All rights reserved.

---

## 🐛 개선 및 문제점

프로젝트의 지속적인 개선을 위해 현재 파악된 문제점 및 개선 사항은 다음과 같습니다.

### 1. 미사용 코드 (Dead Code)
- **`services/transparency_service.py`**: 클래스가 초기화되지만, 애플리케이션 내에서 해당 서비스의 메서드가 호출되지 않아 현재 미사용 중인 것으로 보입니다. 이는 불필요한 코드 복잡성을 증가시킵니다.
- **`services/overlay_temp.py`**: 임시 오버레이 서비스로 명시되어 있으며, 현재 기능적으로 활용되지 않는다면 제거를 고려해야 합니다.

### 2. MVC 패턴 불일치 가능성
- `views/dialogs/`와 `controllers/` 디렉토리 간의 역할 분담이 모호하거나 기능이 중복 구현되었을 가능성이 있습니다. 이는 코드의 유지보수성과 확장성을 저해할 수 있습니다.

### 3. `map_sample/` 디렉토리의 데이터 비일관성
- `map_sample/` 내의 샘플 데이터 파일명들이 체계적이지 않고, 어떤 파일이 필수 샘플이고 어떤 것이 임시 생성물인지 구별하기 어렵습니다. 이는 데이터 관리 및 프로젝트 이해에 어려움을 줍니다.

---

## 👥 기여자

- GNEWSOFT Development Team

---

## 📧 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

**Note**: 이 문서는 프로젝트의 현재 상태를 반영하고 있으며, 지속적으로 업데이트됩니다.
