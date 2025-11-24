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
- [라이선스](#라이선스)

## 🎯 주요 기능

### 이미지 처리
- **ENVI HDR/RAW 파일 로드**: 초분광 이미지 데이터 로드 및 표시
- **RGB 가시화**: 초분광 데이터를 RGB 이미지로 변환하여 표시
- **밴드 뷰어**: 개별 밴드 이미지 확인 및 분석

### 픽셀 분류 및 라벨링
- **픽셀 라벨링**: 개별 픽셀에 클래스 라벨 할당
- **사용자 지정 라벨링**: 사용자가 직접 픽셀을 선택하여 라벨링
- **클래스맵 라벨링**: 분류 결과를 기반으로 한 라벨링
- **신규 클래스 등록**: 새로운 물질 클래스 생성 및 관리

### 분류 및 분석
- **Top-K 유사도 분석**: 특정 픽셀의 상위 K개 유사 클래스 표시
- **자동 분류**: 스펙트럼 라이브러리를 이용한 자동 분류
- **임계값 설정**: 분류 결과 필터링을 위한 임계값 조정
- **분류맵 생성**: 전체 이미지에 대한 분류 결과 시각화

### ROI(Region of Interest) 관리
- **작업 영역 설정**: 분석할 영역을 사각형 또는 자유곡선으로 지정
- **클래스 ROI**: 클래스별 ROI 영역 관리
- **레이어 관리**: 다중 레이어 오버레이 및 가시성 제어

### 스펙트럼 분석
- **스펙트럼 라이브러리**: 참조 스펙트럼 관리 및 활용
- **Continuum Removal**: 스펙트럼 전처리 기능
- **스펙트럼 비교**: 다양한 메트릭(SAD, SID, SAM 등)을 이용한 유사도 계산

### 데이터 관리
- **캐시 시스템**: 처리된 데이터를 `.npz` 및 `.info` 파일로 캐싱
- **라벨 저장**: 라벨링 결과를 로컬 또는 서버에 저장
- **Personal/Server 모드**: 로컬 저장 또는 서버 API 연동 선택

## 💻 시스템 요구사항

- **Python**: 3.8 이상
- **운영체제**: Windows 10/11, Linux, macOS
- **메모리**: 최소 4GB RAM (대용량 이미지 처리 시 8GB 이상 권장)
- **디스크**: 최소 500MB 여유 공간

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
pip install -r requirements.txt
```

주요 의존성 패키지:
- `PyQt5`: GUI 프레임워크
- `numpy`: 수치 연산
- `scipy`: 과학 계산
- `matplotlib`: 그래프 및 스펙트럼 시각화
- `python-dotenv`: 환경변수 관리 (선택사항)
- `requests`: HTTP API 호출 (서버 모드 사용 시)

### 4. 실행

```bash
python app.py
```

## 🚀 사용 방법

### 이미지 로드

1. **File > Image Load** 메뉴 선택
2. HDR 파일과 RAW 파일 경로 지정
3. 이미지 축 설정 (Height, Width, Bands)
4. 파장 범위 입력 (Start/End Wavelength)
5. **Check** 버튼으로 이미지 정보 확인
6. **OK** 버튼으로 이미지 로드

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
camera_list_url=http://your-api-server/camera-list
material_code_name_url=http://your-api-server/material-code-name
material_add_url=http://your-api-server/material-add
image_check_url=http://your-api-server/image-check
image_save_url=http://your-api-server/image-save
label_add_url=http://your-api-server/label-add
spectrum_library_url=http://your-api-server/spectrum-library
labeling_data_url=http://your-api-server/labeling-data
```

또는 환경변수로 직접 설정:

```bash
# Windows
set camera_list_url=http://your-api-server/camera-list

# Linux/macOS
export camera_list_url=http://your-api-server/camera-list
```

## 📁 프로젝트 구조

```
third_pixel_classification_tool/
├── app.py                      # 애플리케이션 엔트리 포인트
├── views/                      # UI 뷰 컴포넌트
│   ├── main_window.py         # 메인 윈도우
│   ├── mapview.py             # 지도 뷰 (이미지 표시)
│   ├── dialogs/               # 다이얼로그
│   │   ├── image_load.py      # 이미지 로드 다이얼로그
│   │   ├── pixel_labeling_dialog.py
│   │   ├── user_labeling_dialog.py
│   │   └── ...
│   └── docks/                 # 도크 위젯
│       ├── layer_dock.py      # 레이어 관리 도크
│       └── pixel_classification_dock.py
├── controllers/               # 컨트롤러 (비즈니스 로직)
│   ├── pixel_click.py         # 픽셀 클릭 처리
│   ├── roi_controller.py      # ROI 관리
│   └── ...
├── services/                  # 서비스 레이어
│   ├── layer_manager.py       # 레이어 관리
│   ├── resampling_cache.py    # 캐시 관리
│   ├── pixel_knn.py          # KNN 분류
│   └── ...
├── core/                      # 핵심 알고리즘
│   ├── autoclass.py          # 자동 분류
│   ├── metrics.py            # 유사도 메트릭
│   ├── resampling_service.py # 리샘플링
│   └── ...
├── data/                      # 데이터 접근
│   └── db.py                 # API 연동
├── models/                    # 데이터 모델
│   └── labels.py             # 라벨 모델
├── ui/                        # UI 파일 (.ui)
├── icons/                     # 아이콘 리소스
├── logs/                      # 로그 파일
└── test/                      # 테스트 스크립트
```

## 🔧 개발 가이드

### 코드 스타일

- **PEP 8** 준수
- **타입 힌트** 사용 권장
- **독스트링** 작성 (클래스 및 주요 함수)

### 주요 아키텍처 패턴

- **MVC 패턴**: Model-View-Controller 구조
- **서비스 레이어**: 비즈니스 로직 분리
- **의존성 주입**: 느슨한 결합 유지

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

1. **UI 추가**: `ui/` 디렉토리에 `.ui` 파일 생성
2. **다이얼로그 추가**: `views/dialogs/` 디렉토리에 Python 파일 생성
3. **서비스 추가**: `services/` 디렉토리에 서비스 클래스 생성
4. **컨트롤러 추가**: `controllers/` 디렉토리에 컨트롤러 클래스 생성

## 🐛 문제 해결

### 이미지가 로드되지 않을 때

- HDR 파일과 RAW 파일 경로 확인
- 이미지 축 설정이 올바른지 확인
- 파일 형식이 ENVI 표준인지 확인

### 분류가 작동하지 않을 때

- 스펙트럼 라이브러리가 로드되었는지 확인
- 파장 범위가 일치하는지 확인
- 캐시 파일이 손상되었을 수 있으므로 삭제 후 재시도

### API 연결 오류

- `.env` 파일의 URL 설정 확인
- 네트워크 연결 상태 확인
- API 서버 상태 확인

## 📝 변경 이력

### Version 0.1.0
- 초기 릴리스
- 기본 이미지 로드 및 표시 기능
- 픽셀 라벨링 기능
- 분류 기능
- ROI 관리 기능

## 📄 라이선스

Copyright (c) 2024 GNEWSOFT. All rights reserved.

## 👥 기여자

- GNEWSOFT Development Team

## 📧 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

**Note**: 이 문서는 프로젝트의 현재 상태를 반영하고 있으며, 지속적으로 업데이트됩니다.

