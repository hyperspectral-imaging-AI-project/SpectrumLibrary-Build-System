# Third Pixel Classification Tool - 프로젝트 구조 문서

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [전체 디렉토리 구조](#전체-디렉토리-구조)
3. [아키텍처 개요](#아키텍처-개요)
4. [주요 디렉토리 상세 설명](#주요-디렉토리-상세-설명)
5. [핵심 컴포넌트 흐름](#핵심-컴포넌트-흐름)
6. [데이터 흐름](#데이터-흐름)
7. [확장 가이드](#확장-가이드)

---

## 프로젝트 개요

**Third Pixel Classification Tool**은 PyQt5 기반의 초분광 이미지(HSI) 처리 및 픽셀 분류 데스크톱 애플리케이션입니다.

### 주요 기능
- 초분광 이미지 로드 및 시각화
- 픽셀 단위 라벨링 및 분류
- ROI(Region of Interest) 관리
- 스펙트럼 분석 및 유사도 계산
- 자동 분류 및 클래스맵 생성
- 레이어 기반 오버레이 시스템

---

## 전체 디렉토리 구조

```
third_pixel_classification_tool/
├── app.py                          # 애플리케이션 엔트리 포인트
├── README.md                       # 프로젝트 README
│
├── views/                          # UI 레이어 (View)
│   ├── main_window.py             # 메인 윈도우 (중앙 컨트롤러 역할)
│   ├── mapview.py                 # 지도 뷰 (이미지 표시 및 상호작용)
│   ├── dialogs/                   # 다이얼로그 컴포넌트
│   │   ├── image_load.py          # 이미지 로드 다이얼로그
│   │   ├── pixel_labeling_dialog.py      # 픽셀 라벨링 다이얼로그
│   │   ├── user_labeling_dialog.py       # 사용자 지정 라벨링 다이얼로그
│   │   ├── classmap_labeling_dialog.py   # 클래스맵 라벨링 다이얼로그
│   │   ├── pixel_classification.py        # 픽셀 분류 다이얼로그
│   │   ├── analysis_selection_dialog.py   # 분석 선택 다이얼로그
│   │   ├── camera_add.py                  # 카메라 추가 다이얼로그
│   │   ├── material_add_dialog.py        # 물질 추가 다이얼로그
│   │   ├── unmixing_dialog.py             # 언믹싱 다이얼로그
│   │   ├── diffusion_dialog.py            # 확산 다이얼로그
│   │   ├── search_labeling_database.py    # 라벨링 데이터베이스 검색
│   │   ├── recommend_label_wizard.py     # 라벨 추천 마법사
│   │   ├── workspace_dialog.py             # 워크스페이스 다이얼로그
│   │   ├── viewer_band_dialog.py          # 밴드 뷰어 다이얼로그
│   │   ├── endmember_detail_dialog.py     # 엔드멤버 상세 다이얼로그
│   │   └── labeling_candidate_review.py   # 라벨링 후보 검토
│   └── docks/                     # 도크 위젯 (사이드바)
│       ├── layer_dock.py          # 레이어 관리 도크
│       ├── pixel_classification_dock.py   # 픽셀 분류 도크
│       └── unmixing_dock.py       # 언믹싱 도크
│
├── controllers/                    # 컨트롤러 레이어 (Controller)
│   ├── pixel_click.py             # 픽셀 클릭 이벤트 처리
│   ├── roi_controller.py          # ROI(작업 영역) 관리 컨트롤러
│   ├── class_roi_controller.py    # 클래스별 ROI 관리 컨트롤러
│   └── analysis_selection_controller.py    # 분석 선택 컨트롤러
│
├── services/                       # 서비스 레이어 (비즈니스 로직)
│   ├── layer_manager.py           # 레이어 관리 서비스
│   ├── resampling_cache.py        # 리샘플링 및 캐시 관리
│   ├── pixel_knn.py              # KNN 기반 픽셀 분류
│   ├── classmap_render.py         # 클래스맵 렌더링
│   ├── paletteService.py         # 팔레트 관리 서비스
│   ├── transparency_service.py   # 투명도 오버레이 서비스
│   ├── region_growing_service.py # 영역 확장 서비스
│   ├── opacity.py                # 투명도 계산 유틸리티
│   ├── spec_library.py            # 스펙트럼 라이브러리 관리
│   ├── label_store.py            # 라벨 저장소
│   ├── label_code.py             # 라벨 코드 생성
│   ├── io_store.py               # 입출력 저장소
│   ├── overlay_temp.py           # 임시 오버레이 서비스
│   └── classmap_render.py        # 클래스맵 렌더러
│
├── core/                          # 핵심 알고리즘 (Core Logic)
│   ├── autoclass.py              # 자동 분류 알고리즘
│   ├── metrics.py                # 유사도 메트릭 (SAD, SID, SAM 등)
│   ├── resampling_service.py     # 스펙트럼 리샘플링
│   ├── vis.py                    # 시각화 유틸리티 (RGB 변환 등)
│   ├── region_growing.py         # 영역 확장 알고리즘
│   └── vlm_generation.py         # VLM 생성 (Vision Language Model)
│
├── data/                          # 데이터 접근 레이어
│   └── db.py                     # API 연동 (서버 모드)
│
├── dataio/                        # 데이터 입출력
│   └── sidecar.py                # 사이드카 파일 처리 (.info 등)
│
├── models/                        # 데이터 모델
│   └── labels.py                 # 라벨 데이터 모델
│
├── ui/                            # UI 정의 파일 (.ui)
│   ├── main_window.ui            # 메인 윈도우 UI
│   ├── image_load.ui             # 이미지 로드 UI
│   ├── pixel_labeling_dialog.ui  # 픽셀 라벨링 UI
│   ├── user_labeling_dialog.ui   # 사용자 지정 라벨링 UI
│   ├── classmap_labeling_dialog.ui   # 클래스맵 라벨링 UI
│   ├── layer_dock.ui             # 레이어 도크 UI
│   ├── pixel_classification_dock.ui  # 픽셀 분류 도크 UI
│   ├── unmixing_dock.ui          # 언믹싱 도크 UI
│   └── ... (기타 다이얼로그 UI 파일들)
│
├── icons/                         # 아이콘 리소스
│   ├── gnew_icon.png
│   └── roi_plus.png
│
├── test/                          # 테스트 스크립트
│   ├── api_test.py               # API 테스트
│   ├── test_region_growing.py   # 영역 확장 테스트
│   ├── load_npz.py               # NPZ 파일 로드 테스트
│   └── ... (기타 테스트 파일들)
│
├── examples/                      # 예제 코드
│   └── region_growing_example.py # 영역 확장 예제
│
├── prompt/                        # 프롬프트 문서
│   ├── asd.md                    # 프로젝트 구조 문서 (현재 파일)
│   └── vlm_prompt.md            # VLM 프롬프트
│
├── logs/                          # 로그 파일 디렉토리
├── map_sample/                    # 샘플 맵 데이터
└── label_local/                   # 로컬 라벨 데이터
```

---

## 아키텍처 개요

이 프로젝트는 **MVC(Model-View-Controller) 패턴**을 기반으로 하며, 다음과 같은 레이어 구조를 가집니다:

```
┌─────────────────────────────────────────┐
│           View Layer (UI)                │
│  - views/main_window.py                 │
│  - views/dialogs/*.py                   │
│  - views/docks/*.py                     │
│  - views/mapview.py                     │
└──────────────┬──────────────────────────┘
               │ 이벤트/시그널
┌──────────────▼──────────────────────────┐
│      Controller Layer                   │
│  - controllers/pixel_click.py            │
│  - controllers/roi_controller.py        │
│  - controllers/class_roi_controller.py  │
└──────────────┬──────────────────────────┘
               │ 호출
┌──────────────▼──────────────────────────┐
│       Service Layer                     │
│  - services/layer_manager.py            │
│  - services/resampling_cache.py         │
│  - services/pixel_knn.py               │
│  - services/*.py                        │
└──────────────┬──────────────────────────┘
               │ 사용
┌──────────────▼──────────────────────────┐
│        Core Layer                       │
│  - core/autoclass.py                    │
│  - core/metrics.py                      │
│  - core/resampling_service.py            │
│  - core/vis.py                          │
└──────────────┬──────────────────────────┘
               │ 접근
┌──────────────▼──────────────────────────┐
│      Data Layer                         │
│  - data/db.py (API)                     │
│  - dataio/sidecar.py (파일 I/O)         │
│  - models/labels.py (데이터 모델)       │
└─────────────────────────────────────────┘
```

### 레이어별 역할

1. **View Layer**: 사용자 인터페이스 및 시각화
2. **Controller Layer**: 사용자 입력 처리 및 이벤트 라우팅
3. **Service Layer**: 비즈니스 로직 및 상태 관리
4. **Core Layer**: 핵심 알고리즘 및 수학적 연산
5. **Data Layer**: 데이터 접근 및 영속성

---

## 주요 디렉토리 상세 설명

### 1. `app.py` - 애플리케이션 엔트리 포인트

**역할**: 애플리케이션 초기화 및 실행

**주요 기능**:
- HighDPI 설정
- 환경변수 로드 (`.env` 파일)
- 로깅 설정
- 리소스 파일 등록 (`icon.rcc`)
- `MainWindow` 인스턴스 생성 및 실행

**의존성**:
- `views.main_window.MainWindow`

---

### 2. `views/` - UI 레이어

#### 2.1 `main_window.py` - 메인 윈도우

**역할**: 애플리케이션의 중앙 컨트롤러 역할

**주요 책임**:
- 전체 애플리케이션 상태 관리
- 모든 다이얼로그 및 도크 위젯 관리
- 서비스 및 컨트롤러 초기화 및 연결
- 레이어 관리자와의 통합
- ROI 컨트롤러 관리
- 이미지 데이터 및 설정 관리

**주요 속성**:
- `layer_manager`: 레이어 관리 서비스
- `roi`: ROI 컨트롤러
- `pixel_click_controller`: 픽셀 클릭 컨트롤러
- `class_roi_controller`: 클래스 ROI 컨트롤러
- `transparency_service`: 투명도 서비스
- `region_growing_service`: 영역 확장 서비스

#### 2.2 `mapview.py` - 지도 뷰

**역할**: 초분광 이미지 표시 및 사용자 상호작용 처리

**주요 기능**:
- QGraphicsView 기반 이미지 렌더링
- 픽셀 클릭 이벤트 처리
- 레이어 오버레이 표시
- 줌/팬 기능
- ROI 그리기 지원

**주요 시그널**:
- `labelPixelPicked`: 라벨링용 픽셀 선택
- `diffusionSeedPicked`: 확산 시드 선택
- `inspectPixelPicked`: 검사용 픽셀 선택

#### 2.3 `dialogs/` - 다이얼로그 컴포넌트

각 다이얼로그는 특정 기능을 담당하는 독립적인 UI 컴포넌트입니다.

**주요 다이얼로그**:

- **`image_load.py`**: ENVI HDR/RAW 파일 로드
- **`pixel_labeling_dialog.py`**: 픽셀 단위 라벨링 인터페이스
- **`user_labeling_dialog.py`**: 사용자 지정 라벨링 (픽셀 선택 및 클래스 할당)
- **`classmap_labeling_dialog.py`**: 클래스맵 기반 라벨링
- **`pixel_classification.py`**: 픽셀 분류 결과 표시
- **`analysis_selection_dialog.py`**: 분석 선택 및 설정
- **`unmixing_dialog.py`**: 언믹싱(Unmixing) 분석
- **`diffusion_dialog.py`**: 확산 기반 영역 선택
- **`search_labeling_database.py`**: 라벨링 데이터베이스 검색
- **`recommend_label_wizard.py`**: AI 기반 라벨 추천

#### 2.4 `docks/` - 도크 위젯

사이드바 형태의 도구 패널입니다.

- **`layer_dock.py`**: 레이어 목록 및 가시성 제어
- **`pixel_classification_dock.py`**: 픽셀 분류 설정 및 결과
- **`unmixing_dock.py`**: 언믹싱 분석 도구

---

### 3. `controllers/` - 컨트롤러 레이어

#### 3.1 `pixel_click.py` - 픽셀 클릭 컨트롤러

**역할**: MapView의 픽셀 클릭 이벤트를 모드별로 처리

**지원 모드**:
- `LABEL`: 라벨링 모드
- `DIFFUSION_SEED`: 확산 시드 선택 모드
- `INSPECT`: 검사 모드 (KNN Top-K 분석)
- `NONE`: 비활성

**주요 기능**:
- LRU 캐시를 통한 성능 최적화
- 모드별 콜백 라우팅

#### 3.2 `roi_controller.py` - ROI 컨트롤러

**역할**: 작업 영역(Work Area) ROI 관리

**주요 기능**:
- 사각형/다각형 ROI 생성
- ROI 추가/제거/수정
- ROI 시그널 방출 (`roiAdded`, `roiRemoved` 등)

#### 3.3 `class_roi_controller.py` - 클래스 ROI 컨트롤러

**역할**: 클래스별 ROI 영역 관리

**차이점**: `roi_controller.py`는 작업 영역용, `class_roi_controller.py`는 클래스별 영역 관리용

---

### 4. `services/` - 서비스 레이어

#### 4.1 `layer_manager.py` - 레이어 관리자

**역할**: 레이어의 생성/삭제/순서/가시성 관리

**주요 기능**:
- 레이어 추가/제거
- 레이어 순서 변경 (Z-order)
- 레이어 가시성 제어
- MapView와 LayersDock 동기화

**아키텍처**: 콜백 기반 설계로 MapView와 LayersDock에 의존성 없음

#### 4.2 `resampling_cache.py` - 리샘플링 및 캐시 관리

**역할**: 스펙트럼 리샘플링 및 결과 캐싱

**주요 기능**:
- `.npz` 파일 기반 캐시 저장/로드
- `.info` 파일 기반 메타데이터 관리
- 클래스 정보 저장/로드
- Continuum Removal 결과 캐싱

**캐시 파일 형식**:
- `{원본파일}.resample.npz`: NumPy 배열 데이터
- `{원본파일}.resample.info`: JSON 메타데이터

#### 4.3 `pixel_knn.py` - KNN 분류 서비스

**역할**: K-Nearest Neighbors 기반 픽셀 분류

**주요 기능**:
- Top-K 유사 클래스 검색
- 스펙트럼 라이브러리 기반 분류
- 유사도 점수 계산

#### 4.4 `classmap_render.py` - 클래스맵 렌더러

**역할**: 분류 결과를 RGB 이미지로 렌더링

**주요 기능**:
- 클래스 ID → 색상 매핑
- 팔레트 기반 렌더링
- 투명도 지원

#### 4.5 `paletteService.py` - 팔레트 서비스

**역할**: 클래스별 색상 팔레트 관리

**주요 기능**:
- 클래스 ID별 색상 할당
- 팔레트 일관성 유지 (SSOT: Single Source of Truth)

#### 4.6 `transparency_service.py` - 투명도 서비스

**역할**: 투명도 오버레이 생성 및 관리

#### 4.7 `region_growing_service.py` - 영역 확장 서비스

**역할**: 시드 픽셀에서 시작하는 영역 확장

#### 4.8 `label_store.py` - 라벨 저장소

**역할**: 라벨링 데이터 저장 및 관리

**지원 모드**:
- Personal: 로컬 저장
- Server: API를 통한 서버 저장

#### 4.9 `spec_library.py` - 스펙트럼 라이브러리 관리

**역할**: 참조 스펙트럼 라이브러리 관리

---

### 5. `core/` - 핵심 알고리즘 레이어

#### 5.1 `autoclass.py` - 자동 분류

**역할**: 스펙트럼 라이브러리 기반 자동 분류

**주요 함수**:
- `classify_cube()`: 전체 큐브 분류
- `UNKNOWN`, `MULTIPLE`: 특수 클래스 ID

#### 5.2 `metrics.py` - 유사도 메트릭

**역할**: 스펙트럼 유사도 계산

**지원 메트릭**:
- SAD (Spectral Angle Distance)
- SID (Spectral Information Divergence)
- SAM (Spectral Angle Mapper)
- 기타 유사도 측정

#### 5.3 `resampling_service.py` - 리샘플링 서비스

**역할**: 스펙트럼 파장 범위 변환

**주요 기능**:
- 파장 범위 리샘플링
- Continuum Removal 전처리

#### 5.4 `vis.py` - 시각화 유틸리티

**역할**: 이미지 시각화 유틸리티

**주요 함수**:
- `make_rgb()`: 초분광 데이터를 RGB 이미지로 변환

#### 5.5 `region_growing.py` - 영역 확장 알고리즘

**역할**: 시드 픽셀에서 유사 픽셀로 영역 확장

---

### 6. `data/` - 데이터 접근 레이어

#### 6.1 `db.py` - API 연동

**역할**: 서버 모드에서 외부 API 호출

**주요 함수**:
- `image_check_and_save()`: 이미지 확인 및 저장
- `material_code_name()`: 물질 코드/이름 조회
- `send_label_add()`: 라벨 추가
- `call_unmixing()`: 언믹싱 API 호출

**환경변수 기반 설정**:
- `material_code_name_url`
- `label_add_url`
- `image_check_url`
- 등등

---

### 7. `dataio/` - 데이터 입출력

#### 7.1 `sidecar.py` - 사이드카 파일 처리

**역할**: `.info` 파일 등 메타데이터 파일 처리

---

### 8. `models/` - 데이터 모델

#### 8.1 `labels.py` - 라벨 모델

**역할**: 라벨 데이터 구조 정의

---

## 핵심 컴포넌트 흐름

### 이미지 로드 흐름

```
1. 사용자: File > Image Load
   ↓
2. ImageLoadDialog 열림
   ↓
3. HDR/RAW 파일 선택 및 설정 입력
   ↓
4. MainWindow.on_image_load() 호출
   ↓
5. resampling_cache.try_load_cache() 시도
   ↓
6. 캐시 없으면 resampling_service.resampling() 실행
   ↓
7. resampling_cache.save_cache() 저장
   ↓
8. MapView에 RGB 이미지 표시
   ↓
9. 레이어 관리자에 RGB 레이어 등록
```

### 픽셀 라벨링 흐름

```
1. 사용자: Tools > Pixel Labeling
   ↓
2. PixelLabelingDialog 또는 UserLabelingDialog 열림
   ↓
3. 사용자가 이미지에서 픽셀 클릭
   ↓
4. MapView가 pixelPicked 시그널 방출
   ↓
5. PixelClickController가 모드에 따라 처리
   ↓
6. 다이얼로그에 픽셀 정보 전달
   ↓
7. 사용자가 클래스 선택 및 등록
   ↓
8. LabelStore를 통해 저장 (Personal/Server 모드)
   ↓
9. 라벨링 완료 시그널 방출
```

### 분류 수행 흐름

```
1. 사용자: Tools > Pixel Classification
   ↓
2. PixelClassificationDialog 열림
   ↓
3. 스펙트럼 라이브러리 로드 확인
   ↓
4. 임계값 설정
   ↓
5. core.autoclass.classify_cube() 호출
   ↓
6. pixel_knn.topk_for_pixel()로 유사도 계산
   ↓
7. 분류 결과 생성 (클래스맵)
   ↓
8. classmap_render로 RGB 렌더링
   ↓
9. 레이어 관리자에 클래스맵 레이어 추가
   ↓
10. MapView에 표시
```

### ROI 설정 흐름

```
1. 사용자: ROI 툴바에서 모드 선택
   ↓
2. ROIController 활성화
   ↓
3. MapView에서 드래그로 영역 지정
   ↓
4. ROIController.roiAdded 시그널 방출
   ↓
5. MainWindow._on_work_roi_added() 처리
   ↓
6. 작업 마스크 업데이트
   ↓
7. 레이어 관리자에 ROI 레이어 추가
```

---

## 데이터 흐름

### 데이터 저장 구조

```
원본 이미지 파일 (HDR/RAW)
    ↓
리샘플링 및 전처리
    ↓
캐시 파일 생성
    ├── {파일명}.resample.npz  (NumPy 배열)
    └── {파일명}.resample.info  (JSON 메타데이터)
        ├── classes: {cid: {name, description}}
        ├── wavelength: [...]
        ├── shape: [H, W, C]
        └── ...
```

### Personal vs Server 모드

**Personal 모드**:
- 모든 데이터 로컬 저장
- `.npz` 및 `.info` 파일 사용
- API 호출 없음

**Server 모드**:
- 서버 API를 통한 데이터 관리
- 로컬 캐시는 보조적으로 사용
- 환경변수로 API 엔드포인트 설정

---

## 확장 가이드

### 새로운 다이얼로그 추가

1. `ui/` 디렉토리에 `.ui` 파일 생성 (Qt Designer 사용)
2. `views/dialogs/` 디렉토리에 Python 파일 생성
3. `main_window.py`에서 다이얼로그 인스턴스 생성 및 연결

**예시**:
```python
# views/dialogs/my_dialog.py
from PyQt5 import QtWidgets, uic
from pathlib import Path

class MyDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, ui_dir=None):
        super().__init__(parent)
        ui_path = Path(ui_dir) / "my_dialog.ui"
        self._root = uic.loadUi(str(ui_path), self)
```

### 새로운 서비스 추가

1. `services/` 디렉토리에 서비스 클래스 생성
2. 필요한 경우 `core/`에 알고리즘 추가
3. `main_window.py`에서 서비스 인스턴스 생성 및 주입

**예시**:
```python
# services/my_service.py
class MyService:
    def __init__(self):
        self.state = {}
    
    def do_something(self):
        pass
```

### 새로운 컨트롤러 추가

1. `controllers/` 디렉토리에 컨트롤러 클래스 생성
2. 필요한 시그널 정의
3. `main_window.py`에서 컨트롤러 인스턴스 생성 및 연결

**예시**:
```python
# controllers/my_controller.py
from PyQt5.QtCore import QObject, pyqtSignal

class MyController(QObject):
    somethingHappened = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
```

### 새로운 메트릭 추가

1. `core/metrics.py`에 메트릭 함수 추가
2. `services/pixel_knn.py`에서 사용

**예시**:
```python
# core/metrics.py
def my_metric(spectrum1, spectrum2):
    """새로운 유사도 메트릭"""
    # 구현
    return similarity_score
```

---

## 주요 설계 원칙

1. **관심사의 분리**: 각 레이어는 명확한 책임을 가짐
2. **의존성 주입**: 서비스와 컨트롤러는 콜백/함수로 주입받음
3. **시그널/슬롯**: PyQt5 시그널을 통한 느슨한 결합
4. **SSOT (Single Source of Truth)**: 팔레트 등은 단일 소스 유지
5. **캐싱**: 성능 최적화를 위한 적극적인 캐싱 활용

---

## 참고 사항

- 모든 UI 파일은 `ui/` 디렉토리에 있으며, Qt Designer로 편집 가능
- 로그는 `logs/app.log`에 저장됨
- 환경변수는 `.env` 파일 또는 시스템 환경변수로 설정
- 캐시 파일은 자동 생성되며, 필요시 삭제 후 재생성 가능

---

**문서 버전**: 1.0  
**최종 업데이트**: 2024  
**작성자**: GNEWSOFT Development Team
