# Third Pixel Classification Tool - Sequential Thinking 분석

## 📌 1단계: 프로젝트 개요 및 목적

### 1.1 프로젝트 정체성
- **이름**: Third Pixel Classification Tool
- **타입**: PyQt5 기반 데스크톱 애플리케이션
- **도메인**: 초분광 이미지(Hyperspectral Image, HSI) 처리 및 픽셀 분류
- **목적**: 초분광 데이터를 활용한 물질 분류, 라벨링, 분석을 위한 전문 도구

### 1.2 핵심 가치 제안
1. **초분광 이미지 처리**: ENVI HDR/RAW 형식 지원
2. **스펙트럼 기반 분류**: SAM/SID/SCC 등 다양한 메트릭 활용
3. **인터랙티브 라벨링**: 사용자 지정, 클래스맵 기반, AI 추천 라벨링
4. **다층 레이어 관리**: RGB, 분류맵, ROI, 오버레이 등 다중 레이어 시각화
5. **서버/로컬 이중 모드**: Personal(로컬) 및 Server(API) 모드 지원

---

## 📐 2단계: 아키텍처 구조 분석

### 2.1 계층 구조 (MVC 패턴 기반)

```
┌─────────────────────────────────────────────────────────┐
│                    View Layer (UI)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ MainWindow   │  │   Dialogs    │  │    Docks     │ │
│  │  MapView     │  │  (다이얼로그) │  │  (도크 위젯) │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                        ↕ Signals/Slots
┌─────────────────────────────────────────────────────────┐
│                 Controller Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ PixelClick   │  │   ROI        │  │  Analysis    │ │
│  │ Controller   │  │ Controller   │  │  Controller  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                        ↕ Service Calls
┌─────────────────────────────────────────────────────────┐
│                  Service Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ LayerManager │  │ PixelKNN     │  │ LabelStore    │ │
│  │ PaletteSvc   │  │ Resampling   │  │ SpecLibrary   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                        ↕ Algorithm Calls
┌─────────────────────────────────────────────────────────┐
│                    Core Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Autoclass     │  │   Metrics    │  │ RegionGrowing│ │
│  │ Resampling    │  │      VIS     │  │   VLM        │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                        ↕ Data Access
┌─────────────────────────────────────────────────────────┐
│                  Data Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │     DB       │  │   Sidecar    │  │   Models     │ │
│  │  (API 연동)  │  │  (.info 파일) │  │  (LabelRow)  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 2.2 주요 컴포넌트 역할

| 컴포넌트 | 경로 | 역할 |
|---------|------|------|
| **MainWindow** | `views/main_window.py` | 애플리케이션 중앙 허브, 모든 서비스/컨트롤러 조율 |
| **MapView** | `views/mapview.py` | 이미지 표시 및 픽셀 상호작용 처리 |
| **LayerManager** | `services/layer_manager.py` | 다중 레이어 생명주기 관리 (SSOT) |
| **PixelClickController** | `controllers/pixel_click.py` | 픽셀 클릭 모드별 처리 (라벨링/분석/검사) |
| **ROIController** | `controllers/roi_controller.py` | 작업 영역(Work Area) ROI 관리 |
| **ClassmapRenderer** | `services/classmap_render.py` | 분류맵 시각화 및 팔레트 적용 |
| **LabelStore** | `services/label_store.py` | 라벨 데이터 영속성 관리 |

---

## 🔧 3단계: 주요 기능 모듈 분류

### 3.1 이미지 처리 모듈

#### 3.1.1 이미지 로드 (`views/dialogs/image_load.py`)
- **입력**: ENVI HDR/RAW 파일 경로
- **처리**: 
  - 이미지 축 설정 (Height, Width, Bands)
  - 파장 정보 파싱 및 검증
  - 카메라 정보 매칭
  - 리샘플링 캐시 확인/생성
- **출력**: `self.cfg` 딕셔너리 (data, wavelength, metadata)

#### 3.1.2 RGB 가시화 (`core/vis.py`)
- **함수**: `make_rgb(cube, bands=[R, G, B])`
- **목적**: 초분광 큐브를 RGB 이미지로 변환
- **저장**: `self.rgb_image` (H, W, 3)

#### 3.1.3 밴드 뷰어 (`views/dialogs/viewer_band_dialog.py`)
- **기능**: 개별 밴드 이미지 확인
- **용도**: 스펙트럼 특성 분석 전 밴드별 품질 검사

### 3.2 분류 모듈

#### 3.2.1 자동 분류 (`core/autoclass.py`)
- **알고리즘**: 스펙트럼 라이브러리 기반 최근접 이웃 분류
- **메트릭**: SAM, SID, SCC (사용자 선택)
- **임계값**: 
  - `tau` (strict): 확정 분류 임계값
  - `delta` (base): 후보 분류 임계값
- **출력**: 분류맵 (H, W) int32 배열

#### 3.2.2 Top-K 유사도 분석 (`services/pixel_knn.py`)
- **함수**: `topk_for_pixel(cube, lib, y, x, k=10, metric='SAD')`
- **목적**: 특정 픽셀의 상위 K개 유사 클래스 반환
- **사용처**: 픽셀 검사 모드에서 실시간 후보 표시

#### 3.2.3 분류맵 렌더링 (`services/classmap_render.py`)
- **입력**: 분류맵 (H, W) + 팔레트
- **처리**: 클래스 ID → 색상 매핑
- **출력**: RGBA 이미지 (H, W, 4)

### 3.3 라벨링 모듈

#### 3.3.1 사용자 지정 라벨링 (`views/dialogs/user_labeling_dialog.py`)
- **모드**: 픽셀 직접 선택 → 클래스 할당
- **저장**: `LabelStore`를 통해 로컬/서버 저장

#### 3.3.2 클래스맵 라벨링 (`views/dialogs/classmap_labeling_dialog.py`)
- **모드**: 분류 결과 기반 일괄 라벨링
- **필터링**: 신뢰도/임계값 기반 선택

#### 3.3.3 라벨링 데이터베이스 검색 (`views/dialogs/search_labeling_database.py`)
- **기능**: 
  - 서버에서 라벨링 데이터 검색
  - 스펙트럼 비교 및 시각화
  - Material 정보 조회 (API 연동)
- **표시**: 
  - 스펙트럼 그래프 (matplotlib)
  - 패치 이미지 (라벨링 픽셀 주변)
  - 상세 정보 (PK, Name, Description, Coord)

#### 3.3.4 AI 기반 라벨 추천 (`views/dialogs/recommend_label_wizard.py`)
- **기술**: Vision Language Model (VLM)
- **프롬프트**: `prompt/vlm_prompt.md` 기반
- **입력**: 스펙트럼 데이터 + Top-K 후보
- **출력**: 물질 추정 및 신뢰도

### 3.4 ROI 관리 모듈

#### 3.4.1 작업 영역 ROI (`controllers/roi_controller.py`)
- **타입**: 사각형(Rect) / 자유곡선(Poly)
- **저장**: `self._work_mask` (H, W) bool 배열
- **용도**: 분석 범위 제한

#### 3.4.2 클래스별 ROI (`controllers/class_roi_controller.py`)
- **목적**: 클래스별 영역 지정
- **저장**: `self._class_roi_masks` {cid: mask}

#### 3.4.3 분석 영역 선택 (`controllers/analysis_selection_controller.py`)
- **모드**: 픽셀 누적 / 사각형 영역
- **연산**: Union / Subtract
- **저장**: `self._analysis_mask` (H, W) bool 배열

### 3.5 레이어 관리 모듈

#### 3.5.1 LayerManager (`services/layer_manager.py`)
- **역할**: 모든 레이어의 SSOT (Single Source of Truth)
- **관리 항목**:
  - 레이어 추가/제거
  - 가시성 제어
  - Z-order 관리
  - 부모-자식 관계 (트리 구조)
- **동기화**: MapView ↔ LayersDock 자동 동기화

#### 3.5.2 레이어 타입
- **RGB 레이어**: `image.rgb` (베이스 레이어)
- **분류맵 레이어**: `classification` (분류 결과)
- **ROI 레이어**: `work.roi`, `class.{cid}.roi`
- **오버레이 레이어**: 투명도/하이라이트 오버레이

### 3.6 스펙트럼 분석 모듈

#### 3.6.1 스펙트럼 라이브러리 (`services/spec_library.py`)
- **소스**: 
  - `splib_raw`: 원본 스펙트럼 라이브러리
  - `splib_cr`: Continuum Removal 적용
  - `label_raw`: 라벨링 데이터 (원본)
  - `label_cr`: 라벨링 데이터 (CR)
- **형식**: `{class_id: (N, C)}` dict (N=스펙트럼 수, C=밴드 수)

#### 3.6.2 Continuum Removal (`core/resampling_service.py`)
- **함수**: `perform_continuum_removal(spectrum, wavelengths)`
- **목적**: 스펙트럼 전처리로 흡수 특성 강조

#### 3.6.3 리샘플링 (`core/resampling_service.py`)
- **목적**: 서로 다른 파장 해상도 간 보간
- **캐시**: `.resample.npz` + `.resample.info` 파일

### 3.7 고급 분석 모듈

#### 3.7.1 영역 확장 (Region Growing) (`core/region_growing.py`)
- **알고리즘**: 시드 픽셀에서 유사 픽셀로 영역 확장
- **파라미터**: 유사도 임계값, 최대 영역 크기
- **서비스**: `services/region_growing_service.py`

#### 3.7.2 확산 기반 선택 (`views/dialogs/diffusion_dialog.py`)
- **목적**: 유사도 기반 영역 선택
- **시드**: 사용자 지정 픽셀

#### 3.7.3 언믹싱 (Unmixing) (`views/dialogs/unmixing_dialog.py`)
- **목적**: 혼합 픽셀 분해 분석
- **알고리즘**: 선형 언믹싱 (Endmember 추출)
- **API**: `call_unmixing()` (서버 연동)

---

## 🔄 4단계: 데이터 흐름 및 워크플로우

### 4.1 이미지 로드 워크플로우

```
[사용자] File > Image Load
    ↓
[ImageLoadDialog] HDR/RAW 경로 입력
    ↓
[검증] 이미지 축, 파장 범위 확인
    ↓
[캐시 확인] .resample.npz 존재 여부
    ├─ 있음: 캐시 로드
    └─ 없음: 리샘플링 수행 → 캐시 저장
    ↓
[MainWindow] self.cfg 업데이트
    ├─ cfg["data"] = (H, W, C) numpy 배열
    ├─ cfg["wavelength"] = [λ₁, λ₂, ..., λC]
    └─ cfg["metadata"] = {...}
    ↓
[RGB 생성] make_rgb() → self.rgb_image
    ↓
[MapView] RGB 레이어 표시
```

### 4.2 분류 워크플로우

```
[사용자] Tools > Pixel Classification
    ↓
[PixelClassificationDock] 파라미터 설정
    ├─ Metric: SAM/SID/SCC
    ├─ Tau (strict)
    └─ Delta (base)
    ↓
[MainWindow._on_pixel_classify] 분류 실행
    ↓
[core/autoclass.classify_cube]
    ├─ 입력: cube (H,W,C), lib {cid: (N,C)}, roi_mask
    ├─ 처리: 각 픽셀에 대해 최근접 이웃 검색
    └─ 출력: classmap (H,W), where_note (H,W) bool
    ↓
[ClassmapRenderer] 분류맵 → RGBA 이미지
    ↓
[LayerManager] "classification" 레이어 추가
    ↓
[MapView] 분류맵 오버레이 표시
```

### 4.3 라벨링 워크플로우

```
[사용자] Tools > Pixel Labeling
    ↓
[PixelLabelingDialog] 라벨링 모드 선택
    ├─ 사용자 지정
    ├─ 클래스맵 기반
    └─ AI 추천
    ↓
[PixelClickController] 클릭 모드 전환 (LABEL)
    ↓
[사용자] 이미지에서 픽셀 클릭
    ↓
[MainWindow._handle_label_pick] 라벨 데이터 수집
    ├─ 좌표: (y, x)
    ├─ 클래스 ID: cid
    ├─ 스펙트럼: cube[y, x, :]
    └─ 이미지 코드: image_cd
    ↓
[LabelStore] 라벨 저장
    ├─ Personal 모드: 로컬 파일 (.npy)
    └─ Server 모드: API 호출 (send_label_add)
    ↓
[업데이트] label_raw, label_coords 갱신
```

### 4.4 스펙트럼 라이브러리 구축 워크플로우

```
[서버 API] spectrum_library_url 호출
    ↓
[응답] 스펙트럼 라이브러리 데이터 (JSON)
    ↓
[services/spec_library.py] build_library_from_spec_libs()
    ├─ 파장 리샘플링 (필요시)
    ├─ Continuum Removal (선택)
    └─ 형식 변환: {cid: (N, C)}
    ↓
[MainWindow] self.splib_raw, self.splib_cr 저장
    ↓
[사용] 분류, Top-K 분석 등에서 활용
```

---

## 🧩 5단계: 핵심 컴포넌트 상세 분석

### 5.1 MainWindow 구조

#### 5.1.1 초기화 순서
1. UI 로드 (`_load_ui`)
2. LayersDock 생성 (`_init_layers_dock`)
3. LayerManager 생성 (MapView/Dock 콜백 주입)
4. 팔레트 서비스 초기화 (`_init_single_palette`)
5. ClassmapRenderer 생성
6. ROI 컨트롤러 생성 (`_ensure_roi_controller`)
7. PixelClickController 생성
8. 분석 컨트롤러 생성 (AnalysisSelectionController)

#### 5.1.2 주요 상태 변수
```python
self.cfg = {}                    # 이미지 설정 (data, wavelength, ...)
self.rgb_image = None            # RGB 이미지 (H, W, 3)
self.lib = {}                    # 스펙트럼 라이브러리 {cid: (N, C)}
self.splib_raw = {}              # 원본 스펙트럼 라이브러리
self.splib_cr = {}                # CR 적용 스펙트럼 라이브러리
self.label_raw = {}               # 라벨링 데이터 (원본)
self.label_cr = {}                 # 라벨링 데이터 (CR)
self.label_coords = {}            # 라벨링 좌표 {cid: [(x, y, img_cd), ...]}
self._work_mask = None            # 작업 영역 마스크 (H, W) bool
self._class_roi_masks = {}        # 클래스별 ROI {cid: mask}
self._analysis_mask = None         # 분석 영역 마스크
self.user_type = "server"        # "server" | "personal"
```

### 5.2 PixelClickController

#### 5.2.1 클릭 모드
```python
class ClickMode(Enum):
    NONE = auto()            # 일반 포인터
    LABEL = auto()           # 라벨링 모드
    DIFFUSION_SEED = auto()  # 확산 시드 지정
    INSPECT = auto()         # Top-K 검사 모드
    ANALYSIS = auto()        # 분석 영역 선택
```

#### 5.2.2 의존성 주입 패턴
```python
@dataclass
class PixelClickDeps:
    get_cube: Callable[[], np.ndarray]
    get_lib: Callable[[], dict]
    get_metric_name: Callable[[], str]
    get_image_code: Callable[[], Optional[str]]
    present_topk_for_pixel: Callable[[int, int], None]
    analysis_callback: Callable[[int, int], None]
    label_callback: Callable[[int, int, int], None]
    seed_callback: Callable[[int, int, int], None]
```

### 5.3 LayerManager

#### 5.3.1 레이어 생명주기
1. **추가**: `add(name, data, opacity, unique=True)`
2. **가시성**: `set_visible(name, visible)`
3. **순서**: `reorder_top_to_bottom(names)`
4. **제거**: `remove(name)`

#### 5.3.2 레이어 트리 구조
```
image.rgb (부모)
├─ classification (자식)
├─ work.roi (자식)
└─ class.1.roi (자식)
```

### 5.4 ClassmapRenderer

#### 5.4.1 렌더링 파이프라인
```
분류맵 (H, W) int32
    ↓
클래스 ID → 팔레트 색상 매핑
    ↓
RGBA 이미지 (H, W, 4)
    ↓
투명도 적용 (opacity)
    ↓
MapView 오버레이 표시
```

#### 5.4.2 팔레트 관리
- **SSOT**: `MainWindow._single_palette` (전역 팔레트)
- **서비스**: `PaletteService` (색상 할당 로직)
- **동기화**: 분류맵, 지도, 뷰어 간 색상 일치 보장

---

## 🛠️ 6단계: 기술 스택 및 의존성

### 6.1 핵심 라이브러리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| PyQt5 | ≥5.15.0 | GUI 프레임워크 |
| NumPy | ≥1.20.0 | 배열 연산 |
| SciPy | ≥1.7.0 | 과학 계산 (리샘플링 등) |
| Matplotlib | ≥3.4.0 | 스펙트럼 그래프 |
| Requests | ≥2.25.0 | HTTP API 호출 |
| Pillow | ≥8.0.0 | 이미지 처리 |
| python-dotenv | ≥0.19.0 | 환경 변수 관리 |
| Paramiko | ≥2.7.0 | SFTP 업로드 |
| OpenAI | ≥1.0.0 | VLM 기능 (선택) |
| hdf5storage | ≥0.1.18 | 대용량 .mat 저장 (선택) |

### 6.2 외부 의존성

#### 6.2.1 서버 API (Server 모드)
- `camera_list_url`: 카메라 목록 조회
- `camera_add_url`: 카메라 등록
- `material_list_url`: 물질 목록 조회
- `material_add_url`: 물질 등록
- `image_check_url`: 이미지 존재 확인
- `image_save_url`: 이미지 저장
- `label_add_url`: 라벨 추가
- `spectrum_library_url`: 스펙트럼 라이브러리 조회
- `labeling_data_url`: 라벨링 데이터 조회
- `unmixing_inference_url`: 언믹싱 추론

#### 6.2.2 환경 변수 (`.env`)
```env
# API 엔드포인트
material_filtering_url=http://...
camera_list_url=http://...

# OpenAI (VLM)
OPENAI_API_KEY=sk-...

# SFTP (선택)
SFTP_HOST=...
SFTP_PORT=22
SFTP_USER=...
SFTP_PASSWORD=...
```

---

## 🎨 7단계: 사용자 인터페이스 구조

### 7.1 메인 윈도우 구성

```
┌─────────────────────────────────────────────────────┐
│  MenuBar: File | Tools | View | Help                │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌─────────────────────────────────┐ │
│  │ Layers   │  │                                 │ │
│  │ Dock     │  │        MapView                  │ │
│  │          │  │    (이미지 표시 영역)            │ │
│  │ [레이어]  │  │                                 │ │
│  │ 트리]    │  │                                 │ │
│  └──────────┘  └─────────────────────────────────┘ │
│  ┌──────────┐  ┌─────────────────────────────────┐ │
│  │ Pixel    │  │                                 │ │
│  │ Classif. │  │                                 │ │
│  │ Dock     │  │                                 │ │
│  └──────────┘  └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────┤
│  StatusBar: Ready                                   │
└─────────────────────────────────────────────────────┘
```

### 7.2 주요 다이얼로그

| 다이얼로그 | 파일 | 용도 |
|-----------|------|------|
| Image Load | `image_load.py` | HSI 이미지 로드 |
| Camera Add | `camera_add.py` | 카메라 정보 등록 |
| Pixel Labeling | `pixel_labeling_dialog.py` | 라벨링 모드 선택 |
| User Labeling | `user_labeling_dialog.py` | 사용자 지정 라벨링 |
| Classmap Labeling | `classmap_labeling_dialog.py` | 클래스맵 기반 라벨링 |
| Search Database | `search_labeling_database.py` | 라벨링 DB 검색 |
| Recommend Label | `recommend_label_wizard.py` | AI 라벨 추천 |
| Pixel Classification | `pixel_classification.py` | 분류 파라미터 설정 |
| Unmixing | `unmixing_dialog.py` | 언믹싱 분석 |
| Diffusion | `diffusion_dialog.py` | 확산 기반 선택 |
| Workspace | `workspace_dialog.py` | 워크스페이스 저장/로드 |

### 7.3 도크 위젯

| 도크 | 파일 | 용도 |
|-----|------|------|
| Layers Dock | `layer_dock.py` | 레이어 트리 및 가시성 제어 |
| Pixel Classification Dock | `pixel_classification_dock.py` | 분류 파라미터 및 실행 |
| Unmixing Dock | `unmixing_dock.py` | 언믹싱 결과 표시 |

---

## 🔬 8단계: 주요 알고리즘 및 처리 파이프라인

### 8.1 분류 알고리즘 (`core/autoclass.py`)

#### 8.1.1 분류 함수 시그니처
```python
def classify_cube(
    cube: np.ndarray,      # (H, W, C) 입력 이미지
    lib: dict,             # {cid: (N, C)} 스펙트럼 라이브러리
    roi_mask: Optional[np.ndarray],  # (H, W) bool 마스크
    metric: str = "SAD",   # "SAD" | "SID" | "SCC"
    tau: float = 0.1,      # strict 임계값
    delta: float = 0.2,     # base 임계값
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        classmap: (H, W) int32 분류맵
        where_note: (H, W) bool 확정 분류 마스크
    """
```

#### 8.1.2 분류 로직
```
각 픽셀 (y, x)에 대해:
    1. 스펙트럼 추출: s = cube[y, x, :]  # (C,)
    2. 모든 클래스 cid에 대해:
        - 라이브러리에서 스펙트럼 검색: lib[cid]  # (N, C)
        - 메트릭 계산: dist = metric(s, lib[cid])
        - 최소 거리: min_dist = min(dist)
    3. 최소 거리 클래스 선택: best_cid = argmin(min_dist)
    4. 임계값 검사:
        - if min_dist < tau: 확정 분류
        - elif min_dist < delta: 후보 분류
        - else: UNKNOWN
```

### 8.2 메트릭 계산 (`core/metrics.py`)

#### 8.2.1 SAM (Spectral Angle Mapper)
```python
def sam(spectrum1, spectrum2):
    """
    스펙트럼 각도 측정
    범위: [0, π/2]
    작을수록 유사
    """
    dot = np.dot(spectrum1, spectrum2)
    norm1 = np.linalg.norm(spectrum1)
    norm2 = np.linalg.norm(spectrum2)
    return np.arccos(dot / (norm1 * norm2))
```

#### 8.2.2 SID (Spectral Information Divergence)
```python
def sid(spectrum1, spectrum2):
    """
    정보 이론 기반 거리
    범위: [0, ∞)
    작을수록 유사
    """
    # 확률 분포로 정규화 후 KL divergence 계산
    p1 = spectrum1 / np.sum(spectrum1)
    p2 = spectrum2 / np.sum(spectrum2)
    return np.sum(p1 * np.log(p1 / p2)) + np.sum(p2 * np.log(p2 / p1))
```

#### 8.2.3 SCC (Spectral Correlation Coefficient)
```python
def scc(spectrum1, spectrum2):
    """
    상관 계수 기반 유사도
    범위: [-1, 1]
    클수록 유사 (1 - abs(corr)로 거리 변환)
    """
    corr = np.corrcoef(spectrum1, spectrum2)[0, 1]
    return 1 - abs(corr)
```

### 8.3 리샘플링 (`core/resampling_service.py`)

#### 8.3.1 목적
- 서로 다른 파장 해상도 간 보간
- 스펙트럼 라이브러리와 이미지 파장 정렬

#### 8.3.2 알고리즘
```python
def resampling(
    data: np.ndarray,           # (..., C_old) 원본 데이터
    old_wavelengths: np.ndarray, # (C_old,) 원본 파장
    new_wavelengths: np.ndarray, # (C_new,) 목표 파장
    method: str = "linear"      # "linear" | "cubic" | "nearest"
) -> np.ndarray:
    """
    SciPy interp1d를 사용한 보간
    """
    from scipy.interpolate import interp1d
    f = interp1d(old_wavelengths, data, axis=-1, kind=method)
    return f(new_wavelengths)
```

### 8.4 Continuum Removal (`core/resampling_service.py`)

#### 8.4.1 목적
- 스펙트럼 전처리로 흡수 특성 강조
- 노이즈 감소 및 특징 추출 향상

#### 8.4.2 알고리즘
```python
def perform_continuum_removal(
    spectrum: np.ndarray,
    wavelengths: np.ndarray
) -> np.ndarray:
    """
    1. Convex hull 계산
    2. 원본 스펙트럼을 hull로 나눔
    3. 정규화된 스펙트럼 반환
    """
    from scipy.spatial import ConvexHull
    # ... 구현 ...
```

---

## 📊 9단계: 데이터 모델 및 저장 형식

### 9.1 라벨 데이터 모델 (`models/labels.py`)

```python
@dataclass
class LabelRow:
    """라벨 데이터 행"""
    pk: int                    # Primary Key
    image_cd: str             # 이미지 코드
    img_x: int                # X 좌표
    img_y: int                # Y 좌표
    class_id: int              # 클래스 ID
    spectrum: np.ndarray       # 스펙트럼 (C,)
    created_at: datetime       # 생성 시간
```

### 9.2 캐시 파일 형식

#### 9.2.1 리샘플링 캐시
- **파일**: `{원본파일}.resample.npz`
- **내용**: 
  - `data`: (H, W, C) 리샘플링된 이미지
  - `wavelength`: (C,) 파장 배열
- **메타데이터**: `{원본파일}.resample.info` (JSON)
  ```json
  {
    "source_file": "...",
    "original_wavelengths": [...],
    "target_wavelengths": [...],
    "method": "linear",
    "created_at": "..."
  }
  ```

#### 9.2.2 라벨 캐시 (Personal 모드)
- **파일**: `label_local/{image_cd}_labels.npy`
- **형식**: NumPy 배열 (N, C+3) [spectrum, x, y, cid]

### 9.3 워크스페이스 형식

#### 9.3.1 저장 내용
- 이미지 설정 (`cfg`)
- 분류맵
- ROI 마스크
- 레이어 상태
- 라벨 데이터

#### 9.3.2 파일 형식
- **형식**: `.npz` (NumPy Zip) 또는 `.json` + `.npy`
- **구조**: 
  ```python
  {
    "cfg": {...},
    "classmap": (H, W) array,
    "work_mask": (H, W) bool array,
    "layers": {...},
    "labels": [...]
  }
  ```

---

## 🔍 10단계: 주요 워크플로우 시퀀스

### 10.1 전체 분석 워크플로우

```
1. [이미지 로드]
   File > Image Load
   → HDR/RAW 파일 선택
   → 이미지 축 설정
   → 파장 범위 입력
   → Check → OK
   → RGB 이미지 표시

2. [스펙트럼 라이브러리 로드]
   (자동 또는 수동)
   → API 호출 또는 로컬 파일
   → splib_raw, splib_cr 생성

3. [ROI 설정] (선택)
   ROI 툴바 → Rect/Poly 선택
   → 이미지에서 영역 지정
   → 작업 영역 마스크 생성

4. [분류 수행]
   Tools > Pixel Classification
   → 메트릭 선택 (SAM/SID/SCC)
   → 임계값 설정 (tau, delta)
   → 적용
   → 분류맵 생성 및 표시

5. [라벨링]
   Tools > Pixel Labeling
   → 모드 선택 (사용자/클래스맵/AI)
   → 픽셀 클릭 또는 영역 선택
   → 라벨 저장

6. [결과 저장]
   File > Save
   → 분류맵, 라벨, ROI 저장
```

### 10.2 픽셀 검사 워크플로우

```
1. [검사 모드 활성화]
   Pixel Classification Dock
   → Inspect 모드 선택

2. [픽셀 클릭]
   MapView에서 픽셀 클릭
   → PixelClickController 처리

3. [Top-K 계산]
   topk_for_pixel() 호출
   → 상위 K개 유사 클래스 반환

4. [결과 표시]
   다이얼로그 팝업
   → 스펙트럼 비교 그래프
   → Material 정보 표시
   → 상세 설명
```

### 10.3 라벨링 데이터베이스 검색 워크플로우

```
1. [다이얼로그 열기]
   Tools > Search Labeling Database

2. [클래스 선택]
   ComboBox에서 클래스 선택
   → 해당 클래스의 스펙트럼 수집
     - splib_raw/cr
     - label_raw/cr

3. [테이블 표시]
   스펙트럼 목록 표시
   - #, Class, From, Material Name

4. [스펙트럼 선택]
   테이블에서 행 선택
   → 스펙트럼 그래프 업데이트
   → 패치 이미지 표시 (라벨링인 경우)
   → 상세 정보 표시

5. [비교 분석]
   여러 스펙트럼 선택하여 비교
   → 그래프에서 시각적 비교
```

---

## 🎯 11단계: 핵심 설계 원칙

### 11.1 아키텍처 원칙

1. **SSOT (Single Source of Truth)**
   - 레이어: `LayerManager`
   - 팔레트: `MainWindow._single_palette`
   - ROI: 각 컨트롤러가 독립 관리

2. **의존성 주입**
   - `PixelClickDeps`: 픽셀 클릭 컨트롤러 의존성
   - 콜백 함수를 통한 느슨한 결합

3. **시그널/슬롯 패턴**
   - PyQt5 시그널을 통한 이벤트 처리
   - 비동기 통신

4. **서비스 레이어 분리**
   - 비즈니스 로직을 서비스로 분리
   - 재사용성 및 테스트 용이성

### 11.2 데이터 관리 원칙

1. **캐싱 전략**
   - 리샘플링 결과 캐시 (`.resample.npz`)
   - Material 정보 캐시 (메모리)
   - Top-K 결과 캐시 (LRU)

2. **영속성**
   - Personal 모드: 로컬 파일 저장
   - Server 모드: API를 통한 서버 저장

3. **메모리 관리**
   - 대용량 이미지: ROI로 작업 영역 제한
   - 필요시 스트리밍 처리

### 11.3 사용자 경험 원칙

1. **인터랙티브 피드백**
   - 실시간 픽셀 검사
   - 즉시 시각화 업데이트

2. **다중 모드 지원**
   - 라벨링: 사용자/클래스맵/AI 추천
   - 분류: SAM/SID/SCC 메트릭 선택
   - ROI: 사각형/자유곡선

3. **레이어 기반 시각화**
   - 다중 레이어 오버레이
   - 가시성 제어
   - Z-order 관리

---

## 📝 12단계: 주요 파일 매핑

### 12.1 진입점
- `app.py`: 애플리케이션 진입점, 초기화, HighDPI 설정

### 12.2 뷰 계층
- `views/main_window.py`: 메인 윈도우 (7110+ 라인)
- `views/mapview.py`: 이미지 표시 및 상호작용
- `views/dialogs/*.py`: 다이얼로그 (15개)
- `views/docks/*.py`: 도크 위젯 (3개)

### 12.3 컨트롤러 계층
- `controllers/pixel_click.py`: 픽셀 클릭 처리
- `controllers/roi_controller.py`: ROI 관리
- `controllers/class_roi_controller.py`: 클래스 ROI
- `controllers/analysis_selection_controller.py`: 분석 영역 선택

### 12.4 서비스 계층
- `services/layer_manager.py`: 레이어 관리
- `services/pixel_knn.py`: Top-K 유사도 계산
- `services/classmap_render.py`: 분류맵 렌더링
- `services/paletteService.py`: 팔레트 관리
- `services/label_store.py`: 라벨 저장
- `services/spec_library.py`: 스펙트럼 라이브러리 구축
- `services/resampling_cache.py`: 리샘플링 캐시 관리

### 12.5 코어 계층
- `core/autoclass.py`: 자동 분류 알고리즘
- `core/metrics.py`: 유사도 메트릭 (SAM/SID/SCC)
- `core/resampling_service.py`: 리샘플링 및 CR
- `core/vis.py`: 시각화 유틸리티
- `core/region_growing.py`: 영역 확장 알고리즘
- `core/vlm_generation.py`: VLM 생성

### 12.6 데이터 계층
- `data/db.py`: API 연동 함수들
- `dataio/sidecar.py`: 사이드카 파일 (.info) 처리
- `models/labels.py`: 라벨 데이터 모델

---

## 🔚 결론

### 프로젝트 특징 요약

1. **전문성**: 초분광 이미지 분석에 특화된 전문 도구
2. **확장성**: 모듈화된 구조로 기능 추가 용이
3. **유연성**: Personal/Server 이중 모드 지원
4. **인터랙티브**: 실시간 피드백 및 다중 모드 지원
5. **시각화**: 다층 레이어 및 다양한 오버레이 지원

### 주요 강점

- ✅ 체계적인 MVC 아키텍처
- ✅ SSOT 기반 상태 관리
- ✅ 의존성 주입을 통한 느슨한 결합
- ✅ 캐싱을 통한 성능 최적화
- ✅ 다양한 분석 알고리즘 지원

### 개선 가능 영역

- 🔄 대용량 이미지 처리 최적화 (스트리밍)
- 🔄 멀티스레딩을 통한 UI 반응성 향상
- 🔄 단위 테스트 커버리지 확대
- 🔄 문서화 보강 (API 문서)

---

**작성일**: 2026-02-09  
**분석 방식**: Sequential Thinking  
**버전**: 0.1.0
