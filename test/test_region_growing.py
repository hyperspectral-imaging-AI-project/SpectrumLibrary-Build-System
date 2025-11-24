# test/test_region_growing.py
"""
Region Growing 알고리즘 테스트 및 사용 예시
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.region_growing import RegionGrowing, RegionGrowingConfig, RegionGrowingResult
from services.region_growing_service import RegionGrowingService


def create_test_hsi_data(height: int = 100, width: int = 100, channels: int = 50) -> np.ndarray:
    """
    테스트용 HSI 데이터 생성
    - 3개의 서로 다른 스펙트럼 영역을 가진 합성 이미지
    """
    hsi_data = np.zeros((height, width, channels), dtype=np.float32)
    
    # 영역 1: 좌상단 (높은 값)
    hsi_data[:height//2, :width//2, :] = 0.8
    
    # 영역 2: 우하단 (중간 값)
    hsi_data[height//2:, width//2:, :] = 0.5
    
    # 영역 3: 나머지 영역 (낮은 값)
    hsi_data[:height//2, width//2:, :] = 0.2
    hsi_data[height//2:, :width//2, :] = 0.2
    
    # 노이즈 추가
    noise = np.random.normal(0, 0.05, hsi_data.shape).astype(np.float32)
    hsi_data += noise
    
    # 값 범위 제한
    hsi_data = np.clip(hsi_data, 0, 1)
    
    return hsi_data


def test_basic_region_growing():
    """기본 Region Growing 테스트"""
    print("=== 기본 Region Growing 테스트 ===")
    
    # 테스트 데이터 생성
    hsi_data = create_test_hsi_data(50, 50, 20)
    print(f"HSI 데이터 크기: {hsi_data.shape}")
    
    # 설정 생성
    config = RegionGrowingConfig(
        metric="SAD",
        threshold=0.1,
        min_region_size=5,
        max_region_size=1000,
        connectivity=4
    )
    
    # Region Growing 수행
    region_grower = RegionGrowing(hsi_data, config)
    
    # 좌상단 영역에서 시작
    result = region_grower.grow_region(seed_x=10, seed_y=10)
    
    print(f"시드 좌표: {result.seed_coordinate}")
    print(f"영역 크기: {result.region_size}")
    print(f"바운딩 박스: {result.bounding_box}")
    print(f"통계 정보: {result.statistics}")
    
    return result


def test_multiple_region_growing():
    """다중 Region Growing 테스트"""
    print("\n=== 다중 Region Growing 테스트 ===")
    
    # 테스트 데이터 생성
    hsi_data = create_test_hsi_data(60, 60, 15)
    
    # 설정
    config = RegionGrowingConfig(
        metric="SAD",
        threshold=0.15,
        min_region_size=3,
        max_region_size=500,
        connectivity=8
    )
    
    # Region Growing 수행
    region_grower = RegionGrowing(hsi_data, config)
    
    # 여러 시드에서 시작
    seeds = [(10, 10), (30, 30), (50, 50)]
    results = region_grower.grow_multiple_regions(seeds)
    
    print(f"시드 개수: {len(seeds)}")
    print(f"성공한 영역 개수: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"영역 {i+1}: 크기={result.region_size}, "
              f"컴팩트니스={result.statistics['compactness']:.3f}")
    
    return results


def test_region_growing_service():
    """Region Growing 서비스 테스트"""
    print("\n=== Region Growing 서비스 테스트 ===")
    
    # 테스트 데이터 생성
    hsi_data = create_test_hsi_data(40, 40, 10)
    
    # 서비스 생성 및 설정
    service = RegionGrowingService()
    service.set_hsi_data(hsi_data)
    service.set_config(
        metric="SAD",
        threshold=0.12,
        min_region_size=5,
        max_region_size=300,
        connectivity=4
    )
    
    # Region Growing 수행
    result = service.grow_region_from_click(x=15, y=15)
    
    if result:
        print(f"서비스를 통한 Region Growing 성공:")
        print(f"  - 영역 크기: {result.region_size}")
        print(f"  - 평균 거리: {result.statistics['average_distance']:.4f}")
        print(f"  - 스펙트럼 분산: {result.statistics['spectral_variance']:.6f}")
    else:
        print("서비스를 통한 Region Growing 실패")
    
    return result


def test_different_metrics():
    """다양한 메트릭으로 Region Growing 테스트"""
    print("\n=== 다양한 메트릭 테스트 ===")
    
    # 테스트 데이터 생성
    hsi_data = create_test_hsi_data(30, 30, 8)
    
    metrics = ["SAD", "SID", "SCC"]
    results = {}
    
    for metric in metrics:
        print(f"\n메트릭: {metric}")
        
        config = RegionGrowingConfig(
            metric=metric,
            threshold=0.1,
            min_region_size=3,
            max_region_size=200,
            connectivity=4
        )
        
        region_grower = RegionGrowing(hsi_data, config)
        result = region_grower.grow_region(seed_x=15, seed_y=15)
        
        results[metric] = result
        print(f"  - 영역 크기: {result.region_size}")
        print(f"  - 평균 거리: {result.statistics['average_distance']:.4f}")
    
    return results


def test_different_thresholds():
    """다양한 임계값으로 Region Growing 테스트"""
    print("\n=== 다양한 임계값 테스트 ===")
    
    # 테스트 데이터 생성
    hsi_data = create_test_hsi_data(35, 35, 12)
    
    thresholds = [0.05, 0.1, 0.2, 0.3]
    results = {}
    
    for threshold in thresholds:
        print(f"\n임계값: {threshold}")
        
        config = RegionGrowingConfig(
            metric="SAD",
            threshold=threshold,
            min_region_size=2,
            max_region_size=500,
            connectivity=4
        )
        
        region_grower = RegionGrowing(hsi_data, config)
        result = region_grower.grow_region(seed_x=17, seed_y=17)
        
        results[threshold] = result
        print(f"  - 영역 크기: {result.region_size}")
        print(f"  - 컴팩트니스: {result.statistics['compactness']:.3f}")
    
    return results


def visualize_region_growing_result(result: RegionGrowingResult, hsi_data: np.ndarray):
    """Region Growing 결과 시각화"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 원본 HSI 데이터 (첫 번째 밴드)
        axes[0].imshow(hsi_data[:, :, 0], cmap='gray')
        axes[0].set_title('Original HSI (First Band)')
        axes[0].axis('off')
        
        # Region Growing 결과 마스크
        axes[1].imshow(result.region_mask, cmap='Reds')
        axes[1].set_title(f'Region Growing Result\nSize: {result.region_size}')
        axes[1].axis('off')
        
        # 시드 위치 표시
        axes[2].imshow(hsi_data[:, :, 0], cmap='gray')
        axes[2].imshow(result.region_mask, cmap='Reds', alpha=0.5)
        axes[2].plot(result.seed_coordinate[1], result.seed_coordinate[0], 
                    'bo', markersize=10, label='Seed')
        axes[2].set_title('Overlay with Seed')
        axes[2].legend()
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib가 설치되지 않아 시각화를 건너뜁니다.")


def main():
    """메인 테스트 함수"""
    print("Region Growing 알고리즘 테스트 시작")
    print("=" * 50)
    
    try:
        # 1. 기본 테스트
        result1 = test_basic_region_growing()
        
        # 2. 다중 영역 테스트
        results2 = test_multiple_region_growing()
        
        # 3. 서비스 테스트
        result3 = test_region_growing_service()
        
        # 4. 다양한 메트릭 테스트
        results4 = test_different_metrics()
        
        # 5. 다양한 임계값 테스트
        results5 = test_different_thresholds()
        
        print("\n" + "=" * 50)
        print("모든 테스트 완료!")
        
        # 시각화 (matplotlib 사용 가능한 경우)
        if result1:
            hsi_data = create_test_hsi_data(50, 50, 20)
            visualize_region_growing_result(result1, hsi_data)
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
