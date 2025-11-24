# examples/region_growing_example.py
"""
Region Growing 사용 예시

이 예시는 다음과 같은 기능을 보여줍니다:
1. HSI 데이터에서 Region Growing 수행
2. 다양한 메트릭과 임계값 사용
3. 결과를 레이어로 추가
"""

import numpy as np
from pathlib import Path
import sys

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.region_growing import RegionGrowing, RegionGrowingConfig
from services.region_growing_service import RegionGrowingService


def example_basic_usage():
    """기본 사용법 예시"""
    print("=== 기본 Region Growing 사용법 ===")
    
    # 1. HSI 데이터 준비 (예시: 100x100x50 크기)
    hsi_data = np.random.rand(100, 100, 50).astype(np.float32)
    
    # 2. Region Growing 설정
    config = RegionGrowingConfig(
        metric="SAD",           # 거리 메트릭: SAD, SID, SCC
        threshold=0.05,         # 임계값 (낮을수록 엄격)
        min_region_size=10,     # 최소 영역 크기
        max_region_size=1000,   # 최대 영역 크기
        connectivity=4          # 연결성: 4 또는 8
    )
    
    # 3. Region Growing 수행
    region_grower = RegionGrowing(hsi_data, config)
    
    # 4. 시드 좌표에서 영역 성장
    result = region_grower.grow_region(seed_x=50, seed_y=50)
    
    # 5. 결과 확인
    print(f"영역 크기: {result.region_size}")
    print(f"바운딩 박스: {result.bounding_box}")
    print(f"컴팩트니스: {result.statistics['compactness']:.3f}")
    
    return result


def example_service_usage():
    """서비스를 통한 사용법 예시"""
    print("\n=== Region Growing 서비스 사용법 ===")
    
    # 1. HSI 데이터 준비
    hsi_data = np.random.rand(80, 80, 30).astype(np.float32)
    
    # 2. 서비스 생성 및 설정
    service = RegionGrowingService()
    service.set_hsi_data(hsi_data)
    service.set_config(
        metric="SAD",
        threshold=0.08,
        min_region_size=15,
        max_region_size=500
    )
    
    # 3. Region Growing 수행
    result = service.grow_region_from_click(x=40, y=40)
    
    if result:
        print(f"서비스를 통한 Region Growing 성공!")
        print(f"  - 영역 크기: {result.region_size}")
        print(f"  - 평균 거리: {result.statistics['average_distance']:.4f}")
    else:
        print("Region Growing 실패")
    
    return result


def example_custom_spectrum():
    """커스텀 스펙트럼을 사용한 예시"""
    print("\n=== 커스텀 스펙트럼 사용 예시 ===")
    
    # 1. HSI 데이터 준비
    hsi_data = np.random.rand(60, 60, 20).astype(np.float32)
    
    # 2. 커스텀 스펙트럼 생성 (예: 특정 파장 대역에서 높은 값)
    custom_spectrum = np.zeros(20, dtype=np.float32)
    custom_spectrum[5:15] = 0.9  # 특정 밴드에서 높은 값
    
    # 3. Region Growing 설정
    config = RegionGrowingConfig(
        metric="SAD",
        threshold=0.1,
        min_region_size=5,
        max_region_size=300
    )
    
    # 4. 커스텀 스펙트럼으로 Region Growing 수행
    region_grower = RegionGrowing(hsi_data, config)
    result = region_grower.grow_region(
        seed_x=30, 
        seed_y=30, 
        seed_spectrum=custom_spectrum
    )
    
    print(f"커스텀 스펙트럼으로 성장된 영역 크기: {result.region_size}")
    
    return result


def example_multiple_regions():
    """다중 영역 성장 예시"""
    print("\n=== 다중 영역 성장 예시 ===")
    
    # 1. HSI 데이터 준비
    hsi_data = np.random.rand(70, 70, 25).astype(np.float32)
    
    # 2. 여러 시드 좌표 정의
    seeds = [(10, 10), (30, 30), (50, 50)]
    
    # 3. Region Growing 설정
    config = RegionGrowingConfig(
        metric="SAD",
        threshold=0.12,
        min_region_size=8,
        max_region_size=400
    )
    
    # 4. 다중 Region Growing 수행
    region_grower = RegionGrowing(hsi_data, config)
    results = region_grower.grow_multiple_regions(seeds)
    
    print(f"시드 개수: {len(seeds)}")
    print(f"성공한 영역 개수: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"영역 {i+1}: 크기={result.region_size}, "
              f"컴팩트니스={result.statistics['compactness']:.3f}")
    
    return results


def example_different_parameters():
    """다양한 파라미터로 Region Growing 비교"""
    print("\n=== 다양한 파라미터 비교 ===")
    
    # 1. HSI 데이터 준비
    hsi_data = np.random.rand(50, 50, 15).astype(np.float32)
    
    # 2. 다양한 설정으로 테스트
    test_configs = [
        {"metric": "SAD", "threshold": 0.05, "name": "SAD 엄격"},
        {"metric": "SAD", "threshold": 0.15, "name": "SAD 관대"},
        {"metric": "SID", "threshold": 0.1, "name": "SID 중간"},
        {"metric": "SCC", "threshold": 0.2, "name": "SCC 관대"}
    ]
    
    results = {}
    
    for config_params in test_configs:
        name = config_params.pop("name")
        
        config = RegionGrowingConfig(
            min_region_size=5,
            max_region_size=200,
            **config_params
        )
        
        region_grower = RegionGrowing(hsi_data, config)
        result = region_grower.grow_region(seed_x=25, seed_y=25)
        
        results[name] = result
        print(f"{name}: 크기={result.region_size}, "
              f"평균거리={result.statistics['average_distance']:.4f}")
    
    return results


def main():
    """메인 예시 함수"""
    print("Region Growing 사용 예시")
    print("=" * 40)
    
    try:
        # 1. 기본 사용법
        result1 = example_basic_usage()
        
        # 2. 서비스 사용법
        result2 = example_service_usage()
        
        # 3. 커스텀 스펙트럼 사용법
        result3 = example_custom_spectrum()
        
        # 4. 다중 영역 성장
        results4 = example_multiple_regions()
        
        # 5. 다양한 파라미터 비교
        results5 = example_different_parameters()
        
        print("\n" + "=" * 40)
        print("모든 예시 완료!")
        
    except Exception as e:
        print(f"예시 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
