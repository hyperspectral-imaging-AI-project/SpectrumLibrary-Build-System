# services/transparency_service.py
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any
from PyQt5.QtGui import QColor
from services.opacity import compute_alpha_from_gmin, build_opacity_overlay_rgba

class TransparencyService:
    """
    투명도 기반 클래스 표출 서비스
    
    투명도 규칙:
    - x < Strict: 투명도 없이 맵에 표출 (alpha = 255)
    - Strict < x < Base: 투명도를 조금 주고 맵에 표출 (alpha = 180)
    - Base < x: 투명도를 많이 주고 맵에 표출 (alpha = 80)
    """
    
    def __init__(self):
        self.strict_threshold: float = 0.02
        self.base_threshold: float = 0.05
        
    def set_thresholds(self, strict: float, base: float):
        """Strict와 Base 임계값 설정"""
        self.strict_threshold = float(strict)
        self.base_threshold = float(base)
        
    def create_transparency_overlay(self, 
                                   classmap: np.ndarray, 
                                   confidence_scores: np.ndarray,
                                   target_class_id: int,
                                   palette: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
        """
        선택된 클래스에 대해서만 투명도 기반 오버레이 생성 (기존 opacity.py 함수 재사용)
        
        Args:
            classmap: 클래스맵 (H, W)
            confidence_scores: 신뢰도 점수 맵 (H, W)
            target_class_id: 대상 클래스 ID
            palette: 클래스 ID -> RGB 색상 팔레트
            
        Returns:
            RGBA 오버레이 (H, W, 4)
        """
        # 크기 불일치 검증 및 처리
        if classmap.shape != confidence_scores.shape:
            print(f"경고: classmap 크기 {classmap.shape}와 confidence_scores 크기 {confidence_scores.shape}가 다릅니다.")
            print("confidence_scores를 classmap 크기에 맞춰 리샘플링합니다.")
            
            # confidence_scores를 classmap 크기에 맞춰 리샘플링
            from scipy.ndimage import zoom
            zoom_factors = (classmap.shape[0] / confidence_scores.shape[0], 
                          classmap.shape[1] / confidence_scores.shape[1])
            confidence_scores = zoom(confidence_scores, zoom_factors, order=1)
        
        # 1) 전체 클래스맵에 대해 alpha 맵 생성 (기존 함수 재사용)
        alpha_map = compute_alpha_from_gmin(confidence_scores, self.strict_threshold, self.base_threshold)
        if alpha_map is None:
            return np.zeros((*classmap.shape, 4), dtype=np.uint8)
        
        # 2) 대상 클래스만 필터링된 클래스맵 생성
        target_classmap = np.where(classmap == target_class_id, classmap, -999)  # 대상 클래스만 유지
        
        # 3) 기존 함수로 RGBA 오버레이 생성
        overlay = build_opacity_overlay_rgba(
            classmap=target_classmap,
            palette_rgb=palette,
            alpha=alpha_map,
            fallback_color_fn=lambda cid: (128, 128, 128) if cid != target_class_id else None
        )
        
        return overlay
        
    def create_multi_class_transparency_overlay(self,
                                              classmap: np.ndarray,
                                              confidence_scores: np.ndarray,
                                              selected_classes: list[int],
                                              palette: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
        """
        여러 선택된 클래스에 대해 투명도 기반 오버레이 생성
        
        Args:
            classmap: 클래스맵 (H, W)
            confidence_scores: 신뢰도 점수 맵 (H, W)
            selected_classes: 선택된 클래스 ID 리스트
            palette: 클래스 ID -> RGB 색상 팔레트
            
        Returns:
            RGBA 오버레이 (H, W, 4)
        """
        # 크기 불일치 검증 및 처리
        if classmap.shape != confidence_scores.shape:
            print(f"경고: classmap 크기 {classmap.shape}와 confidence_scores 크기 {confidence_scores.shape}가 다릅니다.")
            print("confidence_scores를 classmap 크기에 맞춰 리샘플링합니다.")
            
            # confidence_scores를 classmap 크기에 맞춰 리샘플링
            from scipy.ndimage import zoom
            zoom_factors = (classmap.shape[0] / confidence_scores.shape[0], 
                          classmap.shape[1] / confidence_scores.shape[1])
            confidence_scores = zoom(confidence_scores, zoom_factors, order=1)
        
        H, W = classmap.shape
        overlay = np.zeros((H, W, 4), dtype=np.uint8)
        
        for class_id in selected_classes:
            class_mask = (classmap == class_id)
            
            if not class_mask.any():
                continue
                
            # 신뢰도 점수에 따른 투명도 계산
            class_scores = confidence_scores[class_mask]
            alphas = np.array([self.calculate_alpha(score) for score in class_scores])
            
            # 색상 설정
            if class_id in palette:
                r, g, b = palette[class_id]
                overlay[class_mask, 0] = r
                overlay[class_mask, 1] = g
                overlay[class_mask, 2] = b
                overlay[class_mask, 3] = alphas
                
        return overlay
        
    def calculate_alpha(self, score: float) -> int:
        """
        신뢰도 점수에 따른 알파값 계산
        
        Args:
            score: 신뢰도 점수
            
        Returns:
            알파값 (0-255)
        """
        if not np.isfinite(score):
            return 0
        
        if score < self.strict_threshold:
            return 255
        elif score < self.base_threshold:
            return 160
        else:
            return 80


    def build_confmap_for_selected_class(
        self,
        *,
        classmap: np.ndarray,          # (H, W) int
        topk_vals: np.ndarray,         # (H, W, K) 또는 (h, w, K)
        target_class_id: int,
        roi_mask: Optional[np.ndarray] = None,  # (H, W) bool (ROI 분류였으면 필수)
    ) -> np.ndarray:
        """선택 클래스 영역에만 1순위 점수를 채운 (H,W) 신뢰도 맵 반환."""
        if topk_vals.ndim != 3:
            raise ValueError("topk_vals must be (H,W,K) or (h,w,K)")

        H, W = classmap.shape
        # 1등 점수
        if topk_vals.shape[0] == H and topk_vals.shape[1] == W:
            sub0 = topk_vals[:, :, 0]  # (H,W)
            conf_full = sub0.copy()
        else:
            if roi_mask is None or not np.any(roi_mask):
                raise ValueError("ROI-sized topk_vals needs roi_mask")
            ys, xs = np.where(roi_mask)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            sub0 = topk_vals[:, :, 0]               # (h,w)
            conf_full = np.zeros((H, W), dtype=sub0.dtype)
            roi_sub = roi_mask[y0:y1, x0:x1]
            if roi_sub.shape == sub0.shape:
                conf_full[y0:y1, x0:x1][roi_sub] = sub0[roi_sub]
            else:
                # shape 불일치면 bbox 통째로 채움(최소 보장)
                conf_full[y0:y1, x0:x1] = sub0

        # 선택 클래스 외 영역은 0
        conf_full[classmap != int(target_class_id)] = 0
        return conf_full