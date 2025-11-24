# services/layer_manager.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

UNKNOWN  = -1
MULTIPLE = -2

@dataclass
class LayerRec:
    name: str
    type: str
    parent: Optional[str] = None
    map_item_name: Optional[str] = None
    visible: bool = True

class LayerManager:
    """
    - MainWindow 대신 레이어 등록/분해/상태를 전담
    - MapView/LayersDock는 콜백/함수로 주입 받아 호출
    """
    def __init__(self,
                map_add_layer, map_set_visible, map_remove_layer, map_reorder,
                dock_add_layer,           # ← 단일 API
                dock_has_layer=None,
                dock_remove_layer=None):
        self.map_add_layer = map_add_layer
        self.map_set_visible = map_set_visible
        self.map_remove_layer = map_remove_layer
        self.map_reorder = map_reorder
        self.dock_add_layer = dock_add_layer
        self.dock_has_layer = dock_has_layer or (lambda name: False)
        self.dock_remove_layer = dock_remove_layer or (lambda name: None)

        self.recs: Dict[str, LayerRec] = {}
        # ★ 추가: 정렬/데이터 저장소
        self._top_order: list[str] = []          # top-level 이름들의 위→아래 순서
        self._classmap_store: Dict[str, np.ndarray] = {}  # parent(top-level) → int32(H,W)
        self._class_label_map: Dict[str, Dict[int, str]] = {}  # parent → {cid: label}
        self._roi: Optional[np.ndarray] = None   # ROI bool(H,W)
        

    # ---- 외부 서비스 주입(팔레트/렌더러) ----
    def set_services(self, palette_service, classmap_renderer):
        self.palette = palette_service
        self.renderer = classmap_renderer

    # services/layer_manager.py
    def register_rgb_base(self, name: str, rgb_u8: np.ndarray, visible=True, display_label: Optional[str] = None):
        """
        RGB 베이스 레이어 등록.
        
        Args:
            name: 내부 식별자 키 (예: "image.rgb")
            rgb_u8: RGB 이미지 배열 (H, W, 3) uint8
            visible: 초기 가시성
            display_label: Layers Dock에 표시할 이름 (None이면 name 사용)
        """
        if not self.dock_has_layer(name):
            self.dock_add_layer(name=name, checked=True, parent=None, display_label=display_label)

        self.recs[name] = LayerRec(
            name=name, type='rgb_base', parent=None, map_item_name="RGB", visible=bool(visible)
        )

        # ✅ 수정: 이미 _top_order에 있으면 순서 유지, 없을 때만 맨 위에 추가
        if name not in self._top_order:
            self._top_order.insert(0, name)

        try:
            self.map_set_visible("RGB", bool(visible))
        except Exception:
            pass

        self._apply_order_to_map()

    @staticmethod
    def _extract_label(value: Any) -> str:
        """Pick a human-readable label from various structures."""
        if isinstance(value, dict):
            for key in ("mtrl_nm", "name", "label", "desc", "display", "mtrl_name"):
                text = value.get(key)
                if text:
                    return str(text)
            for text in value.values():
                if text:
                    return str(text)
            return ""
        if value is None:
            return ""
        return str(value)

    def _normalize_label_map(self, raw: Any) -> Dict[int, str]:
        """Normalize id→label inputs to {cid:int -> label:str}."""
        normalized: Dict[int, str] = {}
        if not raw:
            return normalized

        if isinstance(raw, dict):
            for key, val in raw.items():
                try:
                    cid = int(key)
                except (TypeError, ValueError):
                    continue
                label = self._extract_label(val)
                if label:
                    normalized[cid] = label
        elif isinstance(raw, (list, tuple)):
            for rec in raw:
                if not isinstance(rec, dict):
                    continue
                cid = rec.get("mtrl_cd") or rec.get("cid") or rec.get("class_id")
                if cid is None:
                    continue
                try:
                    cid_i = int(cid)
                except (TypeError, ValueError):
                    continue
                label = self._extract_label(rec)
                if label:
                    normalized[cid_i] = label

        return normalized

    def register_classmap(self, name: str, classmap_i32: np.ndarray, class_ids: list[int], visible=True, id_to_name: Any | None = None):
        from core.autoclass import UNKNOWN, MULTIPLE

        # ★ 기존 classmap이 있으면 먼저 모든 자식 레이어 완전 제거 (중복 방지)
        is_existing = name in self.recs and name in self._classmap_store
        if is_existing:
            # 기존 자식 레이어들 모두 제거
            old_children = [k for k, v in self.recs.items() if v.parent == name]
            for ch_name in old_children:
                try:
                    ch_rec = self.recs.get(ch_name)
                    if ch_rec and ch_rec.map_item_name:
                        self.map_remove_layer(ch_rec.map_item_name)
                    if self.dock_remove_layer:
                        self.dock_remove_layer(ch_name)
                    self.recs.pop(ch_name, None)
                    self._class_label_map.pop(ch_name, None)
                except Exception:
                    pass

        # 1) 도크 top-level 보장
        if not self.dock_has_layer(name):
            self.dock_add_layer(name=name, checked=True, parent=None)

        # 2) 저장소 업데이트 (항상 2D int32)
        cm = classmap_i32.astype(np.int32, copy=False)
        self._classmap_store[name] = cm

        # 3) 등록 시점에만 최상단
        if name in self._top_order:
            self._top_order.remove(name)
        self._top_order.insert(0, name)

        # ★ 부모 전체 합성 오버레이도 함께 올리기 (없애지 않음, 기본 비가시 + 중복 제거)
        try:
            # 0) 이전 부모 합성 레이어 제거(중복/잔존 방지)
            old_parent = self.recs.get(name)
            if old_parent and old_parent.map_item_name and old_parent.map_item_name != "RGB":
                try:
                    self.map_remove_layer(old_parent.map_item_name)
                except Exception:
                    pass

            # 1) 새 부모 합성 생성
            rgba_all = self.renderer.classmap_rgba(cm, alpha=180)
            parent_item = self.map_add_layer(name, rgba_all, 1.0, False)

            # 2) 기본 비가시로 등록(자식 토글이 즉시 반영되도록)
            self.recs[name] = LayerRec(
                name=name, type="classmap.parent", parent=None,
                map_item_name=parent_item, visible=False
            )
            self.map_set_visible(parent_item, False)
        except Exception:
            pass

        # 4) 실제 등장 클래스(UNKNOWN 포함) 추출
        present = set(int(i) for i in np.unique(cm))
        target_ids = sorted(present)

        # 4-1) 클래스 라벨 매핑 업데이트
        if id_to_name is not None:
            self._class_label_map[name] = self._normalize_label_map(id_to_name)
        label_map = self._class_label_map.get(name, {})
        if name not in self._class_label_map:
            self._class_label_map[name] = label_map
        # UNKNOWN/MULTIPLE 기본 라벨 보강
        label_map.setdefault(UNKNOWN, "미분류")
        label_map.setdefault(MULTIPLE, "중복 클래스")

        # ★ SSOT 팔레트: Renderer에서 직접 참조 (PaletteService 사용 안 함)
        pal_rgb = {}
        try:
            if hasattr(self, "renderer") and hasattr(self.renderer, "get_palette_ref_rgb"):
                pal_rgb = self.renderer.get_palette_ref_rgb()  # {cid: (R,G,B)}
        except Exception:
            pal_rgb = {}

        # 표시 알파(지도 오버레이 블렌딩)
        ALPHA_NORMAL   = 180
        ALPHA_MULTIPLE = 180  # 필요시 220으로 올려도 됨

        # 5) 자식 생성/갱신
        new_children = set()
        for cid in target_ids:
            cid_i = int(cid)

            # SSOT 팔레트만 사용(임의 회색 fallback 금지)
            if cid_i == UNKNOWN:
                rgb = pal_rgb.get(UNKNOWN, (128, 128, 128))
                alpha = ALPHA_NORMAL
            elif cid_i == MULTIPLE:
                rgb = pal_rgb.get(MULTIPLE, (0, 0, 0))  # SSOT에 없으면 마지막 수단: 검정
                alpha = ALPHA_MULTIPLE
            else:
                rgb = pal_rgb.get(cid_i, None)
                if rgb is None:
                    # 엄격 모드: 팔레트에 없는 클래스는 스킵
                    # (느슨 모드 원하면 MainWindow에서 _augment_palette_for로 SSOT에 색을 먼저 채워 넣으세요)
                    continue
                alpha = ALPHA_NORMAL

            # 자식 등록 시: 내부 키는 유지, 도크에는 표시 라벨 전달
            child_key = f"{name}/{cid_i}"
            label_text = label_map.get(cid_i)
            if cid_i == UNKNOWN:
                display_tail = "미분류"
            elif cid_i == MULTIPLE:
                display_tail = "중복 클래스"
            elif label_text:
                display_tail = str(label_text)
            else:
                display_tail = str(cid_i)
            display_label = str(display_tail)

            rgba = self.renderer.class_mask_rgba(cm, cid_i, rgb, alpha)
            _ = self.map_add_layer(child_key, rgba, 1.0, False)
            self.recs[child_key] = LayerRec(
                name=child_key, type="classmap", parent=name, map_item_name=child_key, visible=True
            )

            # LayersDock에 사용자 표시 라벨 제공
            self.dock_add_layer(name=child_key, checked=True, parent=name, display_label=display_label)
            new_children.add(child_key)

        # 6) 이전에 있었지만 이번엔 사라진 자식은 완전 제거 (위에서 이미 제거했으므로 여기는 방어 코드)
        old_children = {k for k, v in self.recs.items() if v.parent == name} - new_children
        for ch in old_children:
            try:
                ch_rec = self.recs.get(ch)
                if ch_rec and ch_rec.map_item_name:
                    self.map_remove_layer(ch_rec.map_item_name)
                if self.dock_remove_layer:
                    self.dock_remove_layer(ch)
                self.recs.pop(ch, None)
                self._class_label_map.pop(ch, None)
            except Exception:
                pass

        # 6-1) 라벨 매핑에서 존재하지 않는 CID 정리
        if name in self._class_label_map:
            kept = {cid: txt for cid, txt in self._class_label_map[name].items() if cid in target_ids}
            if kept:
                self._class_label_map[name] = kept
            else:
                self._class_label_map.pop(name, None)

        # 7) 정렬 적용
        self._apply_order_to_map()

    def register_overlay(self, name: str, rgba_or_rgb_u8: np.ndarray, visible=True, parent: str | None = None):
        if parent:
            # 부모 없으면 생성 + 등록 시점에만 부모를 최상단
            if not self.dock_has_layer(parent):
                self.dock_add_layer(name=parent, checked=True, parent=None)
                if parent in self._top_order:
                    self._top_order.remove(parent)
                self._top_order.insert(0, parent)

            item = self.map_add_layer(name, rgba_or_rgb_u8, 1.0, True)
            self.recs[name] = LayerRec(name=name, type="overlay", parent=parent, map_item_name=item, visible=bool(visible))
            self.dock_add_layer(name=name, checked=bool(visible), parent=parent)

        else:
            if not self.dock_has_layer(name):
                self.dock_add_layer(name=name, checked=bool(visible), parent=None)

            # 부모 없는 단일 오버레이는 자기 자신을 최상단
            if name in self._top_order:
                self._top_order.remove(name)
            self._top_order.insert(0, name)

            item = self.map_add_layer(name, rgba_or_rgb_u8, 1.0, True)
            self.recs[name] = LayerRec(name=name, type="overlay", parent=None, map_item_name=item, visible=bool(visible))
            self.map_set_visible(item, bool(visible))

        # 정렬 반영
        self._apply_order_to_map()

    # ---- 가시성/정렬/삭제 ----
    def set_visible(self, dock_name: str, visible: bool):
        rec = self.recs.get(dock_name)
        if not rec: return
        target = "RGB" if rec.map_item_name == "RGB" else rec.map_item_name
        self.map_set_visible(target, visible)
        rec.visible = visible

    def remove(self, dock_name: str):
        rec = self.recs.get(dock_name)

        # --- 부모(top-level) 또는 rec 누락 케이스 ---
        if rec is None or rec.parent is None:
            # 1) 자식 먼저 제거
            children = [k for k, v in self.recs.items() if v.parent == dock_name]
            for ch in children:
                ch_rec = self.recs.pop(ch, None)
                if ch_rec and ch_rec.map_item_name and ch_rec.map_item_name != "RGB":
                    try:
                        self.map_remove_layer(ch_rec.map_item_name)
                    except Exception:
                        pass

            # 2) 부모 맵 아이템도 제거(※ 누락 지점)
            if rec is not None:
                try:
                    if rec.map_item_name and rec.map_item_name != "RGB":
                        self.map_remove_layer(rec.map_item_name)
                except Exception:
                    pass
                # ROI였다면 내부 상태도 초기화
                if getattr(rec, "type", "") == "mask.roi":
                    try:
                        self._roi = None
                    except Exception:
                        pass
                
                # 수정: classmap_store에서도 제거하여 메모리 누수 방지
                if dock_name in self._classmap_store:
                    try:
                        del self._classmap_store[dock_name]
                    except Exception:
                        pass
                self._class_label_map.pop(dock_name, None)

            # 3) 부모 레코드 제거
            _ = self.recs.pop(dock_name, None)

            # 4) 도크에서 항목 제거
            try:
                self.dock_remove_layer(dock_name)
            except Exception:
                pass
            return

        # --- 자식(leaf) ---
        rec = self.recs.pop(dock_name, None)
        if rec and rec.map_item_name and rec.map_item_name != "RGB":
            try:
                self.map_remove_layer(rec.map_item_name)
            except Exception:
                pass
        parent_name = getattr(rec, "parent", None)
        if parent_name and parent_name in self._class_label_map:
            # 부모 라벨 맵에서 해당 CID 제거
            try:
                cid_str = dock_name.split("/", 1)[1]
                cid_int = int(cid_str)
            except (IndexError, ValueError):
                cid_int = None
            if cid_int is not None:
                labels = self._class_label_map.get(parent_name, {})
                if cid_int in labels:
                    labels.pop(cid_int, None)

    def _apply_order_to_map(self) -> None:
        """
        현재 self._top_order(위→아래)를 기준으로 평탄화된 위→아래 Map 레이어 이름 목록을 만들어
        주입받은 map_reorder(...)에 전달한다.
        - rgb_base: parent의 map_item_name('RGB') 포함
        - classmap: 부모는 자체 픽셀이 없고 자식('parent/cid')들만 포함
        - overlay/mask: 부모 자신을 포함
        """
        # 자식 인덱스 캐시: parent -> [child names]
        children_by_parent: Dict[str, list[str]] = {}
        for k, v in self.recs.items():
            if v.parent is not None:
                children_by_parent.setdefault(v.parent, []).append(k)

        flat_names: list[str] = []
        for parent in self._top_order:
            prec = self.recs.get(parent)

            # 1) 부모 자체가 실제 맵 아이템을 가지면 포함 (rgb_base/overlay 등)
            if prec and prec.map_item_name:
                flat_names.append(prec.map_item_name)

            # 2) 자식들(주로 classmap children). 정책상 이름 정렬로 고정
            for ch in sorted(children_by_parent.get(parent, [])):
                crec = self.recs.get(ch)
                if crec and crec.map_item_name:
                    flat_names.append(crec.map_item_name)

        try:
            if callable(self.map_reorder):
                self.map_reorder(flat_names)  # 위→아래 순서 그대로 전달
        except Exception:
            pass

    def reorder_top_to_bottom(self, names_top_to_bottom: list[str]) -> None:
        """
        LayersDock가 보내는 '부모+자식 평탄화 위→아래' 목록에서
        top-level 부모 순서만 추출해 저장하고 Map에 반영한다.
        """
        # 현재 보유한 top-level만 대상으로 부모 리스트 재구성
        seen = set()
        new_parents: list[str] = []
        for nm in names_top_to_bottom:
            parent = nm.split("/", 1)[0]  # "parent/cid" -> "parent"
            if parent in self._top_order and parent not in seen:
                seen.add(parent)
                new_parents.append(parent)

        # 누락된 기존 부모는 기존 순서 유지하며 뒤에 붙임
        for p in self._top_order:
            if p not in seen:
                new_parents.append(p)

        # 순서 변경 시에만 적용
        if new_parents != self._top_order:
            self._top_order = new_parents
            self._apply_order_to_map()

    def add(self, *, name: str, data: np.ndarray, type: str = "overlay",
            visible: bool = True, meta: dict | None = None) -> None:
        """
        공용 등록:
        - type == "mask.roi": ROI(bool 2D) 칠해서 올리고 내부 ROI 저장
        - type == "mask"    : 일반 마스크(bool 2D) 칠해서 올림
        - 그 외              : RGBA/RGB 배열을 오버레이로 올림
        """
        meta = meta or {}
        opacity = float(meta.get("opacity", 1.0))

        if type in ("mask", "mask.roi"):
            # bool 마스크 보장
            mask = (data.astype(bool) if isinstance(data, np.ndarray)
                    else np.asarray(data).astype(bool))
            if not np.any(mask):
                return  # 빈 마스크면 스킵(혼란 방지)

            h, w = mask.shape
            r, g, b, a = (*meta.get("color", (255, 0, 0, 180)),)
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[mask, 0] = r; rgba[mask, 1] = g; rgba[mask, 2] = b; rgba[mask, 3] = a

            if type == "mask.roi":
                self._roi = mask  # 내부 ROI 저장

            # 이름 그대로(in-place) 1개 아이템
            item = self.map_add_layer(name, rgba, opacity, False)  # force_unique=False

            self.recs[name] = LayerRec(name=name, type=type, parent=None,
                                    map_item_name=item, visible=bool(visible))
            if not self.dock_has_layer(name):
                self.dock_add_layer(name=name, checked=bool(visible), parent=None)

            # 등록 시점에만 최상단
            if name in self._top_order:
                self._top_order.remove(name)
            self._top_order.insert(0, name)

            self.map_set_visible(item, bool(visible))
            self._apply_order_to_map()
            return

        # ---- overlay / rgba ----
        item = self.map_add_layer(name, data, opacity, True)
        self.recs[name] = LayerRec(name=name, type=type, parent=None,
                                map_item_name=item, visible=bool(visible))
        if not self.dock_has_layer(name):
            self.dock_add_layer(name=name, checked=bool(visible), parent=None)

        if name in self._top_order:
            self._top_order.remove(name)
        self._top_order.insert(0, name)

        self.map_set_visible(item, bool(visible))
        self._apply_order_to_map()

    def set_roi(self, mask_bool: np.ndarray):
        self._roi = mask_bool.astype(bool, copy=False)

    def get_roi(self) -> Optional[np.ndarray]:
        return self._roi

    def get_classmap(self, parent_name: str) -> Optional[np.ndarray]:
        return self._classmap_store.get(parent_name)

    def get_mask(self, parent_name: str, cid: int) -> Optional[np.ndarray]:
        cm = self._classmap_store.get(parent_name)
        if cm is None:
            return None
        return (cm == int(cid))

    def get_classmap(self, parent_name: str):
        return getattr(self, "_classmap_store", {}).get(parent_name)

    def _merge_two_classmaps(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """픽셀 규칙: >=0 라벨 / -1 Unknown / -2 Multiple"""
        a = a.astype(np.int32, copy=False); b = b.astype(np.int32, copy=False)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")

        out = np.full_like(a, UNKNOWN, dtype=np.int32)
        # 1) MULTIPLE 우선
        m = (a == MULTIPLE) | (b == MULTIPLE)
        out[m] = MULTIPLE
        undec = (out == UNKNOWN)
        # 2) 동일 라벨
        same = undec & (a >= 0) & (b >= 0) & (a == b)
        out[same] = a[same]
        # 3) 한쪽만 라벨
        only_a = undec & (a >= 0) & (b < 0);  out[only_a] = a[only_a]
        only_b = undec & (a < 0) & (b >= 0);  out[only_b] = b[only_b]
        # 4) 서로 다른 라벨
        conflict = (out == UNKNOWN) & (a >= 0) & (b >= 0) & (a != b)
        out[conflict] = MULTIPLE
        return out

    def _merge_many_classmaps(self, maps: list[np.ndarray]) -> np.ndarray:
        if not maps:
            raise ValueError("no maps to merge")
        merged = maps[0].astype(np.int32, copy=False)
        for cm in maps[1:]:
            merged = self._merge_two_classmaps(merged, cm)
        return merged

    def merge_classmaps(self, parents: list[str], out_name: str | None = None) -> str:
        """
        부모 classmap 이름 목록을 병합해 새 부모 레이어를 등록하고 이름을 반환.
        - 레이어 등록까지 포함 (팔레트/도크/맵 반영)
        """
        # 1) 고유 부모만
        uniq = []
        for n in parents or []:
            p = n.split("/", 1)[0]
            if p not in uniq:
                uniq.append(p)
        if len(uniq) < 2:
            raise ValueError("need at least two parent classmaps")

        # 2) 가져오기/검증
        cms = []
        for p in uniq:
            cm = self.get_classmap(p)
            if cm is None:
                raise ValueError(f"classmap not found: {p}")
            cms.append(cm.astype(np.int32, copy=False))
        shape0 = cms[0].shape
        if any(cm.shape != shape0 for cm in cms):
            raise ValueError("classmap shapes mismatch")

        # 3) 병합
        merged = self._merge_many_classmaps(cms)

        # 4) class_ids(>=0) 계산
        class_ids = sorted(int(i) for i in np.unique(merged) if i >= 0)

        # 5) 이름
        if not out_name:
            out_name = "+".join(uniq[:2]) if len(uniq) == 2 else f"merge({len(uniq)})"

        # 6) 레이어 등록(기존 register_classmap 재사용)
        # 병합 결과 라벨도 합쳐 전달
        merged_labels: Dict[int, str] = {}
        for parent in uniq:
            merged_labels.update(self._class_label_map.get(parent, {}))

        self.register_classmap(
            name=out_name,
            classmap_i32=merged,
            class_ids=class_ids,
            visible=True,
            id_to_name=merged_labels if merged_labels else None,
        )
        return out_name
    
    # services/layer_manager.py (클래스 내부)
    def get_children(self, parent_name: str) -> list[str]:
        """부모 아래의 자식 레이어 이름 목록을 리턴."""
        return [k for k, v in self.recs.items() if v.parent == parent_name]
