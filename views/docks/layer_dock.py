# views/layers_dock.py
from __future__ import annotations
import sip
from typing import Callable, Optional, List, Dict
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QObject, QEvent, QTimer, QSignalBlocker
from PyQt5.QtWidgets import QDockWidget, QTreeWidget, QTreeWidgetItem, QMenu
from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout, QStyle, QHeaderView
from data.db import search_material_filtering_list
from PyQt5.QtGui import QColor, QIcon, QPixmap

class _LayerTree(QTreeWidget):
    def __init__(self, on_drop: Callable[[], None], on_ext: Callable[[List[str]], None] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_drop = on_drop
        self._on_ext  = on_ext
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        md = e.mimeData()
        ACCEPT = (".npz", ".npy")
        if md and md.hasUrls():
            for u in md.urls():
                if u.isLocalFile() and u.toLocalFile().lower().endswith(ACCEPT):
                    e.acceptProposedAction()
                    return
        super().dragEnterEvent(e)

    def dropEvent(self, e):
        """외부 파일(.npz/.npy) 드롭은 on_ext로 전달, 내부 재정렬은 부모 로직 처리."""
        ACCEPT = (".npz", ".npy")
        md = e.mimeData()
        # 1) 외부 파일 드롭 처리
        if md and md.hasUrls() and callable(self._on_ext):
            paths = [u.toLocalFile() for u in md.urls()
                     if u.isLocalFile() and u.toLocalFile().lower().endswith(ACCEPT)]
            if paths:
                try:
                    self._on_ext(paths)  # → LayersDock.requestLoadPaths.emit(paths)
                    e.acceptProposedAction()
                    return
                except Exception:
                    pass
        # 2) 내부 재정렬 드롭 처리 + 드롭 이후 콜백
        super().dropEvent(e)
        try:
            if callable(self._on_drop):
                self._on_drop()
        except Exception:
            pass

class _DropAfterHook(QObject):
    """QTreeWidget을 UI에서 로드했을 때 drop 이후에 콜백을 보장하기 위한 후크."""
    def __init__(self, on_drop: Callable[[], None]):
        super().__init__()
        self._on_drop = on_drop

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Drop:
            # drop 처리가 끝난 직후 호출되도록 0ms 타이머
            QTimer.singleShot(0, self._on_drop)
        return False

from PyQt5.QtCore import QObject, QEvent

class _ExternalNpyDropFilter(QObject):
    """
    QTreeWidget(또는 임의의 위젯)에 부착해서 외부 파일 드롭(.npz/.npy)을 가로채고,
    허용 확장자만 골라 콜백(on_ext)으로 경로 리스트를 넘겨주는 필터.

    사용 예)
        self.tree.setAcceptDrops(True)
        self._ext_filter = _ExternalNpyDropFilter(lambda paths: self.requestLoadPaths.emit(paths))
        self.tree.installEventFilter(self._ext_filter)

    참고:
      - DragEnter/DragMove 에서는 허용 확장자가 하나라도 보이면 acceptProposedAction().
      - Drop 시에는 허용 파일 경로만 추출해 on_ext(paths) 호출.
      - 기본 허용 확장자는 (".npz", ".npy"), 생성자에서 변경 가능.
    """
    def __init__(self, on_ext, accept_exts=(".npz", ".npy")):
        super().__init__()
        self._on_ext = on_ext
        # 소문자로 비교
        self._accept_exts = tuple(e.lower() for e in (accept_exts or (".npz", ".npy")))

    # --- 내부 유틸 ---
    def _iter_local_files(self, mime):
        """QMimeData에서 허용 확장자의 로컬 파일 경로만 추출."""
        try:
            if mime and mime.hasUrls():
                for u in mime.urls():
                    if not u.isLocalFile():
                        continue
                    p = u.toLocalFile()
                    if not p:
                        continue
                    if p.lower().endswith(self._accept_exts):
                        yield p
        except Exception:
            # 드문 케이스(특수 URL 스킴 등)는 조용히 무시
            return

    def eventFilter(self, obj, event):
        et = event.type()

        # 드래그가 진입/이동할 때: 허용 확장자가 하나라도 있으면 수락
        if et in (QEvent.DragEnter, QEvent.DragMove):
            try:
                mime = event.mimeData()
                for _ in self._iter_local_files(mime):
                    event.acceptProposedAction()
                    return True
            except Exception:
                return False
            return False

        # 드롭 시: 허용 파일만 모아서 콜백
        if et == QEvent.Drop:
            try:
                mime = event.mimeData()
                paths = list(self._iter_local_files(mime))
                if paths:
                    if callable(self._on_ext):
                        self._on_ext(paths)  # 예: self.requestLoadPaths.emit(paths)
                    event.acceptProposedAction()
                    return True
            except Exception:
                return False
            return False

        # 그 외 이벤트는 관여하지 않음
        return False


class LayersDock(QDockWidget):
    """
    QGIS 스타일 '트리형' 레이어 패널.
    - Top-level: 일반 레이어 / classmap 부모
    - Child: '부모/클래스명' 형태의 하위 레이어 (자식은 드래그 금지)
    - 체크박스: 가시성 토글 (부모/자식 각각 독립 토글)
    - 드래그&드롭: Top-level 순서 변경 시 drop 직후 on_reorder 호출 보장
    - 컨텍스트 메뉴: 수정 / 저장 / 삭제 / (다중) 클래스맵 합치기
    - ui/layers_dock.ui 가 있으면 그걸 사용하고, 없으면 코드로 생성(폴백)
    """

    # MainWindow로 보내는 액션들
    requestDeleteLayer   = pyqtSignal(str)     # 단일 삭제
    requestEditLayer     = pyqtSignal(str)     # 단일 수정
    requestClassMap      = pyqtSignal(list)    # 다중 클래스맵 병합
    requestSaveLayer     = pyqtSignal(str)     # 단일 저장
    requestROI           = pyqtSignal(str)   # ★ 클래스 속성으로 선언
    requestOpenWorkspace = pyqtSignal(str)   # ★ 추가: + 버튼 클릭 시 부모 이름 전달
    requestLoadPaths     = pyqtSignal(list)   # ★ 외부 파일 드롭 로드용
    requestOpenViewerBand = pyqtSignal(str)

    def __init__(
        self,
        parent=None,
        on_visible_change: Optional[Callable[[str, bool], None]] = None,
        on_reorder: Optional[Callable[[List[str]], None]] = None,
    ):
        super().__init__("Layers", parent)
        self.setObjectName("dockLayers")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self._on_visible_change = on_visible_change
        self._on_reorder = on_reorder

        # ---- UI 로딩 시도 ----
        self._drop_hook: Optional[_DropAfterHook] = None
        tree_from_ui = False
        try:
            ui_dir = getattr(parent, "ui_dir", None)
            if ui_dir is None:
                # 프로젝트 구조: <root>/views/layers_dock.py → <root>/ui
                ui_dir = Path(__file__).resolve().parents[2] / "ui"
            ui_path = Path(ui_dir) / "layers_dock.ui"
            if ui_path.exists():
                uic.loadUi(str(ui_path), self)
                tw = self.findChild(QTreeWidget, "tree")
                if tw is not None:
                    self.tree = tw
                    self._ensure_button_column()  # ★ 추가
                    tree_from_ui = True
        except Exception:
            tree_from_ui = False  # 폴백

        # ---- 폴백: 코드로 트리 생성 ----
        if not tree_from_ui:
            self.tree = _LayerTree(
                on_drop=lambda: (self._on_reorder and self._on_reorder(self._top_level_order_top_to_bottom())),
                on_ext=lambda paths: self.requestLoadPaths.emit(paths),
                parent=self,
            )
            self._ensure_button_column()          # ★ 추가
            self.tree.setHeaderHidden(True)
            self.tree.setExpandsOnDoubleClick(True)
            self.tree.setSelectionMode(self.tree.ExtendedSelection)
            self.tree.setDragEnabled(True)          # Top-level 드래그 허용
            self.tree.setAcceptDrops(True)
            self.tree.setDropIndicatorShown(True)
            self.tree.setDragDropMode(QTreeWidget.InternalMove)
            self.setWidget(self.tree)
        else:
            if self._on_reorder is not None:
                self._drop_hook = _DropAfterHook(
                    lambda: self._on_reorder(self._top_level_order_top_to_bottom())
                )
                self.tree.installEventFilter(self._drop_hook)

            # ★★★ 외부 .npy 드롭 허용 + 필터 설치 (UI 트리 경로에 추가) ★★★
            self.tree.setAcceptDrops(True)
            self._ext_filter = _ExternalNpyDropFilter(lambda paths: self.requestLoadPaths.emit(paths))
            self.tree.installEventFilter(self._ext_filter)
            # 내부 재정렬 동작 유지
            self.tree.setDragEnabled(True)
            self.tree.setDropIndicatorShown(True)

        # 공통 신호 연결
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)

        # name ↔ item 매핑
        self._item_by_name: Dict[str, QTreeWidgetItem] = {}
        self._block_item_changed = False
        self._pinned_viewer = None  # {"parent":"image.rgb","sel":{...}, "fmt":"{:.1f}"}
        # QListWidget 호환(기존 코드가 self.list를 참조하더라도 안전)
        self.list = self.tree

    # ---------- Public API ----------
    def clear(self):
        self._item_by_name.clear()
        self.tree.clear()

    def sync_tree(self, tree_dict: dict):
        """
        tree_dict = {
          "RGB": {"visible": True, "children": {}},
          "CLS:ParentA": {
             "visible": True,
             "children": {"CLS:ParentA/Class1": True, "CLS:ParentA/Class2": False}
          },
          ...
        }
        """
        self._block_item_changed = True
        try:
            self.clear()
            parents_to_expand: List[QTreeWidgetItem] = []

            for top_name, meta in tree_dict.items():
                children = meta.get("children", {}) or {}
                has_children = bool(children)

                # ▶ 부모: 자식이 있어도 '체크 가능 + tri-state'
                parent_item = self._ensure_item(
                    top_name,
                    parent=None,
                    is_child=False,
                    checkable=True,
                )
                if has_children:
                    parent_item.setFlags(parent_item.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
                else:
                    parent_item.setCheckState(0, Qt.Checked if meta.get("visible", True) else Qt.Unchecked)

                # ▶ 자식
                if has_children:
                    for child_name, vis in children.items():
                        child_item = self._ensure_item(
                            child_name, parent=parent_item, is_child=True, checkable=True
                        )
                        child_item.setCheckState(0, Qt.Checked if bool(vis) else Qt.Unchecked)

                    # 부모 classmap 원본은 화면에서 숨김(하위 집합만 표시)
                    if callable(self._on_visible_change):
                        try:
                            self._on_visible_change(top_name, False)
                        except Exception:
                            pass

                    # 부모 체크 상태 합산 반영
                    self._update_parent_check_from_children(parent_item)
                    parents_to_expand.append(parent_item)

            # 한 번에 펼침
            for it in parents_to_expand:
                if it is not None and it.treeWidget() is self.tree:
                    self.tree.expandItem(it)

        finally:
            self._block_item_changed = False
            # 1) viewer 캐시 복원(이미 들어 있음)
            try:
                self._restore_pinned_viewer()
            except Exception:
                pass
            # 2) ★ image.rgb 버튼 재부착
            try:
                rgb_key = self._resolve_parent_for_viewer("image.rgb") or "image.rgb"
                it = self._item_by_name.get(rgb_key)
                if it is not None:
                    # 두 번째 컬럼의 기존 위젯이 지워졌을 수 있으므로 다시 부착
                    self._attach_roi_button(it, parent_name=rgb_key)
            except Exception:
                pass

    # 기존 호출 호환
    def sync(self, names: List[str], visibles: List[bool]):
        tree = {nm: {"visible": bool(v), "children": {}} for nm, v in zip(names, visibles)}
        self.sync_tree(tree)

    # ---------- Internal ----------
    def _ensure_item(
        self,
        name: str,
        parent: Optional[QTreeWidgetItem],
        is_child: bool,
        *,
        checkable: bool = True,
        insert_top: bool = False,          # ★ 추가: 새 항목을 맨 위로 넣을지
    ) -> QTreeWidgetItem:
        it = self._item_by_name.get(name)
        if it is None:
            it = QTreeWidgetItem()
            it.setText(0, name)

            flags = it.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled
            if checkable:
                flags |= Qt.ItemIsUserCheckable
            if not is_child:  # top-level만 드래그/드롭 허용
                flags |= (Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled)
            it.setFlags(flags)

            if checkable:
                it.setCheckState(0, Qt.Checked)

            if parent is None:
                # ★ 최상단으로 삽입
                if insert_top:
                    self.tree.insertTopLevelItem(0, it)
                else:
                    self.tree.addTopLevelItem(it)
            else:
                # ★ 부모의 첫 번째 자식으로 삽입
                if insert_top:
                    parent.insertChild(0, it)
                else:
                    parent.addChild(it)
                parent.setExpanded(True)

            it.setData(0, Qt.UserRole, name)
            self._item_by_name[name] = it
        else:
            # 재사용 시 checkable 동기화
            flags = it.flags()
            if checkable and not (flags & Qt.ItemIsUserCheckable):
                it.setFlags(flags | Qt.ItemIsUserCheckable)
                it.setCheckState(0, Qt.Checked)
            elif (not checkable) and (flags & Qt.ItemIsUserCheckable):
                it.setFlags(flags & ~Qt.ItemIsUserCheckable)
                it.setData(0, Qt.CheckStateRole, None)

        if it.data(0, Qt.UserRole) is None:
            it.setData(0, Qt.UserRole, name)

        return it
    def _on_item_changed(self, item: QTreeWidgetItem, col: int):
        if self._block_item_changed or col != 0:
            return

        # 체크박스 없는 항목은 무시
        state_role = item.data(0, Qt.CheckStateRole)
        if state_role is None:
            return

        is_parent = (item.childCount() > 0)
        state = item.checkState(0)

        if is_parent:
            # 부모 클릭: Checked/Unchecked → 자식 전체에 동일 적용
            if state in (Qt.Checked, Qt.Unchecked):
                want_on = (state == Qt.Checked)
                self._block_item_changed = True
                for ch in self._iter_children(item):
                    ch.setCheckState(0, Qt.Checked if want_on else Qt.Unchecked)
                self._block_item_changed = False

                if callable(self._on_visible_change):
                    for ch in self._iter_children(item):
                        try:
                            self._on_visible_change(self._item_key(ch), want_on)
                        except Exception:
                            pass
        else:
            # 자식 클릭: 해당 레이어만 가시성 변경
            if callable(self._on_visible_change):
                try:
                    self._on_visible_change(self._item_key(item), state == Qt.Checked)
                except Exception:
                    pass

            # 부모 체크 상태 재계산
            parent = item.parent()
            if parent is not None:
                self._update_parent_check_from_children(parent)

    def _top_level_order_top_to_bottom(self) -> List[str]:
        """화면에 보이는 순서: 위→아래"""
        return [self._item_key(self.tree.topLevelItem(i)) for i in range(self.tree.topLevelItemCount())]
    def _on_context_menu(self, pos: QPoint):
        item = self.tree.itemAt(pos)
        if item is None:
            return

        # 드롭 직후 추가 보정(안전망)
        if callable(self._on_reorder):
            try:
                self._on_reorder(self._top_level_order_top_to_bottom())                                      # ★
            except Exception:
                pass

        sel = self.tree.selectedItems() or [item]
        names = [self._item_key(i) for i in sel]
        single = names[0] if len(names) == 1 else None

        menu = QMenu(self)
        act_merge = act_edit = act_save = act_del = None

        if len(names) >= 2:
            act_merge = menu.addAction("선택 클래스맵 합치기")
        if single:
            act_edit   = menu.addAction("수정")
            act_save   = menu.addAction("저장…")
            act_del    = menu.addAction("삭제")

        chosen = menu.exec_(self.tree.viewport().mapToGlobal(pos))
        if not chosen:
            return

        if chosen == act_merge:
            self.requestClassMap.emit(names)
        elif chosen == act_edit and single:
            self.requestEditLayer.emit(single)
        elif chosen == act_save and single:
            self.requestSaveLayer.emit(single)
        elif chosen == act_del and single:
            self.requestDeleteLayer.emit(single)

    def _all_names_top_to_bottom(self) -> List[str]:
        """부모→자식 순으로 위에서부터 평탄화"""
        names: List[str] = []
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            names.append(self._item_key(parent))
            for j in range(parent.childCount()):
                names.append(self._item_key(parent.child(j)))
        return names


    def _iter_children(self, parent_item: QTreeWidgetItem) -> List[QTreeWidgetItem]:
        return [parent_item.child(i) for i in range(parent_item.childCount())]

    @staticmethod
    def _item_key(item: QTreeWidgetItem) -> str:
        key = item.data(0, Qt.UserRole)
        return str(key) if key is not None else item.text(0)

    def _set_item_checked(self, item: QTreeWidgetItem, checked: bool) -> None:
        item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)

    def _update_parent_check_from_children(self, parent_item: QTreeWidgetItem) -> None:
        """자식들의 상태로 부모 체크 상태(Checked/Unchecked/Partially)를 자동 반영"""
        ch = self._iter_children(parent_item)
        if not ch:
            return
        states = [c.checkState(0) == Qt.Checked for c in ch]
        self._block_item_changed = True
        if all(states):
            parent_item.setCheckState(0, Qt.Checked)
        elif any(states):
            parent_item.setCheckState(0, Qt.PartiallyChecked)
        else:
            parent_item.setCheckState(0, Qt.Unchecked)
        self._block_item_changed = False
        
    def add_layer(self, name: str, checked: bool = True, parent: str | None = None, display_label: str | None = None):
        """
        LayersDock 트리에 레이어를 추가/동기화한다.
        - name: 내부 식별자 키(예: "classification/17")
        - parent: 상위 키(예: "classification") 또는 None(top-level)
        - display_label: 사용자에게 보여줄 라벨(없으면 name 사용)
        - checked: 가시성 체크 상태
        """
        label = display_label if display_label else name

        # ------ 재진입/삭제 방어 준비 ------
        from PyQt5.QtCore import Qt
        from PyQt5.QtCore import QSignalBlocker

        sb = QSignalBlocker(self.tree)  # itemChanged 재진입 차단(범위 내)
        it_exist = self._item_by_name.get(name) if hasattr(self, "_item_by_name") else None
        if not hasattr(self, "_item_by_name"):
            self._item_by_name = {}

        # 죽은 포인터 정리
        if it_exist is not None and sip.isdeleted(it_exist):
            self._item_by_name.pop(name, None)
            it_exist = None

        # ------ 이미 존재하면 상태만 동기화 ------
        if it_exist is not None:
            self._block_item_changed = True
            try:
                it_exist.setText(0, label)
                it_exist.setData(0, Qt.UserRole, name)  # 내부 키 보관
                it_exist.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
            finally:
                self._block_item_changed = False

            # 가시성 콜백은 트리 상태가 안정된 뒤 호출
            if callable(getattr(self, "_on_visible_change", None)):
                try:
                    self._on_visible_change(name, bool(checked))
                except Exception:
                    pass
            return it_exist

        # ------ 부모 준비 (없으면 먼저 생성) ------
        parent_item = None
        if parent:
            parent_item = self._item_by_name.get(parent)
            if (parent_item is not None) and sip.isdeleted(parent_item):
                self._item_by_name.pop(parent, None)
                parent_item = None
            if parent_item is None:
                parent_item = self._ensure_item(
                    name=parent, parent=None, is_child=False, checkable=True, insert_top=True
                )
                # 부모는 tristate + 체크 가능
                parent_item.setFlags(parent_item.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
                parent_item.setText(0, parent)
                parent_item.setData(0, Qt.UserRole, parent)
                parent_item.setCheckState(0, Qt.Checked)  # 기본 on
                parent_item.setExpanded(True)
                self._item_by_name[parent] = parent_item

        # ------ 실제 항목 생성 (맨 위에 prepend) ------
        item = self._ensure_item(
            name=name, parent=parent_item, is_child=bool(parent), checkable=True, insert_top=True
        )
        # 표시/내부키/체크 상태 설정 (항상 item 생성 후에!)
        item.setText(0, label)
        item.setData(0, Qt.UserRole, name)  # 내부 식별자(key)
        item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
        self._item_by_name[name] = item

        # image.rgb 전용 버튼/열 보장
        if parent is None and name == "image.rgb":
            if getattr(self.tree, "columnCount", lambda: 1)() < 2:
                self._ensure_button_column()
            # 우측 버튼 컬럼 초기화/부착
            item.setText(1, "")
            self._attach_roi_button(item, parent_name=name)

        # 부모 체크 상태 갱신
        if parent_item is not None:
            self._update_parent_check_from_children(parent_item)

        # ------ 콜백/정렬 동기화 ------
        if callable(getattr(self, "_on_visible_change", None)):
            try:
                self._on_visible_change(name, bool(checked))
            except Exception:
                pass

        if callable(getattr(self, "_on_reorder", None)):
            try:
                # ★ 실제로 그려지는 '자식'까지 포함한 평탄화 순서 전달
                self._on_reorder(self._top_level_order_top_to_bottom())
            except Exception:
                pass

        return item


    def remove_layer(self, name: str) -> None:
        """트리에서 항목 제거(자식인 경우 부모의 체크 상태도 재계산)"""
        it = self._item_by_name.pop(name, None)
        if it is None:
            return
        parent = it.parent()
        if parent is None:
            idx = self.tree.indexOfTopLevelItem(it)
            if idx >= 0:
                self.tree.takeTopLevelItem(idx)
        else:
            parent.removeChild(it)
            self._update_parent_check_from_children(parent)
            
    def _top_level_order_bottom_to_top(self) -> List[str]:
        """top-level 레이어의 아래→위 순서"""
        return [self._item_key(self.tree.topLevelItem(i))
                for i in reversed(range(self.tree.topLevelItemCount()))]

    def _all_names_bottom_to_top(self) -> List[str]:
        """부모/자식 평탄화 목록을 아래→위 순서로 반환"""
        names: List[str] = []
        for i in reversed(range(self.tree.topLevelItemCount())):
            parent = self.tree.topLevelItem(i)
            names.append(self._item_key(parent))
            for j in reversed(range(parent.childCount())):
                names.append(self._item_key(parent.child(j)))
        return names

    def has_layer(self, name: str) -> bool:
        """이름으로 항목 존재 여부 확인."""
        return name in self._item_by_name

    def add_image_layer(self, key: str, checked: bool = True) -> None:
        """
        Top-level 레이어 추가 (컨테이너/단일 레이어 공용).
        MainWindow/LayerManager에서 '부모 없이' 추가할 때 이 API를 기대하므로 지원.
        """
        self.add_layer(name=key, checked=checked, parent=None)

    def add_child_layer(self, parent_key: str, key: str, checked: bool = True) -> None:
        """
        parent_key 아래 자식 레이어 추가.
        MainWindow/LayerManager에서 '부모 지정' 추가할 때 이 API를 기대하므로 지원.
        """
        self.add_layer(name=key, checked=checked, parent=parent_key)
                
    def _attach_roi_button(self, item: QTreeWidgetItem, parent_name: str):
        # ★ 버튼 컬럼 보장
        self._ensure_button_column()
        if self.tree.itemWidget(item, 1) is not None:
            return

        from PyQt5.QtWidgets import QWidget, QHBoxLayout, QToolButton
        from PyQt5.QtCore import QSize

        box = QWidget(self.tree)
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        # [+] 버튼 (ROI) - PNG 아이콘 사용
        btn_plus = QToolButton(box)
        btn_plus.setAutoRaise(True)
        btn_plus.setFocusPolicy(Qt.NoFocus)

        # 아이콘 설정 (리소스를 쓰면 "icons/..." 대신 ":/icons/roi_plus.png")
        btn_plus.setIcon(QIcon("icons/roi_plus.png"))
        btn_plus.setToolButtonStyle(Qt.ToolButtonIconOnly)

        btn_plus.setFixedSize(16, 16)
        btn_plus.setIconSize(QSize(12, 12))
        btn_plus.setStyleSheet("QToolButton{padding:0;margin:0;}")
        btn_plus.setToolTip("ROI 지정")
        btn_plus.clicked.connect(lambda: self.requestOpenWorkspace.emit(parent_name))

        # 🖥 버튼 (뷰어 대역 선택)
        btn_view = QToolButton(box)
        btn_view.setAutoRaise(True)
        btn_view.setFocusPolicy(Qt.NoFocus)
        btn_view.setText("🖥")
        btn_view.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn_view.setFixedSize(18, 16)  # 이모지 폭 약간 넓게
        btn_view.setIconSize(QSize(12, 12))
        btn_view.setStyleSheet("QToolButton{padding:0;margin:0;}")
        btn_view.setToolTip("뷰어 대역 선택")
        btn_view.clicked.connect(lambda: self.requestOpenViewerBand.emit(parent_name))

        # 레이아웃 배치: 우측 정렬
        lay.addStretch(1)
        lay.addWidget(btn_plus)
        lay.addWidget(btn_view)

        # 버튼 위젯을 두 번째 컬럼에 장착
        item.setText(1, "")
        self.tree.setItemWidget(item, 1, box)

    def _ensure_button_column(self):
        if self.tree.columnCount() < 2:
            self.tree.setColumnCount(2)

        hdr = self.tree.header()
        hdr.setVisible(False)                 # 헤더는 숨기되 설정은 유효
        hdr.setMinimumSectionSize(1)          # ★ 기본 최소폭(≈62px) 제거
        hdr.setStretchLastSection(False)      # ★ 마지막 컬럼 자동 늘림 금지
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)  # 이름 컬럼은 남은 폭 모두
        hdr.setSectionResizeMode(1, QHeaderView.Fixed)    # 버튼 컬럼은 고정 폭
        self.tree.setColumnWidth(1, 44)       # ★ 버튼 컬럼을 16px로 고정
                
    # --- LayersDock 클래스 내부에 추가 ---
    def _make_swatch_icon(self, color: QColor, size: int = 12) -> QIcon:
        pm = QPixmap(size, size)
        pm.fill(color)
        return QIcon(pm)

    # LayersDock 클래스 내부 (대체/수정본)
    def set_viewer_bands(self, mode: str, *, r: float=None, g: float=None, b: float=None,
                        gray: float=None, fmt: str="{:.1f}", attach_to: str="image.rgb") -> None:
        """
        실제 Layers(=LayerManager가 만든 트리)에만 viewer 자식을 붙인다.
        부모가 없으면 아무것도 만들지 않고 반환(캐시는 별도 함수에서 복원).
        """
        parent_name = self._resolve_parent_for_viewer(attach_to)
        if parent_name is None:
            return  # 부모가 없으면 지금은 만들지 않음(표준 트리 외 생성 금지)

        parent_item = self._item_by_name.get(parent_name)
        if parent_item is None:
            return

        # 기존 viewer 자식만 제거 (다른 자식/순서는 보존)
        rm_keys = []
        for i in range(parent_item.childCount()):
            ch = parent_item.child(i)
            key = self._item_key(ch)
            if key and "|viewer|" in key:
                rm_keys.append((i, key))
        # 뒤에서부터 제거
        for idx, key in reversed(rm_keys):
            parent_item.takeChild(idx)
            self._item_by_name.pop(key, None)

        # viewer 자식은 "기존 비-viewer 자식 뒤"에 삽입
        non_viewer_count = 0
        for i in range(parent_item.childCount()):
            ch = parent_item.child(i)
            k = self._item_key(ch)
            if not (k and "|viewer|" in k):
                non_viewer_count += 1

        def _add_view_child(suffix: str, label: str, color: QColor):
            name = f"{parent_name}|viewer|{suffix}"
            it = QTreeWidgetItem()
            it.setText(0, label)
            it.setData(0, Qt.UserRole, name)
            it.setIcon(0, self._make_swatch_icon(color))
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # 정보표시 전용
            parent_item.insertChild(non_viewer_count, it)
            self._item_by_name[name] = it
            non_viewer_count_plus = non_viewer_count + 1
            return non_viewer_count_plus

        # 표기 생성
        if mode.lower() == "rgb":
            non_viewer_count = _add_view_child("R", f"R : {fmt.format(float(r))}", QColor(220, 50, 47))
            non_viewer_count = _add_view_child("G", f"G : {fmt.format(float(g))}", QColor(60, 180, 75))
            non_viewer_count = _add_view_child("B", f"B : {fmt.format(float(b))}", QColor(33, 120, 200))
        else:
            non_viewer_count = _add_view_child("Gray", f"Gray : {fmt.format(float(gray))}", QColor(120, 120, 120))

        # 버튼 컬럼/펼침 유지
        self._ensure_button_column()
        self.tree.expandItem(parent_item)

    def set_viewer_bands_from_selection(self, sel: dict, fmt: str="{:.1f}", attach_to: str="image.rgb") -> None:
        """다이얼로그 선택(dict)으로부터 Layers 트리 갱신. 부모 없으면 캐시에만 저장."""
        # 캐시 저장(언제든 복원 가능)
        self._pinned_viewer = {"parent": attach_to, "sel": dict(sel or {}), "fmt": fmt}

        # 부모가 실제 존재할 때만 즉시 반영
        parent_name = self._resolve_parent_for_viewer(attach_to)
        if parent_name is None:
            return  # 지금은 표시 못함(추후 sync_tree에서 복원)

        mode = (sel or {}).get("mode", "gray").lower()
        if mode == "rgb":
            self.set_viewer_bands("rgb", r=sel.get("r"), g=sel.get("g"), b=sel.get("b"),
                                fmt=fmt, attach_to=parent_name)
        else:
            self.set_viewer_bands("gray", gray=sel.get("gray"), fmt=fmt, attach_to=parent_name)

    def _restore_pinned_viewer(self):
        pv = getattr(self, "_pinned_viewer", None)
        if not pv:
            return
        prefer = pv.get("parent") or "image.rgb"
        parent_name = self._resolve_parent_for_viewer(prefer)
        if parent_name is None:
            return  # 아직 부모가 안 올라온 상태

        sel = pv.get("sel") or {}
        fmt = pv.get("fmt") or "{:.1f}"
        # 부모가 생겼으니 이제 실제로 붙인다
        mode = (sel or {}).get("mode", "gray").lower()
        if mode == "rgb":
            self.set_viewer_bands("rgb", r=sel.get("r"), g=sel.get("g"), b=sel.get("b"),
                                fmt=fmt, attach_to=parent_name)
        else:
            self.set_viewer_bands("gray", gray=sel.get("gray"), fmt=fmt, attach_to=parent_name)

        # image.rgb 행에 +/🖥 버튼도 다시 부착
        try:
            it = self._item_by_name.get(parent_name)
            if it is not None:
                self._attach_roi_button(it, parent_name=parent_name)
        except Exception:
            pass

            
    # LayersDock 클래스 안에 추가
    def _resolve_parent_for_viewer(self, prefer: str = "image.rgb") -> Optional[str]:
        """LayerManager가 실제로 만든 top-level 부모 키를 찾아준다."""
        if prefer in self._item_by_name:
            return prefer
        # 흔한 별칭들 (필요 시 추가)
        for alias in ("image.rgb", "RGB", "image/RGB", "base.rgb"):
            if alias in self._item_by_name:
                return alias
        # 휴리스틱: 'rgb' 포함하는 top-level
        for k in self._item_by_name.keys():
            if "/" not in k and "rgb" in k.lower():
                return k
        return None
