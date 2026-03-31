# views/dialogs/material_add_dialog.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional

from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtCore import pyqtSignal
from data.db import material_add
from services.resampling_cache import resample_cache_paths

class MaterialAddDialog(QtWidgets.QDialog):
    """
    신규 클래스(물질) 생성 다이얼로그
    """
    material_created = pyqtSignal(dict)  # {"name": str, "description": str, "mtrl_cd": int}
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, ui_dir: Optional[Path] = None):
        super().__init__(parent)
        
        if ui_dir is None:
            ui_dir = Path(__file__).resolve().parent / ".." / ".." / "ui"
        ui_path = (ui_dir / "material_add_dialog.ui").resolve()
        
        if not ui_path.exists():
            raise FileNotFoundError(f"UI 파일을 찾을 수 없습니다: {ui_path}")
        
        # UI 로드
        self._root = uic.loadUi(str(ui_path), self)
        
        # 위젯 핸들
        self.lineEdit_name = self.findChild(QtWidgets.QLineEdit, "lineEdit_name")
        self.textEdit_description = self.findChild(QtWidgets.QTextEdit, "textEdit_description")
        self.pushButton_create = self.findChild(QtWidgets.QPushButton, "pushButton_create")
        self.pushButton_cancel = self.findChild(QtWidgets.QPushButton, "pushButton_cancel")
        
        # 시그널 연결
        if self.pushButton_create:
            self.pushButton_create.clicked.connect(self._on_create_clicked)
        if self.pushButton_cancel:
            self.pushButton_cancel.clicked.connect(self.reject)
        
        self.setModal(True)
        self._center_to_parent()
    
    def _center_to_parent(self):
        """부모 중앙으로 배치"""
        if self.parent() and isinstance(self.parent(), QtWidgets.QWidget):
            parent_geom = self.parent().frameGeometry()
            self.move(parent_geom.center() - self.frameGeometry().center())

    def _find_main_window(self):
        """
        parent 체인을 타고 MainWindow를 찾는다.
        기존 구현은 parent가 UserLabelingDialog라는 전제를 두었는데,
        PixelLabelingDialog 등 다른 다이얼로그에서 호출할 수도 있어 일반화한다.
        """
        w = self.parent()
        for _ in range(10):
            if w is None:
                return None
            if hasattr(w, "_cache_primary_path") or hasattr(w, "_extract_src_path"):
                return w
            try:
                w = w.parent()
            except Exception:
                return None
        return None
    
    def _get_next_mtrl_cd(self) -> int:
        """
        resample.info 파일에서 기존 클래스들을 확인하고 다음 mtrl_cd를 반환.
        - classes가 비어있거나 없으면 0부터 시작
        - classes가 있으면 가장 큰 cid + 1을 반환
        """
        try:
            main_window = self._find_main_window()
            
            if not main_window:
                import logging
                logging.warning("[MaterialAdd] MainWindow를 찾을 수 없습니다. 0부터 시작합니다.")
                return 0
            
            # primary path 찾기
            cfg = getattr(main_window, "cfg", {})
            primary = None
            
            # _cache_primary_path 메서드가 있으면 사용
            if hasattr(main_window, "_cache_primary_path"):
                primary = main_window._cache_primary_path(cfg)
            
            # 없으면 _extract_src_path 시도
            if not primary and hasattr(main_window, "_extract_src_path"):
                primary = main_window._extract_src_path(cfg)
            
            # 여전히 없으면 data_path에서 찾기
            if not primary:
                data_path = cfg.get("data_path")
                if isinstance(data_path, dict):
                    for v in data_path.values():
                        if isinstance(v, (str, Path)):
                            ap = os.path.abspath(str(v))
                            if os.path.isfile(ap):
                                primary = ap
                                break
                elif isinstance(data_path, (str, Path)):
                    primary = os.path.abspath(str(data_path))
            
            if not primary:
                # primary path를 찾을 수 없으면 0부터 시작
                import logging
                logging.info("[MaterialAdd] primary path not found, starting from 0")
                return 0
            
            # resample.info 파일 경로 찾기
            info_path, _ = resample_cache_paths(primary)
            
            if not os.path.isfile(info_path):
                # .info 파일이 없으면 0부터 시작
                import logging
                logging.info("[MaterialAdd] .info file not found, starting from 0")
                return 0
            
            # .info 파일 읽기
            with open(info_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            classes_info = meta.get("classes", {})
            if not classes_info or not isinstance(classes_info, dict):
                # classes가 비어있거나 없으면 0부터 시작
                import logging
                logging.info("[MaterialAdd] classes empty or not found, starting from 0")
                return 0
            
            # 기존 cid 중 가장 큰 값 찾기
            max_cid = -1
            for cid_str in classes_info.keys():
                try:
                    cid_int = int(cid_str)
                    if cid_int > max_cid:
                        max_cid = cid_int
                except (TypeError, ValueError):
                    continue
            
            # 다음 번호 부여 (max_cid + 1, 최소 0)
            next_cid = max(0, max_cid + 1)
            import logging
            logging.info(f"[MaterialAdd] next mtrl_cd: {next_cid} (max existing: {max_cid})")
            return next_cid
            
        except Exception as e:
            import logging
            logging.exception("[MaterialAdd] _get_next_mtrl_cd failed")
            # 오류 발생 시 0부터 시작
            return 0
    
    def _on_create_clicked(self):
        """생성 버튼 클릭 시"""
        try:
            name = self.lineEdit_name.text().strip()
            if not name:
                QtWidgets.QMessageBox.warning(self, "경고", "물질 이름을 입력하세요.")
                return
            
            description = self.textEdit_description.toPlainText().strip()
            
            # user_type 확인: parent 체인에서 MainWindow 탐색
            main_window = self._find_main_window()
            user_type = getattr(main_window, "user_type", None) if main_window else None
            
            # MainWindow에서 user_type을 찾지 못하면 에러 발생
            if user_type not in ("personal", "server"):
                import logging
                logging.error("[MaterialAdd] MainWindow에서 user_type을 찾을 수 없습니다. parent=%s, main_window=%s", 
                            type(self.parent()).__name__ if self.parent() else None,
                            type(main_window).__name__ if main_window else None)
                QtWidgets.QMessageBox.critical(
                    self, 
                    "오류", 
                    "사용자 타입(user_type)을 확인할 수 없습니다.\n\n"
                    "이미지를 먼저 로드한 후 다시 시도하세요."
                )
                return  # 에러 발생 시 함수 종료
            
            import logging
            logging.info(f"[MaterialAdd] user_type: {user_type}")
            
            mtrl_cd = None
            
            if user_type == "personal":
                # personal 사용자: resample.info에서 다음 번호 부여
                mtrl_cd = self._get_next_mtrl_cd()
                import logging
                logging.info(f"[MaterialAdd] personal user: assigned mtrl_cd={mtrl_cd}")
            elif user_type == 'server':
                # server 사용자: API 호출
                api_url = os.getenv('material_add_url')
                if api_url:
                    mtrl_cd = material_add(api_url, mtrl_nm=name, dsc=description)
                else:
                    # API URL이 없으면 resample.info에서 다음 번호 부여
                    import logging
                    logging.warning("[MaterialAdd] material_add_url not set, using local numbering")
                    mtrl_cd = self._get_next_mtrl_cd()
        
            result = {
                "name": name,
                "description": description,
                "mtrl_cd": mtrl_cd
            }
            self.material_created.emit(result)
            self.accept()
            
        except Exception as e:
            import logging
            logging.exception("[MaterialAdd] create failed")
            QtWidgets.QMessageBox.critical(self, "오류", f"클래스 생성 중 오류가 발생했습니다:\n{e}")
    
    def get_result(self) -> Optional[dict]:
        """다이얼로그 실행 후 결과 반환"""
        if self.exec_() == QtWidgets.QDialog.Accepted:
            return {
                "name": self.lineEdit_name.text().strip(),
                "description": self.textEdit_description.toPlainText().strip(),
            }
        return None

