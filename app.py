# app.py
"""
엔트리 포인트:
- HighDPI 설정
- .env 로드(선택)
- icon.rcc 등록(선택)
- MainWindow 실행
"""
import sys
import os
import logging
from pathlib import Path

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QResource
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon

APP_NAME = "Third Pixel Classification Tool"
ORG_NAME = "GNEWSOFT"
APP_VERSION = "0.1.0"


def _get_app_dir() -> Path:
    # PyInstaller 대응(_MEIPASS) + 일반 실행 모두 지원
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def _load_env(app_dir: Path) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
        env_path = app_dir / ".env"
        if env_path.exists():
            load_dotenv(str(env_path))
            logging.info("Loaded .env from %s", env_path)
    except Exception:
        # dotenv 미설치/미사용 시 무시
        pass


def _setup_logging(app_dir: Path) -> None:
    try:
        # 프로젝트 util이 있으면 우선 사용
        from utils.logging_setup import setup_logging  # type: ignore
        setup_logging()
    except Exception:
        # 기본 로깅
        log_dir = app_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_dir / "app.log", encoding="utf-8"),
            ],
        )
        logging.info("Fallback logging configured (utils.logging_setup not found)")


def _register_rcc(app_dir: Path) -> None:
    # utils.rcc 가 있으면 그걸 쓰고, 없으면 직접 등록
    rcc_path = app_dir / "icon.rcc"
    try:
        from utils.rcc import register_rcc  # type: ignore
        ok = register_rcc(str(rcc_path))
        if ok:
            logging.info("Registered RCC via utils.rcc: %s", rcc_path)
        else:
            logging.warning("Failed to register RCC via utils.rcc: %s", rcc_path)
    except Exception:
        if rcc_path.exists():
            ok = QResource.registerResource(str(rcc_path))
            if ok:
                logging.info("Registered RCC: %s", rcc_path)
            else:
                logging.warning("Failed to register RCC: %s", rcc_path)
        else:
            logging.info("icon.rcc not found (skip)")


def main() -> int:
    app_dir = _get_app_dir()
    print(app_dir)
    _setup_logging(app_dir)
    _load_env(app_dir)
    _register_rcc(app_dir)

    # High DPI
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon("icons/gnew_icon.png"))  # 아이콘 파일 경로
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationVersion(APP_VERSION)

    try:
        from views.main_window import MainWindow  # type: ignore
    except Exception as e:
        logging.exception("MainWindow import failed: %s", e)
        QtWidgets.QMessageBox.critical(
            None,
            "Import Error",
            f"views/main_window.py 로드 실패:\n{e}",
        )
        return 1

    try:
        win = MainWindow(app_dir=app_dir)
        win.show()
        return app.exec_()
    except Exception as e:
        logging.exception("Fatal error while starting MainWindow: %s", e)
        QtWidgets.QMessageBox.critical(
            None,
            "Fatal Error",
            f"애플리케이션 시작 중 치명적 오류:\n{e}",
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
