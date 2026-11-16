import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from streamlit.web import bootstrap


HOST = "127.0.0.1"
PORT = 8501

APP_URL = f"http://localhost:{PORT}/"
HEALTH_URL = f"http://{HOST}:{PORT}/_stcore/health"

logger = logging.getLogger(__name__)


def get_application_directory() -> Path:
    """Папка с exe либо папка с исходным launcher.py."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent


def get_app_file() -> Path:
    """Путь к app.py, включая сборку PyInstaller --onefile."""
    if getattr(sys, "frozen", False):
        bundle_directory = Path(sys._MEIPASS)
        return bundle_directory / "app.py"

    return Path(__file__).resolve().parent / "app.py"


def server_is_ready() -> bool:
    """Проверяет готовность локального сервера без системного прокси."""
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({})
    )

    try:
        with opener.open(HEALTH_URL, timeout=1.0) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def open_browser_when_ready() -> None:
    """Открывает обычную вкладку после запуска сервера."""
    deadline = time.monotonic() + 60.0

    while time.monotonic() < deadline:
        if server_is_ready():
            logger.info("Открываем приложение: %s", APP_URL)

            try:
                opened = webbrowser.open(APP_URL, new=2)

                if not opened:
                    logger.warning(
                        "Не удалось автоматически открыть браузер. "
                        "Откройте вручную: %s",
                        APP_URL,
                    )
            except Exception:
                logger.exception(
                    "Ошибка открытия браузера. Откройте вручную: %s",
                    APP_URL,
                )

            return

        time.sleep(0.5)

    logger.error(
        "Сервер не стал доступен за 60 секунд. "
        "Проверьте сообщения в консоли."
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    application_directory = get_application_directory()
    app_file = get_app_file()

    os.chdir(application_directory)

    if not app_file.is_file():
        raise FileNotFoundError(
            f"Файл приложения не найден: {app_file}"
        )

    flag_options = {
        "global.developmentMode": False,
        "server.address": HOST,
        "server.port": PORT,
        "server.headless": True,
        "server.fileWatcherType": "none",
        "server.runOnSave": False,
        "browser.serverAddress": "localhost",
        "browser.serverPort": PORT,
        "browser.gatherUsageStats": False,
        "logger.level": "info",
    }

    # При программном запуске сначала явно загружаем настройки.
    bootstrap.load_config_options(flag_options=flag_options)

    logger.info("Файл приложения: %s", app_file)
    logger.info("Рабочая папка: %s", application_directory)
    logger.info("Адрес приложения: %s", APP_URL)
    logger.info(
        "Не закрывайте консоль, пока пользуетесь приложением."
    )

    browser_thread = threading.Thread(
        target=open_browser_when_ready,
        name="open-app-browser",
        daemon=True,
    )
    browser_thread.start()

    bootstrap.run(
        str(app_file),
        False,
        [],
        flag_options,
    )


if __name__ == "__main__":
    main()