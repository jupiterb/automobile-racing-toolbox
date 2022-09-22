from pathlib import Path
from racing_toolbox.logger import setup_logger
import logging


ROOT_DIR = Path(__file__).parent

__logger_config_path = ROOT_DIR.parent / "resources/logger.yml"
if __logger_config_path.exists():
    setup_logger(ROOT_DIR.parent / "resources/logger.yml")
    logging.info("logger config file loaded successfully")
else:
    logging.warning(f"logger config file path {__logger_config_path} doesn't exist")
