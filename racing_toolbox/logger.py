from pathlib import Path
import logging.config
import logging
import yaml
from racing_toolbox.const import ROOT_DIR


def setup_logger(logger_config_path: Path) -> None:
    """
    Function initializes logger for the application from provided config file

    Args:
        logger_config_path (Path): path to configuration file for logging

    """

    with open(logger_config_path, "rt") as f:
        config = yaml.safe_load(f.read())
        try:
            # make sure filename exist
            fname = Path(config["handlers"]["file"]["filename"])
            fname = ROOT_DIR / fname
            fname.parent.mkdir(exist_ok=True)
            config["handlers"]["file"]["filename"] = str(fname.absolute())
        except KeyError:
            pass

    logging.config.dictConfig(config)
