from pathlib import Path
import logging.config
import logging 
import yaml


def setup_applevel_logger(logger_config_path: Path) -> None:
    """
    Function initializes logger for the application from provided config file

    Args:
        logger_config_path (Path): path to configuration file for logging

    """
    
    with open(logger_config_path, 'rt') as f:
        config = yaml.safe_load(f.read())
    
    logging.config.dictConfig(config)
