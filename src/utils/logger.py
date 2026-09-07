import logging
import sys
from pathlib import Path

from src.utils.config import LOG_DIR


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to console + a rotating file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers on re-import (e.g. Streamlit reruns)

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(Path(LOG_DIR) / "platform.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
