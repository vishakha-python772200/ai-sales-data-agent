"""
utils/logger.py
----------------
Centralized logger for the whole pipeline.
Every agent should use `get_logger(__name__)` instead of print().
Logs go to both console and a dated file inside logs/.
"""

import logging
import os
import sys
from datetime import datetime

# Import here (not from config) to avoid circular imports
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

_LOG_FILE = os.path.join(LOGS_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.

    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("something happened")
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Logger already configured (avoids duplicate handlers on reimport)
        return logger

    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FORMATTER)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_FORMATTER)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger
