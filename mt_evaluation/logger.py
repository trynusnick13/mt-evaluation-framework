"""
Logging utilities
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mt-evaluation")


def log_hr() -> None:
    """
    Log a horizontal line
    """
    logger.debug("*" * 50)
