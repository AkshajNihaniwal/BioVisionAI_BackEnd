"""
Logging setup for BIOVISION-AI.

Structured logging for API requests, training, and audit.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure root logger.

    Args:
        level: Logging level.
        format_string: Custom format. Default includes timestamp, level, message.
    """
    fmt = format_string or "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
