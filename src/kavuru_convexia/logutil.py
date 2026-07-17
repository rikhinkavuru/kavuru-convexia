"""Structured logging setup. Use ``get_logger(__name__)``; never ``print()``."""
from __future__ import annotations

import logging
import os

_CONFIGURED = False


def _configure_root() -> None:
    """Attach a single formatted handler to the package-root logger, once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("KAVURU_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("kavuru_convexia")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under the package root so one config governs all."""
    _configure_root()
    short = name.split(".")[-1]
    return logging.getLogger(f"kavuru_convexia.{short}")
