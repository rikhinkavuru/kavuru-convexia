"""Repo-root convenience shim: re-exports the canonical configuration.

The single source of truth is :mod:`kavuru_convexia.config`. This module exists
only so scripts and notebooks executed from the repo root can ``import config``
directly without installing the package. Edit values in the package module, not
here.
"""
from kavuru_convexia.config import *  # noqa: F401,F403
