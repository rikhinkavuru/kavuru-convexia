"""Pytest configuration: make the shared stub module importable from any test."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
