"""
Definition:
Backend implementations for MINIM-compatible training and generation.
---
Results:
Keeps orchestration separate from model-specific execution details.
"""

from minim.backends.registry import BACKEND_NAMES, get_backend

__all__ = ["BACKEND_NAMES", "get_backend"]
