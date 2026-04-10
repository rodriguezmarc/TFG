"""
Definition:
Backend registry for MINIM-compatible model drivers.
---
Results:
Provides a single place for selecting model execution backends by name.
"""

from __future__ import annotations

from minim.backends.base import MinimBackend
from minim.backends.mock import MockMinimBackend
from minim.backends.real import RealMinimBackend


BACKENDS: dict[str, MinimBackend] = {
    "real": RealMinimBackend(),
    "mock": MockMinimBackend(),
}
BACKEND_NAMES = tuple(BACKENDS)


def get_backend(name: str) -> MinimBackend:
    normalized_name = name.strip().lower()
    try:
        return BACKENDS[normalized_name]
    except KeyError as exc:
        allowed = ", ".join(BACKEND_NAMES)
        raise ValueError(f"Unknown MINIM backend '{name}'. Expected one of: {allowed}") from exc
