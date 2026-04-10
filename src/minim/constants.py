"""
Definition:
Shared constants for MINIM pipeline configuration.
---
Results:
Avoids importing heavy runtime modules for lightweight configuration lookups.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path("outputs/minim")
METRIC_MODES = ("full", "local")
