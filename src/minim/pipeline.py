"""
Definition:
Public facade for the local MINIM pipeline.
---
Results:
Re-exports the forward pipeline API while implementation lives in focused modules.
"""

from __future__ import annotations

from minim.backends import BACKEND_NAMES
from minim.checkpoints import DEFAULT_BASE_MODEL_ID
from minim.orchestrator import run_evaluation_pipeline, run_full_pipeline
from minim.preparation import prepare_run
from minim.runs import PreparedRun

DEFAULT_BASE_MODEL = DEFAULT_BASE_MODEL_ID
PIPELINE_BACKENDS = BACKEND_NAMES

__all__ = [
    "DEFAULT_BASE_MODEL",
    "PIPELINE_BACKENDS",
    "PreparedRun",
    "prepare_run",
    "run_evaluation_pipeline",
    "run_full_pipeline",
]
