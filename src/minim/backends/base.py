"""
Definition:
Shared backend protocol for MINIM-compatible model execution.
---
Results:
Defines the train/generate contract used by the pipeline orchestrator.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from minim.runs import PreparedRun


class MinimBackend(Protocol):
    """
    ########################################
    Definition:
    Model backend contract consumed by the pipeline.
    ---
    Results:
    Allows real and mock model implementations to be selected like drivers.
    ########################################
    """
    name: str

    def train(
        self,
        prepared_run: "PreparedRun",
        *,
        base_model: str | None = None,
        num_train_epochs: int = 5,
        max_train_steps: int | None = None,
        learning_rate: float = 1e-5,
        train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        resolution: int = 512,
        seed: int = 42,
        mixed_precision: str | None = None,
        validation_epochs: int = 5,
        report_to: str = "tensorboard",
    ) -> subprocess.CompletedProcess:
        ...

    def generate(
        self,
        prepared_run: "PreparedRun",
        *,
        device: str | None = None,
        seed: int | None = 42,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> list[Path]:
        ...
