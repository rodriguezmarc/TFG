"""
Definition:
Real MINIM backend backed by Stable Diffusion fine-tuning and generation.
---
Results:
Launches the heavy training script and generates evaluation images through diffusers.
"""

from __future__ import annotations

import csv
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from minim.checkpoints import resolve_base_model

if TYPE_CHECKING:
    from minim.runs import PreparedRun

logger = logging.getLogger(__name__)

TRAINING_RUNTIME_DEPENDENCIES = (
    "accelerate",
    "datasets",
    "diffusers",
    "huggingface_hub",
    "numpy",
    "torch",
    "torchvision",
    "transformers",
)
MIN_CPU_TRAIN_MEMORY_BYTES = 24 * 1024**3


def _read_validation_prompts(val_manifest: Path, max_prompts: int = 4) -> list[str]:
    prompts: list[str] = []
    with Path.open(val_manifest, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            prompts.append(f"{row['modality']}: {row['text']}")
            if len(prompts) >= max_prompts:
                break
    return prompts


def _missing_training_dependencies() -> list[str]:
    return [module_name for module_name in TRAINING_RUNTIME_DEPENDENCIES if importlib.util.find_spec(module_name) is None]


def _validate_training_environment() -> None:
    missing_modules = _missing_training_dependencies()
    if not missing_modules:
        return

    missing_list = ", ".join(missing_modules)
    install_command = f"{sys.executable} -m pip install -e ."
    raise RuntimeError(
        "MINIM training dependencies are missing from the active Python environment "
        f"({sys.executable}). Missing modules: {missing_list}. "
        f"Install the project dependencies from the repository root with: {install_command}"
    )


def _available_system_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _validate_local_training_resources() -> None:
    try:
        import torch
    except Exception:
        return

    if torch.cuda.is_available():
        return

    available_memory = _available_system_memory_bytes()
    if available_memory is None or available_memory >= MIN_CPU_TRAIN_MEMORY_BYTES:
        return

    available_gib = available_memory / 1024**3
    required_gib = MIN_CPU_TRAIN_MEMORY_BYTES / 1024**3
    raise RuntimeError(
        "MINIM fine-tuning aborted before launch: Stable Diffusion v1.4 full fine-tuning on CPU "
        f"needs substantially more RAM than this machine currently exposes ({available_gib:.1f} GiB available, "
        f"recommended at least {required_gib:.0f} GiB, and swap is strongly recommended). "
        "Without enough memory the process is usually terminated by the OS with exit code 137 right after "
        "training starts. Use a CUDA GPU, add swap and more RAM, or switch to a lighter training method "
        "such as LoRA instead of full-model fine-tuning."
    )


class RealMinimBackend:
    name = "real"

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
        resolved_base_model = resolve_base_model(base_model)
        logger.info("Starting MINIM fine-tuning for run '%s'.", prepared_run.run_name)
        logger.info("Training manifest: %s", prepared_run.train_manifest)
        logger.info("Base model source: %s", resolved_base_model)
        logger.info("Checkpoint output: %s", prepared_run.model_dir)
        _validate_training_environment()
        _validate_local_training_resources()
        try:
            import torch

            if not torch.cuda.is_available():
                logger.info(
                    "Training will run on CPU. This is not a crash, but it can take a long time. "
                    "Use --max-train-steps or lower --num-train-epochs to shorten the run."
                )
        except Exception:
            pass
        validation_prompts = _read_validation_prompts(prepared_run.val_manifest)
        command = [
            sys.executable,
            "-m",
            "minim.train_model",
            "--pretrained_model_name_or_path",
            resolved_base_model,
            "--train_data_dir",
            str(prepared_run.train_manifest),
            "--output_dir",
            str(prepared_run.model_dir),
            "--image_column",
            "path",
            "--caption_column",
            "text",
            "--modality_column",
            "modality",
            "--prepend_modality",
            "--train_batch_size",
            str(train_batch_size),
            "--num_train_epochs",
            str(num_train_epochs),
            "--validation_epochs",
            str(validation_epochs),
            "--gradient_accumulation_steps",
            str(gradient_accumulation_steps),
            "--learning_rate",
            str(learning_rate),
            "--resolution",
            str(resolution),
            "--seed",
            str(seed),
            "--report_to",
            report_to,
            "--checkpoints_total_limit",
            "3",
        ]
        if max_train_steps is not None:
            command.extend(["--max_train_steps", str(max_train_steps)])
        if mixed_precision is not None:
            command.extend(["--mixed_precision", mixed_precision])
        if validation_prompts:
            command.append("--validation_prompts")
            command.extend(validation_prompts)
        result = subprocess.run(command, check=False)
        if result.returncode == 137:
            raise RuntimeError(
                "MINIM fine-tuning was terminated with exit code 137. This is consistent with the OS killing the "
                "training process for running out of memory during the first optimization step. Full Stable "
                "Diffusion v1.4 fine-tuning is too heavy for this CPU-only setup. Use a CUDA GPU, add swap and "
                "more RAM, or switch to a lighter approach such as LoRA."
            )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command)
        logger.info("Fine-tuning completed for run '%s'.", prepared_run.run_name)
        return result

    def generate(
        self,
        prepared_run: "PreparedRun",
        *,
        device: str | None = None,
        seed: int | None = 42,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> list[Path]:
        from minim.generate import generate_from_manifest

        logger.info("Generated samples: %s", prepared_run.generated_test_dir)
        return generate_from_manifest(
            model_path=prepared_run.model_dir,
            manifest_path=prepared_run.test_manifest,
            output_dir=prepared_run.generated_test_dir,
            device=device,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
