"""
Definition:
Brief map of checkpoint and base-model path helpers for the local MINIM pipeline.
---
Results:
Resolves base-model sources and tracks retained best checkpoints.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

DEFAULT_BASE_MODEL_ID = "CompVis/stable-diffusion-v1-4"
BASE_MODEL_ROOT = Path("models/base/stable-diffusion-v1-4")
BEST_CHECKPOINTS_ROOT = Path("outputs/minim/checkpoints/best")
REQUIRED_LOCAL_MODEL_FILES: tuple[tuple[str, ...], ...] = (
    ("model_index.json",),
    ("tokenizer/vocab.json",),
    ("tokenizer/merges.txt",),
    ("text_encoder/model.safetensors", "text_encoder/pytorch_model.bin"),
    ("unet/diffusion_pytorch_model.safetensors", "unet/diffusion_pytorch_model.bin"),
    ("vae/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.bin"),
)


def ensure_model_roots(best_checkpoints_root: Path | None = None) -> None:
    """
    ########################################
    Definition:
    Ensure the base-model and best-checkpoint directories exist.
    ---
    Params:
    None.
    ---
    Results:
    Creates the required directories when they are missing.
    ########################################
    """
    BASE_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    (BEST_CHECKPOINTS_ROOT if best_checkpoints_root is None else best_checkpoints_root).mkdir(parents=True, exist_ok=True)


def missing_local_model_files(model_path: Path) -> list[str]:
    """
    ########################################
    Definition:
    Inspect whether one local Stable Diffusion directory contains the required files.
    ---
    Params:
    model_path: Local model directory to validate.
    ---
    Results:
    Returns the missing required paths or path alternatives.
    ########################################
    """
    missing: list[str] = []
    for candidate_group in REQUIRED_LOCAL_MODEL_FILES:
        if any((model_path / relative_path).exists() for relative_path in candidate_group):
            continue
        missing.append(" or ".join(candidate_group))
    return missing


def resolve_base_model(base_model: str | None = None) -> str:
    """
    ########################################
    Definition:
    Resolve the base-model source for training or inference.
    ---
    Params:
    base_model: Optional explicit model id or local path.
    ---
    Results:
    Returns a local checkpoint path when available, otherwise a model id.
    ########################################
    """
    ensure_model_roots()
    if base_model is None:
        if not BASE_MODEL_ROOT.exists():
            return DEFAULT_BASE_MODEL_ID
        if not any(BASE_MODEL_ROOT.iterdir()):
            raise RuntimeError(
                f"Local base model directory '{BASE_MODEL_ROOT}' exists but is empty. "
                "Download the full Stable Diffusion v1.4 snapshot into that directory or pass "
                f"--base-model {DEFAULT_BASE_MODEL_ID} only when the runtime has Hugging Face access."
            )
        missing_files = missing_local_model_files(BASE_MODEL_ROOT)
        if missing_files:
            missing_summary = ", ".join(missing_files)
            raise RuntimeError(
                f"Local base model directory '{BASE_MODEL_ROOT}' is incomplete. Missing required files: {missing_summary}. "
                f"Re-download the full Stable Diffusion v1.4 snapshot into '{BASE_MODEL_ROOT}' or pass --base-model {DEFAULT_BASE_MODEL_ID}."
            )
        return str(BASE_MODEL_ROOT)

    candidate_path = Path(base_model)
    if candidate_path.exists():
        missing_files = missing_local_model_files(candidate_path)
        if missing_files:
            missing_summary = ", ".join(missing_files)
            raise RuntimeError(
                f"Local base model directory '{candidate_path}' is incomplete. Missing required files: {missing_summary}."
            )
        return str(candidate_path.resolve())
    return base_model


def latest_training_checkpoints(model_dir: Path) -> list[Path]:
    """
    ########################################
    Definition:
    Collect the training-state checkpoints for one model directory.
    ---
    Params:
    model_dir: Directory that may contain `checkpoint-*` subdirectories.
    ---
    Results:
    Returns checkpoints ordered from newest to oldest.
    ########################################
    """
    if not model_dir.exists():
        return []
    checkpoints = [path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")]
    return sorted(checkpoints, key=lambda path: int(path.name.split("-")[1]), reverse=True)


def _metric_or_none(metrics: dict[str, object], metric_name: str) -> float | None:
    value = metrics.get(metric_name)
    if value is None:
        return None
    numeric_value = float(value)
    return None if math.isnan(numeric_value) else numeric_value


def register_best_checkpoint(
    run_name: str,
    model_dir: Path,
    metrics: dict[str, object],
    limit: int = 3,
    registry_root: Path | None = None,
) -> Path:
    """
    ########################################
    Definition:
    Update the retained best-checkpoint registry using evaluation metrics.
    ---
    Params:
    run_name: Logical run identifier.
    model_dir: Directory containing the exported fine-tuned model.
    metrics: Evaluation metrics for ranking the run.
    limit: Maximum number of best runs to retain in the registry.
    registry_root: Optional root directory for the best-checkpoint registry.
    ---
    Results:
    Writes the best-checkpoint registry and returns its path.
    ########################################
    """
    best_checkpoints_root = BEST_CHECKPOINTS_ROOT if registry_root is None else registry_root
    ensure_model_roots(best_checkpoints_root=best_checkpoints_root)
    registry_path = best_checkpoints_root / "registry.json"
    existing: list[dict[str, str | float | int]] = []
    if registry_path.exists():
        existing = json.loads(registry_path.read_text(encoding="utf-8"))

    entry = {
        "run_name": run_name,
        "model_dir": str(model_dir.resolve()),
        "fid": _metric_or_none(metrics, "fid"),
        "is": _metric_or_none(metrics, "is"),
        "ms_ssim": _metric_or_none(metrics, "ms_ssim"),
    }

    filtered = [item for item in existing if item["run_name"] != run_name]
    filtered.append(entry)

    def metric_value(item: dict[str, str | float | int | None], metric_name: str, fallback: float) -> float:
        value = item.get(metric_name)
        if value is None:
            return fallback
        numeric_value = float(value)
        return fallback if math.isnan(numeric_value) else numeric_value

    ranked = sorted(
        filtered,
        key=lambda item: (
            metric_value(item, "fid", math.inf),
            -metric_value(item, "is", -math.inf),
            metric_value(item, "ms_ssim", math.inf),
        ),
    )[:limit]
    registry_path.write_text(json.dumps(ranked, indent=2, allow_nan=False), encoding="utf-8")

    for current_path in best_checkpoints_root.iterdir():
        if current_path.name == "registry.json":
            continue
        if current_path.is_symlink() or current_path.is_file():
            current_path.unlink()

    for index, item in enumerate(ranked, start=1):
        link_path = best_checkpoints_root / f"{index:02d}_{item['run_name']}"
        target_path = Path(str(item["model_dir"]))
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        try:
            os.symlink(target_path, link_path)
        except OSError:
            link_path.write_text(str(target_path), encoding="utf-8")

    return registry_path
