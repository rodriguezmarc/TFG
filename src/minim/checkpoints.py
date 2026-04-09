"""
Definition:
Brief map of checkpoint and base-model path helpers for the local MINIM pipeline.
---
Results:
Resolves base-model sources and tracks retained best checkpoints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_BASE_MODEL_ID = "CompVis/stable-diffusion-v1-4"
BASE_MODEL_ROOT = Path("models/base/stable-diffusion-v1-4")
BEST_CHECKPOINTS_ROOT = Path("outputs/minim/checkpoints/best")


def ensure_model_roots() -> None:
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
    BEST_CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)


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
        return str(BASE_MODEL_ROOT) if BASE_MODEL_ROOT.exists() and any(BASE_MODEL_ROOT.iterdir()) else DEFAULT_BASE_MODEL_ID

    candidate_path = Path(base_model)
    if candidate_path.exists():
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


def register_best_checkpoint(run_name: str, model_dir: Path, metrics: dict[str, float | int], limit: int = 3) -> Path:
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
    ---
    Results:
    Writes the best-checkpoint registry and returns its path.
    ########################################
    """
    ensure_model_roots()
    registry_path = BEST_CHECKPOINTS_ROOT / "registry.json"
    existing: list[dict[str, str | float | int]] = []
    if registry_path.exists():
        existing = json.loads(registry_path.read_text(encoding="utf-8"))

    entry = {
        "run_name": run_name,
        "model_dir": str(model_dir.resolve()),
        "fid": float(metrics["fid"]),
        "is": float(metrics["is"]),
        "ms_ssim": float(metrics["ms_ssim"]),
    }

    filtered = [item for item in existing if item["run_name"] != run_name]
    filtered.append(entry)
    ranked = sorted(filtered, key=lambda item: (float(item["fid"]), -float(item["is"]), float(item["ms_ssim"])))[:limit]
    registry_path.write_text(json.dumps(ranked, indent=2), encoding="utf-8")

    for current_path in BEST_CHECKPOINTS_ROOT.iterdir():
        if current_path.name == "registry.json":
            continue
        if current_path.is_symlink() or current_path.is_file():
            current_path.unlink()

    for index, item in enumerate(ranked, start=1):
        link_path = BEST_CHECKPOINTS_ROOT / f"{index:02d}_{item['run_name']}"
        target_path = Path(str(item["model_dir"]))
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        try:
            os.symlink(target_path, link_path)
        except OSError:
            link_path.write_text(str(target_path), encoding="utf-8")

    return registry_path
