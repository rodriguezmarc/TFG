"""
Definition:
Run artifact contracts for MINIM-compatible pipelines.
---
Results:
Centralizes naming and path construction for prepared and repeated runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class PreparedRun:
    """
    ########################################
    Definition:
    Store the canonical artifact paths for one prepared MINIM run.
    ---
    Results:
    Provides a stable contract between preparation, training, and evaluation stages.
    ########################################
    """
    run_name: str
    run_root: Path
    train_manifest: Path
    val_manifest: Path
    test_manifest: Path
    summary_path: Path
    model_dir: Path
    eval_dir: Path
    metrics_path: Path
    generated_test_dir: Path


def run_name(datasets: tuple[str, ...]) -> str:
    return datasets[0] if len(datasets) == 1 else "_".join(datasets)


def build_prepared_run(datasets: tuple[str, ...], output_root: Path) -> PreparedRun:
    name = run_name(datasets)
    run_root = output_root / "runs" / name
    eval_dir = run_root / "evaluation"
    return PreparedRun(
        run_name=name,
        run_root=run_root,
        train_manifest=output_root / "manifests" / name / "train.csv",
        val_manifest=output_root / "manifests" / name / "val.csv",
        test_manifest=output_root / "manifests" / name / "test.csv",
        summary_path=run_root / "summary.json",
        model_dir=output_root / "checkpoints" / name,
        eval_dir=eval_dir,
        metrics_path=eval_dir / "metrics.json",
        generated_test_dir=eval_dir / "generated",
    )


def write_prepared_summary(prepared_run: PreparedRun, datasets: tuple[str, ...], seed: int, counts: dict[str, int]) -> None:
    summary = {
        "run_name": prepared_run.run_name,
        "datasets": list(datasets),
        "seed": seed,
        "counts": counts,
        "paths": {
            "train_manifest": str(prepared_run.train_manifest.resolve()),
            "val_manifest": str(prepared_run.val_manifest.resolve()),
            "test_manifest": str(prepared_run.test_manifest.resolve()),
            "model_dir": str(prepared_run.model_dir.resolve()),
            "eval_dir": str(prepared_run.eval_dir.resolve()),
            "metrics_path": str(prepared_run.metrics_path.resolve()),
        },
    }
    prepared_run.summary_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_run.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def backend_run_variant(prepared_run: PreparedRun, backend: str, iteration: int) -> PreparedRun:
    name = f"{prepared_run.run_name}_{backend}_{iteration:03d}"
    run_root = prepared_run.run_root.parent / name
    eval_dir = run_root / "evaluation"
    return replace(
        prepared_run,
        run_name=name,
        run_root=run_root,
        summary_path=run_root / "summary.json",
        model_dir=prepared_run.model_dir.parent / name,
        eval_dir=eval_dir,
        metrics_path=eval_dir / "metrics.json",
        generated_test_dir=eval_dir / "generated",
    )


def write_backend_variant_summary(
    base_run: PreparedRun,
    variant_run: PreparedRun,
    backend: str,
    iteration: int,
    total_runs: int,
) -> None:
    summary = {
        "run_name": variant_run.run_name,
        "base_run_name": base_run.run_name,
        "backend": backend,
        "backend_iteration": iteration,
        "backend_runs": total_runs,
        "paths": {
            "train_manifest": str(variant_run.train_manifest.resolve()),
            "val_manifest": str(variant_run.val_manifest.resolve()),
            "test_manifest": str(variant_run.test_manifest.resolve()),
            "model_dir": str(variant_run.model_dir.resolve()),
            "eval_dir": str(variant_run.eval_dir.resolve()),
            "metrics_path": str(variant_run.metrics_path.resolve()),
        },
    }
    variant_run.summary_path.parent.mkdir(parents=True, exist_ok=True)
    variant_run.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
