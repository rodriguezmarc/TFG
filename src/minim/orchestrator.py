"""
Definition:
End-to-end MINIM orchestration across preparation, backend execution, and evaluation.
---
Results:
Runs prepared manifests through selected model backends and records metrics/checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path

from minim.backends import get_backend
from minim.checkpoints import register_best_checkpoint
from minim.metrics import evaluate_from_manifest
from minim.preparation import prepare_run
from minim.runs import PreparedRun, backend_run_variant, write_backend_variant_summary

logger = logging.getLogger(__name__)


def _validate_backend(backend: str) -> str:
    return get_backend(backend).name


def _evaluate_with_backend(
    prepared_run: PreparedRun, backend: str, metrics_mode: str, **kwargs
) -> dict[str, object]:
    selected_backend = get_backend(backend)
    device = kwargs.get("device")
    logger.info(
        "Starting %s evaluation for run '%s'.",
        selected_backend.name,
        prepared_run.run_name,
    )
    logger.info("Test manifest: %s", prepared_run.test_manifest)
    selected_backend.generate(prepared_run=prepared_run, **kwargs)
    metrics = evaluate_from_manifest(
        real_manifest_path=prepared_run.test_manifest,
        generated_dir=prepared_run.generated_test_dir,
        device="cpu" if device is None else device,
        output_json_path=prepared_run.metrics_path,
        mode=metrics_mode,
    )
    logger.info("Metrics written to %s", prepared_run.metrics_path)
    logger.info(
        "%s evaluation completed for run '%s'.",
        selected_backend.name.capitalize(),
        prepared_run.run_name,
    )
    return metrics


def _run_training_and_evaluation(
    prepared_run: PreparedRun,
    backend: str,
    *,
    base_model: str | None,
    num_train_epochs: int,
    max_train_steps: int | None,
    learning_rate: float,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    resolution: int,
    seed: int,
    mixed_precision: str | None,
    validation_epochs: int,
    report_to: str,
    skip_evaluation: bool,
    device: str | None,
    num_inference_steps: int,
    guidance_scale: float,
    metrics_mode: str,
) -> dict[str, object]:
    get_backend(backend).train(
        prepared_run=prepared_run,
        base_model=base_model,
        num_train_epochs=num_train_epochs,
        max_train_steps=max_train_steps,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        resolution=resolution,
        seed=seed,
        mixed_precision=mixed_precision,
        validation_epochs=validation_epochs,
        report_to=report_to,
    )
    if skip_evaluation:
        return {"evaluation_skipped": 1}
    metrics = _evaluate_with_backend(
        prepared_run=prepared_run,
        backend=backend,
        device=device,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        metrics_mode=metrics_mode,
    )
    register_best_checkpoint(
        prepared_run.run_name,
        prepared_run.model_dir,
        metrics,
        registry_root=prepared_run.model_dir.parent / "best",
    )
    return metrics


def _run_backend_repetitions(
    prepared_run: PreparedRun,
    backend: str,
    *,
    runs: int,
    base_model: str | None,
    num_train_epochs: int,
    max_train_steps: int | None,
    learning_rate: float,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    resolution: int,
    seed: int,
    mixed_precision: str | None,
    validation_epochs: int,
    report_to: str,
    skip_evaluation: bool,
    device: str | None,
    num_inference_steps: int,
    guidance_scale: float,
    metrics_mode: str,
) -> tuple[PreparedRun, dict[str, object]]:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if backend != "mock" and runs != 1:
        raise ValueError("--runs is only supported with --backend mock")

    last_run = prepared_run
    last_metrics: dict[str, object] = {"evaluation_skipped": 1}
    for index in range(1, runs + 1):
        current_run = (
            backend_run_variant(prepared_run, backend, index)
            if backend == "mock" and runs > 1
            else prepared_run
        )
        if backend == "mock" and runs > 1:
            write_backend_variant_summary(prepared_run, current_run, backend, index, runs)
            logger.info("Starting mock run %s/%s: %s", index, runs, current_run.run_name)
        current_seed = seed + index - 1 if backend == "mock" else seed
        last_metrics = _run_training_and_evaluation(
            prepared_run=current_run,
            backend=backend,
            base_model=base_model,
            num_train_epochs=num_train_epochs,
            max_train_steps=max_train_steps,
            learning_rate=learning_rate,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            resolution=resolution,
            seed=current_seed,
            mixed_precision=mixed_precision,
            validation_epochs=validation_epochs,
            report_to=report_to,
            skip_evaluation=skip_evaluation,
            device=device,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            metrics_mode=metrics_mode,
        )
        last_run = current_run
    return last_run, last_metrics


def run_full_pipeline(
    datasets: tuple[str, ...],
    images_root: Path | None = None,
    csv_root: Path | None = None,
    internal_root: Path | None = None,
    output_root: Path | None = None,
    modality: str = "Cardiac MRI",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    base_model: str | None = None,
    num_train_epochs: int = 5,
    max_train_steps: int | None = None,
    learning_rate: float = 1e-5,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    resolution: int = 512,
    mixed_precision: str | None = None,
    validation_epochs: int = 5,
    report_to: str = "tensorboard",
    backend: str = "real",
    runs: int = 1,
    skip_evaluation: bool = False,
    device: str | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    metrics_mode: str = "full",
) -> tuple[PreparedRun, dict[str, object]]:
    backend = _validate_backend(backend)
    prepared_run = prepare_run(
        datasets=datasets,
        images_root=images_root,
        csv_root=csv_root,
        internal_root=internal_root,
        output_root=output_root,
        modality=modality,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        refresh_exports=True,
    )
    return _run_backend_repetitions(
        prepared_run=prepared_run,
        backend=backend,
        runs=runs,
        base_model=base_model,
        num_train_epochs=num_train_epochs,
        max_train_steps=max_train_steps,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        resolution=resolution,
        seed=seed,
        mixed_precision=mixed_precision,
        validation_epochs=validation_epochs,
        report_to=report_to,
        skip_evaluation=skip_evaluation,
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        metrics_mode=metrics_mode,
    )


def run_evaluation_pipeline(
    datasets: tuple[str, ...],
    images_root: Path | None = None,
    csv_root: Path | None = None,
    internal_root: Path | None = None,
    output_root: Path | None = None,
    modality: str = "Cardiac MRI",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    base_model: str | None = None,
    num_train_epochs: int = 5,
    max_train_steps: int | None = None,
    learning_rate: float = 1e-5,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    resolution: int = 512,
    mixed_precision: str | None = None,
    validation_epochs: int = 5,
    report_to: str = "tensorboard",
    backend: str = "real",
    runs: int = 1,
    skip_evaluation: bool = False,
    device: str | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    metrics_mode: str = "full",
) -> tuple[PreparedRun, dict[str, object]]:
    backend = _validate_backend(backend)
    prepared_run = prepare_run(
        datasets=datasets,
        images_root=images_root,
        csv_root=csv_root,
        internal_root=internal_root,
        output_root=output_root,
        modality=modality,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        refresh_exports=False,
    )
    return _run_backend_repetitions(
        prepared_run=prepared_run,
        backend=backend,
        runs=runs,
        base_model=base_model,
        num_train_epochs=num_train_epochs,
        max_train_steps=max_train_steps,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        resolution=resolution,
        seed=seed,
        mixed_precision=mixed_precision,
        validation_epochs=validation_epochs,
        report_to=report_to,
        skip_evaluation=skip_evaluation,
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        metrics_mode=metrics_mode,
    )
