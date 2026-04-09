"""
Definition:
Brief map of the end-to-end local MINIM orchestration pipeline.
---
Results:
Connects dataset preparation, fine-tuning, generation, and evaluation.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from data.config import DATASET_PATHS, OUTPUT_PATHS
from data.export.minim_csv import REQUIRED_ROW_COLUMNS
from data.run_pipeline import load_internal_rows, run_csv_pipeline
from data.splits.patient_id_split import split_patient_ids
from minim.checkpoints import DEFAULT_BASE_MODEL_ID, register_best_checkpoint, resolve_base_model

Row = dict[str, str]
DEFAULT_BASE_MODEL = DEFAULT_BASE_MODEL_ID


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


def _normalize_row(row: Row) -> Row:
    """
    ########################################
    Definition:
    Validate that one canonical row satisfies the downstream contract.
    ---
    Params:
    row: Canonical row dictionary to inspect.
    ---
    Results:
    Returns the row unchanged when all required columns are present.
    ########################################
    """
    missing_cols = [column for column in REQUIRED_ROW_COLUMNS if column not in row]
    if missing_cols:
        raise ValueError(f"Row missing required columns: {missing_cols}")
    return row


def _run_name(datasets: tuple[str, ...]) -> str:
    """
    ########################################
    Definition:
    Build the logical run name for one or more datasets.
    ---
    Params:
    datasets: Ordered dataset identifiers included in the run.
    ---
    Results:
    Returns the stable run name used for artifact directories.
    ########################################
    """
    return datasets[0] if len(datasets) == 1 else "_".join(datasets)


def _prepare_training_rows(rows: list[Row], images_root: Path) -> list[dict[str, str]]:
    """
    ########################################
    Definition:
    Convert canonical export rows into training-ready manifest rows.
    ---
    Params:
    rows: Canonical dataset-export rows.
    images_root: Root directory containing processed images.
    ---
    Results:
    Returns rows with absolute image paths and training columns.
    ########################################
    """
    prepared_rows: list[dict[str, str]] = []
    for row in rows:
        normalized = _normalize_row(row)
        prepared_rows.append(
            {
                "path": str((images_root / normalized["path"]).resolve()),
                "text": normalized["text"],
                "modality": normalized["modality"],
                "patient_id": normalized["patient_id"],
                "dataset": normalized["dataset"],
            }
        )
    return prepared_rows


def _write_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    """
    ########################################
    Definition:
    Write one split manifest used by the MINIM pipeline.
    ---
    Params:
    rows: Training-ready rows to serialize.
    output_path: Destination path for the manifest.
    ---
    Results:
    Creates parent directories when needed and writes the CSV file.
    ########################################
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path.open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "text", "modality", "patient_id", "dataset"])
        writer.writeheader()
        writer.writerows(rows)


def _split_rows(rows: list[dict[str, str]], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[dict[str, str]]]:
    """
    ########################################
    Definition:
    Split rows into train, validation, and test partitions by patient id.
    ---
    Params:
    rows: Training-ready rows containing `patient_id`.
    train_ratio: Fraction assigned to the training split.
    val_ratio: Fraction assigned to the validation split.
    test_ratio: Fraction assigned to the test split.
    seed: Random seed used by the patient split helper.
    ---
    Results:
    Returns the split rows grouped by split name.
    ########################################
    """
    patient_ids = sorted({row["patient_id"] for row in rows})
    split = split_patient_ids(patient_ids, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    split_lookup = {patient_id: "train" for patient_id in split.train_ids}
    split_lookup.update({patient_id: "val" for patient_id in split.val_ids})
    split_lookup.update({patient_id: "test" for patient_id in split.test_ids})
    split_rows = {"train": [], "val": [], "test": []}
    for row in rows:
        split_rows[split_lookup[row["patient_id"]]].append(row)
    return split_rows


def prepare_run(
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
    refresh_exports: bool = True,
) -> PreparedRun:
    """
    ########################################
    Definition:
    Prepare manifests and artifact paths for one MINIM run.
    ---
    Params:
    datasets: Dataset identifiers included in the run.
    images_root: Optional root for processed image artifacts.
    csv_root: Optional root for exported public CSV manifests.
    internal_root: Optional root for internal canonical row manifests.
    output_root: Optional root for MINIM run artifacts.
    modality: Modality label used in prompts and manifests.
    train_ratio: Fraction assigned to the training split.
    val_ratio: Fraction assigned to the validation split.
    test_ratio: Fraction assigned to the test split.
    seed: Random seed used by the patient split helper.
    refresh_exports: Whether to rerun dataset export before manifest preparation.
    ---
    Results:
    Returns the prepared-run artifact contract.
    ########################################
    """
    images_root = OUTPUT_PATHS["images"] if images_root is None else images_root
    csv_root = OUTPUT_PATHS["csv"] if csv_root is None else csv_root
    internal_root = OUTPUT_PATHS["internal"] if internal_root is None else internal_root
    output_root = Path("outputs/minim") if output_root is None else output_root

    combined_rows: list[dict[str, str]] = []
    for dataset in datasets:
        if refresh_exports:
            exported_rows = run_csv_pipeline(
                data_path=DATASET_PATHS[dataset],
                images_root=images_root,
                csv_root=csv_root,
                internal_root=internal_root,
                dataset=dataset,
                modality=modality,
            )
        else:
            exported_rows = load_internal_rows(internal_root=internal_root, dataset=dataset)
        combined_rows.extend(_prepare_training_rows(exported_rows, images_root))

    split_rows = _split_rows(combined_rows, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    run_name = _run_name(datasets)
    run_root = output_root / "runs" / run_name
    train_manifest = output_root / "manifests" / run_name / "train.csv"
    val_manifest = output_root / "manifests" / run_name / "val.csv"
    test_manifest = output_root / "manifests" / run_name / "test.csv"
    summary_path = run_root / "summary.json"
    model_dir = output_root / "checkpoints" / run_name
    eval_dir = output_root / "evaluation" / run_name
    metrics_path = output_root / "metrics" / f"{run_name}.json"
    generated_test_dir = eval_dir / "generated_test"

    _write_manifest(split_rows["train"], train_manifest)
    _write_manifest(split_rows["val"], val_manifest)
    _write_manifest(split_rows["test"], test_manifest)

    summary = {
        "run_name": run_name,
        "datasets": list(datasets),
        "seed": seed,
        "counts": {split_name: len(rows) for split_name, rows in split_rows.items()},
        "paths": {
            "train_manifest": str(train_manifest.resolve()),
            "val_manifest": str(val_manifest.resolve()),
            "test_manifest": str(test_manifest.resolve()),
            "model_dir": str(model_dir.resolve()),
            "eval_dir": str(eval_dir.resolve()),
            "metrics_path": str(metrics_path.resolve()),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return PreparedRun(
        run_name=run_name,
        run_root=run_root,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        test_manifest=test_manifest,
        summary_path=summary_path,
        model_dir=model_dir,
        eval_dir=eval_dir,
        metrics_path=metrics_path,
        generated_test_dir=generated_test_dir,
    )


def _read_validation_prompts(val_manifest: Path, max_prompts: int = 4) -> list[str]:
    """
    ########################################
    Definition:
    Read a small prompt sample from the validation manifest.
    ---
    Params:
    val_manifest: Validation manifest path.
    max_prompts: Maximum number of prompts to collect.
    ---
    Results:
    Returns the formatted prompt list used for training-time validation.
    ########################################
    """
    prompts: list[str] = []
    with Path.open(val_manifest, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            prompts.append(f"{row['modality']}: {row['text']}")
            if len(prompts) >= max_prompts:
                break
    return prompts


def train_run(
    prepared_run: PreparedRun,
    base_model: str | None = None,
    num_train_epochs: int = 5,
    learning_rate: float = 1e-5,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    resolution: int = 512,
    seed: int = 42,
    mixed_precision: str | None = None,
) -> subprocess.CompletedProcess:
    """
    ########################################
    Definition:
    Launch MINIM fine-tuning for one prepared run.
    ---
    Params:
    prepared_run: Prepared-run artifact contract.
    base_model: Optional explicit model id or local checkpoint path.
    num_train_epochs: Number of training epochs.
    learning_rate: Learning rate used for optimization.
    train_batch_size: Training batch size per device.
    gradient_accumulation_steps: Number of gradient accumulation steps.
    resolution: Training image resolution.
    seed: Random seed used by the training script.
    mixed_precision: Optional mixed-precision mode.
    ---
    Results:
    Returns the completed subprocess result for the training command.
    ########################################
    """
    resolved_base_model = resolve_base_model(base_model)
    print(f"Starting MINIM fine-tuning for run '{prepared_run.run_name}'.")
    print(f"Training manifest: {prepared_run.train_manifest}")
    print(f"Base model source: {resolved_base_model}")
    print(f"Checkpoint output: {prepared_run.model_dir}")
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
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--learning_rate",
        str(learning_rate),
        "--resolution",
        str(resolution),
        "--seed",
        str(seed),
        "--checkpoints_total_limit",
        "3",
    ]
    if mixed_precision is not None:
        command.extend(["--mixed_precision", mixed_precision])
    if validation_prompts:
        command.append("--validation_prompts")
        command.extend(validation_prompts)
    result = subprocess.run(command, check=True)
    print(f"Fine-tuning completed for run '{prepared_run.run_name}'.")
    return result


def evaluate_run(
    prepared_run: PreparedRun,
    device: str | None = None,
    seed: int | None = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> dict[str, float | int]:
    """
    ########################################
    Definition:
    Generate evaluation samples and compute metrics for one prepared run.
    ---
    Params:
    prepared_run: Prepared-run artifact contract.
    device: Optional explicit device string.
    seed: Optional random seed for reproducible generation.
    num_inference_steps: Number of denoising steps to execute.
    guidance_scale: Classifier-free guidance scale.
    ---
    Results:
    Returns the computed evaluation metrics.
    ########################################
    """
    from minim.generate import generate_from_manifest
    from minim.metrics import evaluate_from_manifest

    print(f"Starting evaluation for run '{prepared_run.run_name}'.")
    print(f"Test manifest: {prepared_run.test_manifest}")
    print(f"Generated samples: {prepared_run.generated_test_dir}")
    generate_from_manifest(
        model_path=prepared_run.model_dir,
        manifest_path=prepared_run.test_manifest,
        output_dir=prepared_run.generated_test_dir,
        device=device,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    metrics = evaluate_from_manifest(
        real_manifest_path=prepared_run.test_manifest,
        generated_dir=prepared_run.generated_test_dir,
        device="cpu" if device is None else device,
        output_json_path=prepared_run.metrics_path,
    )
    print(f"Metrics written to {prepared_run.metrics_path}")
    print(f"Evaluation completed for run '{prepared_run.run_name}'.")
    return metrics


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
    learning_rate: float = 1e-5,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    resolution: int = 512,
    mixed_precision: str | None = None,
    device: str | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> tuple[PreparedRun, dict[str, float | int]]:
    """
    ########################################
    Definition:
    Execute preprocessing, fine-tuning, and evaluation end to end.
    ---
    Params:
    datasets: Dataset identifiers included in the run.
    images_root: Optional root for processed image artifacts.
    csv_root: Optional root for exported public CSV manifests.
    internal_root: Optional root for internal canonical row manifests.
    output_root: Optional root for MINIM run artifacts.
    modality: Modality label used in prompts and manifests.
    train_ratio: Fraction assigned to the training split.
    val_ratio: Fraction assigned to the validation split.
    test_ratio: Fraction assigned to the test split.
    seed: Random seed used across the pipeline.
    base_model: Optional explicit model id or local checkpoint path.
    num_train_epochs: Number of training epochs.
    learning_rate: Learning rate used for optimization.
    train_batch_size: Training batch size per device.
    gradient_accumulation_steps: Number of gradient accumulation steps.
    resolution: Training image resolution.
    mixed_precision: Optional mixed-precision mode.
    device: Optional explicit device string.
    num_inference_steps: Number of denoising steps to execute.
    guidance_scale: Classifier-free guidance scale.
    ---
    Results:
    Returns the prepared-run artifact contract and computed metrics.
    ########################################
    """
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
    train_run(
        prepared_run=prepared_run,
        base_model=base_model,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        resolution=resolution,
        seed=seed,
        mixed_precision=mixed_precision,
    )
    metrics = evaluate_run(
        prepared_run=prepared_run,
        device=device,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    register_best_checkpoint(prepared_run.run_name, prepared_run.model_dir, metrics)
    return prepared_run, metrics


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
    learning_rate: float = 1e-5,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    resolution: int = 512,
    mixed_precision: str | None = None,
    device: str | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> tuple[PreparedRun, dict[str, float | int]]:
    """
    ########################################
    Definition:
    Execute fine-tuning and evaluation using previously prepared exports.
    ---
    Params:
    datasets: Dataset identifiers included in the run.
    images_root: Optional root for processed image artifacts.
    csv_root: Optional root for exported public CSV manifests.
    internal_root: Optional root for internal canonical row manifests.
    output_root: Optional root for MINIM run artifacts.
    modality: Modality label used in prompts and manifests.
    train_ratio: Fraction assigned to the training split.
    val_ratio: Fraction assigned to the validation split.
    test_ratio: Fraction assigned to the test split.
    seed: Random seed used across the pipeline.
    base_model: Optional explicit model id or local checkpoint path.
    num_train_epochs: Number of training epochs.
    learning_rate: Learning rate used for optimization.
    train_batch_size: Training batch size per device.
    gradient_accumulation_steps: Number of gradient accumulation steps.
    resolution: Training image resolution.
    mixed_precision: Optional mixed-precision mode.
    device: Optional explicit device string.
    num_inference_steps: Number of denoising steps to execute.
    guidance_scale: Classifier-free guidance scale.
    ---
    Results:
    Returns the prepared-run artifact contract and computed metrics.
    ########################################
    """
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
    train_run(
        prepared_run=prepared_run,
        base_model=base_model,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        resolution=resolution,
        seed=seed,
        mixed_precision=mixed_precision,
    )
    metrics = evaluate_run(
        prepared_run=prepared_run,
        device=device,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    register_best_checkpoint(prepared_run.run_name, prepared_run.model_dir, metrics)
    return prepared_run, metrics


def run_prompt(
    prepared_run: PreparedRun,
    prompt: str,
    modality: str = "Cardiac MRI",
    count: int = 1,
    output_dir: Path | None = None,
    device: str | None = None,
    seed: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> list[Path]:
    """
    ########################################
    Definition:
    Generate one or more images from an explicit prompt using a prepared run.
    ---
    Params:
    prepared_run: Prepared-run artifact contract.
    prompt: Prompt text to render.
    modality: Optional modality prefix to prepend to prompts.
    count: Number of images to generate.
    output_dir: Optional output directory for generated images.
    device: Optional explicit device string.
    seed: Optional random seed for reproducibility.
    num_inference_steps: Number of denoising steps to execute.
    guidance_scale: Classifier-free guidance scale.
    ---
    Results:
    Returns the output paths of the generated images.
    ########################################
    """
    from minim.generate import generate_images

    output_dir = prepared_run.run_root / "generated" if output_dir is None else output_dir
    return generate_images(
        model_path=prepared_run.model_dir,
        prompts=[prompt] * count,
        modality=modality,
        output_dir=output_dir,
        device=device,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
