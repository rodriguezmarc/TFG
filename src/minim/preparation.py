"""
Definition:
Dataset preparation for MINIM-compatible training runs.
---
Results:
Builds train/validation/test manifests and run summaries from exported dataset rows.
"""

from __future__ import annotations

import csv
from pathlib import Path

from data.config import DATASET_PATHS, OUTPUT_PATHS
from data.export.minim_csv import REQUIRED_ROW_COLUMNS
from data.run_pipeline import load_internal_rows, run_csv_pipeline
from data.splits.patient_id_split import split_patient_ids
from minim.constants import DEFAULT_OUTPUT_ROOT
from minim.runs import PreparedRun, build_prepared_run, write_prepared_summary

Row = dict[str, str]


def normalize_row(row: Row) -> Row:
    missing_cols = [column for column in REQUIRED_ROW_COLUMNS if column not in row]
    if missing_cols:
        raise ValueError(f"Row missing required columns: {missing_cols}")
    return row


def prepare_training_rows(rows: list[Row], images_root: Path) -> list[dict[str, str]]:
    prepared_rows: list[dict[str, str]] = []
    for row in rows:
        normalized = normalize_row(row)
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


def write_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path.open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "text", "modality", "patient_id", "dataset"])
        writer.writeheader()
        writer.writerows(rows)


def split_rows(rows: list[dict[str, str]], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[dict[str, str]]]:
    patient_ids = sorted({row["patient_id"] for row in rows})
    split = split_patient_ids(patient_ids, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    split_lookup = {patient_id: "train" for patient_id in split.train_ids}
    split_lookup.update({patient_id: "val" for patient_id in split.val_ids})
    split_lookup.update({patient_id: "test" for patient_id in split.test_ids})
    grouped_rows = {"train": [], "val": [], "test": []}
    for row in rows:
        grouped_rows[split_lookup[row["patient_id"]]].append(row)
    return grouped_rows


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
    images_root = OUTPUT_PATHS["images"] if images_root is None else images_root
    csv_root = OUTPUT_PATHS["csv"] if csv_root is None else csv_root
    internal_root = OUTPUT_PATHS["internal"] if internal_root is None else internal_root
    output_root = DEFAULT_OUTPUT_ROOT if output_root is None else output_root

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
        combined_rows.extend(prepare_training_rows(exported_rows, images_root))

    grouped_rows = split_rows(combined_rows, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    prepared_run = build_prepared_run(datasets, output_root)
    write_manifest(grouped_rows["train"], prepared_run.train_manifest)
    write_manifest(grouped_rows["val"], prepared_run.val_manifest)
    write_manifest(grouped_rows["test"], prepared_run.test_manifest)
    write_prepared_summary(prepared_run, datasets=datasets, seed=seed, counts={name: len(rows) for name, rows in grouped_rows.items()})
    return prepared_run
