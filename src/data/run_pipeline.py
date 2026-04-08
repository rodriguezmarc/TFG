"""
########################################
Definition:
Brief map of the top-level dataset pipeline orchestration.
---
Params:
None.
---
Results:
Connects dataset builders, validation, split generation, and CSV writing.
########################################
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from data.acdc.pipeline import build_rows as build_acdc_rows
from data.cli.data_cli import parse_args
from data.cli.data_config import DATASET_PATHS, OUTPUT_PATHS
from data.export.minim_csv import validate_minim_csv, write_minim_csv
from data.splits.patient_id_split import split_patient_ids

Row = dict[str, str]
DatasetBuilder = Callable[[Path, Path, str], list[Row]]

DATASET_BUILDERS: dict[str, DatasetBuilder] = {
    "acdc": build_acdc_rows,
}


def _build_output_csv_path(csv_root: Path, dataset: str, split_name: str | None = None) -> Path:
    """
    ########################################
    Definition:
    Build the output CSV path for a dataset and optional split name.
    ---
    Params:
    csv_root: Root directory for generated CSV files.
    dataset: Dataset identifier.
    split_name: Optional split suffix such as train, val, or test.
    ---
    Results:
    Returns the CSV file path that should be written.
    ########################################
    """
    suffix = f"_{split_name}" if split_name else ""
    return csv_root / f"{dataset}{suffix}_minim.csv"


def _partition_rows_by_patient(rows: list[Row], seed: int) -> dict[str, list[Row]]:
    """
    ########################################
    Definition:
    Partition generated rows into train, validation, and test groups by patient id.
    ---
    Params:
    rows: Row dictionaries that already contain a `patient_id` field.
    seed: Random seed forwarded to the patient splitter.
    ---
    Results:
    Returns a dictionary keyed by split name with the corresponding rows.
    ########################################
    """
    patient_ids = sorted({row["patient_id"] for row in rows})
    split_result = split_patient_ids(patient_ids, seed=seed)
    split_sets = {
        "train": set(split_result.train_ids),
        "val": set(split_result.val_ids),
        "test": set(split_result.test_ids),
    }
    return {
        split_name: [row for row in rows if row["patient_id"] in patient_id_set]
        for split_name, patient_id_set in split_sets.items()
    }


def run_csv_pipeline(
    data_path: Path,                # dataset path where training data is stored
    images_root: Path,              # path where images will be stored (output)
    csv_root: Path,                 # path where csv will be stored (output)
    dataset: str = "acdc",          # dataset identifier
    modality: str = "Cardiac MRI",  # modality identifier
    split_mode: str = "combined",
    seed: int = 42,
) -> list[Row]:
    """
    ########################################
    Definition:
    Execute the full dataset-to-CSV export workflow.
    ---
    Params:
    data_path: Dataset input root.
    images_root: Directory where processed images are stored.
    csv_root: Directory where CSV manifests are stored.
    dataset: Dataset identifier used to select the builder.
    modality: Modality label written into exported rows.
    split_mode: Whether to write one combined CSV or one CSV per split.
    seed: Random seed used for patient-level splitting.
    ---
    Results:
    Returns the list of generated rows after validation and CSV writing.
    ---
    Other Information:
    Raises ValueError when the requested dataset builder is not registered.
    ########################################
    """
    try:
        dataset_builder = DATASET_BUILDERS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset '{dataset}'.") from exc

    rows = dataset_builder(
        data_path=data_path,
        images_root=images_root,
        modality=modality,
    )
    validate_minim_csv(rows, images_root)

    if split_mode == "per-split":
        split_rows = _partition_rows_by_patient(rows, seed=seed)
        for split_name, rows_for_split in split_rows.items():
            write_minim_csv(
                rows_for_split,
                _build_output_csv_path(csv_root, dataset, split_name=split_name),
            )
    else:
        write_minim_csv(rows, _build_output_csv_path(csv_root, dataset))

    return rows


if __name__ == "__main__":
    args = parse_args()
    data_path = Path(DATASET_PATHS.get(args.dataset, Path("ACDC")))
    print(f"Processing {args.dataset} dataset...")

    images_root = OUTPUT_PATHS["images"]
    csv_root = OUTPUT_PATHS["csv"]

    rows = run_csv_pipeline(
        data_path,
        images_root,
        csv_root,
        dataset=args.dataset,
        split_mode=args.split,
        seed=args.seed,
    )

    print(f"Processed {len(rows)} patients.")
