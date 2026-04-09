"""
Definition:
Brief map of the top-level dataset pipeline orchestration.
---
Results:
Connects dataset drivers, validation, and CSV writing.
"""

from __future__ import annotations

import csv
from pathlib import Path

from data.config import DATASET_PATHS, OUTPUT_PATHS
from data.datasets.acdc.pipeline import ACDC_DRIVER
from data.datasets.driver_contract import DatasetDriver
from data.datasets.ukbb.pipeline import UKBB_DRIVER
from data.export.minim_csv import validate_minim_csv, write_minim_csv
from data.export.row_contract import DataRow

Row = dict[str, str]
DATASET_DRIVERS: dict[str, DatasetDriver] = {
    "acdc": ACDC_DRIVER,
    "ukbb": UKBB_DRIVER,
}


def _build_output_csv_path(csv_root: Path, dataset: str) -> Path:
    """
    ########################################
    Definition:
    Build the output CSV path for a dataset.
    ---
    Params:
    csv_root: Root directory for generated CSV files.
    dataset: Dataset identifier.
    ---
    Results:
    Returns the CSV file path that should be written.
    ########################################
    """
    return csv_root / f"{dataset}_minim.csv"


def _build_internal_csv_path(internal_root: Path, dataset: str) -> Path:
    """
    ########################################
    Definition:
    Build the internal canonical-row CSV path for a dataset.
    ---
    Params:
    internal_root: Root directory for canonical split-capable manifests.
    dataset: Dataset identifier.
    ---
    Results:
    Returns the CSV path used to persist the full internal row contract.
    ########################################
    """
    return internal_root / f"{dataset}_rows.csv"


def _normalize_rows(rows: list[Row | DataRow]) -> list[Row]:
    """
    ########################################
    Definition:
    Normalize driver outputs into the canonical plain-dictionary row contract.
    ---
    Params:
    rows: Driver outputs as dictionaries or `DataRow` instances.
    ---
    Results:
    Returns a list of dictionaries with canonical row fields.
    ########################################
    """
    normalized_rows: list[Row] = []
    for row in rows:
        if isinstance(row, DataRow):
            normalized_rows.append(row.to_dict())
        else:
            normalized_rows.append(row)
    return normalized_rows


def write_internal_rows(rows: list[Row], output_csv_path: Path) -> None:
    """
    ########################################
    Definition:
    Persist the full canonical row contract required for downstream splits.
    ---
    Params:
    rows: Canonical rows including internal-only fields.
    output_csv_path: Destination path for the internal CSV.
    ---
    Results:
    Writes a CSV with all canonical fields.
    ########################################
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with Path.open(output_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "text", "modality", "patient_id", "dataset"])
        writer.writeheader()
        writer.writerows(rows)


def load_internal_rows(internal_root: Path, dataset: str) -> list[Row]:
    """
    ########################################
    Definition:
    Load the persisted canonical row contract for one dataset.
    ---
    Params:
    internal_root: Root directory containing canonical row CSVs.
    dataset: Dataset identifier.
    ---
    Results:
    Returns the canonical rows required for split generation.
    ########################################
    """
    internal_csv_path = _build_internal_csv_path(internal_root, dataset)
    if not internal_csv_path.exists():
        raise FileNotFoundError(
            f"Missing internal manifest for dataset '{dataset}' at {internal_csv_path}. "
            f"Run `python prepare -d {dataset}` first."
        )
    with Path.open(internal_csv_path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_csv_pipeline(
    data_path: Path,                # dataset path where training data is stored
    images_root: Path,              # path where images will be stored (output)
    csv_root: Path,                 # path where csv will be stored (output)
    internal_root: Path | None = None,
    dataset: str = "acdc",          # dataset identifier
    modality: str = "Cardiac MRI",  # modality identifier
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
    dataset: Dataset identifier used to select the driver.
    modality: Modality label written into exported rows.
    Results:
    Returns the list of generated rows after validation and CSV writing.
    ---
    Other Information:
    Raises ValueError when the requested dataset driver is not registered.
    ########################################
    """
    try:
        dataset_driver = DATASET_DRIVERS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset '{dataset}'.") from exc

    output_csv_path = _build_output_csv_path(csv_root, dataset)
    if internal_root is None:
        internal_root = OUTPUT_PATHS["internal"]
    internal_csv_path = _build_internal_csv_path(internal_root, dataset)
    print(f"Starting {dataset.upper()} preprocessing...")
    print(f"Reading data from {data_path}.")

    rows = _normalize_rows(
        dataset_driver.build_rows(
            data_path=data_path,
            images_root=images_root,
            modality=modality,
        )
    )

    print(f"Prepared {len(rows)} rows. Writing outputs to {images_root}.")
    validate_minim_csv(rows, images_root)
    write_minim_csv(rows, output_csv_path)
    write_internal_rows(rows, internal_csv_path)
    print(f"{dataset.upper()} export completed successfully.")

    return rows
