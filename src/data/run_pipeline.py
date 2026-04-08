"""
Definition:
Brief map of the top-level dataset pipeline orchestration.
---
Results:
Connects dataset drivers, validation, and CSV writing.
"""

from __future__ import annotations

from pathlib import Path

from data.cli.data_cli import parse_args
from data.cli.data_config import DATASET_PATHS, OUTPUT_PATHS
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


def run_csv_pipeline(
    data_path: Path,                # dataset path where training data is stored
    images_root: Path,              # path where images will be stored (output)
    csv_root: Path,                 # path where csv will be stored (output)
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
    print(f"{dataset.upper()} export completed successfully.")

    return rows


if __name__ == "__main__":
    args = parse_args()
    data_path = Path(DATASET_PATHS.get(args.dataset, Path("ACDC")))  # fallback to acdc
    images_root = OUTPUT_PATHS["images"]
    csv_root = OUTPUT_PATHS["csv"]

    rows = run_csv_pipeline(
        data_path,
        images_root,
        csv_root,
        dataset=args.dataset,
    )

    print(f"Finished processing {len(rows)} cases.")
