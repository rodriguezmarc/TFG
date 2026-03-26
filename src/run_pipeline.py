"""
Top-level orchestration for dataset -> split -> preprocess -> prompt -> CSV.
"""
import os
import argparse
from pathlib import Path

from datasets.acdc.pipeline import build_rows
from export.minim_csv import validate_minim_csv, write_minim_csv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", "-d",
    choices=["acdc", "ukbb"],
    default="acdc",
    help="Dataset to be process (default: acdc)"
)
args = parser.parse_args()

dataset_paths = {
    "acdc": "ACDC",
    "ukbb": "UKBB",
}

data_path = dataset_paths.get(args.dataset, "ACDC")   # ACDC dataset fallback
print(f"Processing {args.dataset} dataset...")

print(f"Procesando dataset: {args.dataset}")

def run__csv_pipeline(
    data_path: Path,                # dataset path where training data is stored
    images_root: Path,              # path where images will be stored (output)
    csv_root: Path,                 # path where csv will be stored (output)
    dataset: str = "acdc",          # dataset identifier
    modality: str = "Cardiac MRI",  # modality identifier
) -> list[dict[str, str]]:
    """
    End-to-end routine for CSV generation.

    TODO: Change method description and add more detailed description of the pipeline.
    TODO: Implement STRATEGY PLAN para los distintos datasets
    """
    rows = build_rows(
        data_path=data_path,
        images_root=images_root,
        modality=modality,
    )
    validate_minim_csv(rows, images_root)
    write_minim_csv(rows, csv_root)
    return rows

if __name__ == "__main__":
    data_path = Path(data_path)
    images_root = Path("output/images")
    csv_root = Path("output/csv")

    rows = run__csv_pipeline(
        data_path, 
        images_root,
        csv_root,
    )

    print(f"Processed {len(rows)} patients.")
