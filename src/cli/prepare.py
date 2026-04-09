"""
Definition:
Brief map of the top-level preprocessing command.
---
Results:
Parses CLI arguments and executes dataset preprocessing plus CSV export.
"""

from __future__ import annotations

import argparse

from data.config import DATASET_PATHS, OUTPUT_PATHS
from data.run_pipeline import run_csv_pipeline


def build_parser() -> argparse.ArgumentParser:
    """
    ########################################
    Definition:
    Build the CLI parser for the preprocessing command.
    ---
    Params:
    None.
    ---
    Results:
    Returns the configured argument parser.
    ########################################
    """
    parser = argparse.ArgumentParser(description="Run dataset preprocessing and export MINIM manifests")
    parser.add_argument("--dataset", "-d", choices=["acdc", "ukbb"], action="append", required=True)
    parser.add_argument("--modality", type=str, default="Cardiac MRI")
    return parser


def main() -> None:
    """
    ########################################
    Definition:
    Execute the preprocessing command from parsed CLI arguments.
    ---
    Params:
    None.
    ---
    Results:
    Runs preprocessing for each requested dataset and prints row counts.
    ########################################
    """
    args = build_parser().parse_args()
    for dataset in args.dataset:
        rows = run_csv_pipeline(
            data_path=DATASET_PATHS[dataset],
            images_root=OUTPUT_PATHS["images"],
            csv_root=OUTPUT_PATHS["csv"],
            internal_root=OUTPUT_PATHS["internal"],
            dataset=dataset,
            modality=args.modality,
        )
        print(f"{dataset}: {len(rows)} rows")


if __name__ == "__main__":
    main()
