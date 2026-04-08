"""
########################################
Definition:
Brief map of CLI argument parsing for the dataset pipeline entrypoint.
---
Params:
None.
---
Results:
Builds and returns parsed command-line arguments.
########################################
"""

import argparse


def parse_args():
    """
    ########################################
    Definition:
    Parse supported command-line arguments for the data pipeline.
    ---
    Params:
    None.
    ---
    Results:
    Returns an argparse namespace with dataset, split mode, and seed values.
    ---
    Other Information:
    Centralizes the CLI contract for `python -m data.run_pipeline`.
    ########################################
    """
    parser = argparse.ArgumentParser(
        description="Dataset CSV pipeline"
    )

    parser.add_argument(
        "--dataset", "-d",
        choices=["acdc", "ukbb"],
        default="acdc",
        help="Dataset to process (default: acdc)"
    )

    parser.add_argument(
        "--split", "-s",
        choices=["combined", "per-split"],
        default="combined",
        help="CSV export mode: a single manifest or one CSV per split."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for patient-level splits."
    )

    return parser.parse_args()
