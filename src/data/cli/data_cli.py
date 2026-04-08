"""
Definition:
Brief map of CLI argument parsing for the dataset pipeline entrypoint.
---
Results:
Builds and returns parsed command-line arguments.
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
    Returns an argparse namespace with the selected dataset value.
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

    return parser.parse_args()
