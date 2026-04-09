"""
Shared dataset input and output path configuration.
"""

from pathlib import Path

DATASET_PATHS = {
    "acdc": Path("ACDC"),
    "ukbb": Path("UKBB"),
}

OUTPUT_PATHS = {
    "images": Path("outputs/data/images"),
    "csv": Path("outputs/data/csv"),
    "internal": Path("outputs/data/internal"),
}
