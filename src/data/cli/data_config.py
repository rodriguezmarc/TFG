"""
Definition:
Brief map of dataset input roots and generated output roots.
---
Results:
Provides static path configuration dictionaries for the pipeline.
"""

from pathlib import Path

DATASET_PATHS = {
    "acdc": Path("ACDC"),
    "ukbb": Path("UKBB"),
}

OUTPUT_PATHS =  {
    "images": Path("output/images"),
    "csv": Path("output/csv"),
}
