"""
Top-level orchestration for dataset -> split -> preprocess -> prompt -> CSV.
"""

from __future__ import annotations

from pathlib import Path

from datasets.acdc.pipeline import build_acdc_rows
from export.minim_csv import validate_minim_csv, write_minim_csv


def run_acdc_csv_pipeline(
    config_paths: list[Path],
    images_root: Path,
    output_csv_path: Path,
    image_subdir: str = "acdc",
    modality: str = "Cardiac MRI",
) -> list[dict[str, str]]:
    """
    End-to-end routine for ACDC CSV generation.
    """
    rows = build_acdc_rows(
        config_paths=config_paths,
        images_root=images_root,
        image_subdir=image_subdir,
        modality=modality,
    )
    validate_minim_csv(rows, images_root)
    write_minim_csv(rows, output_csv_path)
    return rows
