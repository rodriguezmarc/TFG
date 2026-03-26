"""
Backward-compatible exports for CSV generation.
"""

from pathlib import Path
from typing import Callable

import SimpleITK as sitk

from datasets.acdc.pipeline import build_acdc_rows as build_acdc_minim_rows
from datasets.acdc.pipeline import default_acdc_prompt as _default_acdc_prompt_from_metadata
from export.minim_csv import validate_minim_csv, write_minim_csv


def build_acdc_minim_csv(
    config_paths: list[Path],
    images_root: Path,
    output_csv_path: Path,
    image_subdir: str = "acdc",
    modality: str = "Cardiac MRI",
    preprocess_fn: Callable[[Path], tuple[sitk.Image, sitk.Image, float]] | None = None,
    metadata_loader_fn: Callable[[Path], dict] | None = None,
    prompt_fn: Callable[[dict, float], str] = _default_acdc_prompt_from_metadata,
) -> list[dict[str, str]]:
    """
    Full routine: preprocess, prompt generation, CSV write, validation.
    """
    kwargs: dict = {
        "config_paths": config_paths,
        "images_root": images_root,
        "image_subdir": image_subdir,
        "modality": modality,
        "prompt_fn": prompt_fn,
    }
    if preprocess_fn is not None:
        kwargs["preprocess_fn"] = preprocess_fn
    if metadata_loader_fn is not None:
        kwargs["metadata_loader_fn"] = metadata_loader_fn

    rows = build_acdc_minim_rows(**kwargs)
    validate_minim_csv(rows, images_root)
    write_minim_csv(rows, output_csv_path)
    return rows


__all__ = [
    "build_acdc_minim_rows",
    "build_acdc_minim_csv",
    "validate_minim_csv",
    "write_minim_csv",
]
