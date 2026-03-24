"""
Dataset-specific ACDC pipeline that standardizes preprocessed outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import SimpleITK as sitk

from data.acdc import load_metadata
from data.acdc.preprocess import preprocess
from prompts.cardiac_prompt import prompt_generation


def _image_to_uint8(image: sitk.Image) -> sitk.Image:
    image_np = sitk.GetArrayFromImage(image).astype(np.float32)
    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = (image_np * 255.0).round().astype(np.uint8)
    out = sitk.GetImageFromArray(image_np)
    out.CopyInformation(image)
    return out


def save_processed_image(image: sitk.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(_image_to_uint8(image), str(output_path))


def default_acdc_prompt(metadata: dict, ef: float) -> str:
    metadata_for_prompt = {
        "Weight": metadata["weight"],
        "Height": metadata["height"] * 100.0,  # m -> cm
        "Group": metadata["pathology"],
        "ef": ef,
    }
    return prompt_generation(metadata_for_prompt)


def build_acdc_rows(
    config_paths: list[Path],
    images_root: Path,
    image_subdir: str = "acdc",
    modality: str = "Cardiac MRI",
    preprocess_fn: Callable[[Path], tuple[sitk.Image, sitk.Image, float]] = preprocess,
    metadata_loader_fn: Callable[[Path], dict] = load_metadata,
    prompt_fn: Callable[[dict, float], str] = default_acdc_prompt,
) -> list[dict[str, str]]:
    """
    Run ACDC preprocess and build standardized rows for CSV export.
    """
    rows: list[dict[str, str]] = []
    for config_path in config_paths:
        es_slice, _, ef = preprocess_fn(config_path)
        metadata = metadata_loader_fn(config_path)

        patient_id = config_path.parent.name
        rel_path = Path(image_subdir) / f"{patient_id}_es_mid.png"
        abs_image_path = images_root / rel_path

        save_processed_image(es_slice, abs_image_path)
        rows.append(
            {
                "path": rel_path.as_posix(),
                "text": prompt_fn(metadata, ef),
                "modality": modality,
            }
        )
    return rows
