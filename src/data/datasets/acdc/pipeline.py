"""
Dataset-specific ACDC pipeline that standardizes preprocessed outputs.
"""


from pathlib import Path
from typing import Callable

import numpy as np
import SimpleITK as sitk

from .load_data import load_metadata
from .preprocess import preprocess
from .prompts.cardiac_prompt import prompt_generation


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

def find_config_files_acdc(data_path: Path) -> list[Path]:
    """
    TODO: move to the ACDC module.
    """
    config_paths = []
    for patient_dir in data_path.glob("patient*"):
        if not patient_dir.is_dir(): continue
        cfg_path = list(patient_dir.glob("*.cfg"))
        if not cfg_path: continue
        config_paths.append(cfg_path[0])

    return config_paths
    

def default_acdc_prompt(metadata: dict, ef: float) -> str:
    metadata_for_prompt = {
        "Weight": metadata["weight"],
        "Height": metadata["height"] * 100.0,  # m -> cm
        "Group": metadata["pathology"],
        "ef": ef,
    }
    return prompt_generation(metadata_for_prompt)


def build_rows(
    data_path: Path,
    images_root: Path,
    modality: str = "Cardiac MRI",
    preprocess_fn: Callable[[Path], tuple[sitk.Image, sitk.Image, float]] = preprocess,
    prompt_fn: Callable[[dict, float], str] = default_acdc_prompt,
) -> list[dict[str, str]]:
    """
    Run ACDC preprocess and build standardized rows for CSV export.

    TODO: Change method description to add more detail.
    TODO: Implement STRATEGY PLAN para los distintos datasets
    """
    rows: list[dict[str, str]] = []

    config_paths = find_config_files_acdc(data_path)
    for config_path in config_paths:
        es_slice, _, ef, metadata = preprocess_fn(config_path)
        img_filename = f"{metadata['pid']}_es_mid.png"
        save_processed_image(es_slice, Path(images_root) / img_filename)
        
        rows.append(
            {
                "path": img_filename,
                "text": prompt_fn(metadata, ef),
                "modality": modality,
            }
        )

    return rows