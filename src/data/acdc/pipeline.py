"""
########################################
Definition:
Brief map of the ACDC export pipeline that saves images and builds CSV rows.
---
Params:
None.
---
Results:
Provides image-export helpers, case discovery, prompt generation, and row construction.
########################################
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import SimpleITK as sitk

from data.acdc.preprocess import preprocess
from data.prompts.cardiac_prompt import build_cardiac_prompt


def _image_to_uint8(image: sitk.Image) -> sitk.Image:
    """
    ########################################
    Definition:
    Convert a normalized scalar image into an 8-bit SimpleITK image.
    ---
    Params:
    image: Input image expected to contain values in the `[0, 1]` range.
    ---
    Results:
    Returns a uint8 image suitable for PNG export.
    ########################################
    """
    image_np = sitk.GetArrayFromImage(image).astype(np.float32)
    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = (image_np * 255.0).round().astype(np.uint8)
    out = sitk.GetImageFromArray(image_np)
    out.CopyInformation(image)
    return out


def _mask_overlay_to_uint8(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    """
    ########################################
    Definition:
    Blend a binary or labeled mask on top of the grayscale image for inspection.
    ---
    Params:
    image: Base image to visualize.
    mask: Segmentation mask aligned with the image.
    ---
    Results:
    Returns an RGB uint8 overlay image.
    ---
    Other Information:
    Masked pixels are tinted red with fixed alpha blending.
    ########################################
    """
    image_np = sitk.GetArrayFromImage(_image_to_uint8(image))
    mask_np = sitk.GetArrayFromImage(mask)

    rgb_np = np.stack([image_np, image_np, image_np], axis=-1)
    overlay_color = np.array([255, 0, 0], dtype=np.float32)
    alpha = 0.35

    mask_region = mask_np > 0
    rgb_np = rgb_np.astype(np.float32)
    rgb_np[mask_region] = (
        (1.0 - alpha) * rgb_np[mask_region] + alpha * overlay_color
    )

    out = sitk.GetImageFromArray(np.clip(rgb_np, 0, 255).astype(np.uint8), isVector=True)
    out.CopyInformation(image)
    return out


def save_processed_image(image: sitk.Image, output_path: Path) -> None:
    """
    ########################################
    Definition:
    Save a processed image slice to disk as an 8-bit image file.
    ---
    Params:
    image: Processed grayscale image to export.
    output_path: Destination file path.
    ---
    Results:
    Creates parent directories and writes the image.
    ########################################
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(_image_to_uint8(image), str(output_path))


def save_mask_overlay(image: sitk.Image, mask: sitk.Image, output_path: Path) -> None:
    """
    ########################################
    Definition:
    Save the inspection overlay that shows the mask on top of the processed image.
    ---
    Params:
    image: Processed base image.
    mask: Processed aligned mask.
    output_path: Destination file path for the overlay preview.
    ---
    Results:
    Creates parent directories and writes the RGB overlay image.
    ########################################
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(_mask_overlay_to_uint8(image, mask), str(output_path))


def discover_cases(data_path: Path) -> list[Path]:
    """
    ########################################
    Definition:
    Discover patient configuration files inside the ACDC dataset root.
    ---
    Params:
    data_path: Root directory that contains patient subdirectories.
    ---
    Results:
    Returns a sorted list of config file paths.
    ########################################
    """
    config_paths: list[Path] = []
    for patient_dir in data_path.glob("patient*"):
        if not patient_dir.is_dir():
            continue
        cfg_path = list(patient_dir.glob("*.cfg"))
        if not cfg_path:
            continue
        config_paths.append(cfg_path[0])

    return sorted(config_paths)


def default_acdc_prompt(metadata: dict, ef: float) -> str:
    """
    ########################################
    Definition:
    Build the default prompt string for an ACDC case.
    ---
    Params:
    metadata: Canonical metadata dictionary for the case.
    ef: Ejection fraction value associated with the case.
    ---
    Results:
    Returns the prompt string used in exported rows.
    ########################################
    """
    return build_cardiac_prompt(metadata, ef)


def build_rows(
    data_path: Path,
    images_root: Path,
    modality: str = "Cardiac MRI",
    preprocess_fn: Callable[[Path], tuple[sitk.Image, sitk.Image, float, dict[str, str | int | float]]] = preprocess,
    prompt_fn: Callable[[dict, float], str] = default_acdc_prompt,
) -> list[dict[str, str]]:
    """
    ########################################
    Definition:
    Run preprocessing across all discovered ACDC cases and build export rows.
    ---
    Params:
    data_path: Root directory containing patient folders.
    images_root: Root directory where processed images are saved.
    modality: Modality label to store in each output row.
    preprocess_fn: Preprocessing function used per case.
    prompt_fn: Prompt builder used per case.
    ---
    Results:
    Returns a list of standardized row dictionaries ready for CSV export.
    ---
    Other Information:
    Each case produces both the processed image and a mask overlay preview in `acdc/maksed`.
    ########################################
    """
    rows: list[dict[str, str]] = []

    config_paths = discover_cases(data_path)
    for config_path in config_paths:
        es_slice, mask_slice, ef, metadata = preprocess_fn(config_path)
        img_filename = f"acdc/{metadata['pid']}_es_mid.png"
        save_processed_image(es_slice, Path(images_root) / img_filename)
        save_mask_overlay(
            es_slice,
            mask_slice,
            Path(images_root) / "acdc" / "maksed" / f"{metadata['pid']}_es_mid.png",
        )
        
        rows.append(
            {
                "path": img_filename,
                "text": prompt_fn(metadata, ef),
                "modality": modality,
                "patient_id": str(metadata["pid"]),
            }
        )

    return rows
