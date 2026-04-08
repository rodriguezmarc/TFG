"""
Definition:
Brief map of the ACDC preprocessing routine from raw images to export-ready slices.
---
Results:
Provides the dataset-specific preprocessing entrypoint.
"""

from __future__ import annotations

from pathlib import Path
import SimpleITK as sitk

from data.datasets.acdc import ACDC_LABEL_MAP, load_data
from data.utilities.image_utilities import (
    extract_slice,
    select_mask_slice,
    resample_slice,
    resize_slice,
    get_lv_center,
    get_mask_crop_size,
    crop_around_center,
    normalize,
)
from data.utilities.medical_utilities import compute_ef


def preprocess(config_path: Path) -> tuple[sitk.Image, sitk.Image, float, dict[str, str | int | float]]:
    """
    ########################################
    Definition:
    Run the full ACDC preprocessing pipeline for one patient config.
    ---
    Params:
    config_path: Path to the patient configuration file.
    ---
    Results:
    Returns the processed ES slice, processed mask slice, EF value, and metadata.
    ---
    Other Information:
    The pipeline relabels masks, selects the ES slice with the target anatomy, crops around the ROI, resizes the outputs, and normalizes image intensities.
    ########################################
    """
    # --- load data ---
    (
        image_4d,
        ed_image,
        ed_mask,
        es_image,
        es_mask,
        metadata,
    ) = load_data(config_path)  # load the images and metadata

    # --- unify labels to standarize masks ---
    ed_mask = sitk.ChangeLabel(ed_mask, ACDC_LABEL_MAP)
    es_mask = sitk.ChangeLabel(es_mask, ACDC_LABEL_MAP)

    # --- frame selection & slice selection ---
    slice_idx = select_mask_slice(es_mask)
    es_slice, mask_slice = extract_slice(es_image, es_mask, slice_idx)
                              
    # --- resampling ---
    es_slice = resample_slice(es_slice)
    mask_slice = resample_slice(mask_slice, is_label=True)

    # --- croping ---
    center = get_lv_center(mask_slice)
    crop_size = get_mask_crop_size(mask_slice)
    es_slice = crop_around_center(es_slice, center, size=crop_size)
    mask_slice = crop_around_center(mask_slice, center, size=crop_size)

    # --- resize to a fixed output canvas after ROI-focused cropping ---
    es_slice = resize_slice(es_slice)
    mask_slice = resize_slice(mask_slice, is_label=True)

    # --- normalization ---
    es_slice = normalize(es_slice)

    # --- EF calculus ---
    ef = compute_ef(ed_mask, es_mask)

    return es_slice, mask_slice, ef, metadata
