"""
Preprocessing for ACDC dataset.
"""

from pathlib import Path
import SimpleITK as sitk

from data.acdc import load_data, ACDC_LABEL_MAP
from data.image_utilities import (
    extract_slice,
    resample_slice,
    get_lv_center,
    crop_around_center,
    normalize,
)
from data.medical_utilities import compute_ef


def preprocess(config_path: Path) -> tuple[sitk.Image, sitk.Image, float, dict[str, str | int | float]]:
    """
    Full preprocessing pipeline for ACDC dataset.
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
    z_size = es_image.GetSize()[2]
    es_slice, mask_slice = extract_slice(es_image, es_mask, z_size // 2)
                              
    # --- resampling ---
    es_slice = resample_slice(es_slice)
    mask_slice = resample_slice(mask_slice)

    # --- croping ---
    center = get_lv_center(mask_slice)
    es_slice = crop_around_center(es_slice, center)
    mask_slice = crop_around_center(mask_slice, center)

    # --- normalization ---
    es_slice = normalize(es_slice)

    # --- EF calculus ---
    ef = compute_ef(ed_mask, es_mask)

    return es_slice, mask_slice, ef, metadata

