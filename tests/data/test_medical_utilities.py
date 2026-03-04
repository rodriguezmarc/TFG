import numpy as np
import SimpleITK as sitk
import pytest

from data import LV_LABEL
from data.medical_utilities import (
    compute_frame_volume,
    compute_ef,
)

TEST_IMAGE_SIZE = (5,5,5) 
TEST_IMAGE_SPACING = (2.0, 2.0, 2.0)

# 1. compute_frame BASIC VOLUME 
def test_compute_frame_volume_basic():
    """
    Mask with exactly 10 LV voxels.
    Spacing = (2,2,2) mm -> voxel volume = 8 mm3.
    Expected LV volume: 10 * 8 = 80 mm3 = 0.08 mL.
    """
    arr = np.zeros(TEST_IMAGE_SIZE, dtype=np.uint8)  # simulates the image
    arr.flat[:10] = LV_LABEL  # exactly the 10 LV voxels
    mask = sitk.GetImageFromArray(arr)
    mask.SetSpacing(TEST_IMAGE_SPACING)  # change all the mask

    volume_ml = compute_frame_volume(mask, LV_LABEL)
    
    assert np.isclose(volume_ml, 0.08), (f"Volume should be 0.08 ml, but got {volume_ml} ml.")

# 2. compute_frame NO LABEL VOLUME 
def test_compute_frame_volume_no_label():
    """
    If no mask voxels exist, volume computed should be 0 ml.
    """
    arr = np.zeros(TEST_IMAGE_SIZE, dtype=np.uint8)
    mask = sitk.GetImageFromArray(arr)
    mask.SetSpacing(TEST_IMAGE_SPACING) 

    volume_ml = compute_frame_volume(mask, LV_LABEL)

    assert volume_ml == 0.0, (f"Volume should be 0.0 ml when no voxels are present, but got {volume_ml} ml.")

# 3. compute_frame ASINTROPIC SPACING
def test_compute_frame_volume_asintropic_spacing():
    """
    Test correct handling of asintropic spacing.
    Spacig = (1,2,3) -> voxel volume = 6 mm3.
    Mask with exactly 5 LV voxels.
    Expected LV volume: 5 * 6 = 30 mm3 = 0.03 mL.
    """
    arr = np.zeros(TEST_IMAGE_SIZE, dtype=np.uint8)
    arr.flat[:5] = LV_LABEL
    mask = sitk.GetImageFromArray(arr)
    mask.SetSpacing((1.0, 2.0, 3.0))

    volume_ml = compute_frame_volume(mask, LV_LABEL)

    assert np.isclose(volume_ml, 0.03), (f"Volume should account for asintropic spacing corretcly. Expected 0.03 ml, but got {volume_ml} ml.")

# 4. compute_ef BASIC CASE
def test_compute_ef_case_basic():
    """
    EDV = 100 mL.
    ESV = 40 mL.
    Expected EF to be 0.6.
    """
    ed_arr = np.zeros(TEST_IMAGE_SIZE, dtype=np.uint8)
    ed_arr.flat[:100] = LV_LABEL  # 100 voxels LV
    ed_mask = sitk.GetImageFromArray(ed_arr)
    ed_mask.SetSpacing((10.0, 10.0, 10.0))  #  1000 mm3 per voxel = 1 mL

    es_arr = np.zeros(TEST_IMAGE_SIZE, dtype=np.uint8)
    es_arr.flat[:40] = LV_LABEL  # 40 voxels LV
    es_mask = sitk.GetImageFromArray(es_arr)
    es_mask.SetSpacing((10.0, 10.0, 10.0))  # 1 mL per voxel

    ef = compute_ef(ed_mask, es_mask)

    assert np.isclose(ef, 0.6), (f"Ejection fraction expected to be 0.6, but got {ef}")


# 5. compute_ef EDV = 0
def test_compute_ef_zero_edv_raises():
    """
    If EDV is zero, EF cannot be computed and should raise ValueError.
    """
    arr = np.zeros(TEST_IMAGE_SIZE, dtype=np.uint8)
    ed_mask = sitk.GetImageFromArray(arr)  # zero LV voxels
    es_mask = sitk.GetImageFromArray(arr)
    ed_mask.SetSpacing(TEST_IMAGE_SPACING)
    es_mask.SetSpacing(TEST_IMAGE_SPACING)

    with pytest.raises(ValueError, match="EDV is zero, cannot compute EF."): compute_ef(ed_mask, es_mask)
    

    