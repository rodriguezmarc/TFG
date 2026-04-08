"""
Definition:
Brief map of medical measurement utilities derived from segmentation masks.
---
Results:
Provides frame-volume and ejection-fraction calculations.
"""

import numpy as np
import SimpleITK as sitk

from data import LV_LABEL


def compute_frame_volume(mask_3d: sitk.Image, label: int = LV_LABEL) -> float:
    """
    ########################################
    Definition:
    Compute the volume of one labeled structure within a 3D mask.
    ---
    Params:
    mask_3d: Three-dimensional segmentation mask.
    label: Label value to measure.
    ---
    Results:
    Returns the structure volume in millilitres.
    ---
    Other Information:
    The computation relies on the voxel spacing stored in the image metadata.
    ########################################
    """
    mask_np = sitk.GetArrayFromImage(mask_3d)  # obtain voxels
    spacing = mask_3d.GetSpacing()
    
    voxel_volume = spacing[0] * spacing[1] * spacing[2]  # compute voxel volume in mm³
    lv_voxels = np.sum(mask_np == label)  # count the voxels that belong to the label

    volume_mm3 = lv_voxels * voxel_volume  # compute the total volume in mm³
    volume_ml = volume_mm3 / 1000.0  # convert to mL

    return volume_ml

def compute_ef(ed_mask: sitk.Image, es_mask: sitk.Image, label: int = LV_LABEL) -> float:
    """
    ########################################
    Definition:
    Compute ejection fraction from end-diastolic and end-systolic masks.
    ---
    Params:
    ed_mask: Mask for the end-diastolic frame.
    es_mask: Mask for the end-systolic frame.
    label: Label value that defines the target cardiac structure.
    ---
    Results:
    Returns ejection fraction as a ratio.
    ---
    Other Information:
    Raises ValueError when the end-diastolic volume is zero.
    ########################################
    """
    edv = compute_frame_volume(ed_mask, label=label)
    esv = compute_frame_volume(es_mask, label=label)

    if edv == 0:
        raise ValueError("EDV is zero, cannot compute EF.")

    ef = (edv - esv) / edv
    return ef
