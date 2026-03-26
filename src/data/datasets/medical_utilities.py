import numpy as np
import SimpleITK as sitk

from data import LV_LABEL


def compute_frame_volume(mask_3d: sitk.Image, label: int = LV_LABEL) -> float:
    """
    Compute the volume of a labelled region in a single 3D frame.

    Parameters
    ----------
    mask_3d : sitk.Image
        3D mask image where voxels belonging to the region of interest have
        value ``label``. Spacing is assumed to be in millimetres.
    label : int, optional
        Label value identifying the region of interest. Defaults to ``LV_LABEL``.

    Returns
    -------
    float
        Volume of the region in millilitres (mL).
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
    Compute ejection fraction from end-diastolic (ED) and end-systolic (ES) masks.

    Parameters
    ----------
    ed_mask : sitk.Image
        3D mask image corresponding to the end-diastolic frame.
    es_mask : sitk.Image
        3D mask image corresponding to the end-systolic frame.
    label : int, optional
        Label value identifying the region of interest. Defaults to ``LV_LABEL``.

    Returns
    -------
    float
        Ejection fraction as a value in the range \[0, 1\].

    Raises
    ------
    ValueError
        If the end-diastolic volume is zero and EF cannot be computed.
    """
    edv = compute_frame_volume(ed_mask, label=label)
    esv = compute_frame_volume(es_mask, label=label)

    if edv == 0:
        raise ValueError("EDV is zero, cannot compute EF.")

    ef = (edv - esv) / edv
    return ef