import numpy as np
import SimpleITK as sitk

from data import LV_LABEL

def compute_frame_volume(mask_3d: sitk.Image, label=LV_LABEL) -> float:
    """
    Compute the volume from a given frame using its 3D mask (LV, MYO; RV)
    It is important to note that this method computes the volume in a given frame.

    Parameters: 
        - mask_3d: A 3D SimpleITK image that defines a certain region.
        - label: Selection of the region to which compute the volume.
        
    Returns: 
        -  volume_ml: A float number which informs the ml of the region in the given frame.
    """
    mask_np = sitk.GetArrayFromImage(mask_3d)  # obtain voxels
    spacing = mask_3d.GetSpacing()
    
    voxel_volume = spacing[0] * spacing[1] * spacing[2]  # compute all volume mm³
    lv_voxels = np.sum(mask_np == label)  # count the voxels that belong to the label

    volume_mm3 = lv_voxels * voxel_volume  # compute the total volume in mm³
    volume_ml = volume_mm3 / 1000.0  # convert to ml

    return volume_ml

def compute_ef(ed_mask: sitk.Image, es_mask: sitk.Image, label=LV_LABEL) -> float:
    """
    Compute ejection fraction from ED and ES masks.

    Parameters: 
        - ed_mask: A 3D SimpleITK image that corresponds to the end-diastolic frame.
        - es_mask: A 3D SimpleITK image that corresponds to the end-systolic frame.
        - label: Selection of the region to which compute the volume.
        
    Returns:
        - ef: A float number whcih informs the ejection fraction.
    """
    edv = compute_frame_volume(ed_mask, LV_LABEL)
    esv = compute_frame_volume(es_mask, LV_LABEL)

    if edv == 0: raise ValueError("EDV is zero, cannot compute EF.")

    ef = (edv - esv) / edv
    return ef