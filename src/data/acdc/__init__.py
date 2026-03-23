"""
ACDC MR image dataset metadata.
ACDC has sapcing 10mm on z-axis
"""

from pathlib import Path
import SimpleITK as sitk

from data import RV_LABEL, MYO_LABEL, LV_LABEL

ACDC_LABEL_MAP = {3: LV_LABEL, 2: MYO_LABEL, 1: RV_LABEL}
ACDC_SPACING = (1.0, 1.0, 10.0)

def load_metadata(config_path: Path) -> dict[str, str | int | float]:
    """
    Load the config file .cfg.
    Example of config file:
        ED: 1
        ES: 12
        Group: DCM
        Height: 184.0
        NbFrame: 30
        Weight: 95.0

    Args: 
        config_path: Path to the config file.

    Returns:
        A dictionary containing the config parameters.
    """
    with Path.open(config_path, encoding="utf-8") as file:
        lines = file.read().splitlines()

    d: dict[str, str | int | float] = {}  # will contain .cfg info
    for l in lines:
        k, v = l.split(": ")  # each line is divided into key and value
        d[k] = v

    # bmi calculus
    bmi = float(d["Weight"]) / (float(d["Height"]) / 100.0)
    if bmi < 18.5: bmi = "underweight"
    elif bmi < 25: bmi = "normal"
    elif bmi < 30: bmi = "overweight"
    else: bmi = "obese"

    # dict with all the config parameters, including bmi
    return {
        "pid": config_path.parent.name,  # patient id is the name of the parent folder
        "pathology": d["Group"],  
        "height": float(d["Height"]) / 100.0,  # to meters
        "weight": float(d["Weight"]),  # in kg
        "n_frames": int(d["NbFrame"]),
        "ed_frame": int(d["ED"]),
        "es_frame": int(d["ES"]),
    }

def load_data(
    config_path: Path
) -> tuple[
    sitk.Image,  # image_4d
    sitk.Image,  # ed_image
    sitk.Image,  # ed_label
    sitk.Image,  # es_image
    sitk.Image,  # es_label
    dict[str, str | int | float],  # metadata
]:
    """
    Load the images and masks for ED and ES frames, as well as the complete 4D image.
    
    Args:
        config_path: Path to the config file.

    Returns:
        A dictionary containing the loaded images and masks.
    """
    metadata = load_metadata(config_path)
    pid = str(metadata["pid"])  # patient id
    ed = int(metadata["ed_frame"])  # end-diastole frame, starting at
    es = int(metadata["es_frame"])  # end-systole frame

    base = config_path.parent  # base path is the parent folder of the config file

    PATHS = {  # paths to the images and labels
        "image_path": base / f"{pid}_4d.nii.gz",  # complete images
        "ed_image_path": base / f"{pid}_frame{ed:02d}.nii.gz",  # end-diastole image
        "ed_mask_path": base / f"{pid}_frame{ed:02d}_gt.nii.gz",  # end-diastole label
        "es_image_path": base / f"{pid}_frame{es:02d}.nii.gz",  # end-systole image
        "es_mask_path": base / f"{pid}_frame{es:02d}_gt.nii.gz",  # end-systole label
    }

    # load the images and labels using SimpleITK
    image_4d = sitk.ReadImage(str(PATHS["image_path"]))
    ed_image = sitk.ReadImage(str(PATHS["ed_image_path"]))
    ed_mask = sitk.ReadImage(str(PATHS["ed_mask_path"]), outputPixelType=sitk.sitkUInt8)
    es_image = sitk.ReadImage(str(PATHS["es_image_path"]))
    es_mask = sitk.ReadImage(str(PATHS["es_mask_path"]), outputPixelType=sitk.sitkUInt8)
    
    return image_4d, ed_image, ed_mask, es_image, es_mask, metadata