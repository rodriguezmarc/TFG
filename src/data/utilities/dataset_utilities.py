"""
########################################
Definition:
Brief map of dataset-agnostic metadata normalization helpers.
---
Params:
None.
---
Results:
Provides BMI and pathology normalization utilities used by prompt generation.
########################################
"""

from __future__ import annotations


def _get_required_float(metadata: dict, *keys: str) -> float:
    """
    ########################################
    Definition:
    Extract the first available numeric metadata value from a key list.
    ---
    Params:
    metadata: Source metadata dictionary.
    keys: Candidate keys to probe in order.
    ---
    Results:
    Returns the selected value converted to float.
    ---
    Other Information:
    Raises KeyError when none of the requested keys are present.
    ########################################
    """
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return float(metadata[key])
    raise KeyError(f"Missing required metadata. Expected one of: {keys}")


def compute_bmi_value(metadata: dict) -> float:
    """
    ########################################
    Definition:
    Compute body mass index from metadata.
    ---
    Params:
    metadata: Metadata dictionary containing weight and height values.
    ---
    Results:
    Returns BMI as a float.
    ---
    Other Information:
    Height is expected in centimetres and must be greater than zero.
    ########################################
    """
    weight_kg = _get_required_float(metadata, "weight", "Weight")
    height_cm = _get_required_float(metadata, "height", "Height")
    height_m = height_cm / 100.0

    if height_m <= 0:
        raise ValueError("Height must be greater than zero to compute BMI.")

    return weight_kg / (height_m ** 2)


def compute_bmi_group(metadata: dict) -> str:
    """
    ########################################
    Definition:
    Convert BMI into the simplified textual group used in prompts.
    ---
    Params:
    metadata: Metadata dictionary containing the measurements needed for BMI.
    ---
    Results:
    Returns one BMI group label.
    ########################################
    """
    bmi = compute_bmi_value(metadata)
    if bmi < 18.5: 
        return "underweight"
    if bmi < 25.0:
        return "normal BMI"
    if bmi < 30.0:
        return "overweight"
    return "obese"


def compute_pathology(metadata: dict) -> str:
    """
    ########################################
    Definition:
    Normalize pathology codes into stable prompt-friendly text.
    ---
    Params:
    metadata: Metadata dictionary containing a pathology field or original dataset group code.
    ---
    Results:
    Returns the normalized pathology description string.
    ---
    Other Information:
    Unknown values fall back to lowercase text or a generic placeholder.
    ########################################
    """
    pathology = str(metadata.get("pathology", metadata.get("Group", ""))).strip().upper()
    mapping = {
        "NOR": "healthy",
        "MINF": "myocardial infarction",
        "DCM": "dilated cardiomyopathy",
        "HCM": "hypertrophic cardiomyopathy",
        "RV": "abnormal right ventricle",
    }
    return mapping.get(pathology, pathology.lower() if pathology else "unknown pathology")
