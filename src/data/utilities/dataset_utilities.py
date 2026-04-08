"""
Definition:
Brief map of dataset-agnostic metadata normalization helpers.
---
Results:
Provides BMI, demographic, and disease normalization utilities used by prompt generation.
"""

from __future__ import annotations


def _get_first_value(metadata: dict, *keys: str) -> object | None:
    """
    ########################################
    Definition:
    Return the first non-empty metadata value available from a key list.
    ---
    Params:
    metadata: Source metadata dictionary.
    keys: Candidate keys to probe in order.
    ---
    Results:
    Returns the selected value or `None` when none are available.
    ########################################
    """
    for key in keys:
        if key not in metadata:
            continue
        value = metadata[key]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


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
    value = _get_first_value(metadata, *keys)
    if value is None:
        raise KeyError(f"Missing required metadata. Expected one of: {keys}")
    return float(value)


def _normalize_free_text(value: object) -> str:
    """
    ########################################
    Definition:
    Normalize free-text metadata values into prompt-friendly lowercase text.
    ---
    Params:
    value: Metadata value to normalize.
    ---
    Results:
    Returns a stripped lowercase string with underscores replaced by spaces.
    ########################################
    """
    return str(value).strip().replace("_", " ").lower()


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


def compute_age_group(metadata: dict) -> str | None:
    """
    ########################################
    Definition:
    Convert a patient age into the simplified textual group used in prompts.
    ---
    Params:
    metadata: Metadata dictionary containing an age field.
    ---
    Results:
    Returns one age-group label or `None` when age is unavailable.
    ########################################
    """
    age = _get_first_value(metadata, "age", "Age")
    if age is None:
        return None

    age_years = float(age)
    if age_years < 18:
        return "child"
    if age_years < 40:
        return "young adult"
    if age_years < 65:
        return "middle-aged"
    return "elderly"


def compute_sex_label(metadata: dict) -> str | None:
    """
    ########################################
    Definition:
    Normalize a sex field into stable prompt-friendly text.
    ---
    Params:
    metadata: Metadata dictionary containing a sex field.
    ---
    Results:
    Returns `male`, `female`, or `None` when unavailable.
    ########################################
    """
    sex = _get_first_value(metadata, "sex", "Sex", "gender", "Gender")
    if sex is None:
        return None

    normalized = _normalize_free_text(sex)
    mapping = {
        "m": "male",
        "male": "male",
        "man": "male",
        "f": "female",
        "female": "female",
        "woman": "female",
    }
    return mapping.get(normalized)


def compute_disease_label(metadata: dict) -> str | None:
    """
    ########################################
    Definition:
    Normalize disease codes into stable prompt-friendly text.
    ---
    Params:
    metadata: Metadata dictionary containing a disease or pathology field.
    ---
    Results:
    Returns the normalized disease description string or `None` when unavailable.
    ---
    Other Information:
    Unknown values fall back to lowercase text.
    ########################################
    """
    disease = _get_first_value(
        metadata,
        "disease",
        "Disease",
        "pathology",
        "Pathology",
        "Group",
    )
    if disease is None:
        return None

    pathology = str(disease).strip().upper()
    mapping = {
        "NOR": "healthy",
        "MINF": "myocardial infarction",
        "DCM": "dilated cardiomyopathy",
        "HCM": "hypertrophic cardiomyopathy",
        "RV": "abnormal right ventricle",
    }
    return mapping.get(pathology, _normalize_free_text(disease))


def compute_pathology(metadata: dict) -> str:
    """
    ########################################
    Definition:
    Provide a backward-compatible pathology label with a generic fallback.
    ---
    Params:
    metadata: Metadata dictionary containing pathology information.
    ---
    Results:
    Returns a normalized pathology string.
    ########################################
    """
    return compute_disease_label(metadata) or "unknown pathology"
