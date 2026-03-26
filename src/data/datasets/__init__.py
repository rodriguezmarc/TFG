"""
Original classes:
- RV: right ventricle, 1
- MYO: myocardium, 2
- LV: left ventricle, 3
"""

# --- labels: important to unify them with the rest of the datasets ---
RV_LABEL = 1
MYO_LABEL = 2
LV_LABEL = 3
LABEL_TO_NAME = {
    RV_LABEL: "RV",
    MYO_LABEL: "MYO",
    LV_LABEL: "LV",
}

def compute_bmi(metadata):
    """
    Compute the BMI from the metadata.

    Args:
        metadata: Metadata dictionary.

    Returns:
        BMI as a string.
    """
    bmi = float(metadata["Weight"]) / (float(metadata["Height"]) / 100.0)

    if bmi < 18.5: bmi = "underweight"
    elif bmi < 25: bmi = "normal"
    elif bmi < 30: bmi = "overweight"
    else: bmi = "obese"

    return bmi

def compute_pathology(metadata):
    """
    Compute the pathology from the metadata.

    Args:
        metadata: Metadata dictionary.

    Returns:
        Pathology as a string.
    """
    group = metadata["Group"]

    if group == "NOR": group = "normal"
    elif group == "MINF": group = "myocardial infarction"
    elif group == "DCM": group = "dilated cardiomyopathy"
    elif group == "HCM": group = "hypertrophic cardiomyopathy"  
    elif group == "RV": group = "abnormal right ventricle"

    return group

