"""
########################################
Definition:
Brief map of prompt helpers for turning cardiac metadata into MINIM text.
---
Params:
None.
---
Results:
Provides EF normalization, EF categorization, and prompt generation helpers.
########################################
"""

from __future__ import annotations

from data.utilities.dataset_utilities import compute_bmi_group, compute_pathology


def to_ef_percentage(ef: float) -> float:
    """
    ########################################
    Definition:
    Normalize an EF value into percentage form.
    ---
    Params:
    ef: Ejection fraction expressed either as a ratio or percentage.
    ---
    Results:
    Returns EF as a percentage in the `[0, 100]` style range.
    ---
    Other Information:
    Values less than or equal to `1.0` are interpreted as ratios.
    ########################################
    """
    return ef * 100.0 if ef <= 1.0 else ef


def classify_ef(ef: float) -> str:
    """
    ########################################
    Definition:
    Map an EF value to the prompt category used by the project.
    ---
    Params:
    ef: Ejection fraction expressed as ratio or percentage.
    ---
    Results:
    Returns one of the supported EF class labels.
    ########################################
    """
    ef_pct = to_ef_percentage(float(ef))

    if ef_pct <= 40.0:
        return "reduced EF"
    if ef_pct <= 49.0:
        return "mildly reduced EF"
    return "normal EF"


def build_cardiac_prompt(metadata: dict, ef: float) -> str:
    """
    ########################################
    Definition:
    Build the normalized cardiac MRI prompt from metadata and EF.
    ---
    Params:
    metadata: Patient or study metadata with pathology and body measurements.
    ef: Ejection fraction value to classify.
    ---
    Results:
    Returns the final prompt string used in CSV export.
    ########################################
    """
    bmi_group = compute_bmi_group(metadata)
    pathology = compute_pathology(metadata)
    ef_category = classify_ef(float(ef))

    return (
        "Cardiac MRI, short-axis view, "
        f"{bmi_group}, {pathology} condition, {ef_category}."
    )
