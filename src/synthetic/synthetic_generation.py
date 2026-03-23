"""
Synthetic Image Generation for model training.
"""

from pathlib import Path

from data import compute_bmi, compute_pathology
from data.acdc import load_metadata
from data.acdc.preprocess import preprocess


def _to_ef_percentage(ef: float) -> float:
    """
    Convert EF to percentage.

    Accepts EF as ratio [0,1] or as percentage [0,100].
    """
    return ef * 100.0 if ef <= 1.0 else ef


def classify_ef(ef: float) -> str:
    """
    Map EF value to clinically-used category.
    """
    ef_pct = _to_ef_percentage(float(ef))

    if ef_pct <= 40.0:
        return "reduced EF"
    if ef_pct <= 49.0:
        return "mildly reduced EF"
    return "normal EF"


def prompt_generation(metadata: dict, ef: float | None = None) -> str:
    """
    Generate a prompt for the synthetic image generation.

    Args:
        metadata: Metadata dictionary.
        ef: Ejection fraction. If None, reads from metadata["ef"].

    Returns:
        Prompt as a string.
    """
    bmi = compute_bmi(metadata)
    group = compute_pathology(metadata)
    ef_value = metadata.get("ef") if ef is None else ef

    if ef_value is None:
        raise ValueError("EF is required to generate the prompt.")

    ef_category = classify_ef(float(ef_value))

    prompt = (
        "Cardiac MRI, short-axis view, "
        f"{bmi} BMI, {group} condition, {ef_category}."
    )
    return prompt


def prompt_generation_from_acdc(config_path: Path) -> str:
    """
    Run ACDC preprocessing and generate a prompt with EF.
    """
    _, _, ef = preprocess(config_path)
    metadata = load_metadata(config_path)

    # `compute_bmi` and `compute_pathology` currently expect ACDC raw keys.
    metadata_for_prompt = {
        "Weight": metadata["weight"],
        "Height": metadata["height"] * 100.0,  # convert m -> cm
        "Group": metadata["pathology"],
        "ef": ef,
    }
    return prompt_generation(metadata_for_prompt)