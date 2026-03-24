"""
Backward-compatible synthetic prompt exports.
"""

from pathlib import Path

from prompts.cardiac_prompt import _to_ef_percentage, classify_ef, prompt_generation
from prompts import cardiac_prompt as _cardiac_prompt

# Compatibility aliases for tests and legacy monkeypatch usage.
preprocess = _cardiac_prompt.preprocess
load_metadata = _cardiac_prompt.load_metadata


def prompt_generation_from_acdc(config_path: Path) -> str:
    """
    Compatibility wrapper that uses module-level patchable dependencies.
    """
    _, _, ef = preprocess(config_path)
    metadata = load_metadata(config_path)
    metadata_for_prompt = {
        "Weight": metadata["weight"],
        "Height": metadata["height"] * 100.0,
        "Group": metadata["pathology"],
        "ef": ef,
    }
    return prompt_generation(metadata_for_prompt)

__all__ = [
    "_to_ef_percentage",
    "classify_ef",
    "prompt_generation",
    "prompt_generation_from_acdc",
    "preprocess",
    "load_metadata",
]