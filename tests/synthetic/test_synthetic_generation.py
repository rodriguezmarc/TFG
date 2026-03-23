from pathlib import Path

import pytest

import synthetic.synthetic_generation as synthetic_generation


def test_to_ef_percentage_handles_ratio_and_percentage():
    assert synthetic_generation._to_ef_percentage(0.55) == pytest.approx(55.0)
    assert synthetic_generation._to_ef_percentage(55.0) == 55.0


@pytest.mark.parametrize(
    ("ef", "expected_category"),
    [
        (0.40, "reduced EF"),
        (0.41, "mildly reduced EF"),
        (0.49, "mildly reduced EF"),
        (0.50, "normal EF"),
        (65.0, "normal EF"),
    ],
)
def test_classify_ef_boundaries(ef, expected_category):
    assert synthetic_generation.classify_ef(ef) == expected_category


def test_prompt_generation_uses_metadata_ef():
    metadata = {
        "Weight": 80.0,
        "Height": 180.0,
        "Group": "NOR",
        "ef": 0.35,
    }

    prompt = synthetic_generation.prompt_generation(metadata)

    assert "Cardiac MRI, short-axis view" in prompt
    assert "normal condition" in prompt
    assert "reduced EF" in prompt


def test_prompt_generation_uses_explicit_ef_over_metadata():
    metadata = {
        "Weight": 80.0,
        "Height": 180.0,
        "Group": "NOR",
        "ef": 0.35,
    }

    prompt = synthetic_generation.prompt_generation(metadata, ef=0.55)

    assert "normal EF" in prompt
    assert "reduced EF" not in prompt


def test_prompt_generation_raises_if_ef_missing():
    metadata = {
        "Weight": 80.0,
        "Height": 180.0,
        "Group": "NOR",
    }

    with pytest.raises(ValueError, match="EF is required to generate the prompt."):
        synthetic_generation.prompt_generation(metadata)


def test_prompt_generation_from_acdc_uses_preprocess_and_metadata(monkeypatch, tmp_path):
    config_path = tmp_path / "Info.cfg"

    observed = {}

    def fake_preprocess(path: Path):
        observed["preprocess_path"] = path
        return "es_slice", "mask_slice", 0.41

    def fake_load_metadata(path: Path):
        observed["metadata_path"] = path
        return {
            "weight": 70.0,
            "height": 1.70,
            "pathology": "NOR",
        }

    monkeypatch.setattr(synthetic_generation, "preprocess", fake_preprocess)
    monkeypatch.setattr(synthetic_generation, "load_metadata", fake_load_metadata)

    prompt = synthetic_generation.prompt_generation_from_acdc(config_path)

    assert observed["preprocess_path"] == config_path
    assert observed["metadata_path"] == config_path
    assert "Cardiac MRI, short-axis view" in prompt
    assert "normal condition" in prompt
    assert "mildly reduced EF" in prompt
