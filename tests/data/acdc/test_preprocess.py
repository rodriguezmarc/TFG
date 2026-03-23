import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import pytest

import data.acdc.preprocess as preprocess_module


# 1. preprocess FUNCTIONAL
def test_preprocess_runs_full_pipeline(monkeypatch, tmp_path):
    """
    Run the ACDC preprocessing pipeline and verify that all steps are
    invoked and the final outputs are correctly propagated.
    """

    # --- arrange: synthetic ACDC-like data ---
    ed_image_arr = np.zeros((10, 20, 30), dtype=np.float32)
    es_image_arr = np.zeros((10, 20, 30), dtype=np.float32)
    ed_image = sitk.GetImageFromArray(ed_image_arr)
    es_image = sitk.GetImageFromArray(es_image_arr)
    ed_mask = "ed_mask_raw"
    es_mask = "es_mask_raw"

    def fake_load_data(config_path):
        # Expect a valid path-like config argument.
        assert config_path == tmp_path, (
            f"preprocess should pass the provided config_path to load_data, but got {config_path}"
        )
        return "image_4d", ed_image, ed_mask, es_image, es_mask, {"meta": "data"}

    def fake_change_label(mask, label_map):
        # Ensure masks are passed through label-unification step.
        assert mask in {ed_mask, es_mask}, (
            f"ChangeLabel should receive either ed_mask or es_mask, but got {mask}"
        )
        return f"changed_{mask}"

    def fake_extract_slice(image, mask, idx):
        # Slice index should be half of the z-dimension.
        assert image is es_image, (
            f"extract_slice should be called with es_image as first argument, but got {image}"
        )
        assert mask == "changed_es_mask_raw", (
            f"extract_slice should receive the relabelled ES mask, but got {mask}"
        )
        expected_idx = es_image.GetSize()[2] // 2
        assert idx == expected_idx, (
            f"Slice index should be half of es_image z-size, expected {expected_idx}, but got {idx}"
        )
        return "es_slice_initial", "mask_slice_initial"

    def fake_resample_slice(image, target_spacing=(1.0, 1.0), is_label=False):
        # Distinguish between image and mask by is_label flag.
        if is_label:
            assert image == "mask_slice_initial", (
                f"resample_slice for label should receive mask_slice_initial, but got {image}"
            )
            return "mask_slice_resampled"
        assert image in {"es_slice_initial", "mask_slice_initial", "mask_slice_resampled"}, (
            f"Unexpected image passed to resample_slice: {image}"
        )
        if image == "es_slice_initial":
            return "es_slice_resampled"
        return "mask_slice_resampled"

    def fake_get_lv_center(mask_slice):
        assert mask_slice == "mask_slice_resampled", (
            f"get_lv_center should be called with the resampled mask slice, but got {mask_slice}"
        )
        return (32, 32)

    def fake_crop_around_center(image, center):
        # Both image and mask should be cropped around the same centre.
        assert center == (32, 32), (
            f"crop_around_center should use the LV centre (32, 32), but got {center}"
        )
        if image == "es_slice_resampled":
            return "es_slice_cropped"
        if image == "mask_slice_resampled":
            return "mask_slice_cropped"
        raise AssertionError(f"Unexpected image passed to crop_around_center: {image}")

    def fake_normalize(image):
        assert image == "es_slice_cropped", (
            f"normalize should be called after cropping the ES slice, but got {image}"
        )
        return "es_slice_normalized"

    def fake_compute_ef(ed_mask_arg, es_mask_arg):
        # EF should be computed from the relabelled 3D masks.
        assert ed_mask_arg == "changed_ed_mask_raw", (
            f"compute_ef should receive the relabelled ED mask, but got {ed_mask_arg}"
        )
        assert es_mask_arg == "changed_es_mask_raw", (
            f"compute_ef should receive the relabelled ES mask, but got {es_mask_arg}"
        )
        return 0.6

    # --- apply monkeypatches on the preprocess module ---
    monkeypatch.setattr(preprocess_module, "load_data", fake_load_data)
    monkeypatch.setattr(preprocess_module.sitk, "ChangeLabel", fake_change_label)
    monkeypatch.setattr(preprocess_module, "extract_slice", fake_extract_slice)
    monkeypatch.setattr(preprocess_module, "resample_slice", fake_resample_slice)
    monkeypatch.setattr(preprocess_module, "get_lv_center", fake_get_lv_center)
    monkeypatch.setattr(preprocess_module, "crop_around_center", fake_crop_around_center)
    monkeypatch.setattr(preprocess_module, "normalize", fake_normalize)
    monkeypatch.setattr(preprocess_module, "compute_ef", fake_compute_ef)

    # --- act ---
    es_slice, mask_slice, ef = preprocess_module.preprocess(tmp_path)

    # --- assert ---
    assert es_slice == "es_slice_normalized", (
        f"Preprocess should return the normalized ES slice, but got {es_slice}"
    )
    assert mask_slice == "mask_slice_cropped", (
        f"Preprocess should return the cropped mask slice, but got {mask_slice}"
    )
    assert ef == 0.6, (
        f"Preprocess should return the EF value computed by compute_ef (0.6), but got {ef}"
    )


# 2. preprocess RAISES
def test_preprocess_propagates_compute_ef_error(monkeypatch, tmp_path):
    """
    If EF computation fails (e.g., EDV is zero), preprocess should surface
    the ValueError from compute_ef.
    """

    def fake_load_data(config_path):
        ed_image_arr = np.zeros((10, 20, 30), dtype=np.float32)
        es_image_arr = np.zeros((10, 20, 30), dtype=np.float32)
        ed_image = sitk.GetImageFromArray(ed_image_arr)
        es_image = sitk.GetImageFromArray(es_image_arr)
        return "image_4d", ed_image, "ed_mask_raw", es_image, "es_mask_raw", {"meta": "data"}

    def fake_change_label(mask, label_map):
        return f"changed_{mask}"

    def fake_extract_slice(image, mask, idx):
        return "es_slice_initial", "mask_slice_initial"

    def fake_resample_slice(image, target_spacing=(1.0, 1.0), is_label=False):
        return image

    def fake_get_lv_center(mask_slice):
        return (32, 32)

    def fake_crop_around_center(image, center):
        return image

    def fake_normalize(image):
        return image

    def fake_compute_ef(ed_mask_arg, es_mask_arg):
        raise ValueError("EDV is zero, cannot compute EF.")

    monkeypatch.setattr(preprocess_module, "load_data", fake_load_data)
    monkeypatch.setattr(preprocess_module.sitk, "ChangeLabel", fake_change_label)
    monkeypatch.setattr(preprocess_module, "extract_slice", fake_extract_slice)
    monkeypatch.setattr(preprocess_module, "resample_slice", fake_resample_slice)
    monkeypatch.setattr(preprocess_module, "get_lv_center", fake_get_lv_center)
    monkeypatch.setattr(preprocess_module, "crop_around_center", fake_crop_around_center)
    monkeypatch.setattr(preprocess_module, "normalize", fake_normalize)
    monkeypatch.setattr(preprocess_module, "compute_ef", fake_compute_ef)

    with pytest.raises(ValueError, match="EDV is zero, cannot compute EF."):
        preprocess_module.preprocess(tmp_path)


# 3. preprocess WITH REAL ACDC DATA (optional)
def test_preprocess_with_real_acdc_data_if_available():
    """
    Optional integration test that runs the preprocessing pipeline against
    real ACDC data if a valid config path is provided via the
    ACDC_CONFIG_PATH environment variable.
    """
    config_env = os.getenv("ACDC_CONFIG_PATH")
    if not config_env:
        pytest.skip("ACDC_CONFIG_PATH not set; skipping integration test with real ACDC data.")

    config_path = Path(config_env)
    if not config_path.exists():
        pytest.skip(f"ACDC config file {config_path} does not exist; skipping.")

    es_slice, mask_slice, ef = preprocess_module.preprocess(config_path)

    assert isinstance(es_slice, sitk.Image), (
        f"Preprocess should return a SimpleITK image for ES slice, but got {type(es_slice)}"
    )
    assert isinstance(mask_slice, sitk.Image), (
        f"Preprocess should return a SimpleITK image for mask slice, but got {type(mask_slice)}"
    )
    assert isinstance(ef, float), (
        f"Preprocess should return EF as a float value, but got {type(ef)}"
    )

