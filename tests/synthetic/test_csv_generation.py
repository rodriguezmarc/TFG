from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from synthetic.csv_generation import (
    build_acdc_minim_csv,
    build_acdc_minim_rows,
    validate_minim_csv,
    write_minim_csv,
)


def _fake_image() -> sitk.Image:
    arr = np.linspace(0, 1, 64 * 64).reshape(64, 64).astype(np.float32)
    return sitk.GetImageFromArray(arr)


def test_build_acdc_minim_rows_runs_pipeline_and_creates_images(tmp_path):
    cfg1 = tmp_path / "patient001" / "Info.cfg"
    cfg2 = tmp_path / "patient002" / "Info.cfg"
    cfg1.parent.mkdir(parents=True, exist_ok=True)
    cfg2.parent.mkdir(parents=True, exist_ok=True)
    cfg1.write_text("dummy", encoding="utf-8")
    cfg2.write_text("dummy", encoding="utf-8")

    def fake_preprocess(path: Path):
        return _fake_image(), _fake_image(), 0.55 if path == cfg1 else 0.35

    def fake_load_metadata(path: Path):
        if path == cfg1:
            return {"weight": 70.0, "height": 1.70, "pathology": "NOR"}
        return {"weight": 90.0, "height": 1.70, "pathology": "DCM"}

    rows = build_acdc_minim_rows(
        config_paths=[cfg1, cfg2],
        images_root=tmp_path / "images",
        preprocess_fn=fake_preprocess,
        metadata_loader_fn=fake_load_metadata,
    )

    assert len(rows) == 2
    assert rows[0]["modality"] == "Cardiac MRI"
    assert "normal EF" in rows[0]["text"]
    assert "reduced EF" in rows[1]["text"]

    for row in rows:
        assert (tmp_path / "images" / row["path"]).exists()


def test_validate_minim_csv_rejects_empty_and_duplicate_paths(tmp_path):
    image_dir = tmp_path / "images"
    (image_dir / "acdc").mkdir(parents=True, exist_ok=True)
    (image_dir / "acdc" / "p1.png").write_text("x", encoding="utf-8")

    rows_ok = [
        {
            "path": "acdc/p1.png",
            "text": "Cardiac MRI, short-axis view, normal BMI, normal condition, normal EF.",
            "modality": "Cardiac MRI",
        }
    ]
    validate_minim_csv(rows_ok, image_dir)

    rows_dup = rows_ok + [rows_ok[0].copy()]
    with pytest.raises(ValueError, match="Duplicated path"):
        validate_minim_csv(rows_dup, image_dir)

    rows_empty = [{"path": "", "text": "x", "modality": "Cardiac MRI"}]
    with pytest.raises(ValueError, match="empty path"):
        validate_minim_csv(rows_empty, image_dir)


def test_write_minim_csv_writes_expected_header_and_rows(tmp_path):
    csv_path = tmp_path / "out" / "cardiac_data.csv"
    rows = [
        {
            "path": "acdc/p1.png",
            "text": "Cardiac MRI, short-axis view, normal BMI, normal condition, normal EF.",
            "modality": "Cardiac MRI",
        }
    ]
    write_minim_csv(rows, csv_path)
    content = csv_path.read_text(encoding="utf-8")

    assert content.splitlines()[0] == "path,text,modality"
    assert "acdc/p1.png" in content


def test_build_acdc_minim_csv_full_flow(tmp_path):
    cfg = tmp_path / "patient010" / "Info.cfg"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("dummy", encoding="utf-8")

    def fake_preprocess(path: Path):
        return _fake_image(), _fake_image(), 0.41

    def fake_load_metadata(path: Path):
        return {"weight": 65.0, "height": 1.70, "pathology": "NOR"}

    csv_path = tmp_path / "exports" / "cardiac_data.csv"
    rows = build_acdc_minim_csv(
        config_paths=[cfg],
        images_root=tmp_path / "images",
        output_csv_path=csv_path,
        image_subdir="acdc",
        preprocess_fn=fake_preprocess,
        metadata_loader_fn=fake_load_metadata,
    )

    assert len(rows) == 1
    assert csv_path.exists()
