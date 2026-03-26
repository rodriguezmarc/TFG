"""
CSV writer and validation utilities for MINIM.
"""

from __future__ import annotations

import csv
from pathlib import Path

MINIM_COLUMNS = ("path", "text", "modality")


def write_minim_csv(rows: list[dict[str, str]], output_csv_path: Path) -> None:
    """
    Write rows to CSV with MINIM-compatible columns.
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with Path.open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(MINIM_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def validate_minim_csv(rows: list[dict[str, str]], images_root: Path) -> None:
    """
    Validate MINIM rows before writing/using the CSV.
    """
    seen_paths: set[str] = set()
    for idx, row in enumerate(rows):
        missing_cols = [col for col in MINIM_COLUMNS if col not in row]
        if missing_cols:
            raise ValueError(f"Row {idx} missing columns: {missing_cols}")

        rel_path = row["path"].strip()
        text = row["text"].strip()
        modality = row["modality"].strip()

        if not rel_path:
            raise ValueError(f"Row {idx} has empty path.")
        if not text:
            raise ValueError(f"Row {idx} has empty text.")
        if not modality:
            raise ValueError(f"Row {idx} has empty modality.")
        if rel_path in seen_paths:
            raise ValueError(f"Duplicated path in CSV rows: {rel_path}")

        image_path = images_root / rel_path
        if not image_path.exists():
            raise ValueError(f"Image path does not exist: {image_path}")
        seen_paths.add(rel_path)
