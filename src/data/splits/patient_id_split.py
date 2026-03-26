"""
Patient-level train/val/test split utilities.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitResult:
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]


def split_patient_ids(
    patient_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitResult:
    """
    Split patient IDs reproducibly at patient level.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    ids = list(patient_ids)
    random.Random(seed).shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val : n_train + n_val + n_test]

    return SplitResult(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)
