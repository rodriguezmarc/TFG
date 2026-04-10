"""
Definition:
Shared CLI argument helpers for MINIM pipeline entrypoints.
---
Results:
Keeps top-level and standalone CLIs aligned as runtime options evolve.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from minim.backends import BACKEND_NAMES
from minim.constants import DEFAULT_OUTPUT_ROOT, METRIC_MODES
from minim.runs import PreparedRun


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def add_prepare_args(parser: argparse.ArgumentParser, *, include_split_ratios: bool = False) -> None:
    parser.add_argument("--dataset", "-d", choices=["acdc", "ukbb"], action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modality", type=str, default="Cardiac MRI")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    if include_split_ratios:
        parser.add_argument("--train-ratio", type=float, default=0.7)
        parser.add_argument("--val-ratio", type=float, default=0.15)
        parser.add_argument("--test-ratio", type=float, default=0.15)


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--num-train-epochs", type=int, default=5)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--validation-epochs", type=int, default=5)
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--backend", choices=BACKEND_NAMES, default="real")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--metrics-mode", choices=METRIC_MODES, default="full")


def prepare_kwargs(args: argparse.Namespace, *, include_split_ratios: bool = False) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "datasets": tuple(args.dataset),
        "output_root": args.output_root,
        "modality": args.modality,
        "seed": args.seed,
    }
    if include_split_ratios:
        kwargs.update(
            {
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
            }
        )
    return kwargs


def runtime_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "base_model": args.base_model,
        "num_train_epochs": args.num_train_epochs,
        "max_train_steps": args.max_train_steps,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "resolution": args.resolution,
        "mixed_precision": args.mixed_precision,
        "validation_epochs": args.validation_epochs,
        "report_to": args.report_to,
        "backend": args.backend,
        "runs": args.runs,
        "skip_evaluation": args.skip_evaluation,
        "device": args.device,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "metrics_mode": args.metrics_mode,
    }


def print_run_result(prepared_run: PreparedRun, metrics: dict[str, object]) -> None:
    print(prepared_run.summary_path)
    print(prepared_run.model_dir)
    print(prepared_run.metrics_path)
    print(json.dumps(metrics, indent=2, allow_nan=False))
