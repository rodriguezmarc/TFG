"""
Definition:
Brief map of the top-level end-to-end pipeline command.
---
Results:
Parses CLI arguments and runs preprocessing, fine-tuning, and evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minim.pipeline import run_full_pipeline


def build_parser() -> argparse.ArgumentParser:
    """
    ########################################
    Definition:
    Build the CLI parser for the end-to-end run command.
    ---
    Params:
    None.
    ---
    Results:
    Returns the configured argument parser.
    ########################################
    """
    parser = argparse.ArgumentParser(description="Run preprocessing, fine-tuning, and evaluation end to end")
    parser.add_argument("--dataset", "-d", choices=["acdc", "ukbb"], action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modality", type=str, default="Cardiac MRI")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/minim"))
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--num-train-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    return parser


def main() -> None:
    """
    ########################################
    Definition:
    Execute the end-to-end run command from parsed CLI arguments.
    ---
    Params:
    None.
    ---
    Results:
    Runs preprocessing, fine-tuning, and evaluation and prints artifact paths and metrics.
    ########################################
    """
    args = build_parser().parse_args()
    prepared_run, metrics = run_full_pipeline(
        datasets=tuple(args.dataset),
        output_root=args.output_root,
        modality=args.modality,
        seed=args.seed,
        base_model=args.base_model,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resolution=args.resolution,
        mixed_precision=args.mixed_precision,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    print(prepared_run.summary_path)
    print(prepared_run.model_dir)
    print(prepared_run.metrics_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
