"""
Definition:
Brief map of the standalone CLI for the local MINIM pipeline.
---
Results:
Parses subcommands that expose preparation, training, generation, and evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from minim.checkpoints import DEFAULT_BASE_MODEL_ID
from minim.pipeline import evaluate_run, prepare_run, run_full_pipeline, run_prompt, train_run


def build_parser() -> argparse.ArgumentParser:
    """
    ########################################
    Definition:
    Build the standalone CLI parser for the local MINIM pipeline.
    ---
    Params:
    None.
    ---
    Results:
    Returns the configured argument parser with all supported subcommands.
    ########################################
    """
    parser = argparse.ArgumentParser(description="Local MINIM pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_prepare_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--dataset", "-d", choices=["acdc", "ukbb"], action="append", required=True)
        target.add_argument("--output-root", type=Path, default=Path("outputs/minim"))
        target.add_argument("--modality", type=str, default="Cardiac MRI")
        target.add_argument("--seed", type=int, default=42)
        target.add_argument("--train-ratio", type=float, default=0.7)
        target.add_argument("--val-ratio", type=float, default=0.15)
        target.add_argument("--test-ratio", type=float, default=0.15)

    prepare_parser = subparsers.add_parser("prepare")
    add_common_prepare_args(prepare_parser)

    train_parser = subparsers.add_parser("train")
    add_common_prepare_args(train_parser)
    train_parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL_ID)
    train_parser.add_argument("--num-train-epochs", type=int, default=5)
    train_parser.add_argument("--learning-rate", type=float, default=1e-5)
    train_parser.add_argument("--train-batch-size", type=int, default=1)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_parser.add_argument("--resolution", type=int, default=512)
    train_parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=None)

    evaluate_parser = subparsers.add_parser("evaluate")
    add_common_prepare_args(evaluate_parser)
    evaluate_parser.add_argument("--device", type=str, default=None)
    evaluate_parser.add_argument("--num-inference-steps", type=int, default=50)
    evaluate_parser.add_argument("--guidance-scale", type=float, default=7.5)

    run_parser = subparsers.add_parser("run")
    add_common_prepare_args(run_parser)
    run_parser.add_argument("--prompt", type=str, required=True)
    run_parser.add_argument("--count", type=int, default=1)
    run_parser.add_argument("--device", type=str, default=None)
    run_parser.add_argument("--num-inference-steps", type=int, default=50)
    run_parser.add_argument("--guidance-scale", type=float, default=7.5)

    all_parser = subparsers.add_parser("all")
    add_common_prepare_args(all_parser)
    all_parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL_ID)
    all_parser.add_argument("--num-train-epochs", type=int, default=5)
    all_parser.add_argument("--learning-rate", type=float, default=1e-5)
    all_parser.add_argument("--train-batch-size", type=int, default=1)
    all_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    all_parser.add_argument("--resolution", type=int, default=512)
    all_parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=None)
    all_parser.add_argument("--device", type=str, default=None)
    all_parser.add_argument("--num-inference-steps", type=int, default=50)
    all_parser.add_argument("--guidance-scale", type=float, default=7.5)

    return parser


def _prepare_from_args(args: argparse.Namespace):
    """
    ########################################
    Definition:
    Convert parsed CLI arguments into a prepared-run contract.
    ---
    Params:
    args: Parsed CLI namespace.
    ---
    Results:
    Returns the prepared run used by downstream subcommands.
    ########################################
    """
    return prepare_run(
        datasets=tuple(args.dataset),
        output_root=args.output_root,
        modality=args.modality,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


def main() -> None:
    """
    ########################################
    Definition:
    Execute the standalone MINIM CLI from parsed arguments.
    ---
    Params:
    None.
    ---
    Results:
    Dispatches to the requested subcommand and prints the resulting artifacts.
    ########################################
    """
    parser = build_parser()
    args = parser.parse_args()
    prepared_run = _prepare_from_args(args)

    if args.command == "prepare":
        print(prepared_run.summary_path)
        return

    if args.command == "train":
        train_run(
            prepared_run=prepared_run,
            base_model=args.base_model,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            resolution=args.resolution,
            seed=args.seed,
            mixed_precision=args.mixed_precision,
        )
        print(prepared_run.model_dir)
        return

    if args.command == "evaluate":
        metrics = evaluate_run(
            prepared_run=prepared_run,
            device=args.device,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        print(metrics)
        return

    if args.command == "run":
        output_paths = run_prompt(
            prepared_run=prepared_run,
            prompt=args.prompt,
            modality=args.modality,
            count=args.count,
            device=args.device,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        for output_path in output_paths:
            print(output_path)
        return

    _, metrics = run_full_pipeline(
        datasets=tuple(args.dataset),
        output_root=args.output_root,
        modality=args.modality,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
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
    print(metrics)


if __name__ == "__main__":
    main()
