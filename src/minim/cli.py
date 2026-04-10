"""
Definition:
Brief map of the standalone CLI for the local MINIM pipeline.
---
Results:
Parses only the public prepare, evaluate, and run commands.
"""

from __future__ import annotations

import argparse

from minim.cli_args import add_prepare_args, add_runtime_args, configure_logging, prepare_kwargs, print_run_result, runtime_kwargs
from minim.pipeline import prepare_run, run_evaluation_pipeline, run_full_pipeline


def _build_parser() -> argparse.ArgumentParser:
    """
    ########################################
    Definition:
    Build the standalone CLI parser for the local MINIM pipeline.
    ---
    Params:
    None.
    ---
    Results:
    Returns the configured parser with the supported commands.
    ########################################
    """
    parser = argparse.ArgumentParser(description="Local MINIM pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    add_prepare_args(prepare_parser, include_split_ratios=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    add_prepare_args(evaluate_parser, include_split_ratios=True)
    add_runtime_args(evaluate_parser)

    run_parser = subparsers.add_parser("run")
    add_prepare_args(run_parser, include_split_ratios=True)
    add_runtime_args(run_parser)

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
    Returns the prepared run used by downstream commands.
    ########################################
    """
    return prepare_run(
        **prepare_kwargs(args, include_split_ratios=True),
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
    Dispatches to the requested command and prints the resulting artifacts.
    ########################################
    """
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging()

    try:
        if args.command == "prepare":
            prepared_run = _prepare_from_args(args)
            print(prepared_run.summary_path)
            return

        if args.command == "evaluate":
            prepared_run, metrics = run_evaluation_pipeline(
                **prepare_kwargs(args, include_split_ratios=True),
                **runtime_kwargs(args),
            )
        else:
            prepared_run, metrics = run_full_pipeline(
                **prepare_kwargs(args, include_split_ratios=True),
                **runtime_kwargs(args),
            )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print_run_result(prepared_run, metrics)


if __name__ == "__main__":
    main()
