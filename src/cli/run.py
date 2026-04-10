"""
Definition:
Brief map of the top-level end-to-end pipeline command.
---
Results:
Parses CLI arguments and runs preprocessing, fine-tuning, and evaluation.
"""

from __future__ import annotations

import argparse

from minim.cli_args import add_prepare_args, add_runtime_args, configure_logging, prepare_kwargs, print_run_result, runtime_kwargs
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
    add_prepare_args(parser)
    add_runtime_args(parser)
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
    configure_logging()
    try:
        prepared_run, metrics = run_full_pipeline(
            **prepare_kwargs(args),
            **runtime_kwargs(args),
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print_run_result(prepared_run, metrics)


if __name__ == "__main__":
    main()
