"""
Definition:
Brief map of the top-level fine-tune and evaluation command.
---
Results:
Parses CLI arguments and runs fine-tuning plus metric evaluation.
"""

from __future__ import annotations

import argparse

from minim.cli_args import add_prepare_args, add_runtime_args, configure_logging, prepare_kwargs, print_run_result, runtime_kwargs
from minim.pipeline import run_evaluation_pipeline


def build_parser() -> argparse.ArgumentParser:
    """
    ########################################
    Definition:
    Build the CLI parser for the evaluation command.
    ---
    Params:
    None.
    ---
    Results:
    Returns the configured argument parser.
    ########################################
    """
    parser = argparse.ArgumentParser(description="Run fine-tuning and evaluation from prepared dataset exports")
    add_prepare_args(parser)
    add_runtime_args(parser)
    return parser

def main() -> None:
    """
    ########################################
    Definition:
    Execute the evaluation command from parsed CLI arguments.
    ---
    Params:
    None.
    ---
    Results:
    Runs fine-tuning plus evaluation and prints artifact paths and metrics.
    ########################################
    """
    args = build_parser().parse_args()
    configure_logging()
    try:
        prepared_run, metrics = run_evaluation_pipeline(
            **prepare_kwargs(args),
            **runtime_kwargs(args),
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print_run_result(prepared_run, metrics)


if __name__ == "__main__":
    main()
