import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dataset CSV pipeline"
    )

    parser.add_argument(
        "--dataset", "-d",
        choices=["acdc", "ukbb"],
        default="acdc",
        help="Dataset to process (default: acdc)"
    )

    return parser.parse_args()