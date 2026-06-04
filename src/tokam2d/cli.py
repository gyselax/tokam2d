# src/tokam2d/cli.py
"""Command-line entry point: a thin wrapper over :func:`run_simulation`."""

import time
from argparse import ArgumentParser
from pathlib import Path

from tokam2d.api import run_simulation


def get_input_path_and_output_folder():
    parser = ArgumentParser(description="Run TOKAM2D")
    parser.add_argument("-i", "--input_file", action="store", nargs="?",
                        default=None, type=Path,
                        help="input file (YAML format)")
    parser.add_argument("-o", "--output_folder", action="store", nargs="?",
                        default=None, type=Path, help="output folder")
    args = parser.parse_args()
    if args.input_file is None:
        raise ValueError("No input file provided.")
    return args.input_file, args.output_folder


def main():
    start = time.time()
    input_path, save_dir = get_input_path_and_output_folder()
    run_simulation(input_path, save_dir=save_dir, quiet=False)
    print(f"Time taken: {time.time() - start:.3f}s")
    print("Simulation complete.")


if __name__ == "__main__":
    main()
