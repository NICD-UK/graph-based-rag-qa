#!/usr/bin/env python3
"""Human evaluation of fuzzy match corrections."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Human evaluation of fuzzy match corrections."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="Correction CSV files to evaluate (supports shell glob expansion).",
    )
    args = parser.parse_args()

    csv_files = sorted(args.csv_files)
    print(f"Found {len(csv_files)} files")

    # 1. Load all corrections and track which files contain each pair
    pair_to_files: dict[tuple[str, str], list[tuple[Path, float]]] = {}

    for csv_path in csv_files:
        path = Path(csv_path)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["original"], row["matched"])
                score = float(row["score"])
                if key not in pair_to_files:
                    pair_to_files[key] = []
                pair_to_files[key].append((path, score))

    # 2. Collect human decisions for each unique pair
    decisions: dict[tuple[str, str], bool] = {}

    total = len(pair_to_files)
    for i, ((original, matched), file_scores) in enumerate(pair_to_files.items(), 1):
        score = file_scores[0][1]  # Same across files
        print(f"\n[{i}/{total}]")
        print(f"  Original: {original!r}")
        print(f"  Matched:  {matched!r}")
        print(f"  Score:    {score:.1f}")

        while True:
            response = input("  Accept? (y/n): ").strip().lower()
            if response in ("y", "n"):
                decisions[(original, matched)] = (response == "y")
                break
            print("  Please enter 'y' or 'n'")

    # 3. Write per-file results
    for csv_path in args.csv_files:
        path = Path(csv_path)
        output_path = path.with_name(path.stem + "-humaneval.csv")

        with (
            open(path, newline="", encoding="utf-8") as infile,
            open(output_path, "w", newline="", encoding="utf-8") as outfile,
        ):
            reader = csv.DictReader(infile)
            writer = csv.DictWriter(outfile, fieldnames=["original", "matched", "score", "accept"])
            writer.writeheader()

            for row in reader:
                key = (row["original"], row["matched"])
                writer.writerow({
                    "original": row["original"],
                    "matched": row["matched"],
                    "score": row["score"],
                    "accept": decisions[key],
                })

        print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
