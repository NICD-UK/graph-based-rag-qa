#!/usr/bin/env python3
"""Extract the MoNaCo questions and gold answers for a matches JSONL subset.

`load_monaco.py` / `filter_by_humaneval.py` record only which Wikipedia pages
each question needs (`ex_num`, `question`, `sources`). The evaluation stage
also needs the benchmark's gold `decomposition` and `validated_answer` for
exactly those questions. Those live in `monaco_version_1_release.jsonl` on the
Hugging Face hub, so this script joins the two on `ex_num` and writes a JSON
list, in the same order as the matches file, of:

    {"ex_num": int, "question": str, "decomposition": [str], "validated_answer": [...]}
"""

import argparse
import json
import os

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

REPO_ID = "allenai/MoNaCo_Benchmark"
RELEASE_FILENAME = "monaco_version_1_release.jsonl"
OUTPUT_FIELDS = ("ex_num", "question", "decomposition", "validated_answer")


def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not found in environment")
    return token


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_release(path: str | None) -> dict[int, dict]:
    """Load the MoNaCo release file keyed by ex_num (downloading it if needed)."""
    if path is None:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=RELEASE_FILENAME,
            repo_type="dataset",
            token=get_hf_token(),
        )
    return {int(row["ex_num"]): row for row in iter_jsonl(path)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Join a MoNaCo matches JSONL with the benchmark release file to "
            "produce the questions + gold answers for that subset."
        )
    )
    parser.add_argument(
        "--matches-jsonl",
        required=True,
        help=(
            "Matches JSONL (from load_monaco.py / filter_by_humaneval.py) whose "
            "ex_nums select and order the output questions."
        ),
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output JSON list of {ex_num, question, decomposition, validated_answer}.",
    )
    parser.add_argument(
        "--release-jsonl",
        help=(
            f"Local copy of {RELEASE_FILENAME}. Default: download it from "
            f"{REPO_ID} on Hugging Face (requires HF_TOKEN in .env)."
        ),
    )
    args = parser.parse_args()

    release = load_release(args.release_jsonl)
    print(f"Loaded {len(release)} MoNaCo questions from {RELEASE_FILENAME}")

    records: list[dict] = []
    missing: list[int] = []
    mismatched: list[int] = []
    for match in iter_jsonl(args.matches_jsonl):
        ex_num = int(match["ex_num"])
        row = release.get(ex_num)
        if row is None:
            missing.append(ex_num)
        elif row["question"] != match["question"]:
            mismatched.append(ex_num)
        else:
            records.append({field: row[field] for field in OUTPUT_FIELDS})

    # The matches file was built from the same benchmark, so every ex_num must
    # resolve to the identical question. Anything else means the release file
    # and the matches file come from different MoNaCo revisions.
    if missing or mismatched:
        if missing:
            print(f"Error: {len(missing)} ex_num(s) not in the release file: {missing}")
        if mismatched:
            print(
                f"Error: {len(mismatched)} ex_num(s) whose question text differs "
                f"from the release file: {mismatched}"
            )
        return 1

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {len(records)} questions to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
