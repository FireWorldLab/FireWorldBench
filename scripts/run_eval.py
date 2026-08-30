#!/usr/bin/env python3
"""Run a model on a FireWorldBench split and compute the six metrics.

Usage (from the repo root):
    python scripts/run_eval.py --split main_synthetic
    python scripts/run_eval.py --split mmodalfire_c06 --group-by physical

Requires the OpenAI-compatible endpoint env vars:
    OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SPLITS = {
    "main_synthetic": ("main_synthetic/full_test_A", "main_synthetic"),
    "mmodalfire_c06": ("mmodalfire_c06/full_test_A", "mmodalfire_c06"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument("--group-by", choices=("task", "physical", "fire"), default="task",
                        help="report cells by task (36), physical axis P1-P5 (20) or fire axis T1-T5 (20)")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default=None, help="default: $OPENAI_MODEL")
    parser.add_argument("--max-requests", type=int, default=None,
                        help="run only the first N questions (for a quick test)")
    parser.add_argument("--data-dir", default=None, help="default: ./data")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else root / "data"
    split, source = SPLITS[args.split]
    questions = data_dir / split / "questions.jsonl"
    gold = data_dir / split / "gold.jsonl"
    if not questions.exists():
        sys.exit(f"Data not found at {questions}\nRun: cd scripts && bash download.sh")
    if not gold.exists():
        sys.exit(f"Gold not found at {gold}")

    model = args.model or os.environ.get("OPENAI_MODEL")
    if not model:
        sys.exit("Missing model. Set OPENAI_MODEL or pass --model.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = out_dir / f"{args.split}.jsonl"
    result = out_dir / f"{args.split}-{args.group_by}.json"

    # Step 1: run inference (resumable; never reads Gold)
    cmd = [sys.executable, str(root / "scripts" / "run_api.py"),
           "--provider", args.provider,
           "--questions", str(questions),
           "--source-root", str(data_dir / source),
           "--output", str(preds),
           "--model", model]
    if args.max_requests:
        cmd += ["--max-requests", str(args.max_requests)]
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # Step 2: compute the six metrics
    cmd = [sys.executable, str(root / "scripts" / "score_six_metrics.py"),
           "--gold", str(gold), "--predictions", str(preds),
           "--output", str(result), "--group-by", args.group_by]
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    print(f"\nDone. Six metrics written to {result}")
    print(json.dumps(json.load(open(result))["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
