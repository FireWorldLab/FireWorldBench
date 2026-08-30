#!/usr/bin/env python3
"""Download the FireWorldBench formal-test dataset from Hugging Face.

Usage:
    python download_dataset.py [--repo-id Guaogua/FireWorldBench]
                               [--output FireWorldBench] [--mirror]

--mirror  sets HF_ENDPOINT to https://hf-mirror.com (recommended in China).
"""
import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Guaogua/FireWorldBench",
                        help="Hugging Face dataset repo id (default: Guaogua/FireWorldBench)")
    parser.add_argument("--output", default="FireWorldBench",
                        help="local output directory (default: FireWorldBench)")
    parser.add_argument("--mirror", action="store_true",
                        help="use HF mirror endpoint hf-mirror.com")
    args = parser.parse_args()

    if args.mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Missing dependency. Run: pip install huggingface_hub", file=sys.stderr)
        return 1

    print(f"Downloading {args.repo_id} -> {args.output} ...")
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.output,
        local_dir_use_symlinks=False,
    )
    print(f"Done: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
