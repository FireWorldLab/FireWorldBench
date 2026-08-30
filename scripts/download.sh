#!/usr/bin/env bash
# Download the FireWorldBench formal-test dataset from Hugging Face.
#
# By downloading the dataset you agree to its license:
#   questions / Gold / assets: CC BY 4.0  https://creativecommons.org/licenses/by/4.0/
# The data may be freely used and redistributed with attribution.
set -euo pipefail
cd "$(dirname "$0")/.."
python download_dataset.py --output data "$@"
echo "Dataset downloaded to ./data"
