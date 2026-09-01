# FireWorldBench

![Task](https://img.shields.io/badge/Task-Fire--Physics--VQA-red)
![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red)
![Dataset](https://img.shields.io/badge/Dataset-FireWorldBench-blue)

<font size=5><div align='center'>[[📊 Dataset](https://huggingface.co/datasets/Guaogua/FireWorldBench)] [[📖 Paper](Paper link to be added)] [[🏆 Leaderboard](Leaderboard link to be added)]</div></font>

> The benchmark's motivation, design and analysis are described in the paper (to be provided by the authors).

## 🚀 Quick Start (evaluate your model)

Full pipeline: **download data → configure a model endpoint → run → get the six metrics.**

```shell
# 1. Install dependencies
pip install -e .

# 2. Download data from Hugging Face (~660MB; add --mirror in mainland China)
cd scripts
bash download.sh --mirror
cd ..

# 3. Configure an OpenAI-compatible model endpoint
export OPENAI_BASE_URL="https://your-endpoint/v1"
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="your-model"

# 4. (Optional) quick check with 2 items
python scripts/run_eval.py --split main_synthetic --max-requests 2

# 5. Run the full test split and compute the six metrics
python scripts/run_eval.py --split main_synthetic
```

After it finishes, the six metrics are written to `results/main_synthetic-physical.json` (by default grouped by the physics axis P1-P5 × 2 tracks × 2 question types = 20 cells; use `--group-by fire` for the fire axis, or `--group-by task` for the fine-grained 36 cells).

> Text-only models must run with `--track S` (I-track items carry images and will fail on a text-only model); vision models may use `--track I`. See `python scripts/run_eval.py --help`.



Data is published on Hugging Face: **[`Guaogua/FireWorldBench`](https://huggingface.co/datasets/Guaogua/FireWorldBench)**

```shell
cd scripts && bash download.sh          # add --mirror in mainland China
```


## Installation

```shell
pip install -e .
```

(Or without pip: `pip install -r scripts/requirements.txt`.)

## Scripts

| Script | Purpose |
|---|---|
| `scripts/download.sh` | Download data from Hugging Face |
| `scripts/run_eval.py` | One-click pipeline: run + compute the six metrics |
| `scripts/run_api.py` | Multimodal API runner (resumable, **never reads Gold**) |
| `scripts/score_six_metrics.py` | Deterministic six-metric scorer |
| `scripts/score_fg9_*.py` | Scorer internals |
| `scripts/contracts/` | (Optional) frozen task contracts |

## Metrics (FWB-FG9-SIX-METRICS)

All six metrics are deterministic (no LLM judge) and scored over `status=ok` responses only:

- **Completion Accuracy** — choice: Jaccard on option sets; open: mean required-field match (L3-1 uses inverse-class-frequency weights).
- **Macro-F1 (Slot-MacroF1)** — choice: Sørensen–Dice on option sets (unchanged); open: per-slot exact-match F1 `2·TP/(2·TP+FP+FN)` (TP=exact match, FP=wrong value, FN=missing) macro-averaged over slots; **overall = (choice + open) / 2**.
- **Evidence-F1** (open) — token P/R against gold evidence anchors.
- **Mechanism Alignment** (open) — content-word overlap with gold mechanism statements.
- **Brier Score** — `(confidence − completion)²`; lower is better.
- **Gold-linked Support (GLS)** (open) — correct fields that are also grounded in the model's explanation.

See `scripts/score_six_metrics.py` and `scripts/score_fg9_predictions.py` for the exact implementation.

## Citation

(Paper BibTeX to be provided by the authors.)
