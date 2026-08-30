# FireWorldBench

![Task](https://img.shields.io/badge/Task-Fire--Physics--VQA-red)
![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red)
![Dataset](https://img.shields.io/badge/Dataset-FireWorldBench-blue)

<font size=5><div align='center'>[[📊 Dataset](https://huggingface.co/datasets/Guaogua/FireWorldBench)] [[📖 Paper](Paper link to be added)] [[🏆 Leaderboard](Leaderboard link to be added)]</div></font>

> The benchmark's motivation, design and analysis are described in the paper (to be provided by the authors). This repo is meant to get you **up and running quickly**.

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

The two formal test splits:

| Split | Description | #Items |
|---|---|---|
| `main_synthetic/full_test_A` | Synthetic fire events (S text + I vision) | 8,360 |
| `mmodalfire_c06/full_test_A` | Real events (C06) | 714 |

## Data

- **Data** (questions / Gold / images): **CC BY 4.0**, see `DATA_LICENSE.md`
- **Code**: **Apache-2.0**, see `LICENSE`

Data is published on Hugging Face: **[`Guaogua/FireWorldBench`](https://huggingface.co/datasets/Guaogua/FireWorldBench)**

```shell
cd scripts && bash download.sh          # add --mirror in mainland China
```

Every item carries **dual-axis five-layer capability labels**: `physical_axis` (physics axis P1-P5) and `fire_axis` (fire axis T1-T5).

### Five-layer dual-axis partition

Physics axis P:

| P | Layer | Tasks |
|---|---|---|
| P1 | Temporal Evolution Forecasting | L3-1, L3-2 |
| P2 | Physical Field Perception and Grounding | L1-1, L1-2 |
| P3 | Cross-Field Coupling Understanding | L1-3, L2-1, L2-2 |
| P4 | Counterfactual Intervention Reasoning | L3-3 |
| P5 | Causal Mechanism Attribution | L2-3 |

Fire axis T:

| T | Layer | Tasks |
|---|---|---|
| T1 | Fire Evolution Prediction | L3-1, L3-2, L1-2 |
| T2 | Fire Early Warning | L1-1 |
| T3 | Fire State Assessment | L1-3, L2-1, L2-2 |
| T4 | Fire Intervention Decision-Making | L3-3 |
| T5 | Fire Mechanism Diagnosis | L2-3 |

## Installation

```shell
pip install -e .
```

(Or without pip: `pip install -r scripts/requirements.txt`.)

## Six metrics

| Metric | Direction | Applies to |
|---|---|---|
| Completion Accuracy (ACC) | Higher is better (primary) | choice / open |
| Macro-F1 | Higher is better | choice / open |
| Evidence-F1 | Higher is better | open |
| Mechanism Alignment | Higher is better | open |
| Brier Score | Lower is better | choice / open |
| Gold-linked Support | Higher is better | open |

- **Completion Accuracy**: for choice, the mean Jaccard similarity between the predicted and Gold option sets; for open, the fraction of required fields predicted correctly.
- Reporting via `--group-by`, never merging or hiding a failed cell: `physical` (P1-P5 × 2 tracks × 2 types = 20 cells, default), `fire` (T1-T5 × 2 tracks × 2 types = 20 cells), `task` (fine-grained, 36 cells).
- All metrics are computed deterministically; no model judge is used.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/download.sh` | Download data from Hugging Face |
| `scripts/run_eval.py` | One-click pipeline: run + compute the six metrics |
| `scripts/run_api.py` | Multimodal API runner (resumable, **never reads Gold**) |
| `scripts/score_six_metrics.py` | Deterministic six-metric scorer |
| `scripts/score_fg9_*.py` | Scorer internals |
| `scripts/contracts/` | (Optional) frozen task contracts |

## Citation

(Paper BibTeX to be provided by the authors.)
