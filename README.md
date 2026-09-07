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


## Dual-axis five-layer labels

Physical axis (P1 easiest → P5 hardest; numbered on controlled-simulation figure Acc, 25 tracks including InternVL3):

- P1 Temporal Evolution Forecasting ← L3-1, L3-2

- P2 Physical Field Perception and Grounding ← L1-1, L2-1

- P3 Cross-Field Coupling Understanding ← L1-2, L2-2

- P4 Causal Mechanism Attribution ← L1-3, L2-3

- P5 Counterfactual Intervention Reasoning ← L3-3



Fire axis (T1 easiest → T5 hardest; numbered on the same controlled-simulation figure Acc as P; T2/T3 swapped vs 2026-08-30):

- T1 Fire Evolution Prediction ← L1-2, L3-1, L3-2

- T2 Fire State Assessment ← L1-3, L2-1, L2-2

- T3 Fire Early Warning ← L1-1

- T4 Fire Intervention Decision-Making ← L3-3

- T5 Fire Mechanism Diagnosis ← L2-3



Real-world C06 formal test: **714** questions (357 choice + 357 open). The older 760 count included 46 L2-2 S-track items whose public history ended before the query target time; those items are not in the formal set.

Labels are stored on each `questions.jsonl` / `gold.jsonl` row as `physical_axis` and `fire_axis`. Scoring uses `--group-by physical` or `--group-by fire`.

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

## Three-stage fire axis and event families

The reproducible relabeling/audit script is `scripts/relabel_event_families.py`,
with its frozen contract in `scripts/contracts/three_stage_event_family.json`.
The three fire-axis labels are `T1 Localized Onset` (局域起火),
`T2 Coupled Propagation` (耦合蔓延), and `T3 Critical Transition` (临界转变).
They are assigned from the question's task semantics and
answer target, independently of `physical_axis`; `physical_axis` is copied
unchanged. The script also builds a stable event-to-family manifest using the
seven fixed paper families and reports unmatched records.

Example:

```shell
python scripts/relabel_event_families.py \
  --gold <testResult gold jsonl files> \
  --questions <benchmark question jsonl files> \
  --item-scores <official per-item score jsonl files> \
  --output <output directory>
```

The output includes relabeled JSONL, `event_family_manifest.json`,
`metrics_by_event_family.csv`, `metric_items.jsonl`, and `audit.json`.

## Citation

(Paper BibTeX to be provided by the authors.)
