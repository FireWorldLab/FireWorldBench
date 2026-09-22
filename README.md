# FireWorldBench

FireWorldBench is a multimodal benchmark for evaluating fire-physics world understanding.

This repository contains evaluation code, metric implementations, task contracts, and schema definitions. Dataset distribution details and author-identifying links are intentionally omitted for anonymous review.

## Evaluation

Install the local package and run the evaluation pipeline after placing the benchmark data under `data/`:

```shell
pip install -e .
python scripts/run_eval.py --split main_synthetic --max-requests 2
python scripts/run_eval.py --split main_synthetic
```

Text-only models should use `--track S`; vision-language models may use `--track I`. Run `python scripts/run_eval.py --help` for the available options.

By default, results are written under `results/` and grouped by the physical-capability axis. Alternative groupings are available through `--group-by fire` and `--group-by task`.

## Benchmark axes

Physical capability axis:

- P1 Temporal Evolution Forecasting
- P2 Physical Field Perception and Grounding
- P3 Cross-Field Coupling Understanding
- P4 Causal Mechanism Attribution
- P5 Counterfactual Intervention Reasoning

Fire scenario task axis:

- T1 Localized Onset
- T2 Coupled Propagation
- T3 Critical Transition

The six reported metrics are Acc, F1, Brier, Evi-F1, Mech, and GLS. Acc, F1, and Brier apply to choice and open questions; the remaining metrics apply to open questions.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/run_eval.py` | Run inference and compute the six metrics |
| `scripts/run_api.py` | Resumable multimodal API runner |
| `scripts/score_six_metrics.py` | Deterministic metric computation |
| `scripts/score_fg9_*.py` | Scoring internals |
| `scripts/relabel_event_families.py` | Reproduce event-family labels and audits |
| `scripts/contracts/` | Frozen task contracts |

The repository does not include the benchmark dataset. Code paths that depend on unavailable data may not run in this anonymous-review snapshot.

## Anonymous review

Author names, affiliations, account identifiers, repository links, dataset-hosting links, acknowledgements, and citation metadata have been omitted to preserve double-blind review.
