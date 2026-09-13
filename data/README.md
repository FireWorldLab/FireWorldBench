# Data

The formal-test dataset is published on Hugging Face and downloaded via `scripts/download.sh`
(which outputs into this `data/` directory):

- `data/main_synthetic/full_test_A/`   — synthetic fire events (8,360 questions)
- `data/mmodalfire_c06/full_test_A/`   — real-world C06 events (714 questions)

Each question carries `physical_axis` (P1–P5) and `fire_axis` (T1–T3) labels matching the paper.
The data itself is not committed to this repository; see `scripts/download.sh`.
