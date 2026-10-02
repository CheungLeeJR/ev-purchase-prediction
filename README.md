# EV Purchase Prediction — Leakage-Safe Ensemble

A research-oriented Kaggle binary-classification project for EV purchase prediction using competition-aware feature engineering, stratified out-of-fold validation, LightGBM, optional CatBoost, nested target encoding, and OOF-only blend selection.

## What this project demonstrates

- Strict train/test/submission schema validation before modelling
- Competition-specific feature engineering without label access
- **Stratified OOF** evaluation with configurable fold count and random seed
- Fold-local preprocessing to prevent validation leakage
- **Nested cross-fitted target encoding** inside each outer training fold
- LightGBM + CatBoost model diversity
- OOF-only probability/rank blending rather than leaderboard-tuned weights
- Hard validation of generated submission files
- Reproducibility metadata including package versions and SHA-256 input fingerprints
- Unit-tested reusable validation, feature-engineering, and target-encoding helpers

## Verified evidence

The supplied project history supports a valid public Kaggle score of **0.94156** for the earlier stable pipeline. The stronger V2 design is the research implementation maintained here, but the supplied artifacts do **not** contain a verified leaderboard result for that stronger version. This repository therefore does not claim an improvement beyond the supported historical result.

See `MODEL_CARD.md` and `RESULTS.md` for the evidence boundary.

## Leakage-control design

For every outer CV fold:

1. validation rows are held out completely;
2. numeric imputation statistics come only from the outer-training partition;
3. categorical dictionaries are learned only from training data;
4. target encoding for training rows is generated with inner cross-fitting;
5. validation/test target encodings use mappings fitted only on outer-training labels;
6. ensemble weights are selected using genuine OOF AUC only.

## Repository layout

```text
.
├── src/
│   ├── train.py                 # full competition/research runner
│   ├── pipeline.py              # tested schema, feature and nested-TE helpers
│   └── reproducibility.py       # environment and input-fingerprint utilities
├── tests/
│   ├── test_pipeline.py
│   └── test_reproducibility.py
├── artifacts/audit/             # aggregate, non-row-level dataset audits
├── RESEARCH_PROTOCOL.md
├── REPRODUCIBILITY.md
├── RESULTS.md
├── MODEL_CARD.md
└── requirements.txt
```

The standalone repository intentionally uses the script-based pipeline as the canonical implementation. Historical exploratory notebooks from the original competition work are not required for reproduction of the maintained research pipeline.

## Run

Attach or place the competition dataset containing `train.csv`, `test.csv`, and `sample_submission.csv`, then run:

```bash
python src/train.py
```

The runner defaults to `/kaggle/input` and `/kaggle/working`. For local use:

```bash
KAGGLE_INPUT_ROOT=/path/to/input OUTPUT_DIR=./outputs python src/train.py
```

For an explicitly recorded experiment configuration:

```bash
EXPERIMENT_SEED=42 CV_FOLDS=5 \
KAGGLE_INPUT_ROOT=/path/to/input OUTPUT_DIR=./outputs \
python src/train.py
```

## Test the reusable core

```bash
pip install numpy pandas scikit-learn pytest
python -m pytest -q
```

The tests use synthetic frames and do not require the competition dataset.

## Generated outputs

A full run writes outputs including:

- `submission_best.csv`
- `submission_safe.csv`
- model-specific submissions
- `oof_predictions.csv`
- `model_comparison.csv`
- `run_summary.txt`
- `run_metadata.json`

Generated submissions and row-level OOF files are excluded from Git by default.

## Research documents

- Experimental questions and ablations: [`RESEARCH_PROTOCOL.md`](./RESEARCH_PROTOCOL.md)
- Reproduction controls and run metadata: [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md)
- Verified-vs-pending evidence ledger: [`RESULTS.md`](./RESULTS.md)
- Intended-use and limitation boundary: [`MODEL_CARD.md`](./MODEL_CARD.md)

## Current limitations

- Competition data may contain synthetic or competition-specific patterns that do not transfer to real EV-adoption behavior.
- A leaderboard score is not evidence of causal validity or real-world calibration.
- The stronger V2 implementation should be rerun and archived with `run_metadata.json` before any new performance claim is made.

## Next research steps

- Re-run V2 and archive reproducible OOF and external-evaluation evidence
- Add repeated-CV stability analysis
- Add calibration/Brier-score diagnostics
- Add feature-importance stability or SHAP analysis
- Compare ablations using matched folds/seeds rather than leaderboard feedback
