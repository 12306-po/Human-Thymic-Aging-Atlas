# Deploying the T05 research predictor

The three files in `release/age_prediction_v1` are held-out **evaluation**
results. They are not a model. The website's `Predict new sample` tab remains
disabled until `release/age_prediction_model_v1/frozen_t05_model.json` exists
and passes its schema check.

## 1. Reproduce T05C and freeze a final model on the analysis server

Run from the checkout that contains the T05B and T05C release directories:

```bash
REPO=/data/zxy/projects/human_thymus_age_ML_DL/Human-Thymic-Aging-Atlas
PYTHON=/data/zxy/environments/micromamba_envs/human_thymus_age_ml_dl/bin/python

"$PYTHON" "$REPO/freeze_t05_web_model.py" --verify-only
```

Review the fold-selection and prediction checks plus the full-training
candidate scores. Only then rerun without `--verify-only` to write the model.

The script expects these inputs by default:

- `release/transformer_annotation_v1/t05b_donor_age_features_v1/X_scgpt_global_plus_composition.tsv.gz`
- `release/transformer_annotation_v1/t05b_donor_age_features_v1/locked_age_outcome.tsv`
- `release/transformer_annotation_v1/t05c_nested_lodo_age_prediction_v1/primary_model_oof_predictions.tsv`

If actual filenames differ, pass `--features`, `--outcomes` and `--oof` with
their exact paths. The inputs must contain the same 18 donors. The script
reconstructs inner leave-one-donor-out tuning using mean absolute error and an
ordered candidate grid of `100, 1000` by default. These are the two values
observed in the available T05C fold selections; the original full candidate
grid was not preserved in the release directory. It **refuses to export**
unless the reconstructed tuning selects the recorded alpha in every outer fold
and the resulting fold-local StandardScaler/Ridge predictions match T05C within
0.001 years. If either check fails, obtain the original T05C grid/scoring rule
and pass the verified candidates with `--alpha-grid`; do not bypass the checks.
After all 18 folds pass, the same inner LODO-MAE rule selects the final alpha
on all 18 donors. The exported JSON records the grid and scores.

On success the script writes one JSON model to the new
`release/age_prediction_model_v1` directory. It refuses to overwrite a
nonempty model release. Review the source hashes and reproduction difference
before publishing it.

## 2. Publish the reviewed model and website code together

Synchronize `frozen_t05_model.json` to the same path in the local website
checkout. Commit the model JSON, `app.py`, `research_extension.py`,
`age_prediction_inference.py`, and this documentation to the GitHub branch
used by Streamlit Community Cloud. Keep unrelated project changes out of
that commit. The model JSON contains coefficients but no uploaded samples or
single-cell count matrices.

Cloud users upload only a one-row 542-feature TSV generated with the same
frozen T05B feature pipeline. The app processes it in memory for that session
and does not write the uploaded values to the repository or server disk.
