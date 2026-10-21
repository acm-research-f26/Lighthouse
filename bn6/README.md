# Lighthouse BN6: merge and score

Compares two merge methods using the same five stratified folds.

## Run

Use Python 3.10+ and install `pip install -r bn6/requirements.txt`.
From the repository root:

```sh
python -m bn6.lighthouse_merge merge --image-text ratings.csv --text-only text_ratings.csv --answer-key cohort_ocr_v2.csv
python -m bn6.lighthouse_merge score preds.csv --answer-key cohort_ocr_v2.csv
python -m pytest bn6 -q
```

Inputs stay outside Git. Each ratings file needs `image`, `label`, `predicted`,
and six integer rating columns: `confirmshaming`, `hidden_information`,
`preselection`, `scarcity`, `urgency`, `social_proof`.
Spaces in category names are accepted. Optional `strength` must equal the
maximum rating; otherwise it is derived. Ties use the category order above;
all zeros means `none`. Metadata and justifications are ignored.
Duplicate/missing screens, invalid ratings, and conflicting labels stop the run.

## Methods

- **Threshold:** keep the image + text prediction unless its strength ≤ A
  AND text-only strength ≤ B, then predict `none`. Each fold tries all 16
  pairs on the other four folds; accuracy ties choose lowest A, then B.
- **Logistic regression:** fit only the 12 category ratings on those same four
  folds, then predict the fifth. Fixed L2 regularization, C=1, seed=42.

Every screen receives one held-out prediction from each method. Scores use
the answer key matched by image name: accuracy, seven-class macro F1,
per-category F1, and honest false alarms (out of 55 for the full cohort).
Invalid predictions count as errors, including false alarms on honest screens.
Default 95% intervals use 1,000 screen-level bootstrap resamples of fixed
predictions; they do not refit models or account for website clustering.
A new holdout is needed to independently evaluate the selected winner.

## Outputs

Saved in `bn6/outputs/` (override with `--output-dir`):

- `merged.csv`: image, true label, threshold prediction, logistic prediction.
- `merge_table.csv`: labels, predictions, strengths, and 12 ratings; no metadata/identifier.
- `folds.csv`: screen-to-fold assignments.
- `report.json`: fold thresholds, all scores/intervals, and winner by accuracy.

To score either merged column, add `--prediction-column threshold_predicted`
or `--prediction-column logistic_predicted`. Other options: `--seed`,
`--bootstrap`, and scoring-only `--output`.

## Verified run

The paired [ratings.csv](https://github.com/acm-research-f26/Lighthouse/blob/83bb2fea0a1e547bacf40dbb52fc39810e53eb1c/ratings.csv)
and [text_only.csv](https://github.com/acm-research-f26/Lighthouse/blob/83bb2fea0a1e547bacf40dbb52fc39810e53eb1c/text_only.csv)
contain all 180 screens and valid ratings. Their notebook uses Qwen3-VL-8B.
Use the shared [answer key](https://drive.google.com/file/d/1PudKmfS9hJp-LMGuR8g3vYO7p146bebF/view).

| Merge | Correct | Accuracy | Macro F1 | Honest false alarms / 55 |
| --- | --- | --- | --- | --- |
| Threshold | 99/180 | 55.0% | 0.556 | 7 |
| Logistic regression | 98/180 | 54.4% | 0.548 | 6 |

Thresholds were A=0, B=0 in every fold, retaining the VL prediction.
Neither merge improved the supplied VL baseline. The one-screen difference
is not decisive. These are real-data scores; the 21 unit tests use synthetic
fixtures solely to verify the implementation. Inputs and generated results
are excluded from Git.

[Assignment source](https://docs.google.com/document/d/19iP03Apuj336OwEmDcR4egBq_ei4Xprhp9uU0aaSwn8/edit)
