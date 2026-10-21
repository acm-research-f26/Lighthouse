"""Validate, combine, and score Lighthouse screen predictions.

Run with ``python -m bn6.lighthouse_merge --help`` from the repository root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


# This order also defines the prescribed rating tie-break.
CATEGORIES = (
    "confirmshaming", "hidden_information", "preselection",
    "scarcity", "urgency", "social_proof",
)
LABELS = (*CATEGORIES, "none")
DEFAULT_KEY = Path("cohort_ocr_v2.csv")


def image_key(value: str) -> str:
    """Normalize paths/case, retaining extensions to avoid ambiguous matches."""
    return str(value).strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()


def label_key(value: str) -> str:
    return "_".join(str(value).strip().lower().split())


def read_table(file: str | Path | pd.DataFrame) -> pd.DataFrame:
    table = file.copy() if isinstance(file, pd.DataFrame) else pd.read_csv(
        file, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    table.columns = [label_key(c) for c in table.columns]
    if table.columns.duplicated().any():
        raise ValueError("Duplicate column names after normalization")
    if "image" not in table:
        raise ValueError("Missing required column: image")
    if len(table) == 0:
        raise ValueError("Input has no screens")
    if table["image"].isna().any():
        raise ValueError("Missing image identifier")
    table["image"] = table["image"].astype(str).str.strip()
    table["_key"] = table["image"].map(image_key)
    if table["_key"].eq("").any() or table["_key"].duplicated().any():
        raise ValueError("Missing or duplicate screen identifiers after normalization")
    return table


def read_answer_key(file: str | Path | pd.DataFrame = DEFAULT_KEY) -> pd.DataFrame:
    table = read_table(file)
    if "label" not in table or table["label"].isna().any():
        raise ValueError("Answer key needs a label for every screen")
    table["label"] = table["label"].map(label_key)
    unknown = set(table["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown true labels: {sorted(unknown)}")
    return table[["image", "label", "_key"]].sort_values("_key").reset_index(drop=True)


def align(table: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Require the complete cohort; never silently discard unmatched screens."""
    missing = set(key["_key"]) - set(table["_key"])
    extra = set(table["_key"]) - set(key["_key"])
    if missing or extra:
        raise ValueError(f"Screen mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    return table.set_index("_key").loc[key["_key"]].reset_index()


def read_ratings(file: str | Path | pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    table = align(read_table(file), key)
    required = {"label", "predicted", *CATEGORIES}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Ratings missing columns: {sorted(missing)}")
    if table[list(required)].isna().any().any():
        raise ValueError("Ratings contain missing values")
    table["label"] = table["label"].map(label_key)
    if not table["label"].eq(key["label"]).all():
        raise ValueError("Ratings labels disagree with the answer key")
    for category in CATEGORIES:
        values = pd.to_numeric(table[category], errors="coerce")
        if not (values.isin([0, 1, 2, 3])).all():
            raise ValueError(f"{category}: ratings must be integers from 0 to 3")
        table[category] = values.astype(int)
    ratings = table[list(CATEGORIES)].to_numpy()
    strengths = ratings.max(axis=1)
    derived = np.asarray(CATEGORIES, dtype=object)[ratings.argmax(axis=1)]
    derived[strengths == 0] = "none"
    table["predicted"] = table["predicted"].map(label_key)
    if not np.array_equal(table["predicted"], derived):
        raise ValueError("Predicted labels disagree with ratings or the prescribed tie-break")
    if "strength" in table and not np.array_equal(
        pd.to_numeric(table["strength"], errors="coerce"), strengths
    ):
        raise ValueError("Strength must equal the maximum of the six ratings")
    table["strength"] = strengths
    return table


def build_merge_table(image_text, text_only, answer_key=DEFAULT_KEY) -> pd.DataFrame:
    key = read_answer_key(answer_key)
    vision = read_ratings(image_text, key)
    text = read_ratings(text_only, key)
    table = key[["image", "label"]].copy()
    for prefix, source in (("image_text", vision), ("text_only", text)):
        for column in ("predicted", "strength", *CATEGORIES):
            table[f"{prefix}_{column}"] = source[column].to_numpy()
    return table


def threshold_predict(table: pd.DataFrame, a: int, b: int) -> np.ndarray:
    if a not in range(4) or b not in range(4):
        raise ValueError("A and B must each be 0, 1, 2, or 3")
    suppress = (table["image_text_strength"].to_numpy() <= a) & (
        table["text_only_strength"].to_numpy() <= b
    )
    return np.where(suppress, "none", table["image_text_predicted"].to_numpy())


def choose_thresholds(table: pd.DataFrame) -> tuple[int, int, float]:
    """Fit only on the supplied training rows; break accuracy ties by A, then B."""
    candidates = [
        (float(np.mean(threshold_predict(table, a, b) == table["label"])), a, b)
        for a in range(4) for b in range(4)
    ]
    accuracy, a, b = min(candidates, key=lambda item: (-item[0], item[1], item[2]))
    return a, b, accuracy


def cross_validate(table: pd.DataFrame, seed: int = 42):
    """Produce one held-out prediction per screen for each merge method."""
    counts = table["label"].value_counts()
    if len(counts) < 2 or counts.min() < 5:
        raise ValueError("Five stratified folds need at least two classes and five screens per present class")
    features = [f"{prefix}_{c}" for prefix in ("image_text", "text_only") for c in CATEGORIES]
    x = table[features].to_numpy(dtype=float)
    y = table["label"].to_numpy()
    rule = np.empty(len(table), dtype=object)
    logistic = np.empty(len(table), dtype=object)
    folds = np.zeros(len(table), dtype=int)
    details = []
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(split.split(x, y), start=1):
        a, b, train_accuracy = choose_thresholds(table.iloc[train])
        rule[test] = threshold_predict(table.iloc[test], a, b)
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x[train], y[train])
        logistic[test] = model.predict(x[test])
        folds[test] = fold
        details.append({
            "fold": fold, "train_n": len(train), "test_n": len(test),
            "A": a, "B": b, "threshold_train_accuracy": train_accuracy,
            "threshold_test_accuracy": float(np.mean(rule[test] == y[test])),
            "logistic_test_accuracy": float(np.mean(logistic[test] == y[test])),
        })
    merged = table[["image", "label"]].copy()
    merged["threshold_predicted"] = rule
    merged["logistic_predicted"] = logistic
    assignments = table[["image"]].copy()
    assignments["fold"] = folds
    return merged, assignments, details


def _metrics(truth: np.ndarray, predicted: np.ndarray) -> dict:
    # Unknown/failed predictions occupy a separate column and count as errors.
    class_count = len(LABELS)
    matrix = confusion_matrix(truth, predicted, labels=range(class_count + 1))[:class_count]
    correct = np.diag(matrix[:, :class_count])
    actual = matrix.sum(axis=1)
    called = matrix[:, :class_count].sum(axis=0)
    f1 = np.divide(2 * correct, actual + called, out=np.zeros(class_count), where=actual + called != 0)
    honest = LABELS.index("none")
    honest_n = actual[honest]
    honest_wrong = matrix[honest].sum() - matrix[honest, honest]
    # An invalid prediction on an honest screen is conservatively a false alarm.
    return {
        "accuracy": float(correct.sum() / len(truth)),
        "macro_f1": float(f1.mean()),
        "honest_wrongly_flagged": int(honest_wrong),
        "honest_false_alarm_rate": float(honest_wrong / honest_n) if honest_n else np.nan,
        **{f"f1_{label}": float(value) for label, value in zip(LABELS, f1)},
    }


def evaluate(truth, predicted, n_bootstrap: int = 1000, seed: int = 42) -> dict:
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if len(truth) == 0 or len(truth) != len(predicted):
        raise ValueError("Truth and predictions must have the same nonzero length")
    lookup = {label: i for i, label in enumerate(LABELS)}
    if set(truth) - set(LABELS):
        raise ValueError("Unknown true labels")
    y = np.array([lookup[label] for label in truth])
    failed = len(LABELS)
    honest = lookup["none"]
    p = np.array([lookup.get(label, failed) for label in predicted])
    metrics = _metrics(y, p)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        sample = rng.integers(0, len(y), size=len(y))
        draws.append(_metrics(y[sample], p[sample]))
    draws = pd.DataFrame(draws)
    intervals = {}
    for name in draws:
        values = draws[name].dropna().to_numpy()
        intervals[name] = np.quantile(values, [0.025, 0.975]).tolist() if len(values) else None
    return {
        "n": len(y), "correct": int((y == p).sum()),
        "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
        "honest_n": int((y == honest).sum()),
        "honest_wrongly_flagged": metrics["honest_wrongly_flagged"],
        "honest_false_alarm_rate": (metrics["honest_false_alarm_rate"]
                                    if np.isfinite(metrics["honest_false_alarm_rate"]) else None),
        "invalid_predictions": int((p == failed).sum()),
        "per_category": {label: {"f1": metrics[f"f1_{label}"], "support": int((y == i).sum())}
                         for i, label in enumerate(LABELS)},
        "bootstrap": {"resamples": n_bootstrap, "seed": seed,
                      "method": "screen-level percentile", "confidence": 0.95,
                      "intervals": intervals},
    }


def score(file, answer_key=DEFAULT_KEY, prediction_column="predicted", n_bootstrap=1000, seed=42):
    """Score against the authoritative key, ignoring any embedded input labels."""
    key = read_answer_key(answer_key)
    predictions = align(read_table(file), key)
    if prediction_column not in predictions:
        raise ValueError(f"Missing prediction column: {prediction_column}")
    predicted = predictions[prediction_column].fillna("").map(label_key).to_numpy()
    report = evaluate(key["label"], predicted, n_bootstrap, seed)
    if "label" in predictions:
        report["input_label_disagreements"] = int(
            (~predictions["label"].fillna("").map(label_key).eq(key["label"])).sum()
        )
    return report


def run_merge(image_text, text_only, answer_key=DEFAULT_KEY, output_dir="bn6/outputs", seed=42, n_bootstrap=1000):
    table = build_merge_table(image_text, text_only, answer_key)
    merged, assignments, fold_details = cross_validate(table, seed)
    reports = {
        "image_text": evaluate(table["label"], table["image_text_predicted"], n_bootstrap, seed),
        "text_only": evaluate(table["label"], table["text_only_predicted"], n_bootstrap, seed),
        "threshold": score(merged, answer_key, "threshold_predicted", n_bootstrap, seed),
        "logistic": score(merged, answer_key, "logistic_predicted", n_bootstrap, seed),
    }
    winner = ("tie" if reports["threshold"]["correct"] == reports["logistic"]["correct"]
              else max(("threshold", "logistic"), key=lambda name: reports[name]["accuracy"]))
    report = {
        "seed": seed, "folds": fold_details, "selection_metric": "accuracy",
        "threshold_tie_break": "lowest A, then lowest B",
        "winner_by_oof_accuracy": winner, "scores": reports,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Display table excludes identifiers and metadata; row order matches merged.csv.
    table.drop(columns="image").to_csv(output / "merge_table.csv", index=False)
    merged.to_csv(output / "merged.csv", index=False)
    assignments.to_csv(output / "folds.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    merge_parser = commands.add_parser("merge", help="Run both merge methods with the same five folds")
    merge_parser.add_argument("--image-text", required=True, type=Path)
    merge_parser.add_argument("--text-only", required=True, type=Path)
    merge_parser.add_argument("--output-dir", type=Path, default=Path("bn6/outputs"))
    score_parser = commands.add_parser("score", help="Score one prediction column against the cohort key")
    score_parser.add_argument("file", type=Path)
    score_parser.add_argument("--prediction-column", default="predicted")
    score_parser.add_argument("--output", type=Path)
    for sub in (merge_parser, score_parser):
        sub.add_argument("--answer-key", type=Path, default=DEFAULT_KEY)
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    try:
        if args.command == "merge":
            report = run_merge(args.image_text, args.text_only, args.answer_key,
                               args.output_dir, args.seed, args.bootstrap)
        else:
            report = score(args.file, args.answer_key, args.prediction_column, args.bootstrap, args.seed)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, OSError, pd.errors.ParserError, ConvergenceWarning) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
