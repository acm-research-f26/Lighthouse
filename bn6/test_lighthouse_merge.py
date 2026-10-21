"""Synthetic tests for screen matching, fold isolation, and scoring."""

import json
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from bn6 import lighthouse_merge as lm


@pytest.fixture
def inputs():
    # Synthetic only: enough examples per class for five stratified folds.
    labels = list(lm.LABELS) * 10
    key = pd.DataFrame({"image": [f"screen_{i:03}.png" for i in range(len(labels))], "label": labels})
    sources = []
    for offset in (1, 3):
        frame = key.copy()
        scores = np.zeros((len(key), 6), dtype=int)
        for i, label in enumerate(labels):
            if label != "none":
                scores[i, lm.CATEGORIES.index(label)] = 1 + i % 3
            if i % (offset + 4) == 0:
                scores[i, (i + offset) % 6] = 3
        predicted = np.asarray(lm.CATEGORIES, dtype=object)[scores.argmax(axis=1)]
        predicted[scores.max(axis=1) == 0] = "none"
        frame["predicted"] = predicted
        frame["strength"] = scores.max(axis=1)
        frame["collector"] = "should not be a feature"
        frame["image_size"] = "100x100"
        frame["filename"] = "unused.png"
        for i, category in enumerate(lm.CATEGORIES):
            frame[category] = scores[:, i]
        sources.append(frame)
    return key, *sources


def test_join_uses_screen_identity_and_preserves_ratings(inputs):
    key, vision, text = inputs
    reordered = text.sample(frac=1, random_state=7)
    reordered["image"] = "C:\\screens\\" + reordered["image"].str.upper()
    actual = lm.build_merge_table(vision, reordered, key)
    expected = lm.build_merge_table(vision, text, key)
    pd.testing.assert_frame_equal(actual, expected)
    assert not {"filename", "collector", "image_size"} & set(actual.columns)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "extra", "wrong_label", "bad_rating", "bad_strength", "bad_prediction"])
def test_bad_ratings_fail_instead_of_dropping_rows(inputs, problem):
    key, vision, text = inputs
    if problem == "duplicate":
        text = pd.concat([text, text.iloc[[0]]], ignore_index=True)
    elif problem == "missing":
        text = text.iloc[:-1]
    elif problem == "extra":
        text = pd.concat([text, text.iloc[[0]].assign(image="extra.png")], ignore_index=True)
    elif problem == "wrong_label":
        text.loc[0, "label"] = "none"
    elif problem == "bad_rating":
        text[lm.CATEGORIES[0]] = text[lm.CATEGORIES[0]].astype(float)
        text.loc[0, lm.CATEGORIES[0]] = 1.5
    elif problem == "bad_strength":
        text.loc[0, "strength"] = -1
    elif problem == "bad_prediction":
        text.loc[0, "predicted"] = "none"
    with pytest.raises(ValueError):
        lm.build_merge_table(vision, text, key)


def test_rating_ties_and_zero_strength(inputs):
    key, vision, _ = inputs
    vision.loc[0, list(lm.CATEGORIES)] = [2, 2, 2, 2, 2, 2]
    vision.loc[0, "predicted"] = "confirmshaming"
    vision.loc[0, "strength"] = 2
    vision.loc[1, list(lm.CATEGORIES)] = 0
    vision.loc[1, "predicted"] = "none"
    vision.loc[1, "strength"] = 0
    result = lm.read_ratings(vision.drop(columns="strength"), lm.read_answer_key(key))
    assert result.loc[0, "strength"] == 2
    assert result.loc[1, "strength"] == 0


def test_threshold_requires_both_inclusive_conditions():
    table = pd.DataFrame({
        "image_text_strength": [1, 1, 2, 0], "text_only_strength": [0, 1, 0, 0],
        "image_text_predicted": ["urgency", "scarcity", "preselection", "none"],
    })
    assert lm.threshold_predict(table, 1, 0).tolist() == ["none", "scarcity", "preselection", "none"]


def test_threshold_search_checks_every_pair_and_has_stable_ties():
    table = pd.DataFrame({"image_text_strength": [1, 2], "text_only_strength": [0, 1],
                          "image_text_predicted": ["scarcity", "urgency"], "label": ["none", "urgency"]})
    assert lm.choose_thresholds(table) == (1, 0, 1.0)


def test_shared_folds_and_no_heldout_rows_used_for_fitting(inputs, monkeypatch):
    key, vision, text = inputs
    table = lm.build_merge_table(vision, text, key)
    original_choose = lm.choose_thresholds
    original_model = lm.LogisticRegression
    training_rows, feature_sets = [], []
    rule_inputs = []
    original_predict = lm.threshold_predict

    def choose(train):
        training_rows.append(set(train.index))
        return original_choose(train)

    def predict(frame, a, b):
        rule_inputs.append(set(frame.index))
        return original_predict(frame, a, b)

    class CheckedModel:
        def __init__(self, **kwargs):
            self.model = original_model(**kwargs)

        def fit(self, x, y):
            expected = table.loc[sorted(training_rows[-1])]
            features = [f"{p}_{c}" for p in ("image_text", "text_only") for c in lm.CATEGORIES]
            np.testing.assert_array_equal(x, expected[features].to_numpy())
            np.testing.assert_array_equal(y, expected["label"].to_numpy())
            feature_sets.append(x.shape[1])
            self.model.fit(x, y)

        def predict(self, x):
            return self.model.predict(x)

    monkeypatch.setattr(lm, "choose_thresholds", choose)
    monkeypatch.setattr(lm, "threshold_predict", predict)
    monkeypatch.setattr(lm, "LogisticRegression", CheckedModel)
    merged, assignments, details = lm.cross_validate(table)
    assert feature_sets == [12] * 5
    assert len(merged) == len(table)
    assert set(assignments["fold"]) == {1, 2, 3, 4, 5}
    for i, detail in enumerate(details):
        heldout = set(assignments.index[assignments["fold"] == detail["fold"]])
        assert heldout.isdisjoint(training_rows[i])
        assert training_rows[i] | heldout == set(table.index)
        # Each fold invokes the rule 16 times on train, then once on test.
        assert rule_inputs[i * 17:i * 17 + 16] == [training_rows[i]] * 16
        assert rule_inputs[i * 17 + 16] == heldout
    again, folds_again, _ = lm.cross_validate(table)
    pd.testing.assert_frame_equal(merged, again)
    pd.testing.assert_frame_equal(assignments, folds_again)


def test_score_uses_authoritative_key_and_counts_parse_failures(inputs):
    key, vision, _ = inputs
    predictions = key.copy()
    predictions["predicted"] = predictions["label"]
    predictions.loc[0, "predicted"] = "PARSE_FAIL"
    predictions.loc[6, "predicted"] = "urgency"
    predictions["label"] = "none"  # Deliberately wrong embedded labels.
    report = lm.score(predictions.sample(frac=1, random_state=4), key, n_bootstrap=30)
    assert report["correct"] == len(key) - 2
    assert report["honest_wrongly_flagged"] == 1
    assert report["honest_n"] == 10
    assert report["invalid_predictions"] == 1
    assert report["input_label_disagreements"] == 60
    assert report["accuracy"] == accuracy_score(key["label"], predictions["predicted"])
    assert report["macro_f1"] == pytest.approx(f1_score(
        key["label"], predictions["predicted"], labels=list(lm.LABELS), average="macro", zero_division=0
    ))


def test_bootstrap_matches_independent_percentile_calculation():
    truth = np.array(["none", "none", "urgency", "scarcity", "preselection"])
    predicted = np.array(["none", "urgency", "urgency", "none", "preselection"])
    report = lm.evaluate(truth, predicted, n_bootstrap=1000, seed=10)
    rng = np.random.default_rng(10)
    draws = []
    for _ in range(1000):
        indices = rng.integers(0, 5, size=5)
        draws.append(accuracy_score(truth[indices], predicted[indices]))
    assert report["bootstrap"]["intervals"]["accuracy"] == np.quantile(draws, [0.025, 0.975]).tolist()
    assert report == lm.evaluate(truth, predicted, n_bootstrap=1000, seed=10)


def test_no_honest_screens_and_absent_categories_have_defined_metrics():
    report = lm.evaluate(["urgency"] * 5, ["urgency"] * 5, n_bootstrap=20)
    assert report["accuracy"] == 1
    assert report["macro_f1"] == pytest.approx(1 / 7)
    assert report["honest_false_alarm_rate"] is None
    assert report["bootstrap"]["intervals"]["honest_false_alarm_rate"] is None


def test_outputs_are_out_of_fold_and_exclude_metadata(inputs, tmp_path):
    key, vision, text = inputs
    report = lm.run_merge(vision, text, key, tmp_path, n_bootstrap=30)
    merged = pd.read_csv(tmp_path / "merged.csv")
    assert merged.columns.tolist() == ["image", "label", "threshold_predicted", "logistic_predicted"]
    table = pd.read_csv(tmp_path / "merge_table.csv")
    assert not {"image", "filename", "collector", "image_size"} & set(table.columns)
    assert len(table) == len(key)
    assert report["scores"]["threshold"]["accuracy"] == np.mean(merged["label"] == merged["threshold_predicted"])
    assert json.loads((tmp_path / "report.json").read_text()) == report


def test_missing_or_duplicated_screen_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        lm.read_table(pd.DataFrame({"image": ["a.png", "C:\\images\\A.PNG"]}))
    with pytest.raises(ValueError, match="Missing"):
        lm.read_table(pd.DataFrame({"image": [None]}))


def test_insufficient_class_support_is_actionable(inputs):
    key, vision, text = inputs
    table = lm.build_merge_table(vision, text, key)
    with pytest.raises(ValueError, match="five screens"):
        lm.cross_validate(table.iloc[:14])


@pytest.mark.parametrize("correct,percentage", [(127, 70.6), (62, 34.4)])
def test_score_percentage_contract(correct, percentage):
    # Verify expected count-to-percentage conversion without inventing real predictions.
    key = pd.DataFrame({"image": [f"screen_{i}.png" for i in range(180)],
                        "label": ["none"] * 55 + ["urgency"] * 125})
    predictions = key.copy()
    predictions["predicted"] = predictions["label"]
    predictions.loc[correct:, "predicted"] = "scarcity"
    report = lm.score(predictions, key, n_bootstrap=10)
    assert report["correct"] == correct
    assert report["honest_n"] == 55
    assert round(report["accuracy"] * 100, 1) == percentage


def test_cli_help_is_available(monkeypatch, capsys):
    for command in ([], ["merge"], ["score"]):
        monkeypatch.setattr(sys, "argv", ["lighthouse_merge", *command, "--help"])
        with pytest.raises(SystemExit) as result:
            lm.main()
        assert result.value.code == 0
        assert "usage:" in capsys.readouterr().out
