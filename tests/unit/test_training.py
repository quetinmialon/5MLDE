import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from bank_marketing.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from bank_marketing.training import load_training_data, split_dataset, train_model
from bank_marketing.validation import DataValidationError


def test_sequential_partitions_are_fixed_and_disjoint(training_setup):
    _, csv_path, _ = training_setup
    frame, _ = load_training_data(csv_path)
    partitions = split_dataset(frame)
    assert partitions["train"].index.tolist() == list(range(28))
    assert partitions["validation"].index.tolist() == list(range(28, 34))
    assert partitions["test"].index.tolist() == list(range(34, 40))
    pd.testing.assert_frame_equal(pd.concat(partitions.values()), frame)


@pytest.mark.parametrize("train_fraction,validation_fraction", [
    (0, 0.15), (1, 0.15), (0.7, 0), (0.7, 0.4),
    (float("nan"), 0.15), (True, 0.15), ("0.7", 0.15),
])
def test_invalid_partition_fractions_are_rejected(train_fraction, validation_fraction, training_setup):
    _, csv_path, _ = training_setup
    frame, _ = load_training_data(csv_path)
    with pytest.raises(ValueError):
        split_dataset(frame, train_fraction=train_fraction, validation_fraction=validation_fraction)


@pytest.mark.parametrize("name,start,end", [("train", 0, 28), ("validation", 28, 34), ("test", 34, 40)])
def test_single_class_partition_is_rejected(name, start, end, training_setup):
    _, csv_path, _ = training_setup
    frame, _ = load_training_data(csv_path)
    frame.loc[start:end - 1, "y"] = "no"
    with pytest.raises(ValueError, match=name):
        split_dataset(frame)


def test_too_small_dataset_is_rejected(training_setup):
    _, csv_path, _ = training_setup
    frame, _ = load_training_data(csv_path)
    with pytest.raises(ValueError, match="vide"):
        split_dataset(frame.iloc[:1])


def test_invalid_raw_data_stops_before_parsing_or_fitting(monkeypatch, training_setup, make_csv):
    config, csv_path, rows = training_setup
    csv_path.write_bytes(make_csv([{**rows[0], "age": "invalid"}]))

    def unexpected_call(*args, **kwargs):
        pytest.fail("Le parsing et l'entraînement ne doivent pas commencer.")

    monkeypatch.setattr("bank_marketing.training.pd.read_csv", unexpected_call)
    monkeypatch.setattr("bank_marketing.training.build_pipeline", unexpected_call)
    with pytest.raises(DataValidationError):
        train_model(config)
    assert not (config.parent / "models").exists()


def test_training_uses_only_train_statistics_and_categories(training_setup):
    config, csv_path, _ = training_setup
    run_dir = train_model(config)
    model = joblib.load(run_dir / "pipeline.joblib")
    frame, _ = load_training_data(csv_path)
    preprocessor = model.named_steps["preprocessing"]
    means = preprocessor.named_transformers_["numeric"].mean_
    expected = frame.iloc[:28][NUMERIC_FEATURES].mean().to_numpy()
    np.testing.assert_allclose(means, expected)
    assert means[0] != frame["age"].mean()
    categories = preprocessor.named_transformers_["categorical"].categories_
    assert "retired" not in categories[CATEGORICAL_FEATURES.index("job")]
    assert np.isfinite(model.predict_proba(frame.iloc[34:])).all()
    np.testing.assert_allclose(preprocessor.named_transformers_["numeric"].mean_, expected)


def test_artifacts_trace_input_and_do_not_evaluate_test_by_default(training_setup):
    config, csv_path, _ = training_setup
    run_dir = train_model(config)
    report = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert report["dataset"]["sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert report["splits"]["test"]["start_inclusive"] == 34
    assert report["splits"]["test"]["end_exclusive"] == 40
    assert report["excluded_columns"] == ["duration", "y"]
    assert report["test_evaluated"] is False
    for model in ("logistic_regression", "dummy"):
        assert set(report["metrics"][model]) == {"train", "validation"}
    assert report["metrics"]["dummy"]["validation"]["roc_auc"] == 0.5
    assert report["environment"]["scikit-learn"]


def test_test_evaluation_requires_explicit_request(training_setup):
    config, _, _ = training_setup
    run_dir = train_model(config, evaluate_test=True)
    report = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert report["test_evaluated"]
    assert report["metrics"]["logistic_regression"]["test"]["row_count"] == 6


def test_nonconverged_fit_does_not_publish_model(training_setup):
    config, _, _ = training_setup
    config.write_text(
        config.read_text(encoding="utf-8").replace("max_iter = 1000", "max_iter = 1"), encoding="utf-8",
    )
    with pytest.raises(ConvergenceWarning):
        train_model(config)
    assert not (config.parent / "models").exists()


def test_multiple_runs_preserve_previous_artifacts(training_setup):
    config, _, _ = training_setup
    first = train_model(config)
    first_content = (first / "pipeline.joblib").read_bytes()
    second = train_model(config)
    assert first != second
    assert (first / "pipeline.joblib").read_bytes() == first_content
