import joblib
import numpy as np

from bank_marketing.preprocessing import FEATURE_COLUMNS, NUMERIC_FEATURES
from bank_marketing.training import build_pipeline, load_training_data, split_dataset


def test_pipeline_roundtrip_preserves_probabilities(tmp_path, training_setup):
    _, csv_path, _ = training_setup
    frame, _ = load_training_data(csv_path)
    partitions = split_dataset(frame)
    model = build_pipeline()
    model.fit(
        partitions["train"][FEATURE_COLUMNS],
        partitions["train"]["y"].map({"no": 0, "yes": 1}),
    )
    expected = model.predict_proba(partitions["test"][FEATURE_COLUMNS])
    model_path = tmp_path / "pipeline.joblib"
    joblib.dump(model, model_path)
    restored = joblib.load(model_path)
    np.testing.assert_allclose(restored.predict_proba(partitions["test"][FEATURE_COLUMNS]), expected)
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(expected.sum(axis=1), 1)


def test_target_and_call_duration_do_not_change_predictions(training_setup):
    _, csv_path, _ = training_setup
    frame, _ = load_training_data(csv_path)
    model = build_pipeline()
    model.fit(frame[FEATURE_COLUMNS], frame["y"].map({"no": 0, "yes": 1}))
    modified = frame.assign(duration=999999, y="irrelevant")
    np.testing.assert_allclose(model.predict_proba(frame), model.predict_proba(modified))
    assert "duration" not in model.feature_names_in_
    assert "y" not in model.feature_names_in_
    assert len(model.feature_names_in_) == 15
    assert len(NUMERIC_FEATURES) == 6
