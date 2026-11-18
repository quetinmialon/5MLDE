import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd


def test_training_command_and_reloaded_model(training_setup, tmp_path):
    config, _, rows = training_setup
    result = subprocess.run(
        [sys.executable, "-I", "-m", "bank_marketing.training", "--config", str(config), "--evaluate-test"],
        cwd=tmp_path, capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    runs = list((tmp_path / "models").iterdir())
    assert len(runs) == 1
    report = json.loads((runs[0] / "metrics.json").read_text(encoding="utf-8"))
    assert report["test_evaluated"]
    model = joblib.load(runs[0] / "pipeline.joblib")
    prediction = model.predict_proba(pd.DataFrame([rows[-1]]))
    assert prediction.shape == (1, 2)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction.sum(axis=1), 1)


def test_invalid_training_command_reports_error_without_artifacts(training_setup, tmp_path):
    config, csv_path, _ = training_setup
    csv_path.write_bytes(b"invalid data")
    result = subprocess.run(
        [sys.executable, "-I", "-m", "bank_marketing.training", "--config", str(config)],
        cwd=tmp_path, capture_output=True,
    )
    assert result.returncode == 1
    assert b"Traceback" not in result.stderr
    assert not (tmp_path / "models").exists()
