import pytest

from bank_marketing.settings import load_tracking_config
from bank_marketing.registry import change_champion
from bank_marketing.tracking import validation_decision


@pytest.mark.parametrize("candidate,baseline,gain,expected", [
    (0.4, 0.2, 0.0, True), (0.2, 0.2, 0.0, True),
    (0.1, 0.2, 0.0, False), (0.4, 0.2, 0.3, False),
])
def test_validation_rule_is_relative(candidate, baseline, gain, expected):
    assert validation_decision(candidate, baseline, gain)["passed"] is expected


@pytest.mark.parametrize("candidate,baseline,gain", [
    (float("nan"), 0.2, 0.0), (0.3, float("inf"), 0.0),
    (1.1, 0.2, 0.0), (0.3, -0.1, 0.0), (0.3, 0.2, -0.1),
])
def test_invalid_scores_are_not_accepted(candidate, baseline, gain):
    with pytest.raises(ValueError):
        validation_decision(candidate, baseline, gain)


@pytest.mark.parametrize("uri", ["file:///tmp/mlruns", "http://user:secret@localhost:5000", "invalid"])
def test_tracking_rejects_invalid_uri_or_embedded_credentials(uri, tracking_setup, monkeypatch):
    config, _, _, _, _ = tracking_setup
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    with pytest.raises(ValueError):
        load_tracking_config(config)


def test_tracking_uri_can_be_overridden_by_environment(tracking_setup, monkeypatch):
    config, _, _, _, _ = tracking_setup
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    assert load_tracking_config(config)["mlflow"]["tracking_uri"] == "http://127.0.0.1:5001"


def test_registry_requires_a_reason_before_contacting_the_server():
    with pytest.raises(ValueError, match="raison"):
        change_champion("missing.toml", version="1", reason=" ")
