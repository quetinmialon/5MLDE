import json
from pathlib import Path

import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

from bank_marketing.registry import change_champion, get_champion, load_champion
from bank_marketing.tracking import track_training
from bank_marketing.validation import DataValidationError


def test_tracked_run_contains_lineage_and_reloadable_model(tracking_setup):
    config, _, rows, client, _ = tracking_setup
    receipt = track_training(config)
    run = client.get_run(receipt["run_id"])
    assert run.info.status == "FINISHED"
    assert receipt["validation_status"] == "passed"
    assert run.data.params["classifier.C"] == "1.0"
    assert run.data.params["dummy.strategy"] == "prior"
    assert len(run.inputs.dataset_inputs) == 2
    assert "logistic_regression.validation.average_precision" in run.data.metrics
    assert not any(".test." in key for key in run.data.metrics)
    assert get_champion(client, receipt["model_name"]) is None
    assert len(receipt["validation_digest"]) == 64
    artifacts = {item.path for item in client.list_artifacts(run.info.run_id)}
    assert {"metrics.json", "configuration", "environment", "quality", "source", "validation"} <= artifacts
    source_files = client.list_artifacts(run.info.run_id, "source/bank_marketing")
    assert any(item.path.endswith("training.py") for item in source_files)
    assert not any(item.path.endswith(".env") for item in source_files)
    frame = pd.DataFrame(rows[:3])
    # La signature MLflow attend les mêmes types numériques que le pipeline validé.
    for column in ("age", "balance", "day", "campaign", "pdays", "previous"):
        frame[column] = frame[column].astype(float)
    native = mlflow.sklearn.load_model(receipt["model_uri"])
    pyfunc = mlflow.pyfunc.load_model(receipt["model_uri"])
    np.testing.assert_allclose(pyfunc.predict(frame), native.predict_proba(frame))


def test_two_versions_promotion_and_rollback_are_audited(tracking_setup):
    config, _, _, client, experiment_id = tracking_setup
    first = track_training(config)
    change_champion(config, version=first["model_version"], reason="Premier modèle de démonstration")
    second = track_training(config)
    assert str(get_champion(client, first["model_name"]).version) == first["model_version"]
    change_champion(config, version=second["model_version"], reason="Nouvelle version validée")
    _, active_version = load_champion(config)
    assert active_version == second["model_version"]
    decision = change_champion(config, version=first["model_version"], reason="Incident contrôlé", operation="rollback")
    assert decision["from_version"] == second["model_version"]
    assert decision["to_version"] == first["model_version"]
    _, active_version = load_champion(config)
    assert active_version == first["model_version"]
    audit = client.get_run(decision["audit_run_id"])
    assert audit.info.status == "FINISHED"
    path = client.download_artifacts(audit.info.run_id, "decision.json")
    assert json.loads(Path(path).read_text(encoding="utf-8"))["reason"] == "Incident contrôlé"
    assert len(client.search_runs([experiment_id])) == 5


def test_invalid_data_creates_failed_run_and_preserves_champion(tracking_setup, make_csv):
    config, csv_path, rows, client, experiment_id = tracking_setup
    first = track_training(config)
    change_champion(config, version=first["model_version"], reason="Modèle initial")
    csv_path.write_bytes(make_csv([{**rows[0], "day": "0"}]))
    with pytest.raises(DataValidationError):
        track_training(config)
    assert str(get_champion(client, first["model_name"]).version) == first["model_version"]
    failed = client.search_runs([experiment_id], filter_string="attributes.status = 'FAILED'")
    assert len(failed) == 1
    report_path = client.download_artifacts(failed[0].info.run_id, "quality/validation.json")
    assert not json.loads(Path(report_path).read_text(encoding="utf-8"))["valid"]
    versions = client.search_model_versions(f"name = '{first['model_name']}'")
    assert len(versions) == 1


def test_rejected_candidate_cannot_be_promoted(tracking_setup):
    config, _, _, client, _ = tracking_setup
    config.write_text(
        config.read_text(encoding="utf-8").replace("min_average_precision_gain = 0.0", "min_average_precision_gain = 1.0"),
        encoding="utf-8",
    )
    receipt = track_training(config)
    assert receipt["validation_status"] == "rejected"
    with pytest.raises(ValueError, match="validée"):
        change_champion(config, version=receipt["model_version"], reason="Test de refus")
    client.set_model_version_tag(receipt["model_name"], receipt["model_version"], "validation_status", "passed")
    with pytest.raises(ValueError, match="validée"):
        change_champion(config, version=receipt["model_version"], reason="Un tag seul ne suffit pas")
    assert get_champion(client, receipt["model_name"]) is None


def test_failed_source_run_cannot_be_promoted(tracking_setup):
    config, _, _, client, _ = tracking_setup
    receipt = track_training(config)
    client.set_terminated(receipt["run_id"], status="FAILED")
    with pytest.raises(ValueError, match="terminé avec succès"):
        change_champion(config, version=receipt["model_version"], reason="Run en échec")
    assert get_champion(client, receipt["model_name"]) is None


def test_loading_without_champion_is_explicitly_rejected(tracking_setup):
    config, _, _, _, _ = tracking_setup
    with pytest.raises(ValueError, match="Aucun modèle champion"):
        load_champion(config)


def test_different_validation_data_blocks_comparison(tracking_setup, make_csv):
    config, csv_path, rows, client, _ = tracking_setup
    first = track_training(config)
    change_champion(config, version=first["model_version"], reason="Modèle initial")
    rows[28]["balance"] = "9999"
    csv_path.write_bytes(make_csv(rows))
    second = track_training(config)
    assert second["validation_digest"] != first["validation_digest"]
    with pytest.raises(ValueError, match="validation différents"):
        change_champion(config, version=second["model_version"], reason="Comparaison interdite")
    assert str(get_champion(client, first["model_name"]).version) == first["model_version"]


def test_unserved_version_cannot_be_used_for_rollback(tracking_setup):
    config, _, _, client, _ = tracking_setup
    first = track_training(config)
    second = track_training(config)
    change_champion(config, version=second["model_version"], reason="Premier modèle actif")
    with pytest.raises(ValueError, match="jamais été champion"):
        change_champion(config, version=first["model_version"], reason="Retour non justifié", operation="rollback")
    assert str(get_champion(client, first["model_name"]).version) == second["model_version"]


def test_regression_against_champion_is_blocked(tracking_setup):
    config, _, _, client, _ = tracking_setup
    first = track_training(config)
    change_champion(config, version=first["model_version"], reason="Modèle initial")
    second = track_training(config)
    # Scénario synthétique : simuler une métrique inférieure, au-dessus du naïf.
    baseline = client.get_run(second["run_id"]).data.metrics["dummy.validation.average_precision"]
    champion = client.get_run(first["run_id"]).data.metrics["logistic_regression.validation.average_precision"]
    client.log_metric(second["run_id"], "logistic_regression.validation.average_precision", (baseline + champion) / 2)
    with pytest.raises(ValueError, match="Régression"):
        change_champion(config, version=second["model_version"], reason="Régression simulée")
    assert str(get_champion(client, first["model_name"]).version) == first["model_version"]
