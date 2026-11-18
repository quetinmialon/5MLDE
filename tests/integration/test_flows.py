import json
from pathlib import Path
from urllib.error import URLError

import pytest
from prefect import flow
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import ArtifactFilter, ArtifactFilterKey

from bank_marketing.flows import acquire_source, bank_marketing_pipeline, retry_ingestion
from bank_marketing.registry import change_champion, get_champion
from bank_marketing.replay import prepare_replay
from bank_marketing.tracking import track_training
from bank_marketing.validation import DataValidationError


def test_flow_replays_without_duplicate_version(prefect_backend, replay_setup):
    config, _, _, client, _ = replay_setup
    first = bank_marketing_pipeline(str(config))
    assert first["status"] == "completed"
    assert first["batch"]["training_rows"] == 10
    second = bank_marketing_pipeline(str(config), batch_number=1)
    assert second["status"] == "skipped"
    assert len(client.search_model_versions("name = 'bank-marketing-test'")) == 1
    third = bank_marketing_pipeline(str(config))
    assert third["batch"]["training_rows"] == 20
    first_run = client.get_run(first["receipt"]["run_id"])
    second_run = client.get_run(third["receipt"]["run_id"])
    assert first_run.data.tags["validation_digest"] == second_run.data.tags["validation_digest"]
    assert first_run.data.tags["prefect_flow_run_id"]


def test_failed_batch_alert_is_persistent_and_champion_is_preserved(prefect_backend, replay_setup):
    config, source, _, client, _ = replay_setup
    initial = track_training(config)
    change_champion(config, version=initial["model_version"], reason="Modèle initial")
    prepare_replay(config, source)
    (config.parent / "replay" / "batches" / "batch-00001.csv").write_bytes(b"invalid CSV")
    with pytest.raises(DataValidationError):
        bank_marketing_pipeline(str(config))
    assert str(get_champion(client, initial["model_name"]).version) == initial["model_version"]
    alerts = list((config.parent / "replay" / "alerts").glob("*.json"))
    assert len(alerts) == 1
    alert = json.loads(alerts[0].read_text(encoding="utf-8"))
    assert alert["severity"] == "critical"
    assert not (config.parent / "replay" / "progress.json").exists()
    with get_client(sync_client=True) as prefect_client:
        artifacts = prefect_client.read_artifacts(
            artifact_filter=ArtifactFilter(key=ArtifactFilterKey(any_=["bank-marketing-alerts"]))
        )
    assert any(alert["flow_run_id"] in item.data for item in artifacts)


def test_completed_mlflow_batch_recovers_after_checkpoint_loss(prefect_backend, replay_setup, monkeypatch):
    config, _, _, client, _ = replay_setup

    def fail_commit(*args, **kwargs):
        raise OSError("checkpoint unavailable")

    with monkeypatch.context() as patch:
        patch.setattr("bank_marketing.flows.commit_batch", fail_commit)
        with pytest.raises(OSError):
            bank_marketing_pipeline(str(config))
    assert len(client.search_model_versions("name = 'bank-marketing-test'")) == 1
    result = bank_marketing_pipeline(str(config))
    assert result["status"] == "completed"
    assert len(client.search_model_versions("name = 'bank-marketing-test'")) == 1


def test_transient_ingestion_is_retried_but_invalid_data_is_not(prefect_backend, replay_setup, monkeypatch):
    config, source, _, _, _ = replay_setup
    attempts = []

    def transient(*args, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise URLError("temporary network failure")
        return source

    @flow
    def execute_acquisition():
        return acquire_source.with_options(
            retries=2, retry_delay_seconds=0, retry_condition_fn=retry_ingestion,
        )(str(config), "local.zip")

    monkeypatch.setattr("bank_marketing.flows.ingest_dataset", transient)
    assert execute_acquisition() == str(source)
    assert len(attempts) == 3

    attempts.clear()

    def invalid(*args, **kwargs):
        attempts.append(1)
        raise ValueError("invalid archive checksum")

    monkeypatch.setattr("bank_marketing.flows.ingest_dataset", invalid)
    with pytest.raises(ValueError):
        execute_acquisition()
    assert len(attempts) == 1
