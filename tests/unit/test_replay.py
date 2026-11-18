import hashlib
import json
from pathlib import Path

import joblib
import pytest

from bank_marketing.replay import commit_batch, next_batch, prepare_replay, read_checked_csv
from bank_marketing.training import load_training_data, split_dataset, train_model
from bank_marketing.validation import DataValidationError


def test_replay_contains_only_train_rows_and_preserves_holdouts(replay_setup):
    config, source, _, _, _ = replay_setup
    manifest = prepare_replay(config, source)
    assert [item["row_count"] for item in manifest["batches"]] == [10, 10, 8]
    assert manifest["validation"]["row_count"] == manifest["test"]["row_count"] == 6
    first = next_batch(config, manifest)
    train = read_checked_csv(Path(first["train_csv"]))
    validation = read_checked_csv(config.parent / "replay" / "validation.csv")
    test = read_checked_csv(config.parent / "replay" / "test.csv")
    assert train["age"].tolist() == [str(age) for age in range(25, 35)]
    assert validation["age"].tolist() == [str(age) for age in range(53, 59)]
    assert test["age"].tolist() == [str(age) for age in range(59, 65)]
    frozen = (config.parent / "replay" / "validation.csv").read_bytes()
    commit_batch(config, first, {"run_id": "test-run", "model_version": "1", "validation_status": "passed"})
    second = next_batch(config, prepare_replay(config, source))
    assert second["training_rows"] == 20
    assert next_batch(config, manifest, batch_number=1) is None
    assert (config.parent / "replay" / "validation.csv").read_bytes() == frozen


def test_fixed_holdout_training_does_not_resplit_the_batch(replay_setup):
    config, source, _, _, _ = replay_setup
    batch = next_batch(config, prepare_replay(config, source))
    run = train_model(
        config, csv_path=batch["train_csv"], validation_csv=batch["validation_csv"], test_csv=batch["test_csv"],
    )
    report = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert report["split_mode"] == "fixed_holdouts"
    assert [report["splits"][name]["row_count"] for name in ("train", "validation", "test")] == [10, 6, 6]
    model = joblib.load(run / "pipeline.joblib")
    assert model.named_steps["preprocessing"].named_transformers_["numeric"].mean_[0] == pytest.approx(29.5)
    full, _ = load_training_data(source)
    reference = split_dataset(full)["validation"]
    expected = hashlib.sha256(reference.to_csv(index=False, lineterminator="\r\n").encode()).hexdigest()
    assert report["splits"]["validation"]["sha256"] == expected


def test_corrupt_batch_blocks_processing_without_advancing(replay_setup):
    config, source, _, _, _ = replay_setup
    manifest = prepare_replay(config, source)
    batch_path = config.parent / "replay" / "batches" / "batch-00001.csv"
    batch_path.write_text("bad header\n", encoding="utf-8")
    with pytest.raises(DataValidationError):
        next_batch(config, manifest)
    assert not (config.parent / "replay" / "progress.json").exists()


def test_changed_holdout_is_rejected_instead_of_recreated(replay_setup):
    config, source, _, _, _ = replay_setup
    prepare_replay(config, source)
    reference = config.parent / "replay" / "validation.csv"
    original = reference.read_bytes()
    reference.write_bytes(original.replace(b"280", b"281", 1) if b"280" in original else original + b"\n")
    with pytest.raises((ValueError, DataValidationError)):
        prepare_replay(config, source)


def test_replay_configuration_changes_require_new_directory(replay_setup):
    config, source, _, _, _ = replay_setup
    prepare_replay(config, source)
    config.write_text(config.read_text(encoding="utf-8").replace("batch_size = 10", "batch_size = 12"), encoding="utf-8")
    with pytest.raises(ValueError, match="référence de rejeu"):
        prepare_replay(config, source)


def test_batches_must_be_processed_in_order(replay_setup):
    config, source, _, _, _ = replay_setup
    manifest = prepare_replay(config, source)
    with pytest.raises(ValueError, match="ordre"):
        next_batch(config, manifest, batch_number=2)


def test_incomplete_fixed_holdouts_are_rejected(replay_setup):
    config, source, _, _, _ = replay_setup
    with pytest.raises(ValueError, match="ensemble"):
        train_model(config, csv_path=source, validation_csv=source)


def test_batch_id_depends_on_training_configuration(replay_setup):
    config, source, _, _, _ = replay_setup
    manifest = prepare_replay(config, source)
    first = next_batch(config, manifest)
    config.write_text(config.read_text(encoding="utf-8").replace("regularization_c = 1.0", "regularization_c = 2.0"), encoding="utf-8")
    second = next_batch(config, manifest)
    assert first["batch_id"] != second["batch_id"]
