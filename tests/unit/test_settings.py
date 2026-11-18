import pytest

from bank_marketing.settings import load_config, load_training_config


def test_paths_are_resolved_from_config_location(tmp_path, monkeypatch, make_config):
    config = make_config(b"archive", b"csv")
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)
    loaded = load_config(config)
    assert loaded["data"]["raw_dir"] == tmp_path / "raw"
    assert loaded["data"]["report_dir"] == tmp_path / "reports"


@pytest.mark.parametrize("before,after", [
    ('[validation]', '[other]'),
    ('[data]', '[other]'),
    ('min_rows = 1', 'min_rows = 0'),
    ('min_rows = 1', 'min_rows = true'),
    ('min_rows = 1', 'min_rows = 1.5'),
    ('raw_dir = "raw"', 'raw_dir = ""'),
    ('report_dir = "reports"', 'report_dir = 123'),
    ('https://example.invalid/bank.zip', 'http://example.invalid/bank.zip'),
])
def test_invalid_config_is_rejected(before, after, make_config):
    config = make_config(b"archive", b"csv")
    config.write_text(config.read_text(encoding="utf-8").replace(before, after), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(config)


def test_invalid_hash_format_is_rejected(make_config):
    config = make_config(b"archive", b"csv")
    content = config.read_text(encoding="utf-8")
    lines = [line if not line.startswith("csv_sha256") else 'csv_sha256 = "bad-hash"' for line in content.splitlines()]
    config.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_config(config)


@pytest.mark.parametrize("before,after", [
    ("[training]", "[other_training]"),
    ("train_fraction = 0.70", "train_fraction = 0.90"),
    ("train_fraction = 0.70", "train_fraction = true"),
    ("validation_fraction = 0.15", "validation_fraction = 0"),
    ("regularization_c = 1.0", "regularization_c = 0"),
    ("regularization_c = 1.0", "regularization_c = inf"),
    ("threshold = 0.5", "threshold = 1.5"),
    ("max_iter = 1000", "max_iter = false"),
    ("random_state = 42", "random_state = -1"),
    ('output_dir = "models"', 'output_dir = ""'),
])
def test_invalid_training_settings_are_rejected(before, after, training_setup):
    config, _, _ = training_setup
    config.write_text(config.read_text(encoding="utf-8").replace(before, after), encoding="utf-8")
    with pytest.raises(ValueError):
        load_training_config(config)


def test_ingestion_does_not_require_training_settings(make_config):
    config = make_config(b"archive", b"csv")
    assert load_config(config)["validation"]["min_rows"] == 1
    with pytest.raises(ValueError, match="training"):
        load_training_config(config)
