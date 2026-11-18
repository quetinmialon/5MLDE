import hashlib
import io
import json
from pathlib import Path
from urllib.error import URLError

import pytest

from bank_marketing.ingestion import ingest_dataset
from bank_marketing.validation import DataValidationError


def test_local_acquisition_is_reproducible(tmp_path, make_csv, make_archive, make_config):
    content = make_csv()
    archive = make_archive(content)
    config = make_config(archive, content)
    archive_path = tmp_path / "source.zip"
    archive_path.write_bytes(archive)
    first = ingest_dataset(config, archive_path=archive_path)
    second = ingest_dataset(config, archive_path=archive_path)
    assert first == second
    assert second.read_bytes() == content
    manifest = json.loads((tmp_path / "raw" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["archive_sha256"] == hashlib.sha256(archive).hexdigest()
    assert manifest["csv_sha256"] == hashlib.sha256(content).hexdigest()
    assert manifest["row_count"] == 2
    assert manifest["acquisition"] == "local_archive"
    assert json.loads(Path(manifest["validation_report"]).read_text(encoding="utf-8"))["valid"]


def test_download_can_be_exercised_without_external_network(monkeypatch, make_csv, make_archive, make_config):
    content = make_csv()
    archive = make_archive(content)
    config = make_config(archive, content)
    monkeypatch.setattr("bank_marketing.ingestion.urlopen", lambda *args, **kwargs: io.BytesIO(archive))
    assert ingest_dataset(config).read_bytes() == content


@pytest.mark.parametrize("mismatch", ["archive", "csv"])
def test_hash_mismatch_preserves_existing_dataset(mismatch, tmp_path, make_csv, make_archive, make_config):
    content = make_csv()
    archive = make_archive(content)
    config = make_config(archive, content if mismatch == "archive" else b"different CSV")
    archive_path = tmp_path / "source.zip"
    archive_path.write_bytes(b"corrupted archive" if mismatch == "archive" else archive)
    output = tmp_path / "raw" / "bank-full.csv"
    output.parent.mkdir()
    output.write_bytes(b"previous dataset")
    with pytest.raises(ValueError, match="SHA-256"):
        ingest_dataset(config, archive_path=archive_path)
    assert output.read_bytes() == b"previous dataset"
    assert not (output.parent / "manifest.json").exists()


@pytest.mark.parametrize("kind", ["corrupt", "missing_member"])
def test_invalid_archive_is_rejected(kind, tmp_path, make_csv, make_archive, make_config):
    content = make_csv()
    archive = b"not a ZIP" if kind == "corrupt" else make_archive(content, include_csv=False)
    config = make_config(archive, content)
    archive_path = tmp_path / "source.zip"
    archive_path.write_bytes(archive)
    with pytest.raises(ValueError, match="Archive invalide"):
        ingest_dataset(config, archive_path=archive_path)
    assert not (tmp_path / "raw" / "bank-full.csv").exists()


def test_invalid_data_is_not_published_even_if_hashes_match(tmp_path, valid_rows, make_csv, make_archive, make_config):
    content = make_csv([{**valid_rows[0], "campaign": "0"}])
    archive = make_archive(content)
    config = make_config(archive, content)
    archive_path = tmp_path / "source.zip"
    archive_path.write_bytes(archive)
    output = tmp_path / "raw" / "bank-full.csv"
    output.parent.mkdir()
    output.write_bytes(b"previous dataset")
    with pytest.raises(DataValidationError):
        ingest_dataset(config, archive_path=archive_path)
    assert output.read_bytes() == b"previous dataset"
    reports = list((tmp_path / "reports").glob("*.json"))
    assert len(reports) == 1
    assert not json.loads(reports[0].read_text(encoding="utf-8"))["valid"]


def test_network_failure_preserves_previous_dataset(monkeypatch, tmp_path, make_csv, make_archive, make_config):
    content = make_csv()
    config = make_config(make_archive(content), content)
    output = tmp_path / "raw" / "bank-full.csv"
    output.parent.mkdir()
    output.write_bytes(b"previous dataset")

    def fail(*args, **kwargs):
        raise URLError("source unavailable")

    monkeypatch.setattr("bank_marketing.ingestion.urlopen", fail)
    with pytest.raises(URLError):
        ingest_dataset(config)
    assert output.read_bytes() == b"previous dataset"
