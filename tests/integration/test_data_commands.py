import json
import subprocess
import sys


def test_ingestion_and_validation_commands(tmp_path, make_csv, make_archive, make_config):
    content = make_csv()
    archive = make_archive(content)
    config = make_config(archive, content)
    archive_path = tmp_path / "source.zip"
    archive_path.write_bytes(archive)
    ingestion = subprocess.run(
        [sys.executable, "-I", "-m", "bank_marketing.ingestion", "--config", str(config), "--archive", str(archive_path)],
        cwd=tmp_path, capture_output=True,
    )
    assert ingestion.returncode == 0, ingestion.stderr.decode("utf-8", errors="replace")
    csv_path = tmp_path / "raw" / "bank-full.csv"
    assert csv_path.read_bytes() == content
    report_path = tmp_path / "quality.json"
    validation = subprocess.run(
        [sys.executable, "-I", "-m", "bank_marketing.validation", str(csv_path), "--report", str(report_path)],
        cwd=tmp_path, capture_output=True,
    )
    assert validation.returncode == 0, validation.stderr.decode("utf-8", errors="replace")
    assert json.loads(report_path.read_text(encoding="utf-8"))["row_count"] == 2


def test_invalid_batch_command_fails_with_report(tmp_path, valid_rows, make_csv):
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_bytes(make_csv([{**valid_rows[0], "y": "invalid"}]))
    report_path = tmp_path / "quality.json"
    result = subprocess.run(
        [sys.executable, "-I", "-m", "bank_marketing.validation", str(csv_path), "--report", str(report_path)],
        cwd=tmp_path, capture_output=True,
    )
    assert result.returncode == 1
    assert not json.loads(report_path.read_text(encoding="utf-8"))["valid"]
    assert b"Traceback" not in result.stderr


def test_ingestion_command_handles_missing_archive(tmp_path, make_config):
    config = make_config(b"archive", b"csv")
    result = subprocess.run(
        [sys.executable, "-I", "-m", "bank_marketing.ingestion", "--config", str(config), "--archive", str(tmp_path / "missing.zip")],
        cwd=tmp_path, capture_output=True,
    )
    assert result.returncode == 1
    assert b"Traceback" not in result.stderr
    assert not (tmp_path / "raw").exists()
