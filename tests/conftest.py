"""Jeux synthétiques indépendants du téléchargement UCI."""

import csv
import hashlib
import io
import zipfile

import pytest


@pytest.fixture
def valid_rows():
    row = {
        "age": "35", "job": "technician", "marital": "single",
        "education": "secondary", "default": "no", "balance": "100",
        "housing": "yes", "loan": "no", "contact": "cellular", "day": "15",
        "month": "may", "duration": "120", "campaign": "1", "pdays": "-1",
        "previous": "0", "poutcome": "unknown", "y": "no",
    }
    return [row, {**row, "age": "72", "balance": "-200", "job": "unknown", "y": "yes"}]


@pytest.fixture
def make_csv(valid_rows):
    def build(rows=None, fields=None):
        rows = valid_rows if rows is None else rows
        fields = list(valid_rows[0]) if fields is None else fields
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue().encode("utf-8")
    return build


@pytest.fixture
def make_archive():
    def build(csv_content, *, include_csv=True):
        nested = io.BytesIO()
        with zipfile.ZipFile(nested, "w") as archive:
            if include_csv:
                archive.writestr("bank-full.csv", csv_content)
            else:
                archive.writestr("bank.csv", csv_content)
        outer = io.BytesIO()
        with zipfile.ZipFile(outer, "w") as archive:
            archive.writestr("bank.zip", nested.getvalue())
        return outer.getvalue()
    return build


@pytest.fixture
def make_config(tmp_path):
    def build(archive_content, csv_content):
        config_path = tmp_path / "project.toml"
        config_path.write_text(
            '[data]\nsource_url = "https://example.invalid/bank.zip"\n'
            f'archive_sha256 = "{hashlib.sha256(archive_content).hexdigest()}"\n'
            f'csv_sha256 = "{hashlib.sha256(csv_content).hexdigest()}"\n'
            'raw_dir = "raw"\nreport_dir = "reports"\n'
            '[validation]\nmin_rows = 1\n',
            encoding="utf-8",
        )
        return config_path
    return build
