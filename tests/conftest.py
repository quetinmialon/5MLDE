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


@pytest.fixture
def training_setup(tmp_path, valid_rows, make_csv, make_config):
    rows = [
        {
            **valid_rows[index % 2], "age": str(25 + index),
            "balance": str(index * 100 - 500), "day": str(index % 28 + 1),
            "duration": str(100 + index),
        }
        for index in range(40)
    ]
    for row in rows[34:]:
        row["job"] = "retired"
    content = make_csv(rows)
    config_path = make_config(b"archive", content)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + '\n[training]\ntrain_fraction = 0.70\nvalidation_fraction = 0.15\n'
        'regularization_c = 1.0\nmax_iter = 1000\nrandom_state = 42\n'
        'threshold = 0.5\noutput_dir = "models"\n',
        encoding="utf-8",
    )
    csv_path = tmp_path / "raw" / "bank-full.csv"
    csv_path.parent.mkdir()
    csv_path.write_bytes(content)
    return config_path, csv_path, rows


@pytest.fixture
def tracking_setup(training_setup, tmp_path, monkeypatch):
    import mlflow
    from mlflow import MlflowClient

    config_path, csv_path, rows = training_setup
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + f'\n[mlflow]\ntracking_uri = "{uri}"\n'
        'experiment_name = "bank-marketing-tests"\nmodel_name = "bank-marketing-test"\n'
        '[registry]\nmin_average_precision_gain = 0.0\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_ENABLE_TELEMETRY", "false")
    monkeypatch.chdir(tmp_path)
    previous_tracking_uri = mlflow.get_tracking_uri()
    previous_registry_uri = mlflow.get_registry_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment_id = client.create_experiment(
        "bank-marketing-tests", artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    yield config_path, csv_path, rows, client, experiment_id
    mlflow.end_run()
    mlflow.set_tracking_uri(previous_tracking_uri)
    mlflow.set_registry_uri(previous_registry_uri)


@pytest.fixture
def replay_setup(tracking_setup):
    config, csv_path, rows, client, experiment_id = tracking_setup
    config.write_text(
        config.read_text(encoding="utf-8")
        + '\n[orchestration]\napi_url = "http://127.0.0.1:4200/api"\n'
        'replay_dir = "replay"\nbatch_size = 10\ninterval_seconds = 3600\n'
        'ingestion_retries = 2\nretry_delay_seconds = 0\n',
        encoding="utf-8",
    )
    return config, csv_path, rows, client, experiment_id


@pytest.fixture(scope="session")
def prefect_backend(tmp_path_factory):
    from prefect.testing.utilities import prefect_test_harness
    from prefect.settings import PREFECT_HOME, PREFECT_SERVER_ANALYTICS_ENABLED, temporary_settings

    with temporary_settings(updates={
        PREFECT_HOME: tmp_path_factory.mktemp("prefect-home"),
        PREFECT_SERVER_ANALYTICS_ENABLED: False,
    }):
        with prefect_test_harness(server_startup_timeout=90):
            yield
