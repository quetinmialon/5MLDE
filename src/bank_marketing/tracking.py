"""Suivi explicite des entraînements et des candidats dans MLflow."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from importlib.metadata import distributions, version
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.exceptions import ConvergenceWarning

from bank_marketing.preprocessing import FEATURE_COLUMNS
from bank_marketing.settings import load_tracking_config
from bank_marketing.training import load_partitions, train_model
from bank_marketing.validation import DataValidationError


def configure_tracking(config: dict) -> tuple[MlflowClient, str]:
    uri = config["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    experiment = mlflow.set_experiment(config["mlflow"]["experiment_name"])
    return MlflowClient(tracking_uri=uri, registry_uri=uri), experiment.experiment_id


def validation_decision(candidate: float, baseline: float, minimum_gain: float) -> dict:
    """Règle relative de démonstration, indépendante du jeu de test."""
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (candidate, baseline, minimum_gain)):
        raise ValueError("Les scores et le gain minimal doivent être finis et compris entre 0 et 1.")
    return {
        "metric": "validation.average_precision",
        "candidate": candidate,
        "baseline": baseline,
        "minimum_gain": minimum_gain,
        "passed": candidate >= baseline + minimum_gain,
    }


def log_code_and_environment(config_path: Path) -> None:
    """Tracer le code et les versions sans lire .env ni les fichiers de travail."""
    root = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    git = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    try:
        commit = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True)
        status = subprocess.run(git + ["status", "--porcelain"], capture_output=True, text=True)
        mlflow.set_tags({
            "code_commit": commit.stdout.strip() if commit.returncode == 0 else "unavailable",
            "code_dirty": str(bool(status.stdout.strip())) if status.returncode == 0 else "unknown",
        })
    except FileNotFoundError:
        mlflow.set_tags({"code_commit": "unavailable", "code_dirty": "unknown"})
    package_dir = Path(__file__).resolve().parent
    code_digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        code_digest.update(path.relative_to(package_dir).as_posix().encode("utf-8") + b"\0" + path.read_bytes())
        artifact_path = (Path("source/bank_marketing") / path.parent.relative_to(package_dir)).as_posix()
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    mlflow.set_tag("code_sha256", code_digest.hexdigest())
    for path in (root / "pyproject.toml", root / "requirements.txt", config_path):
        if path.is_file():
            mlflow.log_artifact(str(path), artifact_path="configuration")
    installed = sorted(
        f"{dist.metadata['Name']}=={dist.version}" for dist in distributions() if dist.metadata.get("Name")
    )
    mlflow.log_text("\n".join(installed) + "\n", "environment/installed-packages.txt")


def track_training(
    config_path: str | Path = "config/project.toml",
    *,
    csv_path: str | Path | None = None,
    evaluate_test: bool = False,
    validation_csv: str | Path | None = None,
    test_csv: str | Path | None = None,
    batch_id: str | None = None,
    flow_run_id: str | None = None,
) -> dict:
    """Entraîner dans un run MLflow ; enregistrer un candidat sans le promouvoir."""
    config_path = Path(config_path).resolve()
    config = load_tracking_config(config_path)
    client, experiment_id = configure_tracking(config)
    settings = config["mlflow"]
    with mlflow.start_run(experiment_id=experiment_id, run_name="training", tags={"operation": "training"}) as run:
        try:
            if batch_id is not None:
                mlflow.set_tag("batch_id", batch_id)
            if flow_run_id is not None:
                mlflow.set_tag("prefect_flow_run_id", flow_run_id)
            log_code_and_environment(config_path)
            run_dir = train_model(
                config_path, csv_path=csv_path, evaluate_test=evaluate_test,
                validation_csv=validation_csv, test_csv=test_csv,
            )
            report = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            pipeline = joblib.load(run_dir / "pipeline.joblib")
            source = Path(csv_path) if csv_path is not None else config["data"]["raw_dir"] / "bank-full.csv"
            partitions, quality = load_partitions(config, csv_path, validation_csv, test_csv)
            if quality["sha256"] != report["dataset"]["sha256"]:
                raise ValueError("Les données ont changé depuis l'entraînement ; suivi refusé.")
            for name, partition in partitions.items():
                digest = hashlib.sha256(partition.to_csv(index=False, lineterminator="\r\n").encode("utf-8")).hexdigest()
                if digest != report["splits"][name]["sha256"]:
                    raise ValueError(f"La partition {name} a changé depuis l'entraînement.")
            validation_digest = report["splits"]["validation"]["sha256"]
            mlflow.set_tags({
                "dataset_sha256": quality["sha256"],
                "validation_digest": validation_digest,
                "quality_status": "passed",
                "model_name": settings["model_name"],
            })
            mlflow.log_params(report["parameters"])
            mlflow.log_params({f"classifier.{key}": value for key, value in pipeline.named_steps["classifier"].get_params().items()})
            mlflow.log_param("dummy.strategy", "prior")
            mlflow.log_param("dataset.source_url", config["data"]["source_url"])
            mlflow.log_metric("training_seconds", report["training_seconds"])
            for model_name, results in report["metrics"].items():
                for partition, metrics in results.items():
                    mlflow.log_metrics({
                        f"{model_name}.{partition}.{name}": value
                        for name, value in metrics.items() if type(value) in (int, float)
                    })
            for name in (["train", "validation"] + (["test"] if evaluate_test else [])):
                dataset = mlflow.data.from_pandas(
                    partitions[name], source=(
                        quality["partition_sources"][name]["source"]
                        if "partition_sources" in quality else str(source.resolve())
                    ),
                    # MLflow limite ce champ à 36 caractères ; le SHA-256 complet reste dans le rapport.
                    digest=report["splits"][name]["sha256"][:32], name=f"bank-marketing-{name}", targets="y",
                )
                mlflow.log_input(dataset, context=name)
            mlflow.log_artifact(str(run_dir / "metrics.json"))
            mlflow.log_dict(quality, "quality/validation.json")

            example = partitions["train"].loc[:, FEATURE_COLUMNS].head(3)
            expected = pipeline.predict_proba(example)
            requirements = config_path.parent.parent / "requirements.txt" if config_path.parent.name == "config" else config_path.parent / "requirements.txt"
            pip_requirements = str(requirements) if requirements.is_file() else [
                f"{name}=={version(name)}" for name in ("mlflow", "scikit-learn", "pandas", "numpy", "joblib", "cloudpickle")
            ]
            model_info = mlflow.sklearn.log_model(
                pipeline, name="pipeline", signature=infer_signature(example, expected),
                input_example=example, pip_requirements=pip_requirements,
                serialization_format="cloudpickle", pyfunc_predict_fn="predict_proba",
                metadata={"class_labels": [0, 1], "threshold": report["parameters"]["threshold"]},
            )
            restored = mlflow.sklearn.load_model(model_info.model_uri)
            if not np.allclose(restored.predict_proba(example), expected):
                raise ValueError("Les probabilités diffèrent après rechargement du modèle MLflow.")
            mlflow.set_tag("model_reload_status", "passed")
            decision = validation_decision(
                report["metrics"]["logistic_regression"]["validation"]["average_precision"],
                report["metrics"]["dummy"]["validation"]["average_precision"],
                config["registry"]["min_average_precision_gain"],
            )
            mlflow.log_dict(decision, "validation/decision.json")
            status = "passed" if decision["passed"] else "rejected"
            model_version = mlflow.register_model(model_info.model_uri, settings["model_name"])
            for key, value in {
                "validation_status": status, "validation_digest": validation_digest,
                "dataset_sha256": quality["sha256"], "use_case": "local_demonstration",
            }.items():
                client.set_model_version_tag(settings["model_name"], model_version.version, key, value)
            mlflow.set_tag("validation_status", status)
            receipt = {
                "run_id": run.info.run_id, "experiment_id": experiment_id,
                "model_name": settings["model_name"], "model_version": str(model_version.version),
                "model_uri": f"models:/{settings['model_name']}/{model_version.version}",
                "validation_status": status, "validation_digest": validation_digest,
                "run_dir": str(run_dir),
            }
            mlflow.log_dict(receipt, "tracking.json")
        except Exception as error:
            if isinstance(error, DataValidationError):
                mlflow.log_dict(error.report, "quality/validation.json")
            mlflow.set_tag("validation_status", "failed")
            mlflow.log_dict({"type": type(error).__name__, "message": str(error)}, "failure.json")
            raise
    (run_dir / "tracking.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entraîner et enregistrer un candidat MLflow.")
    parser.add_argument("--config", type=Path, default=Path("config/project.toml"))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = track_training(args.config, csv_path=args.csv, evaluate_test=args.evaluate_test)
    except (OSError, ValueError, MlflowException, ConvergenceWarning) as error:
        print(f"Suivi refusé : {error}", file=sys.stderr)
        return 1
    print(f"Run {receipt['run_id']} ; version {receipt['model_version']} ; validation {receipt['validation_status']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
