"""Entraînement local d'un pipeline simple et comparaison à une référence naïve."""

import argparse
import hashlib
import io
import json
import math
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from bank_marketing.evaluation import evaluate_model
from bank_marketing.preprocessing import FEATURE_COLUMNS, build_preprocessor
from bank_marketing.settings import load_training_config
from bank_marketing.validation import CATEGORIES, INTEGER_BOUNDS, validate_csv


def load_training_data(
    csv_path: str | Path, *, min_rows: int = 1, report_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Valider et lire les mêmes octets, avant toute transformation apprenante."""
    csv_path = Path(csv_path)
    if report_path is not None and Path(report_path).resolve() == csv_path.resolve():
        raise ValueError("Le rapport ne doit pas écraser le CSV source.")
    content = csv_path.read_bytes()
    quality = validate_csv(
        content, source=str(csv_path), min_rows=min_rows, report_path=report_path,
    )
    dtypes = {name: "float64" for name in INTEGER_BOUNDS}
    dtypes.update({name: str for name in CATEGORIES})
    frame = pd.read_csv(io.BytesIO(content), sep=";", dtype=dtypes, keep_default_na=False)
    return frame, quality


def split_dataset(
    frame: pd.DataFrame, *, train_fraction: float = 0.70, validation_fraction: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Découper par positions dans l'ordre source, sans mélange ni chevauchement."""
    for fraction in (train_fraction, validation_fraction):
        if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("Les fractions de partition doivent être entre 0 et 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Les fractions doivent laisser une partition de test.")
    train_end = int(len(frame) * train_fraction)
    validation_end = int(len(frame) * (train_fraction + validation_fraction))
    partitions = {
        "train": frame.iloc[:train_end].copy(),
        "validation": frame.iloc[train_end:validation_end].copy(),
        "test": frame.iloc[validation_end:].copy(),
    }
    check_partitions(partitions)
    return partitions


def check_partitions(partitions: dict[str, pd.DataFrame]) -> None:
    for name, partition in partitions.items():
        if partition.empty:
            raise ValueError(f"Partition {name} vide : jeu de données trop petit.")
        if set(partition["y"]) != {"no", "yes"}:
            raise ValueError(f"La partition {name} doit contenir les deux classes no et yes.")


def load_partitions(
    config: dict, csv_path: str | Path | None = None,
    validation_csv: str | Path | None = None, test_csv: str | Path | None = None,
    *, report_path: str | Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Utiliser des références fixes pour le rejeu, ou la découpe initiale."""
    if (validation_csv is None) != (test_csv is None):
        raise ValueError("Validation et test fixes doivent être fournis ensemble.")
    source = Path(csv_path) if csv_path is not None else config["data"]["raw_dir"] / "bank-full.csv"
    if validation_csv is not None:
        paths = {"train": source, "validation": Path(validation_csv), "test": Path(test_csv)}
        if len({path.resolve() for path in paths.values()}) != 3:
            raise ValueError("Les trois fichiers de partitions doivent être distincts.")
        partitions, qualities = {}, {}
        for name, path in paths.items():
            partitions[name], qualities[name] = load_training_data(
                path, min_rows=config["validation"]["min_rows"],
                report_path=report_path if name == "train" else None,
            )
        check_partitions(partitions)
        quality = qualities["train"]
        quality["partition_sources"] = {
            name: {key: item[key] for key in ("source", "sha256", "row_count")}
            for name, item in qualities.items()
        }
        return partitions, quality
    frame, quality = load_training_data(
        source, min_rows=config["validation"]["min_rows"], report_path=report_path,
    )
    return split_dataset(
        frame, train_fraction=config["training"]["train_fraction"],
        validation_fraction=config["training"]["validation_fraction"],
    ), quality


def build_pipeline(
    *, regularization_c: float = 1.0, max_iter: int = 1000, random_state: int = 42,
) -> Pipeline:
    return Pipeline([
        ("preprocessing", build_preprocessor()),
        ("classifier", LogisticRegression(
            C=regularization_c, max_iter=max_iter, solver="lbfgs", random_state=random_state,
        )),
    ])


def train_model(
    config_path: str | Path = "config/project.toml",
    *,
    csv_path: str | Path | None = None,
    evaluate_test: bool = False,
    validation_csv: str | Path | None = None,
    test_csv: str | Path | None = None,
) -> Path:
    """Ajuster sur train seulement, puis conserver modèle, partitions et métriques."""
    config = load_training_config(config_path)
    settings = config["training"]
    partitions, quality = load_partitions(
        config, csv_path, validation_csv, test_csv,
        report_path=config["data"]["report_dir"] / "training-validation.json",
    )
    features = {name: part.loc[:, FEATURE_COLUMNS] for name, part in partitions.items()}
    targets = {name: part["y"].map({"no": 0, "yes": 1}).astype(int) for name, part in partitions.items()}
    pipeline = build_pipeline(
        regularization_c=settings["regularization_c"], max_iter=settings["max_iter"],
        random_state=settings["random_state"],
    )
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(features["train"], targets["train"])
    training_seconds = time.perf_counter() - started
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(features["train"], targets["train"])

    evaluation_partitions = ["train", "validation"] + (["test"] if evaluate_test else [])
    metrics = {
        model_name: {
            name: evaluate_model(model, features[name], targets[name], threshold=settings["threshold"])
            for name in evaluation_partitions
        }
        for model_name, model in (("logistic_regression", pipeline), ("dummy", dummy))
    }
    created_at = datetime.now(timezone.utc)
    run_dir = settings["output_dir"] / created_at.strftime("%Y%m%dT%H%M%S%fZ")
    split_summary = {}
    start = 0
    for name, partition in partitions.items():
        if validation_csv is not None:
            start = 0
        split_summary[name] = {
            "start_inclusive": start,
            "end_exclusive": start + len(partition),
            "row_count": len(partition),
            "positive_rate": float(targets[name].mean()),
            # Format canonique identique sous Windows/Linux, compatible avec les premiers runs.
            "sha256": hashlib.sha256(partition.to_csv(index=False, lineterminator="\r\n").encode("utf-8")).hexdigest(),
        }
        start += len(partition)
    report = {
        "created_at": created_at.isoformat(),
        "dataset": quality,
        "split_mode": "fixed_holdouts" if validation_csv is not None else "sequential",
        "splits": split_summary,
        "feature_columns": FEATURE_COLUMNS,
        "excluded_columns": ["duration", "y"],
        "parameters": {key: value for key, value in settings.items() if key != "output_dir"},
        "dummy_strategy": "prior",
        "training_seconds": training_seconds,
        "metrics": metrics,
        "test_evaluated": evaluate_test,
        "environment": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("scikit-learn", "pandas", "numpy", "joblib")},
        },
        "model_file": "pipeline.joblib",
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(pipeline, run_dir / "pipeline.joblib")
    (run_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entraîner et évaluer le modèle Bank Marketing.")
    parser.add_argument("--config", type=Path, default=Path("config/project.toml"))
    parser.add_argument("--csv", type=Path, help="CSV brut à utiliser à la place du fichier configuré.")
    parser.add_argument("--evaluate-test", action="store_true", help="Évaluer aussi le test, réservé au bilan final.")
    args = parser.parse_args(argv)
    try:
        run_dir = train_model(args.config, csv_path=args.csv, evaluate_test=args.evaluate_test)
    except (OSError, ValueError, ConvergenceWarning) as error:
        print(f"Entraînement refusé : {error}", file=sys.stderr)
        return 1
    print(f"Pipeline et métriques enregistrés : {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
