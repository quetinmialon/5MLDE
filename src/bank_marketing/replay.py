"""Rejeu ordonné des seules données d'entraînement, avec références fixes."""

import hashlib
import io
import json
import tempfile
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from bank_marketing.settings import load_orchestration_config
from bank_marketing.training import split_dataset
from bank_marketing.validation import validate_csv


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_checked_csv(path: Path, expected_sha256: str | None = None) -> pd.DataFrame:
    content = path.read_bytes()
    quality = validate_csv(content, source=str(path))
    if expected_sha256 is not None and quality["sha256"] != expected_sha256:
        raise ValueError(f"Empreinte de rejeu différente de la référence : {path.name}.")
    # Conserver les entiers textuels du contrat brut lors de l'écriture des lots.
    return pd.read_csv(io.BytesIO(content), sep=";", dtype=str, keep_default_na=False)


def prepare_replay(config_path: str | Path, source: str | Path) -> dict:
    config = load_orchestration_config(config_path)
    settings = config["orchestration"]
    replay_dir = settings["replay_dir"]
    source = Path(source)
    frame = read_checked_csv(source, config["data"]["csv_sha256"])
    signature = {
        "source_sha256": config["data"]["csv_sha256"],
        "train_fraction": config["training"]["train_fraction"],
        "validation_fraction": config["training"]["validation_fraction"],
        "batch_size": settings["batch_size"],
    }
    manifest_path = replay_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["reference"] != signature:
            raise ValueError("La référence de rejeu a changé ; utiliser un autre répertoire de rejeu.")
        for name in ("validation", "test"):
            read_checked_csv(replay_dir / f"{name}.csv", manifest[name]["sha256"])
        return manifest

    partitions = split_dataset(
        frame, train_fraction=signature["train_fraction"],
        validation_fraction=signature["validation_fraction"],
    )
    replay_dir.mkdir(parents=True, exist_ok=True)
    (replay_dir / "batches").mkdir(exist_ok=True)
    manifest = {"reference": signature, "batches": []}
    for name in ("validation", "test"):
        content = partitions[name].to_csv(sep=";", index=False, lineterminator="\n").encode("utf-8")
        (replay_dir / f"{name}.csv").write_bytes(content)
        manifest[name] = {"row_count": len(partitions[name]), "sha256": hashlib.sha256(content).hexdigest()}
    training = partitions["train"]
    for start in range(0, len(training), settings["batch_size"]):
        number = len(manifest["batches"]) + 1
        batch = training.iloc[start:start + settings["batch_size"]]
        content = batch.to_csv(sep=";", index=False, lineterminator="\n").encode("utf-8")
        (replay_dir / "batches" / f"batch-{number:05}.csv").write_bytes(content)
        manifest["batches"].append({
            "number": number, "start_inclusive": start, "end_exclusive": start + len(batch),
            "row_count": len(batch), "sha256": hashlib.sha256(content).hexdigest(),
        })
    write_json_atomic(manifest_path, manifest)
    return manifest


def read_progress(replay_dir: Path) -> dict:
    path = replay_dir / "progress.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"completed": []}


def next_batch(config_path: str | Path, manifest: dict, batch_number: int | None = None) -> dict | None:
    config = load_orchestration_config(config_path)
    replay_dir = config["orchestration"]["replay_dir"]
    completed = read_progress(replay_dir)["completed"]
    if [item["number"] for item in completed] != list(range(1, len(completed) + 1)):
        raise ValueError("Progression de rejeu incohérente.")
    if batch_number is not None and (type(batch_number) is not int or not 1 <= batch_number <= len(manifest["batches"])):
        raise ValueError("Numéro de lot invalide.")
    if batch_number is not None and batch_number <= len(completed):
        return None
    number = len(completed) + 1
    if number > len(manifest["batches"]):
        return None
    if batch_number is not None and batch_number != number:
        raise ValueError("Les lots doivent être traités dans l'ordre.")
    frames = []
    for batch in manifest["batches"][:number]:
        path = replay_dir / "batches" / f"batch-{batch['number']:05}.csv"
        frames.append(read_checked_csv(path, batch["sha256"]))
    cumulative = pd.concat(frames, ignore_index=True)
    code_hash = hashlib.sha256()
    for filename in ("training.py", "preprocessing.py", "evaluation.py"):
        code_hash.update(Path(__file__).with_name(filename).read_bytes())
    context = {
        "reference": manifest["reference"], "batch_number": number,
        "training_rows": len(cumulative), "validation_sha256": manifest["validation"]["sha256"],
        "test_sha256": manifest["test"]["sha256"],
        "parameters": {key: value for key, value in config["training"].items() if key != "output_dir"},
        "registry": config["registry"], "model_name": config["mlflow"]["model_name"],
        "code_sha256": code_hash.hexdigest(),
        "environment": {name: version(name) for name in ("scikit-learn", "pandas", "numpy", "joblib")},
    }
    batch_id = hashlib.sha256(json.dumps(context, sort_keys=True).encode("utf-8")).hexdigest()
    attempt_dir = replay_dir / "attempts" / batch_id
    attempt_dir.mkdir(parents=True, exist_ok=True)
    train_csv = attempt_dir / "train.csv"
    train_csv.write_text(cumulative.to_csv(sep=";", index=False, lineterminator="\n"), encoding="utf-8")
    return {
        "number": number, "batch_id": batch_id, "training_rows": len(cumulative),
        "train_csv": str(train_csv), "validation_csv": str(replay_dir / "validation.csv"),
        "test_csv": str(replay_dir / "test.csv"),
    }


def commit_batch(config_path: str | Path, batch: dict, receipt: dict) -> None:
    replay_dir = load_orchestration_config(config_path)["orchestration"]["replay_dir"]
    progress = read_progress(replay_dir)
    if len(progress["completed"]) + 1 != batch["number"]:
        raise ValueError("Progression modifiée pendant le traitement.")
    progress["completed"].append({
        "number": batch["number"], "batch_id": batch["batch_id"],
        "training_rows": batch["training_rows"], "run_id": receipt["run_id"],
        "model_version": receipt["model_version"], "validation_status": receipt["validation_status"],
    })
    write_json_atomic(replay_dir / "progress.json", progress)
