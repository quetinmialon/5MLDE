"""Lecture des paramètres d'ingestion et d'entraînement."""

import math
import re
import tomllib
from pathlib import Path
from urllib.parse import urlparse


def load_config(config_path: str | Path = "config/project.toml") -> dict:
    """Charger les paramètres et résoudre les chemins par rapport au TOML."""
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)

    for section in ("data", "validation"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Section [{section}] absente ou invalide.")

    data = config["data"]
    for key in ("source_url", "archive_sha256", "csv_sha256", "raw_dir", "report_dir"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"data.{key} doit être une chaîne non vide.")
    for key in ("archive_sha256", "csv_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", data[key]):
            raise ValueError(f"data.{key} doit être une empreinte SHA-256 en minuscules.")
    url = urlparse(data["source_url"])
    if url.scheme != "https" or not url.netloc:
        raise ValueError("data.source_url doit être une URL HTTPS.")

    min_rows = config["validation"].get("min_rows")
    if type(min_rows) is not int or min_rows < 1:
        raise ValueError("validation.min_rows doit être un entier strictement positif.")
    for key in ("raw_dir", "report_dir"):
        data[key] = (config_path.parent / data[key]).resolve()
    return config


def load_training_config(config_path: str | Path = "config/project.toml") -> dict:
    """Valider les paramètres du modèle sans les imposer à l'ingestion."""
    config = load_config(config_path)
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("Section [training] absente ou invalide.")
    for key in ("train_fraction", "validation_fraction", "regularization_c", "threshold"):
        value = training.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"training.{key} doit être un nombre fini.")
    for key in ("train_fraction", "validation_fraction"):
        if not 0 < training[key] < 1:
            raise ValueError(f"training.{key} doit être entre 0 et 1, bornes exclues.")
    if training["train_fraction"] + training["validation_fraction"] >= 1:
        raise ValueError("Les fractions doivent laisser une partition de test non vide.")
    if training["regularization_c"] <= 0 or not 0 <= training["threshold"] <= 1:
        raise ValueError("C doit être positif et le seuil compris entre 0 et 1.")
    if type(training.get("max_iter")) is not int or training["max_iter"] < 1:
        raise ValueError("training.max_iter doit être un entier strictement positif.")
    if type(training.get("random_state")) is not int or not 0 <= training["random_state"] < 2**32:
        raise ValueError("training.random_state doit être un entier entre 0 et 2**32 - 1.")
    if not isinstance(training.get("output_dir"), str) or not training["output_dir"].strip():
        raise ValueError("training.output_dir doit être une chaîne non vide.")
    training["output_dir"] = (Path(config_path).resolve().parent / training["output_dir"]).resolve()
    return config
