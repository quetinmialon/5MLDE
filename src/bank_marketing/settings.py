"""Lecture de la configuration de l'ingestion."""

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
