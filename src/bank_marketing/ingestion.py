"""Récupération vérifiée de bank-full.csv depuis l'archive officielle UCI."""

import argparse
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from bank_marketing.settings import load_config
from bank_marketing.validation import validate_csv


def ingest_dataset(
    config_path: str | Path = "config/project.toml",
    *,
    archive_path: str | Path | None = None,
) -> Path:
    """Vérifier archive, CSV et qualité avant de publier le fichier brut local."""
    config = load_config(config_path)
    data = config["data"]
    if archive_path is None:
        with urlopen(data["source_url"], timeout=30) as response:
            archive_content = response.read()
    else:
        archive_content = Path(archive_path).read_bytes()
    archive_sha256 = hashlib.sha256(archive_content).hexdigest()
    if archive_sha256 != data["archive_sha256"]:
        raise ValueError("Empreinte SHA-256 de l'archive différente de la référence.")

    try:
        with zipfile.ZipFile(io.BytesIO(archive_content)) as outer:
            nested_content = outer.read("bank.zip")
        with zipfile.ZipFile(io.BytesIO(nested_content)) as nested:
            csv_content = nested.read("bank-full.csv")
    except (zipfile.BadZipFile, KeyError) as error:
        raise ValueError("Archive invalide : bank.zip ou bank-full.csv absent ou illisible.") from error

    csv_sha256 = hashlib.sha256(csv_content).hexdigest()
    if csv_sha256 != data["csv_sha256"]:
        raise ValueError("Empreinte SHA-256 du CSV différente de la référence.")
    csv_path = data["raw_dir"] / "bank-full.csv"
    report_path = data["report_dir"] / f"validation-{csv_sha256}.json"
    report = validate_csv(
        csv_content, source=str(csv_path), min_rows=config["validation"]["min_rows"],
        report_path=report_path,
    )

    # Le fichier précédent reste intact si l'acquisition ou les contrôles échouent.
    data["raw_dir"].mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=data["raw_dir"], prefix=".bank-full-", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(csv_content)
        temporary_path.replace(csv_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    manifest = {
        "source_url": data["source_url"],
        "acquisition": "download" if archive_path is None else "local_archive",
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": archive_sha256,
        "archive_members": ["bank.zip", "bank-full.csv"],
        "csv_file": csv_path.name,
        "csv_sha256": csv_sha256,
        "row_count": report["row_count"],
        "validation_report": str(report_path),
    }
    (data["raw_dir"] / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Récupérer et valider le dataset UCI bank-full.")
    parser.add_argument("--config", type=Path, default=Path("config/project.toml"))
    parser.add_argument("--archive", type=Path, help="Archive locale bank+marketing.zip, pour fonctionner hors ligne.")
    args = parser.parse_args(argv)
    try:
        csv_path = ingest_dataset(args.config, archive_path=args.archive)
    except (OSError, ValueError) as error:
        print(f"Ingestion refusée : {error}", file=sys.stderr)
        return 1
    print(f"Données vérifiées et disponibles : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
