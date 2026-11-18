"""Contrat bloquant du CSV brut bank-full, avant tout prétraitement."""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


COLUMNS = (
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration", "campaign", "pdays", "previous",
    "poutcome", "y",
)
CATEGORIES = {
    "job": {
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    },
    "marital": {"married", "divorced", "single"},
    "education": {"primary", "secondary", "tertiary", "unknown"},
    "default": {"yes", "no"},
    "housing": {"yes", "no"},
    "loan": {"yes", "no"},
    "contact": {"cellular", "telephone", "unknown"},
    "month": {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"},
    "poutcome": {"unknown", "other", "failure", "success"},
    "y": {"yes", "no"},
}
# Bornes métier, et non minima/maxima observés sur le fichier historique.
INTEGER_BOUNDS = {
    "age": (1, None), "balance": (None, None), "day": (1, 31),
    "duration": (0, None), "campaign": (1, None), "pdays": (-1, None),
    "previous": (0, None),
}


class DataValidationError(ValueError):
    """Données refusées, avec le rapport disponible pour l'orchestrateur."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__(
            f"{report['error_count']} erreur(s) de données : {report['errors'][0]['message']}"
        )


def validate_csv(
    content: bytes,
    *,
    source: str = "<memory>",
    min_rows: int = 1,
    report_path: str | Path | None = None,
) -> dict:
    """Valider les octets sans les modifier ; lever une erreur si non conformes."""
    if type(min_rows) is not int or min_rows < 1:
        raise ValueError("min_rows doit être un entier strictement positif.")
    report = {
        "source": source,
        "sha256": hashlib.sha256(content).hexdigest(),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "valid": False,
        "row_count": 0,
        "class_counts": {"no": 0, "yes": 0},
        "unknown_counts": {},
        "duplicate_count": 0,
        "error_count": 0,
        "errors": [],
        "warnings": [],
    }

    def add_error(message: str, line: int | None = None, field: str | None = None):
        report["error_count"] += 1
        # Compter toutes les erreurs mais limiter la taille des exemples conservés.
        if len(report["errors"]) < 20:
            report["errors"].append({"line": line, "field": field, "message": message})

    seen_rows = set()
    unknown_counts = Counter()
    try:
        text = content.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=";", strict=True)
        header = next(reader, [])
        if len(header) != len(COLUMNS) or set(header) != set(COLUMNS):
            add_error("Colonnes absentes, supplémentaires ou dupliquées ; schéma bank-full attendu.", 1)
        else:
            for values in reader:
                report["row_count"] += 1
                line = reader.line_num
                if len(values) != len(header):
                    add_error("Nombre de champs incorrect ou ligne vide.", line)
                    continue
                row_key = tuple(values)
                if row_key in seen_rows:
                    report["duplicate_count"] += 1
                seen_rows.add(row_key)
                row = dict(zip(header, values))
                for field, value in row.items():
                    if not value.strip():
                        add_error(f"Valeur vide dans {field}.", line, field)
                        continue
                    if field in CATEGORIES:
                        if value not in CATEGORIES[field]:
                            add_error(f"Catégorie non autorisée dans {field}.", line, field)
                        elif value == "unknown":
                            unknown_counts[field] += 1
                    else:
                        if not re.fullmatch(r"[+-]?[0-9]+", value):
                            add_error(f"Entier fini attendu dans {field}.", line, field)
                            continue
                        try:
                            number = int(value)
                            if not math.isfinite(float(number)):
                                raise ValueError("Nombre non fini.")
                        except (ValueError, OverflowError):
                            add_error(f"Entier fini attendu dans {field}.", line, field)
                            continue
                        lower, upper = INTEGER_BOUNDS[field]
                        if (lower is not None and number < lower) or (upper is not None and number > upper):
                            add_error(f"Valeur hors domaine dans {field}.", line, field)
                if row["y"] in report["class_counts"]:
                    report["class_counts"][row["y"]] += 1
    except UnicodeDecodeError:
        add_error("Encodage UTF-8 invalide.")
    except csv.Error as error:
        add_error(f"CSV mal formé : {error}.")

    if report["row_count"] < min_rows:
        add_error(f"Volume insuffisant : au moins {min_rows} ligne(s) requise(s).")
    report["unknown_counts"] = dict(unknown_counts)
    if report["duplicate_count"]:
        report["warnings"].append("Doublons exacts observés ; aucune suppression automatique.")
    report["valid"] = report["error_count"] == 0
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["valid"]:
        raise DataValidationError(report)
    return report


def validate_dataset(
    csv_path: str | Path,
    *,
    min_rows: int = 1,
    report_path: str | Path | None = None,
) -> dict:
    """Valider un fichier ; sa lecture ou sa validation doivent réussir pour continuer."""
    csv_path = Path(csv_path)
    if report_path is not None and Path(report_path).resolve() == csv_path.resolve():
        raise ValueError("Le rapport ne doit pas écraser le CSV source.")
    return validate_csv(
        csv_path.read_bytes(), source=str(csv_path), min_rows=min_rows, report_path=report_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Valider un lot au format bank-full.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-rows", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        report = validate_dataset(args.csv_path, min_rows=args.min_rows, report_path=args.report)
    except (OSError, ValueError) as error:
        print(f"Validation refusée : {error}", file=sys.stderr)
        return 1
    print(f"Validation réussie : {report['row_count']} lignes ; SHA-256 {report['sha256']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
