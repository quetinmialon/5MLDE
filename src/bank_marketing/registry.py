"""Promotion et retour arrière explicites, avec historique dans MLflow."""

import argparse
import sys

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException

from bank_marketing.settings import load_tracking_config
from bank_marketing.tracking import configure_tracking, validation_decision


def get_champion(client, model_name: str):
    try:
        registered = client.get_registered_model(model_name)
    except MlflowException as error:
        if error.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise
    version = registered.aliases.get("champion")
    return client.get_model_version(model_name, version) if version is not None else None


def checked_version(client, model_name: str, version: str, minimum_gain: float):
    """Vérifier le run et la règle ; un tag seul ne suffit pas à autoriser l'usage."""
    model_version = client.get_model_version(model_name, str(version))
    run = client.get_run(model_version.run_id)
    if model_version.status != "READY" or run.info.status != "FINISHED":
        raise ValueError("Le modèle doit être prêt et issu d'un run terminé avec succès.")
    if model_version.tags.get("validation_status") != "passed" or run.data.tags.get("validation_status") != "passed":
        raise ValueError("Version non validée ; changement d'alias refusé.")
    if run.data.tags.get("quality_status") != "passed" or run.data.tags.get("model_reload_status") != "passed":
        raise ValueError("Contrôles de qualité ou de rechargement absents.")
    digest = model_version.tags.get("validation_digest")
    if not digest or digest != run.data.tags.get("validation_digest"):
        raise ValueError("Référence de validation absente ou incohérente.")
    try:
        candidate = run.data.metrics["logistic_regression.validation.average_precision"]
        baseline = run.data.metrics["dummy.validation.average_precision"]
    except KeyError as error:
        raise ValueError("Métriques de validation absentes.") from error
    decision = validation_decision(candidate, baseline, minimum_gain)
    if not decision["passed"]:
        raise ValueError("Le modèle ne satisfait pas la comparaison à la référence naïve.")
    return model_version, candidate


def change_champion(
    config_path: str = "config/project.toml", *, version: str, reason: str, operation: str = "promote",
) -> dict:
    """Valider puis modifier un seul alias ; aucune promotion à l'entraînement."""
    if operation not in {"promote", "rollback"}:
        raise ValueError("Opération de registre inconnue.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Une raison explicite est requise.")
    if not str(version).isdigit() or int(version) < 1:
        raise ValueError("La version doit être un entier positif.")
    config = load_tracking_config(config_path)
    client, experiment_id = configure_tracking(config)
    name = config["mlflow"]["model_name"]
    previous = get_champion(client, name)
    if previous is not None and str(previous.version) == str(version):
        raise ValueError("Cette version est déjà champion.")
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{operation}-v{version}", tags={
        "operation": operation, "model_name": name, "to_version": str(version),
        "from_version": str(previous.version) if previous else "none",
    }) as run:
        minimum_gain = config["registry"]["min_average_precision_gain"] if operation == "promote" else 0.0
        target, score = checked_version(client, name, str(version), minimum_gain)
        if operation == "promote" and previous is not None:
            current, current_score = checked_version(client, name, previous.version, 0.0)
            if current.tags["validation_digest"] != target.tags["validation_digest"]:
                raise ValueError("Jeux de validation différents ; comparaison de promotion refusée.")
            if score < current_score:
                raise ValueError("Régression de l'average precision par rapport au champion.")
        if operation == "rollback":
            if previous is None:
                raise ValueError("Aucun champion à remplacer.")
            history = client.search_runs(
                [experiment_id],
                filter_string=f"tags.model_name = '{name}' AND tags.to_version = '{version}' AND attributes.status = 'FINISHED'",
                max_results=1,
            )
            if not history:
                raise ValueError("Cette version n'a jamais été champion dans l'historique tracé.")
        # Charger la version exacte avant de changer la référence servie.
        mlflow.sklearn.load_model(f"models:/{name}/{version}")
        decision = {
            "operation": operation, "model_name": name,
            "from_version": str(previous.version) if previous else None,
            "to_version": str(version), "reason": reason.strip(),
            "validation_digest": target.tags["validation_digest"],
            "validation_average_precision": score,
            "audit_run_id": run.info.run_id,
        }
        mlflow.log_dict(decision, "decision.json")
        # Un seul opérateur à la fois : les contrôles et la mutation ne sont pas une transaction.
        latest = get_champion(client, name)
        if (latest.version if latest else None) != (previous.version if previous else None):
            raise ValueError("Le champion a changé pendant les contrôles ; recommencer l'opération.")
        client.set_registered_model_alias(name, "champion", str(version))
    return decision


def load_champion(config_path: str = "config/project.toml"):
    """Résoudre l'alias puis charger la version immuable correspondante."""
    config = load_tracking_config(config_path)
    client, _ = configure_tracking(config)
    name = config["mlflow"]["model_name"]
    champion = get_champion(client, name)
    if champion is None:
        raise ValueError("Aucun modèle champion défini.")
    checked_version(client, name, champion.version, 0.0)
    model = mlflow.sklearn.load_model(f"models:/{name}/{champion.version}")
    return model, str(champion.version)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Changer explicitement le modèle champion.")
    parser.add_argument("operation", choices=("promote", "rollback"))
    parser.add_argument("version")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--config", default="config/project.toml")
    args = parser.parse_args(argv)
    try:
        decision = change_champion(args.config, version=args.version, reason=args.reason, operation=args.operation)
    except (OSError, ValueError, MlflowException) as error:
        print(f"Changement refusé : {error}", file=sys.stderr)
        return 1
    print(f"{decision['operation']} : champion {decision['from_version']} -> {decision['to_version']} ; audit {decision['audit_run_id']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
