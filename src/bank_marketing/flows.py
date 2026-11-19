"""Pipeline Prefect de rejeu, avec retries ciblés, alerte locale et résumé en base."""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError

from filelock import FileLock, Timeout
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact
from prefect.cache_policies import NO_CACHE
from prefect.runtime import flow_run as runtime_flow_run
from prefect.types.entrypoint import EntrypointType

from bank_marketing import database
from bank_marketing.ingestion import ingest_dataset
from bank_marketing.replay import commit_batch, next_batch, prepare_replay, write_json_atomic
from bank_marketing.settings import load_orchestration_config, load_tracking_config
from bank_marketing.tracking import configure_tracking, track_training
from bank_marketing.validation import validate_dataset


def retry_ingestion(task, task_run, state) -> bool:
    try:
        state.result()
    except HTTPError as error:
        return error.code in {408, 429} or error.code >= 500
    except (URLError, TimeoutError, ConnectionError):
        return True
    except Exception:
        return False
    return False


def alert_failure(flow, flow_run, state) -> None:
    config_path = Path(flow_run.parameters.get("config_path", "config/project.toml"))
    try:
        replay_dir = load_orchestration_config(config_path)["orchestration"]["replay_dir"]
    except (OSError, ValueError):
        replay_dir = config_path.resolve().parent / "data" / "replay"
    alert = {
        "severity": "critical", "flow_run_id": str(flow_run.id),
        "flow_run_name": flow_run.name, "state": state.name,
        "message": (state.message or "Échec du pipeline")[:2000],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(replay_dir / "alerts" / f"{flow_run.id}.json", alert)
    duration = (state.timestamp - flow_run.start_time).total_seconds() if flow_run.start_time else None
    database.save_pipeline_run(str(flow_run.id), flow_run.name, state.name, duration, message=alert["message"])
    try:
        create_markdown_artifact(
            key="bank-marketing-alerts", description=f"Échec du pipeline {flow_run.name}",
            markdown=(
                f"# Alerte critique\n\nRun : `{flow_run.id}`\n\nÉtat : **{state.name}**\n\n"
                f"{alert['message']}\n\nLe champion n'est pas modifié par ce pipeline."
            ),
        )
    except Exception:
        get_run_logger().exception("Alerte conservée localement ; publication Prefect impossible.")


@task(name="Ingestion et contrôle du snapshot", cache_policy=NO_CACHE)
def acquire_source(config_path: str, archive_path: str | None = None) -> str:
    config = load_orchestration_config(config_path)
    source = config["data"]["raw_dir"] / "bank-full.csv"
    if archive_path is None and source.is_file():
        if hashlib.sha256(source.read_bytes()).hexdigest() == config["data"]["csv_sha256"]:
            validate_dataset(source, min_rows=config["validation"]["min_rows"])
            return str(source)
    return str(ingest_dataset(config_path, archive_path=archive_path))


@task(name="Préparer les références fixes", cache_policy=NO_CACHE)
def prepare_replay_task(config_path: str, source: str) -> dict:
    return prepare_replay(config_path, source)


@task(name="Valider le prochain lot", cache_policy=NO_CACHE)
def select_batch(config_path: str, manifest: dict, batch_number: int | None) -> dict | None:
    return next_batch(config_path, manifest, batch_number)


@task(name="Entraînement et version MLflow", retries=0, cache_policy=NO_CACHE)
def train_candidate(config_path: str, batch: dict, flow_run_id: str) -> dict:
    config = load_tracking_config(config_path)
    client, experiment_id = configure_tracking(config)
    previous = client.search_runs(
        [experiment_id],
        filter_string=(f"tags.batch_id = '{batch['batch_id']}' AND tags.operation = 'training' "
                       f"AND tags.model_name = '{config['mlflow']['model_name']}' AND attributes.status = 'FINISHED'"),
        max_results=1, order_by=["attributes.start_time DESC"],
    )
    if previous:
        path = client.download_artifacts(previous[0].info.run_id, "tracking.json")
        get_run_logger().info("Lot déjà enregistré dans MLflow : reprise du reçu existant.")
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return track_training(
        config_path, csv_path=batch["train_csv"], validation_csv=batch["validation_csv"],
        test_csv=batch["test_csv"], batch_id=batch["batch_id"], flow_run_id=flow_run_id,
    )


@task(name="Enregistrer la progression", cache_policy=NO_CACHE)
def commit_batch_task(config_path: str, batch: dict, receipt: dict) -> None:
    commit_batch(config_path, batch, receipt)


def record_run(state: str, started: float, **details) -> None:
    database.save_pipeline_run(
        str(runtime_flow_run.id), runtime_flow_run.name, state, time.perf_counter() - started, **details,
    )


@flow(name="bank-marketing-pipeline", retries=0, on_failure=[alert_failure], on_crashed=[alert_failure])
def bank_marketing_pipeline(
    config_path: str = "config/project.toml", archive_path: str | None = None,
    batch_number: int | None = None,
) -> dict:
    started = time.perf_counter()
    config_path = str(Path(config_path).resolve())
    config = load_orchestration_config(config_path)
    settings = config["orchestration"]
    settings["replay_dir"].mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(settings["replay_dir"] / ".pipeline.lock"), timeout=0)
    try:
        with lock:
            source = acquire_source.with_options(
                retries=settings["ingestion_retries"], retry_delay_seconds=settings["retry_delay_seconds"],
                retry_condition_fn=retry_ingestion,
            )(config_path, archive_path)
            manifest = prepare_replay_task(config_path, source)
            batch = select_batch(config_path, manifest, batch_number)
            if batch is None:
                get_run_logger().info("Aucun nouveau lot : aucune version supplémentaire créée.")
                record_run("Skipped", started, message="Aucun nouveau lot.")
                return {"status": "skipped", "reason": "no_new_batch"}
            receipt = train_candidate(config_path, batch, str(runtime_flow_run.id))
            create_markdown_artifact(
                key="bank-marketing-batches", description=f"Lot {batch['number']} traité",
                markdown=(f"Lot **{batch['number']}** — {batch['training_rows']} lignes d'entraînement.\n\n"
                          f"Run MLflow : `{receipt['run_id']}`\n\nVersion : **{receipt['model_version']}**\n\n"
                          f"Validation : **{receipt['validation_status']}**. Références de validation et de test fixes."),
            )
            commit_batch_task(config_path, batch, receipt)
            record_run(
                "Completed", started, batch_number=batch["number"],
                model_version=receipt["model_version"], validation_status=receipt["validation_status"],
            )
            return {"status": "completed", "batch": batch, "receipt": receipt}
    except Timeout:
        get_run_logger().info("Une exécution utilise déjà ce rejeu ; aucune modification.")
        record_run("Skipped", started, message="Exécution déjà en cours.")
        return {"status": "skipped", "reason": "already_running"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exécuter ou planifier le pipeline Prefect.")
    parser.add_argument("mode", choices=("run", "serve"))
    parser.add_argument("--config", default="config/project.toml")
    parser.add_argument("--archive")
    parser.add_argument("--batch-number", type=int)
    parser.add_argument("--interval", type=int, help="Intervalle de démonstration, en secondes.")
    parser.add_argument("--deployment-name", default="bank-marketing-hourly")
    args = parser.parse_args(argv)
    try:
        config = load_orchestration_config(args.config)
    except (OSError, ValueError) as error:
        print(f"Configuration refusée : {error}", file=sys.stderr)
        return 1
    os.environ.setdefault("PREFECT_API_URL", config["orchestration"]["api_url"])
    if args.mode == "serve":
        interval = args.interval if args.interval is not None else config["orchestration"]["interval_seconds"]
        if interval < 1:
            parser.error("L'intervalle doit être strictement positif.")
        from bank_marketing.flows import bank_marketing_pipeline as deployed_pipeline

        deployed_pipeline.serve(
            name=args.deployment_name, interval=interval, limit=1, pause_on_shutdown=False,
            parameters={"config_path": str(Path(args.config).resolve()), "archive_path": args.archive},
            global_limit=1, entrypoint_type=EntrypointType.MODULE_PATH,
        )
        return 0
    try:
        bank_marketing_pipeline(args.config, args.archive, args.batch_number)
    except Exception as error:
        print(f"Pipeline refusé : {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
