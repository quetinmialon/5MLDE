"""Persistance des prédictions, feedbacks et erreurs de l'API dans PostgreSQL."""

import logging
import os
from contextlib import contextmanager
from uuid import UUID

import psycopg2
from psycopg2.extras import Json

logger = logging.getLogger(__name__)


class PersistenceError(RuntimeError):
    """Base indisponible ou requête impossible : la trace n'est pas enregistrée."""


class UnknownPredictionError(LookupError):
    pass


class DuplicateFeedbackError(ValueError):
    pass


@contextmanager
def transaction():
    """Une connexion courte par opération ; commit en sortie, rollback en cas d'erreur."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise PersistenceError("DATABASE_URL non défini.")
    try:
        connection = psycopg2.connect(url, connect_timeout=3)
    except psycopg2.Error as error:
        raise PersistenceError("Base de données indisponible.") from error
    try:
        with connection, connection.cursor() as cursor:
            yield cursor
    except psycopg2.IntegrityError:
        raise
    except psycopg2.Error as error:
        raise PersistenceError("Échec de la requête en base.") from error
    finally:
        connection.close()


def check_database() -> None:
    with transaction() as cursor:
        cursor.execute("SELECT 1")


def save_prediction(
    prediction_id: UUID, model_version: str, features: dict,
    probability: float, predicted_label: str, duration_ms: float,
) -> None:
    with transaction() as cursor:
        cursor.execute(
            "INSERT INTO predictions (id, model_version, features, probability, predicted_label, duration_ms)"
            " VALUES (%s, %s, %s, %s, %s, %s)",
            (str(prediction_id), model_version, Json(features), probability, predicted_label, duration_ms),
        )


def save_feedback(prediction_id: UUID, actual_label: str) -> None:
    try:
        with transaction() as cursor:
            cursor.execute(
                "INSERT INTO feedbacks (prediction_id, actual_label) VALUES (%s, %s)",
                (str(prediction_id), actual_label),
            )
    except psycopg2.errors.ForeignKeyViolation as error:
        raise UnknownPredictionError("Prédiction inconnue.") from error
    except psycopg2.errors.UniqueViolation as error:
        raise DuplicateFeedbackError("Feedback déjà enregistré pour cette prédiction.") from error


def save_error(route: str, status_code: int, error: str, model_version: str | None, duration_ms: float) -> None:
    """Best effort : une base indisponible ne doit pas masquer l'erreur d'origine."""
    try:
        with transaction() as cursor:
            cursor.execute(
                "INSERT INTO api_errors (route, status_code, error, model_version, duration_ms)"
                " VALUES (%s, %s, %s, %s, %s)",
                (route, status_code, error[:500], model_version, duration_ms),
            )
    except (PersistenceError, psycopg2.Error):
        logger.warning("Erreur API non enregistrée en base : %s %s", route, status_code)
