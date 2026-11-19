"""API de prédiction : modèle champion gardé en mémoire, traces dans PostgreSQL."""

import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool

from bank_marketing import database
from bank_marketing.api.schemas import ClientFeatures, FeedbackRequest, PredictionResponse
from bank_marketing.preprocessing import FEATURE_COLUMNS
from bank_marketing.registry import load_champion
from bank_marketing.settings import load_training_config

CONFIG_PATH = "config/project.toml"
TRACED_ROUTES = {"/predict", "/feedback", "/model/refresh"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bank_marketing.api")


def load_serving_model(app: FastAPI) -> None:
    model, version = load_champion(CONFIG_PATH)
    # Une seule affectation : une requête ne voit jamais un modèle avec la version d'un autre.
    app.state.serving = {"model": model, "version": version}
    logger.info("Modèle champion chargé : version %s.", version)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.serving = None
    app.state.threshold = load_training_config(CONFIG_PATH)["training"]["threshold"]
    try:
        load_serving_model(app)
    except Exception:
        logger.exception("Aucun modèle chargé au démarrage ; /predict répondra 503.")
    yield


app = FastAPI(title="Bank Marketing API", lifespan=lifespan)


def current_version(request: Request) -> str | None:
    serving = request.app.state.serving
    return serving["version"] if serving else None


def fail(request: Request, status_code: int, message: str):
    request.state.error = message
    raise HTTPException(status_code, message)


async def record_error(request: Request, status_code: int, start: float) -> None:
    if request.url.path in TRACED_ROUTES:
        await run_in_threadpool(
            database.save_error, request.url.path, status_code,
            getattr(request.state, "error", f"HTTP {status_code}"),
            current_version(request), (time.perf_counter() - start) * 1000,
        )


@app.middleware("http")
async def record_errors(request: Request, call_next):
    """Conserver en base chaque erreur des routes métier, avec sa durée."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as error:
        request.state.error = f"Erreur interne : {type(error).__name__}"
        await record_error(request, 500, start)
        raise
    if response.status_code >= 400:
        await record_error(request, response.status_code, start)
    return response


@app.exception_handler(RequestValidationError)
async def record_validation_error(request: Request, error: RequestValidationError):
    request.state.error = "; ".join(
        f"{'.'.join(str(part) for part in item['loc'][1:])}: {item['msg']}" for item in error.errors()
    )
    return await request_validation_exception_handler(request, error)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request, response: Response) -> dict:
    try:
        database.check_database()
        database_status = "ok"
    except database.PersistenceError:
        database_status = "unavailable"
    version = current_version(request)
    is_ready = version is not None and database_status == "ok"
    response.status_code = 200 if is_ready else 503
    return {"status": "ready" if is_ready else "not_ready", "model_version": version, "database": database_status}


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientFeatures, request: Request) -> PredictionResponse:
    start = time.perf_counter()
    serving = request.app.state.serving
    if serving is None:
        fail(request, 503, "Aucun modèle chargé.")
    features = client.model_dump()
    frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    probability = float(serving["model"].predict_proba(frame)[0, 1])
    threshold = request.app.state.threshold
    label = "yes" if probability >= threshold else "no"
    prediction_id = uuid4()
    duration_ms = (time.perf_counter() - start) * 1000
    try:
        database.save_prediction(prediction_id, serving["version"], features, probability, label, duration_ms)
    except database.PersistenceError:
        fail(request, 503, "Prédiction non enregistrée : base de données indisponible.")
    return PredictionResponse(
        prediction_id=prediction_id, probability=probability, predicted_label=label,
        threshold=threshold, model_version=serving["version"],
    )


@app.post("/feedback", status_code=201)
def feedback(body: FeedbackRequest, request: Request) -> dict:
    try:
        database.save_feedback(body.prediction_id, body.actual_label)
    except database.UnknownPredictionError:
        fail(request, 404, "Prédiction inconnue.")
    except database.DuplicateFeedbackError:
        fail(request, 409, "Feedback déjà enregistré pour cette prédiction.")
    except database.PersistenceError:
        fail(request, 503, "Feedback non enregistré : base de données indisponible.")
    return {"status": "recorded", "prediction_id": str(body.prediction_id)}


@app.post("/model/refresh")
def refresh_model(request: Request) -> dict:
    """Rafraîchissement forcé après promotion ou retour arrière (l'alias seul ne suffit pas)."""
    previous = current_version(request)
    try:
        load_serving_model(request.app)
    except Exception:
        logger.exception("Rechargement du champion impossible.")
        fail(request, 503, f"Rechargement impossible ; version {previous} conservée.")
    return {"previous_version": previous, "model_version": current_version(request)}
