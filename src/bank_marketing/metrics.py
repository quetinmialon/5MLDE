"""Métriques techniques de l'API exposées à Prometheus sur /metrics.

L'historique métier (prédictions, feedbacks, traitements) est lu dans PostgreSQL.
Sous Linux, prometheus_client expose aussi CPU et mémoire du processus.
"""

from prometheus_client import Counter, Gauge, Histogram

# Labels bornés : route déclarée (pas l'URL brute) et code HTTP, jamais d'identifiant.
API_REQUESTS = Counter("api_requests", "Requêtes HTTP traitées.", ["route", "status"])
API_LATENCY = Histogram(
    "api_request_duration_seconds", "Durée de traitement des requêtes.", ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
MODEL_VERSION = Gauge("api_model_version", "Version du champion servie (0 si aucun modèle).")
