"""Parcours de charge : prédictions avec persistance, puis feedback sur une prédiction obtenue."""

import random

from locust import HttpUser, between, task

JOBS = ["admin.", "blue-collar", "management", "retired", "services", "student", "technician"]
MONTHS = ["feb", "mar", "may", "jun", "aug", "sep", "oct", "nov"]


def random_client() -> dict:
    return {
        "age": random.randint(20, 80), "job": random.choice(JOBS),
        "marital": random.choice(["married", "single", "divorced"]),
        "education": random.choice(["primary", "secondary", "tertiary", "unknown"]),
        "default": "no", "balance": random.randint(-500, 10000),
        "housing": random.choice(["yes", "no"]), "loan": random.choice(["yes", "no"]),
        "contact": random.choice(["cellular", "telephone", "unknown"]),
        "day": random.randint(1, 31), "month": random.choice(MONTHS),
        "campaign": random.randint(1, 5), "pdays": random.choice([-1, -1, -1, 90, 180]),
        "previous": random.choice([0, 0, 1, 3]), "poutcome": random.choice(["unknown", "failure", "success"]),
    }


class BankMarketingUser(HttpUser):
    # Environ une requête par seconde et par utilisateur : débit offert ≈ nombre d'utilisateurs.
    wait_time = between(0.5, 1.5)

    def on_start(self):
        self.prediction_ids = []

    @task(5)
    def predict(self):
        with self.client.post("/predict", json=random_client(), catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            body = response.json()
            # Vérifier le contenu, pas seulement le code HTTP.
            if not (0 <= body.get("probability", -1) <= 1 and body.get("predicted_label") in {"yes", "no"}
                    and body.get("model_version")):
                response.failure("Réponse de prédiction incomplète")
                return
            self.prediction_ids.append(body["prediction_id"])

    @task(1)
    def feedback(self):
        if not self.prediction_ids:
            return
        prediction_id = self.prediction_ids.pop()
        payload = {"prediction_id": prediction_id, "actual_label": random.choice(["yes", "no"])}
        with self.client.post("/feedback", json=payload, catch_response=True) as response:
            if response.status_code != 201:
                response.failure(f"HTTP {response.status_code}")
