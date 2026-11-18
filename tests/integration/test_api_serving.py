"""Version servie par l'API après promotion et retour arrière dans un vrai registre MLflow."""

from fastapi.testclient import TestClient

from bank_marketing.api import app as api_module
from bank_marketing.registry import change_champion
from bank_marketing.tracking import track_training


def test_api_serves_exact_champion_and_follows_rollback_after_refresh(tracking_setup, fake_database, monkeypatch):
    config, _, rows, _, _ = tracking_setup
    monkeypatch.setattr(api_module, "CONFIG_PATH", str(config))
    first = track_training(config)["model_version"]
    change_champion(config, version=first, reason="Premier modèle")
    second = track_training(config)["model_version"]
    payload = {key: rows[0][key] for key in api_module.FEATURE_COLUMNS}
    payload = {key: int(value) if key in {"age", "balance", "day", "campaign", "pdays", "previous"} else value
               for key, value in payload.items()}

    with TestClient(api_module.app) as client:
        assert client.post("/predict", json=payload).json()["model_version"] == first
        change_champion(config, version=second, reason="Nouvelle version")
        # Modifier l'alias ne recharge pas le modèle en mémoire.
        assert client.post("/predict", json=payload).json()["model_version"] == first
        assert client.post("/model/refresh").json()["model_version"] == second
        promoted = client.post("/predict", json=payload).json()
        change_champion(config, version=first, reason="Retour arrière", operation="rollback")
        client.post("/model/refresh")
        rolled_back = client.post("/predict", json=payload).json()

    assert promoted["model_version"] == second and rolled_back["model_version"] == first
    assert 0 <= rolled_back["probability"] <= 1
    versions = [saved["model_version"] for saved in fake_database["predictions"].values()]
    assert versions == [first, first, second, first]
