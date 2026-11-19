"""Contrat d'entrée et de sortie de l'API, aligné sur le contrat de données."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from bank_marketing.validation import CATEGORIES


def one_of(column: str):
    return Literal[tuple(sorted(CATEGORIES[column]))]


Label = Literal["no", "yes"]


class ClientFeatures(BaseModel):
    """Les 15 variables du modèle ; duration et y sont refusées (champs inconnus)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "age": 35, "job": "management", "marital": "married", "education": "tertiary",
        "default": "no", "balance": 1200, "housing": "yes", "loan": "no", "contact": "cellular",
        "day": 15, "month": "may", "campaign": 2, "pdays": -1, "previous": 0, "poutcome": "unknown",
    }]})

    age: StrictInt = Field(ge=1)
    job: one_of("job")
    marital: one_of("marital")
    education: one_of("education")
    default: one_of("default")
    balance: StrictInt
    housing: one_of("housing")
    loan: one_of("loan")
    contact: one_of("contact")
    day: StrictInt = Field(ge=1, le=31)
    month: one_of("month")
    campaign: StrictInt = Field(ge=1)
    pdays: StrictInt = Field(ge=-1)
    previous: StrictInt = Field(ge=0)
    poutcome: one_of("poutcome")


class PredictionResponse(BaseModel):
    prediction_id: UUID
    probability: float
    predicted_label: Label
    threshold: float
    model_version: str


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: UUID
    actual_label: Label
