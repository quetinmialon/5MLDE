"""Métriques binaires calculées à partir des probabilités de souscription."""

import math
from numbers import Real

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, features, target, *, threshold: float = 0.5) -> dict:
    """Évaluer sans réajuster le modèle ni modifier le seuil fourni."""
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise ValueError("Le seuil doit être un nombre entre 0 et 1.")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Le seuil doit être un nombre fini entre 0 et 1.")
    target = np.asarray(target)
    if target.ndim != 1 or set(np.unique(target)) != {0, 1}:
        raise ValueError("L'évaluation nécessite les deux classes 0 et 1.")
    classes = list(model.classes_)
    if len(classes) != 2 or set(classes) != {0, 1}:
        raise ValueError("Le modèle doit prédire les classes 0 et 1.")
    probabilities = np.asarray(model.predict_proba(features))
    if probabilities.shape != (len(target), 2):
        raise ValueError("Dimensions des probabilités incompatibles avec la cible.")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError("Probabilités invalides ou non finies.")
    positive_probability = probabilities[:, classes.index(1)]
    prediction = (positive_probability >= threshold).astype(int)
    return {
        "row_count": len(target),
        "positive_rate": float(target.mean()),
        "threshold": float(threshold),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, positive_probability)),
        "average_precision": float(average_precision_score(target, positive_probability)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": confusion_matrix(target, prediction, labels=[0, 1]).tolist(),
    }
