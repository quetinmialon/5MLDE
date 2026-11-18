from types import SimpleNamespace

import numpy as np
import pytest

from bank_marketing.evaluation import evaluate_model


def make_model(probabilities=None, classes=(0, 1)):
    if probabilities is None:
        positive = np.array([0.1, 0.6, 0.4, 0.9])
        probabilities = np.column_stack((1 - positive, positive))
    return SimpleNamespace(classes_=np.array(classes), predict_proba=lambda features: np.array(probabilities))


def test_metrics_match_known_predictions():
    report = evaluate_model(make_model(), np.zeros((4, 1)), [0, 0, 1, 1])
    assert report["precision"] == pytest.approx(0.5)
    assert report["recall"] == pytest.approx(0.5)
    assert report["f1"] == pytest.approx(0.5)
    assert report["roc_auc"] == pytest.approx(0.75)
    assert report["average_precision"] == pytest.approx(5 / 6)
    assert report["confusion_matrix"] == [[1, 1], [1, 1]]


def test_threshold_is_explicit_and_not_fitted():
    report = evaluate_model(make_model(), np.zeros((4, 1)), [0, 0, 1, 1], threshold=0.3)
    assert report["recall"] == 1
    assert report["precision"] == pytest.approx(2 / 3)
    assert report["threshold"] == 0.3


def test_positive_class_column_is_resolved_from_classes():
    positive = np.array([0.1, 0.6, 0.4, 0.9])
    model = make_model(np.column_stack((positive, 1 - positive)), classes=(1, 0))
    assert evaluate_model(model, np.zeros((4, 1)), [0, 0, 1, 1])["roc_auc"] == 0.75


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="seuil"):
        evaluate_model(make_model(), np.zeros((4, 1)), [0, 0, 1, 1], threshold=threshold)


@pytest.mark.parametrize("probabilities", [
    [[0.5, float("nan")]] * 4,
    [[1.1, -0.1]] * 4,
    [[0.2, 0.2]] * 4,
    [[0.5, 0.5]] * 3,
    [0.1, 0.6, 0.4, 0.9],
])
def test_invalid_probability_output_is_rejected(probabilities):
    with pytest.raises(ValueError):
        evaluate_model(make_model(probabilities), np.zeros((4, 1)), [0, 0, 1, 1])


@pytest.mark.parametrize("target", [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 2, 2], []])
def test_invalid_or_single_class_target_is_rejected(target):
    with pytest.raises(ValueError, match="deux classes"):
        evaluate_model(make_model(), np.zeros((4, 1)), target)
