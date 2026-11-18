"""Prétraitement partagé par l'entraînement et la prédiction."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_marketing.validation import CATEGORIES, INTEGER_BOUNDS


NUMERIC_FEATURES = [name for name in INTEGER_BOUNDS if name != "duration"]
CATEGORICAL_FEATURES = [name for name in CATEGORIES if name != "y"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Créer un prétraitement non ajusté ; les autres colonnes sont exclues."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
