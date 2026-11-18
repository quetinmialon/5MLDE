-- Traces d'utilisation de l'API (idempotent).
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version TEXT NOT NULL,
    features JSONB NOT NULL,
    probability DOUBLE PRECISION NOT NULL CHECK (probability BETWEEN 0 AND 1),
    predicted_label TEXT NOT NULL CHECK (predicted_label IN ('yes', 'no')),
    duration_ms DOUBLE PRECISION NOT NULL
);

-- Un seul retour terrain par prédiction.
CREATE TABLE IF NOT EXISTS feedbacks (
    prediction_id UUID PRIMARY KEY REFERENCES predictions (id),
    actual_label TEXT NOT NULL CHECK (actual_label IN ('yes', 'no')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS api_errors (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    route TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    error TEXT NOT NULL,
    model_version TEXT,
    duration_ms DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS predictions_created_at ON predictions (created_at);
CREATE INDEX IF NOT EXISTS api_errors_created_at ON api_errors (created_at);
