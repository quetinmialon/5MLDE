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

-- Résumé de chaque exécution du pipeline Prefect, pour le dashboard et l'alerte.
CREATE TABLE IF NOT EXISTS pipeline_runs (
    flow_run_id UUID PRIMARY KEY,
    flow_run_name TEXT NOT NULL,
    state TEXT NOT NULL,
    batch_number INTEGER,
    model_version TEXT,
    validation_status TEXT,
    message TEXT,
    duration_seconds DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Grafana lit les traces sans pouvoir les modifier.
GRANT USAGE ON SCHEMA public TO grafana;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana;
