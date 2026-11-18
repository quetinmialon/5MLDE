FROM python:3.12.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_ENABLE_TELEMETRY=false

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --no-deps -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-build-isolation --no-deps .
RUN python -m pip check

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /mlflow/artifacts /app/data /app/models \
    && chown -R appuser:appuser /mlflow /app/data /app/models
USER appuser

CMD ["mlflow", "server"]
