#!/bin/sh
set -e

psql --set=ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  --set=mlflow_password="$MLFLOW_DB_PASSWORD" <<'SQL'
CREATE ROLE mlflow LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD :'mlflow_password';
CREATE DATABASE mlflow OWNER mlflow;
SQL
