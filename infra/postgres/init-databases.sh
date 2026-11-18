#!/bin/sh
set -e

# Idempotent : fonctionne aussi avec le volume PostgreSQL déjà utilisé par MLflow.
psql --set=ON_ERROR_STOP=1 --set=prefect_password="$PREFECT_DB_PASSWORD" <<'SQL'
SELECT format('CREATE ROLE prefect LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD %L', :'prefect_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'prefect')
\gexec
SELECT 'CREATE DATABASE prefect OWNER prefect'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'prefect')
\gexec
SQL
