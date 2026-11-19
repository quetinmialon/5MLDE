#!/bin/sh
set -e

# Idempotent : fonctionne aussi sur un volume PostgreSQL déjà initialisé.
psql --set=ON_ERROR_STOP=1 \
  --set=prefect_password="$PREFECT_DB_PASSWORD" --set=app_password="$APP_DB_PASSWORD" \
  --set=grafana_password="$GRAFANA_DB_PASSWORD" <<'SQL'
SELECT format('CREATE ROLE prefect LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD %L', :'prefect_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'prefect')
\gexec
SELECT 'CREATE DATABASE prefect OWNER prefect'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'prefect')
\gexec
SELECT format('CREATE ROLE app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD %L', :'app_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app')
\gexec
SELECT 'CREATE DATABASE bank_marketing OWNER app'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'bank_marketing')
\gexec
SELECT format('CREATE ROLE grafana LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD %L', :'grafana_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'grafana')
\gexec
SQL

# Tables de l'API créées par leur propriétaire, pas par le superutilisateur.
PGUSER=app PGPASSWORD="$APP_DB_PASSWORD" PGDATABASE=bank_marketing \
  psql --set=ON_ERROR_STOP=1 --file=/app-schema.sql
