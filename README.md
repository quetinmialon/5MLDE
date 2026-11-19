# Bank Marketing — 5MLDE

Projet MLOps de Master 2 SUPINFO : estimer la probabilité de souscription à un
dépôt à terme à partir du dataset UCI Bank Marketing.

L'objectif est de démontrer une chaîne reproductible, traçable et exploitable,
de l'ingestion à la supervision. La complexité du modèle et la recherche du
meilleur score ne sont pas les objectifs du projet.

## Données

[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
(Moro, Cortez et Rita, 2014), licence CC BY 4.0 : 45 211 appels d'une campagne
de banque portugaise (2008-2010), 16 variables et la cible `y`. Le fichier
`bank-full.csv` est téléchargé puis contrôlé (SHA-256 de l'archive et du CSV,
contrat de données) ; il n'est pas versionné.

```powershell
.\.venv\Scripts\python.exe -m bank_marketing.ingestion   # ou --archive chemin\bank+marketing.zip
```

Les données sont historiques : l'arrivée de nouvelles données est **simulée**
en rejouant la partition d'entraînement par lots (voir Orchestration).

## Architecture

```text
UCI ──► ingestion + contrôles ──► Prefect (rejeu par lots) ──► entraînement ──► MLflow (runs, registre, alias champion)
                                         │                                              │ promotion / retour arrière manuels
                                         ▼                                              ▼
                                  pipeline_runs ◄──────── PostgreSQL ◄──────── API FastAPI (prédictions, feedbacks, erreurs)
                                         │                     │                        │ /metrics
                                         └──────── Grafana ◄───┴──────── Prometheus ◄───┘          Locust ──► API
```

Tout tourne en local avec Docker Compose : une instance PostgreSQL avec une
base et un compte par service (MLflow, Prefect, application, Grafana en lecture).

## Prérequis

- Python **3.12**, avec `venv` et `pip`.
- Git pour récupérer le dépôt.
- Accès à PyPI pour l'installation et à UCI pour télécharger les données.

La version mineure 3.12 est la cible actuelle. Docker Desktop et Docker Compose
sont nécessaires pour les services (PostgreSQL, MLflow, Prefect, API, supervision).

## Installation locale

Exécuter les commandes depuis le dossier contenant ce README et
`pyproject.toml`, après avoir récupéré le dépôt. Utiliser un environnement
virtuel neuf pour éviter les dépendances d'autres projets.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-deps -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
```

Si la commande `py` n'est pas disponible, utiliser `python -m venv .venv`
après avoir vérifié que `python --version` indique bien Python 3.12.


## Vérifications

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -I -c "import bank_marketing; from importlib.metadata import version; print(version('bank-marketing')); print(bank_marketing.__file__)"
.\.venv\Scripts\python.exe -m pytest -q
```


## Modèle simple

```powershell
.\.venv\Scripts\python.exe -m bank_marketing.training
```

Le pipeline combine standardisation, encodage des catégories et régression
logistique. Il est ajusté uniquement sur les 70 % premières lignes ; les 15 %
suivantes servent à la validation et les 15 % restantes au test, sans mélange.
Cette découpe suit l'ordre chronologique annoncé par UCI, sans dates complètes
ni identifiant client. `duration` et `y` sont exclus des 15 variables d'entrée.

Chaque exécution crée `models/<exécution>/pipeline.joblib` et `metrics.json`,
ignorés par Git : paramètres, empreinte des données, partitions, versions et
comparaison à `DummyClassifier(strategy="prior")`. Le seuil est fixé à 0,5
dans la configuration. Le test est évalué uniquement avec `--evaluate-test`,
à réserver au bilan final ; les comparaisons courantes utilisent la validation.

Les métriques incluent précision, rappel, F1, ROC-AUC, average precision et
matrice de confusion. La non-convergence bloque la sauvegarde. Le modèle reste
volontairement simple : pas d'optimisation ni de calibration des probabilités,
et `pdays = -1` est conservé numériquement. Recharger uniquement des fichiers
Joblib de confiance, avec les versions de bibliothèques enregistrées.

Mesure initiale du 18/11/2026 sur la validation : ROC-AUC 0,674 et average
precision 0,264 (référence : 0,115). Au seuil de 0,5, le rappel est nul : le
modèle ne prédit aucune souscription sur cette partition. Cette limite doit
être conservée dans l'interprétation des résultats.

## MLflow et versions

À la première installation, lorsque `.env` n'existe pas, générer les secrets :

```powershell
@'
from pathlib import Path
import secrets
with Path(".env").open("x", encoding="utf-8") as stream:
    for key in ("POSTGRES_PASSWORD", "MLFLOW_DB_PASSWORD", "PREFECT_DB_PASSWORD", "APP_DB_PASSWORD",
                "GRAFANA_DB_PASSWORD", "GRAFANA_ADMIN_PASSWORD"):
        stream.write(f"{key}={secrets.token_hex(32)}\n")
'@ | python -
```

```powershell
docker compose up -d --build --wait
.\.venv\Scripts\python.exe -m bank_marketing.tracking
.\.venv\Scripts\python.exe -m bank_marketing.registry promote 1 --reason "Premier candidat valide"
# Après la promotion d'une autre version :
.\.venv\Scripts\python.exe -m bank_marketing.registry rollback 1 --reason "Retour à la version précédente"
```

Préparer les données avec `python -m bank_marketing.ingestion` avant le premier
entraînement et adapter les numéros aux versions affichées. L'interface
[MLflow](http://127.0.0.1:5000) conserve empreintes des données, code, dépendances,
paramètres, métriques et artefacts ; les échecs restent visibles. PostgreSQL et
les artefacts disposent de volumes persistants. `.env` reste hors de Git,
de l'image et des artefacts ; le rôle MLflow n'est pas superutilisateur.

L'entraînement crée un candidat sans déplacer `champion`. La promotion exige
des contrôles réussis et une average precision de validation au moins égale
au naïf, puis au champion sur la même validation. Le retour arrière cible une
version validée déjà utilisée. Chaque changement conserve sa raison dans un
run d'audit. La limite de rappel constatée ci-dessus reste inchangée.
Effectuer ces opérations une à la fois : contrôle et mutation d'alias ne sont
pas transactionnels. `load_champion` renvoie le pipeline et sa version exacte.

## Orchestration Prefect

```powershell
docker compose up -d --build --wait   # PostgreSQL, MLflow, serveur Prefect, runner planifié
```

Le service `pipeline-runner` expose le flow `bank-marketing-pipeline` avec
`flow.serve()` (intervalle : `orchestration.interval_seconds`, 1 h par défaut).
Suivi des exécutions, états, durées et logs dans l'[interface Prefect](http://127.0.0.1:4200).
Exécution manuelle : `python -m bank_marketing.flows run`.

Étapes : ingestion et contrôle du snapshot → préparation du rejeu → validation
du prochain lot → entraînement et version MLflow → enregistrement de la progression.

- **Rejeu** : seule la partition d'entraînement est découpée en lots de 10 000
  lignes (simulation d'arrivée de données). Chaque exécution entraîne sur les
  lots cumulés ; validation et test restent fixes pour comparer les versions.
- **Retries** : seulement l'ingestion, sur erreurs réseau transitoires
  (2 tentatives). Un lot invalide n'est jamais retenté.
- **Échec** : le flow s'arrête avant l'entraînement, écrit une alerte dans
  `data/replay/alerts/` et un artefact Prefect `bank-marketing-alerts`.
  Le pipeline ne déplace jamais `champion` : la promotion reste manuelle.
- **Doublons** : une exécution à la fois (verrou), et un lot déjà enregistré
  dans MLflow n'est pas réentraîné.

Limites : l'alerte est locale (fichier + artefact), sans envoi d'e-mail ; l'image
ne contient pas Git, le code est donc identifié par son empreinte `code_sha256`.

## API de prédiction

Service `api` (FastAPI) sur http://127.0.0.1:8000, documentation interactive
sur `/docs`. Si le port est pris : `$env:API_PORT=8001; docker compose up -d api`.

| Route | Rôle |
|---|---|
| `POST /predict` | probabilité, classe (seuil 0,5), `prediction_id` et version exacte du modèle |
| `POST /feedback` | vraie réponse (`yes`/`no`) liée à un `prediction_id` |
| `POST /model/refresh` | recharge le champion après promotion ou retour arrière |
| `GET /health`, `GET /ready` | processus vivant ; modèle chargé et base joignable |

- Le champion est chargé une fois en mémoire au démarrage. Déplacer l'alias ne
  change rien tant que `/model/refresh` n'est pas appelé ; un échec de
  rechargement conserve la version en cours.
- Entrées strictes : les 15 variables du modèle, catégories et bornes du contrat
  de données, entiers JSON uniquement ; `duration` et `y` sont refusés (422).
- PostgreSQL (base `bank_marketing`, compte `app`) conserve les prédictions
  (entrées, version, probabilité, durée), les feedbacks et les erreurs des
  routes. Un feedback sur une prédiction inconnue renvoie 404, un doublon 409.
- Base indisponible : `/predict` renvoie 503 sans résultat, car une prédiction
  non tracée n'est pas servie ; `/ready` passe en `not_ready`.

Limites : pas d'authentification (service exposé uniquement en local), une
connexion PostgreSQL par requête, et les erreurs survenues pendant une panne
de la base ne sont visibles que dans les journaux.

## Supervision

[Grafana](http://127.0.0.1:3000) (compte `admin`, mot de passe `GRAFANA_ADMIN_PASSWORD`
du `.env`) → dossier *Bank Marketing*. Dashboard, sources et alertes sont
provisionnés depuis `infra/grafana/` ; [Prometheus](http://127.0.0.1:9090) lit
`/metrics` de l'API toutes les 5 s.

| Niveau | Source | Contenu |
|---|---|---|
| Service | Prometheus | disponibilité, débit par route, taux de 5xx, latences p50/p95/p99, CPU et mémoire de l'API, version servie |
| Modèle et usage | PostgreSQL | prédictions par version, distribution des probabilités, feedbacks et accord, dernières erreurs |
| Traitements | PostgreSQL | exécutions du pipeline (`pipeline_runs`) : état, lot, version, durée, message |

Alertes Grafana :
- **Échec du pipeline** : une exécution `Failed` ou `Crashed` dans les 15 dernières minutes.
- **API indisponible** : Prometheus ne joint plus l'API pendant 30 s.
- **Taux d'erreurs serveur** : plus de 5 % de réponses 5xx sur une minute (ex. base arrêtée).

Les métriques Prometheus n'ont que des labels bornés (route déclarée, code
HTTP), jamais d'identifiant de requête. Grafana lit PostgreSQL avec un compte
`grafana` en lecture seule.

Limites : les alertes sont visibles dans Grafana mais aucun canal d'envoi
(e-mail…) n'est configuré. CPU et mémoire ne couvrent que le processus de
l'API ; pour les autres services, utiliser `docker stats`. Pendant une panne
de PostgreSQL, les panneaux et l'alerte qui le lisent passent en erreur.

## Charge et dimensionnement

```powershell
.\.venv\Scripts\python.exe load_tests/run_campaign.py --cpus 0.5 1 2 --users 10 30 60 90 --repeats 2
```

La campagne recrée l'API pour chaque limite CPU (`API_CPUS`, vérifiée par
`docker inspect`), lance Locust en conteneur (`load_tests/locustfile.py`) et
échantillonne `docker stats`. Résultats : `reports/load/summary.csv` et exports
Locust bruts. Durée : environ 35 minutes.

Recommandation mesurée : **1 vCPU et 512 Mo par instance de l'API** pour environ
10 req/s avec un p95 < 200 ms. Protocole, résultats et limites sont détaillés dans
[docs/synthese-ops.md](docs/synthese-ops.md).

## Organisation actuelle

```text
.
├── .github/workflows/tests.yml
├── .env.example
├── .gitattributes
├── .gitignore
├── config/project.toml
├── compose.yaml
├── Dockerfile
├── docs/synthese-ops.md           # Choix, preuves, charge, dimensionnement, limites
├── infra/                         # PostgreSQL, Prometheus, Grafana
├── load_tests/                    # Locust et campagne de charge
├── pyproject.toml
├── requirements.txt
├── README.md
├── reports/load/                  # Résultats de charge mesurés
├── src/bank_marketing/             # Données, modèle, suivi, orchestration
│   └── api/                       # API FastAPI
└── tests/                         # Tests unitaires et commandes en intégration
```
