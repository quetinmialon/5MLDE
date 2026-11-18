# Bank Marketing — 5MLDE

Projet MLOps de Master 2 SUPINFO : estimer la probabilité de souscription à un
dépôt à terme à partir du dataset UCI Bank Marketing.

L'objectif est de démontrer une chaîne reproductible, traçable et exploitable,
de l'ingestion à la supervision. La complexité du modèle et la recherche du
meilleur score ne sont pas les objectifs du projet.

## Prérequis

- Python **3.12**, avec `venv` et `pip`.
- Git pour récupérer le dépôt.
- Accès à PyPI pour l'installation et à UCI pour télécharger les données.

La version mineure 3.12 est la cible actuelle. Docker Desktop et Docker Compose
sont nécessaires pour PostgreSQL et MLflow.

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
    for key in ("POSTGRES_PASSWORD", "MLFLOW_DB_PASSWORD", "PREFECT_DB_PASSWORD"):
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
├── infra/postgres/
├── pyproject.toml
├── requirements.txt
├── README.md
├── src/bank_marketing/             # Données, prétraitement, modèle et métriques
└── tests/                         # Tests unitaires et commandes en intégration
```
