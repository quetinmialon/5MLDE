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

La version mineure 3.12 est la cible actuelle. Les autres versions de Python ne
sont pas prises en charge à ce stade. Docker Compose sera nécessaire lorsque
les services seront introduits ; il n'est pas utilisé pour le socle actuel.

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

## Organisation actuelle

```text
.
├── .github/workflows/tests.yml
├── .env.example
├── .gitattributes
├── .gitignore
├── config/project.toml
├── pyproject.toml
├── requirements.txt
├── README.md
├── src/bank_marketing/             # Données, prétraitement, modèle et métriques
└── tests/                         # Tests unitaires et commandes en intégration
```
