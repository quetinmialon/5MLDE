# Bank Marketing — 5MLDE

Projet MLOps de Master 2 SUPINFO : estimer la probabilité de souscription à un
dépôt à terme à partir du dataset UCI Bank Marketing.

L'objectif est de démontrer une chaîne reproductible, traçable et exploitable,
de l'ingestion à la supervision. La complexité du modèle et la recherche du
meilleur score ne sont pas les objectifs du projet.

## État du projet

Le paquet Python, l'ingestion reproductible et les contrôles de qualité sont
disponibles. Le modèle et les services MLOps seront ajoutés progressivement.

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

La CI reprend ces contrôles sous Windows et Linux. Les tests utilisent des
données synthétiques et ne dépendent pas du réseau ni d'un dataset téléchargé.

## Données

Source : [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing),
licence CC BY 4.0. Attribution : Moro, S., Rita, P., & Cortez, P. (2014),
*Bank Marketing*, DOI : 10.24432/C5K306.

Depuis la racine du dépôt, télécharger et valider `bank-full.csv` :

```powershell
.\.venv\Scripts\python.exe -m bank_marketing.ingestion
```

Pour utiliser l'archive fournie à côté du dépôt, ajouter
`--archive ../bank+marketing.zip`. Les paramètres et empreintes SHA-256 sont dans
`config/project.toml` ; ses chemins sont relatifs à ce fichier. L'ingestion
vérifie l'archive, le CSV et sa qualité avant de remplacer les données locales.
Elle génère `data/raw/bank-full.csv`, `data/raw/manifest.json` et un rapport
identifié par l'empreinte du CSV dans `data/reports/`. Le dossier `data/` est ignoré.

Pour valider un autre lot au même format :

```powershell
.\.venv\Scripts\python.exe -m bank_marketing.validation chemin/lot.csv --report data/reports/lot.json
```

Le contrat vérifie les 17 colonnes, les catégories, les entiers finis, les
domaines et l'absence de cellules vides. Les soldes négatifs, `pdays = -1` et les
catégories `unknown` documentées sont acceptés. Les doublons sont signalés sans
suppression. Un lot non conforme produit une exception et un code de sortie 1 ;
le rapport conserve les erreurs. Une classe unique est autorisée dans un lot ;
l'exploitabilité du jeu d'entraînement sera vérifiée à l'étape du modèle.

Les statistiques de qualité sont descriptives, sans détection de dérive à ce
stade. `duration` reste dans le CSV brut et sera exclue des entrées du modèle.

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
├── src/bank_marketing/             # Configuration, ingestion et validation
└── tests/                         # Tests unitaires et commandes en intégration
```
