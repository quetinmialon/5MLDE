# Synthèse OPS — Bank Marketing

Ce document présente les choix techniques du projet, les preuves que la chaîne
fonctionne de bout en bout, les résultats des tests de charge, le
dimensionnement recommandé et les limites connues. Les commandes d'installation
et d'exécution sont dans le [README](../README.md).

Toutes les valeurs citées ont été **mesurées ou constatées** sur des exécutions
réelles du 18 et du 19 novembre 2026 (Windows 11, Docker Desktop WSL2), sauf
mention explicite *hypothèse*.

---

## 1. Périmètre

- **Problème** : estimer la probabilité qu'un client souscrive à un dépôt à terme
  après un appel de campagne (classification binaire, cible `y`).
- **Données** : UCI Bank Marketing, `bank-full.csv`. 45 211 lignes, 16 variables,
  39 922 `no` et 5 289 `yes` (11,7 % de positifs). Licence CC BY 4.0.
- **Objectif** : démontrer une chaîne MLOps reproductible, traçable et
  exploitable. Le modèle est volontairement simple ; la performance prédictive
  n'est pas l'objectif.
- **Exécution** : entièrement locale avec Docker Compose (9 services, dont
  Locust à la demande).

### Circulation des données et des artefacts

| Étape | Produit | Stockage | Identifiant |
|---|---|---|---|
| Ingestion | `bank-full.csv` + manifeste + rapport qualité | `data/` (volume, non versionné) | SHA-256 de l'archive et du CSV |
| Rejeu | Lots d'entraînement, validation et test fixes | `data/replay/` | SHA-256 de chaque lot ; `batch_id` |
| Entraînement | Pipeline scikit-learn, métriques, rapport | MLflow (PostgreSQL + volume d'artefacts) | `run_id`, version du modèle |
| Mise en service | Alias `champion` + run d'audit | Registre MLflow | Version, raison, versions avant/après |
| Prédiction | Probabilité, classe, version | PostgreSQL `predictions` | `prediction_id` (UUID) |
| Retour terrain | Vraie classe | PostgreSQL `feedbacks` | Clé étrangère vers la prédiction |
| Exploitation | Erreurs, exécutions du pipeline | PostgreSQL `api_errors`, `pipeline_runs` | Route, code, `flow_run_id` |
| Supervision | Métriques techniques | Prometheus (rétention 7 jours) | Labels `route`, `status` |

---

## 2. Décisions techniques

Pour chaque décision : le choix, sa raison, l'alternative écartée et la limite
acceptée.

### 2.1 Données et qualité

- **Choix** : contrat de données **bloquant**, appliqué avant tout prétraitement.
  Il vérifie les 17 colonnes exactes, des entiers finis dans leurs bornes métier
  (par exemple `day` entre 1 et 31, `pdays` ≥ −1), les catégories documentées et
  l'absence de cellule vide. Il contrôle aussi l'empreinte SHA-256 de l'archive
  et du CSV. Les doublons sont signalés mais pas supprimés.
- **Raison** : la consigne impose d'interrompre l'entraînement si les données ne
  sont pas conformes. Les bornes viennent de la documentation métier et non des
  minima et maxima observés, pour ne pas refuser une valeur légitime.
- **Écarté** : Great Expectations ou Pandera. Un module de quelques fonctions
  suffit pour 17 colonnes et reste lisible.
- **Limite** : pas de détection de dérive statistique. Les distributions sont
  seulement décrites dans le rapport.

### 2.2 Modèle et évaluation

- **Choix** : un `Pipeline` scikit-learn (StandardScaler sur les 6 numériques,
  OneHotEncoder sur les 9 catégorielles, LogisticRegression), ajusté **uniquement
  sur l'entraînement**. La référence est un `DummyClassifier(strategy="prior")`.
- **Fuite de cible** : `duration` est exclue, car la durée de l'appel n'est
  connue qu'après l'appel. `y` est également exclue. Il reste 15 variables.
- **Découpe** : 70 / 15 / 15, **sans mélange**, dans l'ordre chronologique de la
  source (31 647 / 6 782 / 6 782 lignes). Les prévalences diffèrent fortement
  (5,8 % / 11,5 % / 39,3 % de positifs) : c'est une conséquence assumée de
  l'ordre temporel.
- **Métriques** : précision, rappel, F1, ROC-AUC, **average precision (AP)** et
  matrice de confusion. L'accuracy seule serait trompeuse avec 11,7 % de positifs.
- **Limite** : au seuil de 0,5, le modèle ne prédit aucun `yes` sur la
  validation (rappel nul). Ce résultat est conservé et documenté, sans
  réglage a posteriori.

### 2.3 Orchestration (Prefect 3)

- **Choix** : un flow en 5 tâches :
  1. ingestion et contrôle ;
  2. préparation du rejeu ;
  3. validation du prochain lot ;
  4. entraînement et version MLflow ;
  5. enregistrement de la progression.

  Il est servi par `flow.serve()` avec un intervalle (1 h par défaut), sur un
  serveur Prefect adossé à PostgreSQL.
- **Simulation d'arrivée de données** : seule la partition d'entraînement est
  découpée en lots de 10 000 lignes. Chaque exécution entraîne sur les lots
  **cumulés**, alors que la validation et le test restent **fixes** : sans cela,
  deux versions ne seraient pas comparables.
- **Retries ciblés** : seule l'ingestion est réessayée (2 fois, 3 s d'intervalle),
  et seulement pour des erreurs transitoires (réseau, HTTP 408, 429 ou 5xx). Un
  lot invalide n'est **jamais** réessayé : réessayer ne corrige pas une donnée
  fausse.
- **Doublons** : un verrou de fichier garantit une seule exécution à la fois.
  Un lot déjà présent dans MLflow (même `batch_id`) n'est pas réentraîné, même si
  le fichier de progression a été perdu.
- **Écarté** : un work pool avec des workers distribués, inutile sur une seule
  machine.
- **Limite** : les alertes du flow sont locales (fichier, artefact Prefect, ligne
  `pipeline_runs`) ; aucune notification n'est envoyée.

### 2.4 Traçabilité et mise en service (MLflow)

- **Choix** : chaque entraînement est un run MLflow qui contient :
  - les paramètres, les métriques du modèle et du naïf, et le rapport qualité ;
  - les jeux train et validation référencés par leur empreinte ;
  - le code source (instantané et `code_sha256`), le commit Git s'il est
    disponible, la configuration et les versions installées ;
  - le pipeline avec sa signature.

  Chaque candidat devient une **version** avec un tag `validation_status`. Le
  rechargement du modèle et l'égalité des probabilités sont vérifiés avant
  validation.
- **Règle de mise en service**, définie avant les expériences : la promotion est
  manuelle et n'est acceptée que si l'AP de validation est au moins égale à celle
  du naïf, **puis** au moins égale à celle du champion actuel, sur une validation
  identique (même empreinte). Les métriques du test ne sont jamais utilisées pour
  décider.
- **Alias vs tag** : un tag décrit une version ; l'alias `champion` désigne la
  version servie. Le retour arrière consiste à réassigner l'alias à une version
  validée déjà utilisée. Chaque changement crée un **run d'audit** (versions
  avant et après, raison).
- **Écarté** : la promotion automatique en fin de pipeline. Le pipeline crée des
  candidats, la décision reste humaine et tracée.
- **Limite** : le contrôle puis la modification de l'alias ne sont pas
  transactionnels. Un seul opérateur doit agir à la fois.

### 2.5 API de prédiction (FastAPI)

- **Choix** : le champion est chargé **une fois en mémoire** au démarrage, comme le
  recommande le cours (pas de rechargement à chaque requête). Le rafraîchissement
  est forcé via `POST /model/refresh`. Modèle et version sont remplacés en une
  seule affectation, pour qu'une requête ne voie jamais le modèle d'une version
  associé au numéro d'une autre.
- **Validation stricte** (Pydantic) :
  - catégories et bornes reprises du contrat de données ;
  - entiers JSON stricts (`"35"` ou `35.5` refusés) ;
  - champs inconnus refusés, donc `duration` et `y` aussi (réponse 422).
- **Erreurs maîtrisées** : 422 pour une entrée invalide, 404 pour un feedback sur
  une prédiction inconnue, 409 pour un doublon de feedback, 503 si aucun modèle
  n'est chargé ou si la base est indisponible, 500 pour une erreur inattendue.
  Toutes sont tracées.
- **Choix assumé** : si la prédiction ne peut pas être enregistrée, l'API répond
  503 sans résultat. Une prédiction non tracée ne doit pas être servie, car elle
  ne pourrait recevoir ni feedback ni audit.
- **Limite** : pas d'authentification (le service n'est exposé que sur
  127.0.0.1) et un seul worker uvicorn (voir §4).

### 2.6 Persistance (PostgreSQL)

- **Choix** : une instance PostgreSQL 17 avec **une base et un compte par usage**
  (`mlflow`, `prefect`, `app`, et `grafana` en lecture seule). Aucun de ces comptes
  n'est superutilisateur. Les données sont sur des volumes persistants.
- **Schéma applicatif** : 4 tables en SQL simple via psycopg2, sans ORM :
  `predictions`, `feedbacks`, `api_errors` et `pipeline_runs`. Les contraintes
  sont en base (clé étrangère et unicité des feedbacks, domaines des classes,
  probabilité entre 0 et 1).
- **Initialisation idempotente** (service `db-init`) : relancer la pile sur un
  volume existant ajoute ce qui manque sans rien recréer.
- **Limite** : une connexion par requête, mesurée comme une part du coût par
  requête (§4) ; une seule instance PostgreSQL pour tous les services.

### 2.7 Supervision (Prometheus et Grafana)

- **Choix** : deux sources complémentaires, sur les trois niveaux du cours
  (infrastructure, API, métier) :
  - **Prometheus** pour les métriques techniques de l'API : débit par route et
    code, histogramme des latences, version servie, CPU et mémoire du processus ;
  - **PostgreSQL** pour l'historique métier : prédictions par version,
    distribution des probabilités, feedbacks et accord, erreurs, exécutions du
    pipeline.
- **Labels bornés** : route déclarée et code HTTP, jamais d'identifiant de
  requête, pour ne pas faire exploser le nombre de séries.
- **Tout est provisionné par fichiers** : sources, dashboard et 3 règles d'alerte
  (échec de pipeline, API indisponible, plus de 5 % de réponses 5xx).
- **Écarté** : cAdvisor ou node-exporter. Les métriques du processus et
  `docker stats` suffisent au besoin.
- **Limite** : pas de canal de notification ; CPU et mémoire suivis pour l'API
  seulement dans Grafana.

### 2.8 Reproductibilité, déploiement et sécurité

- **Dépendances figées** : `requirements.txt` épingle les 167 paquets et les
  installe avec `--no-deps`, ce qui donne une résolution identique partout ;
  `pip check` est exécuté en CI.
- **Versions figées** : Python 3.12 ; les images Docker (PostgreSQL, Prometheus,
  Grafana, Locust) ont des tags exacts.
- **CI GitHub Actions** (Ubuntu 24.04 et Windows 2022) : installation dans un
  environnement virtuel isolé, contrôle des dépendances, import du paquet et
  tests.
- **Secrets** : tous dans `.env`, ignoré par Git. `.env.example` ne contient que
  les noms.
- **Exposition locale** : les ports ne sont publiés que sur 127.0.0.1.
- **Conteneurs** : utilisateur non-root (UID 10001) et
  `restart: unless-stopped` sur les services permanents.

---

## 3. Preuves d'intégration

### 3.1 Fonctionnement nominal

| Exigence | Démonstration réalisée | Résultat constaté |
|---|---|---|
| Ingestion reproductible | Téléchargement UCI **et** archive locale | 45 211 lignes, SHA-256 `d1513ec6…` identique, 0 erreur, 0 doublon |
| Entraînement traçable | Run initial | Validation : ROC-AUC 0,674 et AP 0,264 (naïf 0,115). Fit en 0,31 s |
| Versionnement | Pipeline planifié sur les lots 1 à 4 | Versions 3 (10 000 lignes), 4 (20 000), 5 (30 000) et 6 (31 647), toutes `passed` |
| Reproductibilité | Comparer la v6 (4 lots cumulés) et la v1 (découpe initiale) | **Même AP de validation (0,2643)** et même empreinte de validation, alors que l'entraînement est passé par le rejeu |
| Planification | `flow.serve()` avec un intervalle de 60 s, puis 1 h | Exécutions déclenchées sans intervention (lots 1 et 2, puis lot 4 à une échéance horaire) |
| Idempotence | Exécutions sans nouveau lot | État `Skipped`, aucune nouvelle version |
| Règle de promotion | `promote 4` | **Refusée** : AP 0,1295 inférieure à celle du champion (0,2643) |
| Promotion | `promote 2` (AP 0,2643 = champion) | Acceptée, run d'audit `f24d3533…` |
| Version servie | `/predict` avant et après `/model/refresh` | Version 1 avant le refresh, 2 après : l'alias seul ne change pas le modèle servi |
| Retour arrière | `rollback 1` puis refresh | La version 1 est constatée dans les réponses et en base. Audit `ee8390ae…` |
| Persistance | Prédictions et feedbacks | Chaque ligne contient la version, la probabilité, la durée et les entrées ; le feedback est lié à sa prédiction |
| Persistance des services | Conteneurs PostgreSQL et MLflow recréés | Runs, versions et alias retrouvés ; le modèle se recharge et prédit à l'identique |
| Supervision | Trafic réel puis lecture de chaque panneau via l'API Grafana | 14 panneaux alimentés par des données réelles |

L'évolution de l'AP au fil des lots (0,127 → 0,130 → 0,226 → 0,264) montre ce
que la traçabilité apporte. Avec peu de données récentes, les premières versions
sont moins bonnes que le champion. La règle de promotion bloquerait les
versions 3 à 5 ; la version 6 égale le champion et serait acceptée.

### 3.2 Incidents contrôlés

| Incident provoqué | Détection | Réaction du système | Champion | Retour à la normale |
|---|---|---|---|---|
| Lot avec `day=0` (CLI) | Contrat de données | Code 1, rapport d'erreur ; aucun modèle ni CSV publié | Inchangé (SHA vérifié) | — |
| Lot avec `day=0` dans le pipeline | Tâche « Valider le prochain lot » | Flow `Failed`, alerte en fichier, artefact Prefect, ligne `pipeline_runs`, **alerte Grafana déclenchée** | Inchangé (v1) | Alerte résolue après sa fenêtre de 15 min |
| PostgreSQL arrêté sous trafic | `/ready` → `not_ready` ; métriques 5xx | `/predict` en 503 avec un message explicite ; **alerte 5xx déclenchée** | Inchangé | Automatique au redémarrage, sans relancer l'API |
| API arrêtée | Prometheus (`up = 0`) | **Alerte « API indisponible » déclenchée** | — | Résolue au redémarrage |
| Rechargement impossible | `/model/refresh` en 503 | La version en cours est conservée | Inchangé | — |
| Erreur réseau transitoire à l'ingestion | Test d'intégration | Ingestion réessayée ; un lot invalide n'est pas réessayé | — | — |

**Défaut découvert pendant ces tests.** L'arrêt de PostgreSQL a aussi fait
tomber le runner Prefect : le serveur Prefect, sans base, répondait 500, et le
runner s'est arrêté après son seuil d'erreurs. Correction :
`restart: unless-stopped` sur les 7 services permanents. Le redémarrage
automatique après un arrêt du processus a ensuite été vérifié (`RestartCount=1`).

### 3.3 Tests automatisés

**172 cas de test** (89 fonctions pytest, certaines paramétrées) passent en
local (Windows, Python 3.12.4) **et dans la CI GitHub Actions**, sous Ubuntu
24.04 et Windows 2022. Ils sont déterministes, n'utilisent pas le réseau et
s'appuient sur des jeux synthétiques.

| Domaine | Fonctions | Exemples de comportements vérifiés |
|---|---|---|
| Validation et ingestion | 22 | Colonnes manquantes, types invalides, valeurs non finies, cible invalide, mauvaise empreinte, fichier précédent préservé |
| Prétraitement, entraînement, évaluation | 18 | Ajustement sur train uniquement, `duration`/`y` absents, rechargement, non-convergence bloquante, seuil |
| Suivi et registre MLflow | 14 | Lien données/code, refus de promotion, retour arrière, échec tracé |
| Orchestration et rejeu | 12 | Lots dans l'ordre, références fixes, pas de version en double, alerte, retry ciblé |
| API | 13 | 422/404/409/503/500, version exacte, persistance, refresh, métriques bornées |
| Configuration et commandes | 10 | Paramètres invalides refusés, codes de retour des CLI |

---

## 4. Tests de charge

### 4.1 Objectif et protocole

**Objectif** : trouver les ressources nécessaires à une instance de l'API pour
tenir un seuil de service, en mesurant le parcours réel, persistance comprise.

**Seuil de service, fixé avant les essais** : p95 de `/predict` < **200 ms** et
taux d'erreur < **1 %**.

| Élément | Valeur |
|---|---|
| Hôte | AMD Ryzen 7 4800H (8 cœurs, 16 threads), 15,4 Go de RAM, Windows 11 |
| Docker | Docker Desktop 28.3.3, WSL2, 16 vCPU et 7,5 Go alloués |
| Configurations comparées | API limitée à **0,5 / 1 / 2 vCPU**, 1 Go de RAM, 1 worker uvicorn ; même image, mêmes services annexes, mêmes données |
| Contrôle de configuration | Conteneur recréé pour chaque configuration ; limite réellement appliquée lue avec `docker inspect` (arrêt de la campagne sinon) |
| Générateur | Locust 2.37.10 en conteneur limité à 2 vCPU, sur le réseau Docker |
| Parcours | 5 `/predict` (entrées aléatoires valides) pour 1 `/feedback` sur une prédiction obtenue ; environ 1 requête/s par utilisateur |
| Vérification | Contenu des réponses contrôlé (probabilité entre 0 et 1, classe, version), pas seulement le code HTTP |
| Déroulé | Échauffement de 20 s par configuration (non retenu), paliers de 10, 20*, 30, 60 et 90 utilisateurs, montée de 10 s exclue, **60 s stables**, **2 répétitions** |
| Ressources | `docker stats` échantillonné pendant les phases stables (API et PostgreSQL) |
| Volume | 31 412 requêtes mesurées au total |

\* Le palier à 20 utilisateurs a été ajouté après la campagne principale, pour
1 et 2 vCPU seulement, afin de préciser la zone de bascule.

**Rejouer la campagne** : `python load_tests/run_campaign.py`. Les résultats sont
dans `reports/load/summary.csv` et `summary-u20.csv`, avec les exports Locust
bruts dans `reports/load/raw/`.

### 4.2 Résultats

Aucune erreur HTTP et aucune réponse invalide dans aucun essai. Les débits sont
les moyennes des 2 répétitions ; les percentiles sont donnés pour la répétition 1
puis la 2. « Débit / offert » compare le débit obtenu au débit nominal de
1 req/s par utilisateur.

| vCPU | Utilisateurs | Débit total (req/s) | Débit / offert | `/predict` p50 (ms) | `/predict` p95 (ms) | `/predict` p99 (ms) | CPU API moyen | Seuil |
|---|---|---|---|---|---|---|---|---|
| 0,5 | 10 | 9,3 | 93 % | 35 / 36 | 380 / 740 | 930 / 1 300 | 25 % | ✗ |
| 0,5 | 30 | 12,6 | 42 % | 1 500 / 1 400 | 3 300 / 3 400 | 3 500 / 4 000 | 48 % | ✗ |
| 0,5 | 60 | 10,8 | 18 % | 4 600 / 4 700 | 7 700 / 6 700 | 8 000 / 7 300 | 50 % | ✗ |
| 0,5 | 90 | 11,4 | 13 % | 6 800 / 7 500 | 8 100 / 8 800 | 8 600 / 9 100 | 50 % | ✗ |
| 1 | 10 | 9,6 | 96 % | 35 / 36 | **95 / 74** | 150 / 97 | 26 % | ✓ |
| 1 | 20 | 16,0 | 80 % | 110 / 96 | 1 000 / 1 000 | 1 200 / 1 400 | 74 % | ✗ |
| 1 | 30 | 25,7 | 86 % | 57 / 53 | 670 / 720 | 1 100 / 950 | 69 % | ✗ |
| 1 | 60 | 28,9 | 48 % | 1 200 / 1 100 | 2 300 / 2 100 | 2 700 / 2 400 | 98 % | ✗ |
| 1 | 90 | 22,2 | 25 % | 2 600 / 3 600 | 4 800 / 6 600 | 5 300 / 8 300 | 101 % | ✗ |
| 2 | 10 | 9,5 | 95 % | 48 / 32 | **160 / 60** | 210 / 130 | 29 % | ✓ |
| 2 | 20 | 18,3 | 92 % | 64 / 71 | 210 / 290 | 310 / 380 | 73 % | ✗ (proche) |
| 2 | 30 | 26,3 | 88 % | 81 / 96 | 420 / 460 | 590 / 580 | 94 % | ✗ |
| 2 | 60 | 32,6 | 54 % | 920 / 920 | 1 500 / 1 700 | 1 800 / 2 000 | 139 % | ✗ |
| 2 | 90 | 29,0 | 32 % | 1 900 / 2 400 | 2 600 / 3 600 | 2 800 / 4 000 | 145 % | ✗ |

**Ressources** (maximum sur tous les essais) :

| Mesure | 0,5 vCPU | 1 vCPU | 2 vCPU |
|---|---|---|---|
| Débit total maximal observé | 12,7 req/s | 29,5 req/s | 33,2 req/s |
| CPU API instantané maximal | 54 % | 106 % | 179 % |
| Mémoire API maximale | 213 Mio | 213 Mio | 213 Mio |
| CPU PostgreSQL (moyenne d'un essai, maximum) | 17 % | 34 % | 50 % |

**Détail par route** (1 vCPU, 10 utilisateurs, répétition 1) : `/predict` p50
35 ms et p95 95 ms ; `/feedback` p50 23 ms et p95 45 ms. Le feedback ne fait
qu'une insertion SQL, sans inférence.

### 4.3 Analyse

1. **Utilisateurs ≠ débit.** Avant saturation, le débit suit la charge offerte
   (93 à 96 % à 10 utilisateurs). Après saturation, il plafonne, puis **baisse**
   à 90 utilisateurs (1 vCPU : 28,9 puis 22,2 req/s). Les requêtes attendent dans
   la file : la latence explose (p50 de plusieurs secondes), sans erreur HTTP,
   car aucun délai d'expiration n'est configuré côté serveur.
2. **Le CPU de l'API est le goulot d'étranglement.** À chaque saturation, le CPU
   de l'API atteint sa limite (50 % à 0,5 vCPU, environ 100 % à 1 vCPU), alors
   que PostgreSQL reste au plus à 0,5 cœur.
3. **Coût par requête.** Sans saturation, une requête consomme environ
   **27 ms de CPU** (1 vCPU, 10 utilisateurs : 26 % de CPU pour 9,6 req/s). Sous
   contention, ce coût monte à 34–50 ms. Ce temps comprend l'inférence, la
   construction d'un DataFrame, la connexion PostgreSQL et l'insertion.
4. **2 vCPU apportent peu avec un seul worker.** Le débit maximal passe de 29,5 à
   33,2 req/s (+13 %) et le CPU utilisé plafonne vers 1,45 cœur. Un processus
   Python exécute son code sur un seul cœur à la fois (GIL) ; seules les parties
   en C (numpy, psycopg2) profitent du second cœur.
5. **Zone de bascule entre 10 et 20 req/s.** À 1 vCPU, le seuil est tenu à 10
   utilisateurs mais pas à 20 (p95 de 1 s). À 2 vCPU, 20 utilisateurs sont tout
   proches du seuil (p95 de 210 et 290 ms).
6. **Instabilité près de la saturation.** À 1 vCPU, le p95 est moins bon à 20
   (1 000 ms) qu'à 30 utilisateurs (670–720 ms). Près de la saturation, les files
   d'attente rendent le p95 très sensible ; 2 répétitions de 60 s ne suffisent
   pas à lisser cet effet. Je retiens donc une capacité **prudente**.
7. **Mémoire stable.** La mémoire de l'API ne dépend pas de la charge : 193 Mio
   à 10 utilisateurs, 213 Mio à 90.

### 4.4 Validité des mesures

- **Campagne invalidée puis refaite.** Lors de la première campagne, le
  lancement de Locust recréait l'API avec sa configuration par défaut (1 vCPU),
  à cause de la dépendance Compose entre les deux services. La configuration
  « 0,5 vCPU » n'était donc pas réellement appliquée. Ces résultats ont été
  écartés, et le script lance désormais Locust sans recréer ses dépendances,
  puis **vérifie la limite appliquée** avant chaque série.
- **Critère de débit mal calibré.** Un critère de débit (au moins 95 % de 1 req/s
  par utilisateur) avait été annoncé avec les seuils. En boucle fermée, la
  latence réduit elle-même le débit, donc 95 à 96 % est le maximum atteignable.
  La décision repose sur le p95 et le taux d'erreur, comme prévu.
- **Biais connus** : Locust et l'API partagent le même hôte (Locust est limité à
  2 vCPU) ; tout tourne sous WSL2 ; PostgreSQL contient déjà les traces des
  essais précédents (le volume de la table grossit pendant la campagne) ; les entrées sont aléatoires mais
  valides. Les chiffres valent **pour cet environnement**.

---

## 5. Dimensionnement recommandé

### 5.1 Instance de l'API

| Ressource | Recommandation | Justification mesurée |
|---|---|---|
| CPU | **1 vCPU** | Seuil tenu à environ 10 req/s (p95 74–95 ms) avec 26 % de CPU moyen. 0,5 vCPU échoue dès 10 req/s (p95 380–740 ms). 2 vCPU n'apportent que 13 % de débit maximal en plus. |
| Mémoire | **512 Mo** | Maximum mesuré : 213 Mio, stable quelle que soit la charge, soit une marge de plus de 2 fois. |
| Capacité retenue | **10 req/s** (environ 8 req/s de `/predict`) | Dernier palier où le seuil est tenu sur les deux répétitions. |
| Processus | 1 worker uvicorn | Les métriques Prometheus actuelles sont propres à un processus. |

### 5.2 Montée en charge

- **Au-delà de 10 req/s, il faut ajouter des instances plutôt que des vCPU.** Le
  gain mesuré de 1 à 2 vCPU est faible, car le processus est limité par le GIL.
  Pour un besoin de *D* req/s, l'*hypothèse* est de prévoir ⌈*D* / 10⌉ instances
  de 1 vCPU derrière un répartiteur, plus une de réserve. Par exemple, 30 req/s
  donneraient 3 + 1 instances. **Non mesuré** : à valider par une campagne
  multi-instances.
- **Ce qu'il faudrait adapter pour cela** : passer `prometheus_client` en mode
  multiprocessus (ou scraper chaque réplica), ajouter un répartiteur, et passer
  à un pool de connexions PostgreSQL. Avec N instances, le nombre de connexions
  ouvertes est multiplié.
- **PostgreSQL** n'est pas le facteur limitant : au plus 50 % d'un cœur, à
  plus de 29 req/s. À 10 req/s, il consomme environ 13 % d'un cœur.

### 5.3 Pile complète (exécution locale)

Mesure au repos après la campagne (`docker stats`, instantané) :

| Service | Mémoire | CPU |
|---|---|---|
| MLflow | 1 946 Mio | 1 % |
| pipeline-runner | 243 Mio | 0 % |
| Prefect | 182 Mio | 10 % |
| API | 173 Mio | 0 % |
| Grafana | 105 Mio | 0 % |
| PostgreSQL | 105 Mio | 1 % |
| Prometheus | 32 Mio | 0 % |
| **Total** | **≈ 2,8 Gio** | |

Pour faire tourner toute la pile avec une instance d'API à 10 req/s, je
recommande au minimum **4 vCPU et 4 Go de RAM**. Le CPU se répartit ainsi :
1 vCPU pour l'API, environ 0,15 cœur pour PostgreSQL, le reste pour les services
peu actifs et pour la marge. Ce besoin est *déduit* de mesures au repos et sous
charge de l'API, et non d'une mesure de la pile entière sous charge.

MLflow occupe à lui seul environ 2 Gio au repos. Cette valeur a été observée
mais pas analysée ; c'est le premier poste à étudier pour réduire l'empreinte.

### 5.4 Pistes d'optimisation (non mises en œuvre)

Ces pistes s'attaquent au coût par requête, d'environ 27 ms de CPU :

- un pool de connexions PostgreSQL au lieu d'une connexion par requête ;
- éviter de construire un DataFrame pandas pour une seule ligne ;
- un délai d'expiration côté serveur, pour échouer vite plutôt que de laisser la
  file s'allonger sous surcharge.

---

## 6. Exploitation : détecter et réagir

| Signal | Où le voir | Action |
|---|---|---|
| Alerte « Échec du pipeline » | Grafana, table `pipeline_runs`, artefact Prefect, `data/replay/alerts/` | Lire le message, corriger ou écarter le lot, relancer le flow. Le champion n'a pas bougé. |
| Alerte « API indisponible » | Grafana, `up{job="api"}` | `docker compose ps` et logs de l'API ; redémarrage automatique si le processus s'est arrêté. |
| Alerte « Taux d'erreurs serveur » | Grafana, `/ready` | Si `database: unavailable`, vérifier PostgreSQL. Si `model_version: null`, vérifier MLflow puis `/model/refresh`. |
| p95 en hausse | Panneau « Latence /predict » et CPU de l'API | Proche de 10 req/s par instance : ajouter une instance (§5). |
| Nouvelle version candidate | MLflow (`validation_status`) | `registry promote` (règle appliquée), puis `/model/refresh`. |
| Régression constatée après promotion | Grafana (accord des feedbacks), traces | `registry rollback <version>`, puis `/model/refresh`. Vérifier la version dans les réponses. |

---

## 7. Limites connues

| Limite | Impact | Piste |
|---|---|---|
| Rappel nul au seuil de 0,5 ; probabilités non calibrées | Le modèle n'est pas utilisable tel quel pour décider qui appeler | Choisir le seuil sur la validation selon un coût métier ; calibration |
| Données de 2008-2010, ordre temporel sans dates complètes | Validité métier actuelle faible ; prévalences très différentes entre partitions | Données récentes ; suivi de dérive |
| Arrivée de données simulée par rejeu | Pas de vraie fraîcheur ni de vrai délai de vérité terrain | Source réellement alimentée |
| Feedbacks simulés pendant la démonstration ; vérité terrain partielle | L'« accord » du dashboard n'est pas une performance réelle | Délai et biais de sélection à prendre en compte |
| Machine unique, pas de haute disponibilité, PostgreSQL partagé | Une panne de la base touche l'API, Prefect et Grafana (constaté) | Instances séparées, réplication |
| Promotion et retour arrière non transactionnels | Risque si deux opérateurs agissent en même temps | Verrou ou procédure à opérateur unique |
| Pas d'authentification sur l'API ni sur `/model/refresh` | Acceptable seulement en local (127.0.0.1) | Jeton ou réseau privé |
| Alertes sans canal de notification | Il faut regarder Grafana pour les voir | Point de contact e-mail ou webhook |
| Erreurs survenues pendant une panne de la base | Visibles uniquement dans les journaux | Journalisation centralisée |
| Dimensionnement : une seule instance mesurée, hôte partagé avec Locust, 2 × 60 s | La capacité au-delà d'une instance est une hypothèse | Campagne multi-instances, essais plus longs |
| L'image Docker ne contient pas Git | Les runs lancés depuis le conteneur n'ont pas de commit, seulement `code_sha256` | Passer le commit au build de l'image |
