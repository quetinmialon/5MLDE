# Synthèse OPS — Bank Marketing

Choix techniques, preuves d'intégration, tests de charge, dimensionnement
recommandé et limites connues. Les commandes sont dans le [README](../README.md).

## 1. Décisions techniques

| Sujet | Choix | Raison |
|---|---|---|
| Modèle | Pipeline scikit-learn : standardisation + one-hot + régression logistique | Simple et explicable ; le sujet est la chaîne, pas le score |
| Fuite de cible | `duration` et `y` exclus ; découpe 70/15/15 sans mélange | `duration` n'est connue qu'après l'appel |
| Qualité des données | Contrat bloquant (colonnes, types, domaines, catégories), SHA-256 | Arrêt avant prétraitement si non conforme |
| Arrivée de données | Rejeu par lots de la seule partition d'entraînement | Dataset historique ; validation et test fixes pour comparer les versions |
| Orchestration | Prefect 3, `flow.serve()` planifié, retries sur l'ingestion seulement | Un lot invalide ne se corrige pas en réessayant |
| Traçabilité | MLflow : runs, données, code, dépendances, registre, alias `champion` | Une version précise par décision, avec un run d'audit |
| Mise en service | Promotion manuelle si AP validation ≥ naïf et ≥ champion | Règle fixée avant les expériences ; retour arrière par alias |
| Service | FastAPI, champion en mémoire, `/model/refresh` | Pas de rechargement à chaque requête |
| Persistance | PostgreSQL, SQL simple (psycopg2), une base et un compte par service | Pas d'ORM pour 4 tables ; droits minimaux |
| Supervision | Prometheus (technique) + PostgreSQL (historique métier) → Grafana | Labels bornés ; alertes provisionnées par fichiers |
| Déploiement | Docker Compose local, `restart: unless-stopped`, secrets dans `.env` | Exécution locale suffisante pour le projet |

## 2. Preuves d'intégration (exécutions réelles)

- **Données** : 45 211 lignes, SHA-256 conforme, aucune erreur de contrat. Un lot avec `day=0` est refusé (code 1) et le fichier publié n'est pas modifié.
- **Modèle** (validation) : ROC-AUC 0,674 et AP 0,264, contre 0,115 pour le naïf. Au seuil de 0,5, **le rappel est nul** : cette limite est conservée.
- **MLflow** : les versions sont créées avec `validation_status`. La promotion de la version 4 est **refusée** (AP inférieure à celle du champion). Les essais promotion 1 → 2 puis retour arrière 2 → 1 sont tracés dans des runs d'audit.
- **Prefect** : deux exécutions réellement planifiées (lots 1 et 2) ont produit les versions 3 et 4. Un lot invalide entraîne l'état `Failed`, une alerte (fichier et artefact Prefect, puis alerte Grafana), aucune nouvelle version, et le champion reste inchangé.
- **API** : la réponse contient la version exacte du modèle. Après une promotion, la version servie ne change qu'après `/model/refresh`. Après un retour arrière, la version 1 est constatée dans les réponses et dans les traces.
- **Incident « base indisponible »** : `/predict` renvoie 503 sans résultat et `/ready` passe en `not_ready`. L'alerte 5xx se déclenche dans Grafana. Au redémarrage de la base, le service revient à la normale sans intervention.
- **Incident « API arrêtée »** : l'alerte « API indisponible » se déclenche, puis se résout au redémarrage.
- **Constat pendant les tests** : la panne de PostgreSQL a aussi arrêté le runner Prefect (le serveur Prefect renvoyait des 500). Correction : `restart: unless-stopped`, avec redémarrage automatique vérifié.
- **Tests automatisés** : 172 tests (pytest) passent localement sous Windows et sont exécutés en CI sous Ubuntu et Windows.

## 3. Tests de charge

**Protocole** (`load_tests/run_campaign.py`, exports dans `reports/load/`) :

- **Hôte** : AMD Ryzen 7 4800H (8 cœurs, 16 threads), 15,4 Go de RAM, Docker Desktop WSL2 (16 vCPU, 7,5 Go).
- **Configurations comparées** : l'API seule est limitée à **0,5, 1 et 2 vCPU**, avec 1 Go de RAM et 1 worker uvicorn. La limite réellement appliquée est contrôlée avec `docker inspect`. Les services annexes et les données sont identiques d'une configuration à l'autre.
- **Charge** : Locust dans un conteneur limité à 2 vCPU. Chaque utilisateur envoie environ 1 requête par seconde, avec 5 `/predict` pour 1 `/feedback`. La persistance est incluse dans le parcours, et le contenu des réponses est vérifié (pas seulement le code HTTP).
- **Déroulé** : un échauffement de 20 s par configuration, puis des paliers de 10, 20, 30, 60 et 90 utilisateurs. Chaque palier comprend une montée de 10 s exclue des mesures, puis **60 s stables**, et il est **répété 2 fois**. Le palier à 20 utilisateurs a été ajouté ensuite, pour 1 et 2 vCPU seulement.
- **Ressources** : `docker stats` est échantillonné pendant les phases stables, pour l'API et pour PostgreSQL.
- **Seuil de service fixé avant les essais** : p95 de `/predict` < 200 ms et moins de 1 % d'erreurs.

**Résultats** : aucune erreur HTTP dans aucun essai. Le débit est la moyenne des deux répétitions ; le p95 indique la répétition 1 puis la 2.

| vCPU | Utilisateurs | Débit total (req/s) | `/predict` p95 (ms) | CPU API moyen | Seuil |
|---|---|---|---|---|---|
| 0,5 | 10 | 9,3 | 380 / 740 | 25 % | ✗ |
| 0,5 | 30 | 12,6 | 3 300 / 3 400 | 48 % (plafond) | ✗ |
| 0,5 | 90 | 11,4 | 8 100 / 8 800 | 50 % (plafond) | ✗ |
| 1 | 10 | 9,6 | 95 / 74 | 26 % | ✓ |
| 1 | 20 | 16,0 | 1 000 / 1 000 | 74 % | ✗ |
| 1 | 30 | 25,7 | 670 / 720 | 69 % | ✗ |
| 1 | 60 | 28,9 | 2 300 / 2 100 | 98 % (plafond) | ✗ |
| 1 | 90 | 22,2 | 4 800 / 6 600 | 101 % (plafond) | ✗ |
| 2 | 10 | 9,5 | 160 / 60 | 30 % | ✓ |
| 2 | 20 | 18,3 | 210 / 290 | 73 % | ✗ (proche) |
| 2 | 30 | 26,3 | 420 / 460 | 94 % | ✗ |
| 2 | 60 | 32,6 | 1 500 / 1 700 | 139 % | ✗ |
| 2 | 90 | 29,0 | 2 600 / 3 600 | 145 % | ✗ |

**Lecture des résultats**

- Le nombre d'utilisateurs n'est pas le débit obtenu. Au-delà de la saturation, les requêtes attendent : la latence augmente et le débit n'augmente plus, voire baisse à 90 utilisateurs.
- La mémoire de l'API reste sous **213 Mio** dans tous les essais. PostgreSQL consomme au plus environ 0,5 cœur (49 %) au débit maximal ; ce n'est pas le goulot d'étranglement.
- Avec 2 vCPU, le débit maximal ne gagne qu'environ 13 % (32,6 contre 28,9 req/s). Un seul processus Python n'utilise jamais plus de 1,45 cœur environ, à cause du GIL.
- Une fois saturé, le service est instable : à 1 vCPU, le p95 est moins bon à 20 qu'à 30 utilisateurs. Deux répétitions ne suffisent pas à lisser cet effet.
- Le critère de débit (≥ 95 % de 1 req/s par utilisateur), annoncé avec les seuils, était mal calibré. En boucle fermée, la latence réduit elle-même le débit : à 10 utilisateurs on obtient 93 à 96 %. Je l'indique sans changer la décision, qui repose sur le p95 et le taux d'erreur.

## 4. Dimensionnement recommandé

- **Par instance de l'API (1 worker) : 1 vCPU et 512 Mo de RAM.** Le seuil est tenu jusqu'à environ **10 req/s au total (environ 8 req/s sur `/predict`)**, avec un CPU moyen d'environ 26 %. La mémoire mesurée est de 213 Mio au maximum, soit plus du double de marge avec 512 Mo.
- **0,5 vCPU est insuffisant** : le p95 dépasse déjà 380 ms à 10 utilisateurs.
- **Passer à 2 vCPU pour un seul worker n'est pas rentable** : on double le CPU pour un débit maximal à peine supérieur, et le seuil n'est pas tenu à 20 utilisateurs.
- **Au-delà de 10 req/s**, il faut ajouter des instances de 1 vCPU (réplicas ou workers derrière un répartiteur) plutôt que des CPU. *Hypothèse non mesurée* : environ une instance par tranche de 10 req/s, à valider par une nouvelle campagne. Avec plusieurs workers, il faudrait aussi passer `prometheus_client` en mode multiprocessus.
- **Pistes pour réduire le coût par requête**, environ 30 ms de CPU, non mises en œuvre : un pool de connexions PostgreSQL (aujourd'hui une connexion par requête) et éviter de construire un DataFrame pandas à chaque requête.

## 5. Scénario de démonstration

1. `docker compose up -d --build --wait`, puis ouvrir Prefect (:4200), MLflow (:5000) et Grafana (:3000).
2. Exécuter le flow (`python -m bank_marketing.flows run`) : contrôles, entraînement, puis nouvelle version dans MLflow.
3. `registry promote <version>`, puis `POST /model/refresh` : la nouvelle version apparaît dans `/predict`.
4. Envoyer une prédiction, puis un feedback : on les retrouve dans le dashboard (sections usage et historique).
5. Corrompre un lot de rejeu, puis exécuter le flow : état `Failed`, alerte « Échec du pipeline » et champion inchangé.
6. `registry rollback <version>`, puis `/model/refresh` : l'ancienne version est constatée dans les réponses et dans les traces.
7. Arrêter `postgres` sous trafic : réponses 503, alerte 5xx, puis retour à la normale au redémarrage.
8. Présenter `reports/load/summary.csv` et le dimensionnement ci-dessus.

## 6. Limites connues

- **Modèle** : rappel nul au seuil de 0,5 ; probabilités non calibrées ; données de 2008-2010 qui ne reflètent pas une banque actuelle.
- **Feedbacks** : vérité terrain partielle, et dans la démonstration ils sont simulés.
- **Exploitation** : machine unique, sans haute disponibilité ; une seule instance PostgreSQL partagée par les services ; promotion et retour arrière non transactionnels (un seul opérateur à la fois).
- **Sécurité** : l'API n'a pas d'authentification ; les ports sont exposés uniquement sur 127.0.0.1.
- **Alertes** : visibles dans Grafana, sans canal de notification (e-mail…) ; les erreurs survenues pendant une panne de la base ne figurent que dans les journaux.
- **Tests de charge** : Locust et l'API partagent le même hôte ; seulement 2 répétitions de 60 s ; une seule machine, sous WSL2. Les chiffres valent pour cet environnement.
