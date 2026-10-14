# Guide du projet : maintenance prédictive

Ce document explique ce qui est en place dans le repo, les raisons de chaque choix, et la façon de travailler au quotidien.

## Objectif

Le projet entraîne un modèle qui détecte les pannes d'une machine à partir de données de capteurs (dataset AI4I 2020), et construit autour de lui une chaîne MLOps complète : versioning des données, suivi des entraînements, registre de modèles, puis API, CI/CD et monitoring.

Le score du modèle n'est pas le but. Le but est l'infrastructure : pouvoir dire à tout moment quel modèle tourne, avec quelles données et quel code il a été produit, et comment le reproduire.

## Les trois outils et leur rôle

| Outil | Rôle | Ce qu'il conserve |
|---|---|---|
| Git | Versionne le code et la configuration | `train.py`, `params.yaml`, `dvc.yaml`, `dvc.lock`, fichiers `.dvc` |
| DVC | Versionne les données et décrit le pipeline | Le contenu du CSV (sur le remote), l'empreinte de chaque entraînement |
| MLflow | Trace les entraînements et stocke les modèles | Paramètres, métriques, modèles, versions, alias |

En une phrase : DVC suit ce qui entre dans l'entraînement, MLflow garde ce qui en sort, et git relie le tout.

## Structure du repo

```
MLOPS-predictive-maintenance/
├── .dvc/                               configuration DVC
├── data/raw/
│   ├── data.csv                        les données (ignorées par git)
│   └── data.csv.dvc                    le pointeur vers les données (versionné par git)
├── notebook/
│   └── predictive_maintenance.ipynb    EDA et décision sur la cible
├── src/predictive_maintenance/
│   ├── __init__.py
│   └── train.py                        script d'entraînement
├── tests/
├── params.yaml                         réglages de l'entraînement
├── dvc.yaml                            description du pipeline
├── dvc.lock                            empreinte du dernier entraînement
└── pyproject.toml                      dépendances du projet
```

Le code suit un layout `src/` : le package doit être installé pour être importé. Les tests tournent donc contre la version installée, comme ce sera le cas en CI et dans Docker.

`mlflow.db` et `mlartifacts/` (les données du serveur MLflow) sont dans le `.gitignore`.

## Installation

```powershell
python -m venv .env
.env\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Le `-e` installe le projet en mode éditable : les modifications du code sont prises en compte sans réinstaller.

Les dépendances sont séparées en deux groupes dans `pyproject.toml`. Les `dependencies` de base contiennent uniquement ce dont l'API aura besoin en production. Les outils de développement (pytest, ruff, dvc, pyyaml, jupyter) sont dans l'extra `dev`, et ne finiront donc pas dans l'image Docker. Chaque paquet en trop dans une image alourdit le build, ralentit la CI et ajoute des vulnérabilités potentielles.

## Les données

### Le dataset

AI4I 2020 contient 10 000 lignes **synthétiques** qui simulent une machine-outil. Il y a 339 pannes, soit un taux de 3,39 %.

| Colonne | Signification | Utilisée ? |
|---|---|---|
| `UDI`, `Product ID` | Identifiants | Non |
| `Type` | Qualité du produit : L, M ou H | Oui |
| `Air temperature [K]` | Température ambiante | Oui |
| `Process temperature [K]` | Température du process | Oui |
| `Rotational speed [rpm]` | Vitesse de rotation | Oui |
| `Torque [Nm]` | Couple | Oui |
| `Tool wear [min]` | Usure de l'outil | Oui |
| `Machine failure` | 1 si la machine est en panne | Cible |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | Cause de la panne (5 modes) | Non |

Les colonnes de mode de panne décrivent la cause d'une panne qui a déjà eu lieu. Les donner au modèle reviendrait à lui fournir la réponse : c'est une fuite de cible. Les règles de HDF, PWF et OSF ont été vérifiées sur les données : elles reproduisent ces colonnes à l'identique, ce qui confirme que le dataset est synthétique.

### Décision sur la cible

En théorie, `Machine failure` vaut 1 dès qu'au moins un mode vaut 1. Le contrôle sur les données montre trois écarts : 9 pannes sans aucun mode coché, 18 lignes avec RNF coché mais sans panne, et 24 lignes avec plusieurs modes.

La décision a été de **garder `Machine failure` tel quel**. C'est l'événement observé, celui qu'on aurait en production, alors que les modes sont un diagnostic posé après coup. Recalculer la cible à partir des modes ajouterait 18 pannes aléatoires, impossibles à prédire avec les capteurs. Enfin, une panne sans cause identifiée reste une vraie panne. Le principe : on ne modifie pas les étiquettes sans raison métier, et on documente le choix.

### Versioning avec DVC

`dvc add data/raw/data.csv` crée le fichier `data.csv.dvc`, qui contient le hash md5 du CSV. Git versionne ce petit fichier pointeur, et DVC stocke le contenu réel dans son cache puis sur le remote. Le hash sert aussi d'adresse de stockage : le fichier se retrouve dans `files/md5/f0/adaa9c...` sur le remote.

Le remote est pour l'instant un dossier local, configuré avec `--local`. Cette configuration est écrite dans `.dvc/config.local`, qui n'est pas versionné, parce que ce chemin n'existe que sur cette machine.

Hash de la version actuelle des données : `f0adaa9c5e25366c6df1cc2684479871`.

## L'entraînement (`train.py`)

Le script enchaîne les étapes suivantes : lecture de `params.yaml`, chargement des données, split train/test, construction du pipeline, entraînement, évaluation, log dans MLflow, puis pose de l'alias `champion`.

### Les choix techniques et leurs justifications

**Split stratifié (80/20, `random_state` fixe).** Avec 3,4 % de pannes, un split non stratifié fait varier le nombre de pannes dans le test d'un run à l'autre, et les métriques bougent pour des raisons qui n'ont rien à voir avec le modèle. Avec la stratification, le train et le test ont tous les deux 3,4 % de pannes (271 et 68).

**Prétraitement inclus dans le pipeline.** Le `ColumnTransformer` (OneHotEncoder sur `Type`, colonnes numériques inchangées) et le Random Forest forment un seul `Pipeline` sklearn. Le modèle loggé accepte donc directement les données brutes. L'API n'aura pas à dupliquer le prétraitement, et il ne peut pas y avoir d'écart entre l'entraînement et la production.

**Pas de normalisation.** Un Random Forest découpe sur des seuils, donc l'échelle des variables ne change rien à ses découpages. Un scaler serait une étape inutile.

**`class_weight="balanced"`.** Pendant l'entraînement, une panne pèse environ 30 fois plus qu'une ligne normale. Le modèle détecte ainsi plus de pannes, au prix de quelques fausses alertes (voir les résultats plus bas).

**Paramètres dans `params.yaml`.** Les réglages ne sont plus écrits dans le code. Ils sont loggés en une seule fois avec `mlflow.log_params(params)` : ce qui est loggé est forcément ce qui a été utilisé. Le fichier est lu avec `yaml.safe_load`, et non `yaml.load`, qui peut créer des objets Python arbitraires.

**Format de sauvegarde skops.** MLflow sauvegarde le modèle au format skops plutôt qu'en pickle. Un pickle peut exécuter n'importe quel code au chargement. Skops refuse au contraire les types non déclarés, d'où le paramètre `skops_trusted_types=["sklearn.tree._tree.Tree"]` : on déclare précisément ce qu'on accepte de charger.

## MLflow

### Lancer le serveur

Dans un terminal dédié, à la racine du repo :

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

Dans le terminal de travail :

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

L'interface est accessible sur http://127.0.0.1:5000. Le script ne contient pas l'URL du serveur : il la lit dans la variable d'environnement. Le même code tournera ainsi en local, en CI et dans Docker, et seule la variable changera.

### Ce qui est enregistré

Chaque entraînement crée un **run** dans l'expérience `predictive-maintenance`. Un run contient les paramètres, les métriques, le tag `mlflow.source.git.commit` (le commit en cours au moment du run) et le modèle. Le modèle est loggé avec sa **signature** (le schéma d'entrée attendu, vérifié à chaque prédiction) et un exemple d'entrée.

Le modèle est aussi enregistré dans le **registry** sous le nom `predictive-maintenance`. Chaque entraînement y crée une nouvelle version.

### Versions et alias

Une version est un historique qu'on ne modifie pas. Un **alias** est un pointeur qu'on peut déplacer. L'alias `champion` désigne la version utilisée en production : l'API chargera `models:/predictive-maintenance@champion`, quel que soit le numéro de version derrière.

Les anciens « stages » de MLflow (Staging, Production) sont dépréciés depuis MLflow 2.9, et les alias les remplacent.

## Lire les métriques

### La matrice de confusion

| | Prédit panne | Prédit normal |
|---|---|---|
| **Vraie panne** | Vrai positif (VP) : panne détectée | Faux négatif (FN) : panne ratée |
| **Vraie normale** | Faux positif (FP) : fausse alerte | Vrai négatif (VN) : rien à signaler |

Pour s'en souvenir : le second mot (positif ou négatif) est ce que le modèle a prédit, et le premier (vrai ou faux) dit s'il avait raison.

### Les trois métriques

**Rappel** = VP / (VP + FN). Parmi les vraies pannes, combien le modèle en a-t-il détecté ?

**Précision** = VP / (VP + FP). Quand le modèle donne l'alerte, à quelle fréquence a-t-il raison ?

**F1** = 2 × VP / (2 × VP + FP + FN). C'est la moyenne harmonique des deux précédentes. Une panne ratée et une fausse alerte y pèsent exactement autant.

Ces valeurs sont loggées dans MLflow sous les noms `recall`, `precision` et `f1`. Les comptages le sont aussi : `detected_failures` (VP), `missed_failures` (FN) et `false_alarms` (FP).

### Résultats sur le test (68 pannes sur 2000 machines)

| Run | Détectées | Ratées | Fausses alertes | Précision | Rappel | F1 |
|---|---|---|---|---|---|---|
| Sans pondération | 36 | 32 | 4 | 0,90 | 0,53 | 0,67 |
| `class_weight="balanced"` | 46 | 22 | 15 | 0,75 | 0,68 | 0,71 |

Le modèle pondéré est le champion. Il détecte 10 pannes de plus en échange de 11 fausses alertes supplémentaires. Pour un centre de tri, une panne ratée sur un broyeur provoque un arrêt de ligne, alors qu'une fausse alerte se limite à une inspection pour rien : l'échange est rentable.

Le F1 est aussi meilleur, mais ce n'est pas le bon argument. Le F1 suppose qu'une panne ratée coûte autant qu'une fausse alerte, ce qui est faux ici. Choisir un modèle sur le F1, c'est faire un choix métier sans l'avoir décidé.

L'accuracy n'est pas utilisée. Un modèle qui répond toujours « pas de panne » obtiendrait 96,6 % sans jamais détecter une seule panne.

## Le pipeline DVC

### `dvc.yaml` : la recette

Ce fichier décrit l'étape d'entraînement : la commande à lancer et tout ce dont dépend son résultat.

```yaml
stages:
  train:
    cmd: python -m predictive_maintenance.train
    deps:
    - data/raw/data.csv
    - src/predictive_maintenance/train.py
    params:
    - class_weight
    - max_depth
    - n_estimators
    - random_state
    - test_size
```

Il a été généré par cette commande (en PowerShell, les guillemets autour de la liste des paramètres sont obligatoires) :

```powershell
dvc stage add -n train -d data/raw/data.csv -d src/predictive_maintenance/train.py -p "random_state,test_size,n_estimators,max_depth,class_weight" python -m predictive_maintenance.train
```

L'étape ne déclare pas de sortie, et c'est voulu : le modèle part dans MLflow, pas dans un fichier du repo.

### `dvc.lock` : ce qui a réellement servi

Après chaque exécution, DVC écrit dans `dvc.lock` l'empreinte exacte de ce qui a été utilisé : le md5 du CSV, le md5 de `train.py` et la valeur de chaque paramètre. On retrouve la même logique qu'avec `pyproject.toml` et un fichier lock : `dvc.yaml` décrit l'intention, `dvc.lock` enregistre ce qui s'est réellement passé.

### Les commandes

**`dvc repro`** compare l'empreinte actuelle des dépendances au `dvc.lock`. Si rien n'a changé, il saute l'étape (`Stage 'train' didn't change, skipping`). Si quelque chose a changé (données, code ou paramètre listé), il relance l'entraînement et met le `dvc.lock` à jour. DVC raisonne sur des empreintes de fichiers : un simple commentaire ajouté dans `train.py` suffit à relancer l'étape.

**`dvc status`** indique si l'étape est périmée et pour quelle raison.

**`dvc params diff`** compare les paramètres actuels à ceux du dernier commit.

### Reproduire un ancien modèle

```powershell
git checkout <commit>
dvc pull
dvc repro --force
```

Le `--force` est nécessaire parce que le modèle vit dans MLflow et non dans le repo : DVC ne peut pas vérifier qu'il existe déjà, et sans ce flag il conclurait que tout est à jour.

## Le travail au quotidien

### En début de session

```powershell
git pull
dvc pull
dvc status
```

`git pull` récupère le code et les pointeurs, puis `dvc pull` récupère les données qui leur correspondent. `dvc status` indique si le pipeline est à jour. Lancer ensuite le serveur MLflow et définir `MLFLOW_TRACKING_URI`.

### Modifier un paramètre ou le code d'entraînement

```powershell
# après avoir modifié params.yaml ou train.py
dvc status
git add params.yaml
git commit -m "description du changement"
dvc repro
git add dvc.lock
git commit -m "dvc.lock après entraînement"
```

On commite **avant** `dvc repro` pour que le run MLflow soit rattaché au commit qui contient vraiment le code et les paramètres utilisés. Le `dvc.lock` n'est mis à jour qu'à la fin de l'entraînement, d'où un second commit : le premier contient la décision, le second le résultat de l'exécution.

### Utiliser une nouvelle version des données

```powershell
# après avoir remplacé data/raw/data.csv
dvc add data/raw/data.csv
git add data/raw/data.csv.dvc
git commit -m "données v2"
dvc repro
git add dvc.lock
git commit -m "dvc.lock après réentraînement"
```

### En fin de session, ou pour partager

```powershell
dvc push
git push
```

Toujours dans cet ordre, pour qu'aucun commit sur GitHub ne pointe vers des données absentes du remote. `dvc install` installe un hook git qui lance le `dvc push` automatiquement avant chaque `git push`.

### La règle principale

On ne lance plus `python -m predictive_maintenance.train` directement, sauf pour déboguer. Un lancement direct ne met pas à jour le `dvc.lock`, qui décrirait alors un autre entraînement que celui qui a produit le modèle. Pour entraîner, on passe par `dvc repro`.

## Limites actuelles

**L'alias est posé sans condition.** Chaque `dvc repro` place `champion` sur le nouveau modèle, même s'il est moins bon. En attendant la règle de promotion, après un essai décevant, il faut remettre l'alias sur la bonne version à la main, dans l'onglet Models de MLflow.

**Le run MLflow et le `dvc.lock` ne sont pas reliés.** Le run connaît le commit, mais pas le hash des données. Et rien n'empêche d'entraîner avec du code non commité.

**Le remote DVC est un dossier local.** Ni un collègue ni la CI ne peuvent faire de `dvc pull`.

## Prochaines étapes

Une étape d'évaluation dans le pipeline, qui ne promeut un nouveau modèle que s'il bat le champion selon une règle décidée à l'avance. La traçabilité complète : le hash des données loggé dans le run, et un refus d'entraîner si le repo contient des modifications non commitées. Une API FastAPI qui charge `@champion`, avec un schéma Pydantic strict et des tests Pytest. Une CI GitHub Actions (Ruff, Pytest, build Docker). Enfin, le monitoring avec Prometheus et Grafana, métriques système et dérive des données.
