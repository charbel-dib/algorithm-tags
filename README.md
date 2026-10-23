# Algorithm tags

Classification multilabel d’exercices d’algorithmique à partir de leur énoncé. Un exercice peut recevoir plusieurs tags ou une liste vide.

Cibles : `math`, `graphs`, `strings`, `number theory`, `trees`, `geometry`, `games`, `probabilities`.

## Modèle retenu et résultats

Le modèle retenu est **TF-IDF + SVM linéaire**, avec un seuil global réglé sur le **F1 macro9 hors pli du train**. Il est sélectionné sur validation pour son équilibre entre les huit tags, les sorties sans cible et la simplicité de l’inférence sur CPU.

| Mesure | Validation — 997 exercices | Test — 1 004 exercices |
|---|---:|---:|
| **F1 macro8** | **64,95 %** | **65,17 %** |
| **F1 macro9** | **64,46 %** | **64,81 %** |
| F1 micro | 60,28 % | 60,44 % |
| F1 weighted | 60,24 % | 60,68 % |
| F1 sans cible | 60,60 % | 61,95 % |
| Exact match | 49,75 % | 49,60 % |
| Exercices sans cible recevant au moins un tag | 218 / 461 | 207 / 461 |

Macro8 est la moyenne des F1 des huit tags. Macro9 ajoute le F1 de l’état « aucun tag » à cette moyenne ; cet état est déduit des huit sorties, sans neuvième classe apprise. Micro agrège les décisions, weighted pondère les F1 par leurs supports, et exact match exige que tous les tags soient corrects.

### Résultats par classe sur test

| Tag | Support | Précision | Rappel | F1 |
|---|---:|---:|---:|---:|
| math | 282 | 49,38 % | 70,57 % | 58,10 % |
| graphs | 108 | 53,66 % | 61,11 % | 57,14 % |
| strings | 84 | 55,28 % | 80,95 % | 65,70 % |
| number theory | 70 | 47,95 % | 50,00 % | 48,95 % |
| trees | 65 | 70,31 % | 69,23 % | 69,77 % |
| geometry | 33 | 63,41 % | 78,79 % | 70,27 % |
| games | 21 | 100,00 % | 66,67 % | 80,00 % |
| probabilities | 18 | 100,00 % | 55,56 % | 71,43 % |

`number theory` reste la classe la plus difficile selon le F1. La précision de 100 % sur `games` et `probabilities` porte sur seulement 14 et 10 prédictions positives ; leur rappel reste limité.

Le choix du modèle et du seuil repose sur train et validation. Le test a également servi à des comparaisons exploratoires : ses résultats sont descriptifs et ne constituent pas une estimation indépendante de toute l’exploration. Aucun seuil n’est ajusté sur test.

## Installation

Python **3.11 ou plus récent** est nécessaire. Un environnement virtuel Python suffit ; aucune machine virtuelle ni aucun GPU n’est requis pour le SVM. Toutes les commandes suivantes se lancent depuis la racine du dépôt.

**Windows PowerShell :**

```powershell
# Créer puis activer un environnement Python isolé.
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Installer le projet et les dépendances nécessaires au SVM.
python -m pip install -e .
```

**Linux / macOS, Bash :**

```bash
# Créer puis activer un environnement Python isolé.
python3 -m venv .venv
source .venv/bin/activate

# Installer le projet et les dépendances nécessaires au SVM.
python -m pip install -e .
```

Les versions de scikit-learn et joblib sont fixées pour relire les modèles exportés. L’environnement de l’artefact distribué est décrit dans `models/environment.json`. Des écarts numériques restent possibles entre plateformes.

## Réentraîner depuis les JSON bruts

Les données brutes et les poids ne sont pas versionnés. Le parcours ci-dessous ne nécessite **aucun modèle téléchargé** et entraîne le SVM retenu à partir des données.

### 1. Préparer les données

`../corpus/exercices` désigne ici un dossier contenant les JSON bruts, éventuellement dans des sous-dossiers. Remplacer ce chemin par celui des données : un chemin absolu ou un dossier extérieur au dépôt convient aussi. `data/raw` est une convention possible, pas une obligation. Mettre les chemins contenant des espaces entre guillemets.

```bash
# Valider les JSON, normaliser le texte et encoder les huit cibles.
python -m algo_tags preprocess --input "../corpus/exercices" --output data/processed
```

La commande produit `data/processed/dataset.jsonl` et `preprocessing_summary.json`. Les champs bruts attendus sont décrits dans [data/README.md](data/README.md). Une liste de tags vide ou ne contenant aucune cible devient huit zéros ; une annotation manquante ou mal formée provoque une erreur.

### 2. Utiliser le découpage adapté au corpus

Pour **le corpus de référence de 4 982 exercices**, conserver `data/splits/splits.csv` : **2 981 train, 997 validation, 1 004 test**. Le JSONL complet est fourni à la commande ; le split détermine les exercices autorisés pour chaque étape.

Pour **un autre corpus**, créer un découpage dans un dossier distinct :

```bash
# Créer un split multilabel 60/20/20 pour un autre corpus ; ne pas écraser le split fourni.
python -m algo_tags split --data data/processed/dataset.jsonl --output data/custom_splits --seed 42
```

Dans ce cas, remplacer `data/splits/splits.csv` par `data/custom_splits/splits.csv` dans toutes les commandes suivantes. Ce nouveau corpus ne reproduira pas les résultats chiffrés du corpus de référence.

### 3. Entraîner le modèle et mesurer la validation

```bash
# Apprendre le TF-IDF et le SVM sur train, régler le seuil hors pli, puis évaluer sur validation.
python -m algo_tags train --data data/processed/dataset.jsonl --splits data/splits/splits.csv --config configs/models.json --model tfidf_svm --metric f1_macro9 --output runs/svm_macro9
```

Trois plis du train fournissent les scores hors pli. Le vocabulaire et les IDF sont réappris dans chaque pli. Le seuil global maximise macro9 sur ces scores ; le modèle est ensuite appris sur tout le train. La validation n’est pas utilisée pour apprendre les poids, et le test n’est pas utilisé par cette commande.

Les sorties comprennent :

- `runs/svm_macro9/tfidf_svm/model.joblib` : encodeur et classifieurs entraînés ;
- `runs/svm_macro9/tfidf_svm/registry.json` : chemin des poids, empreinte et seuils à utiliser ;
- `runs/svm_macro9/tfidf_svm/validation/` : métriques, prédictions, résultats par classe et matrices de confusion ;
- `runs/svm_macro9/global.csv` et `per_label.csv` : tableaux de comparaison ;
- `metadata.json`, `oof.npz` et `protocol.json` : paramètres, scores hors pli et traçabilité.

`--data`, `--splits`, `--config` et `--output` acceptent des chemins personnalisés. Un dossier de sortie non vide est refusé ; `--resume` permet une reprise uniquement si les données, paramètres, versions et fichiers Python correspondent.

### 4. Évaluer sur test

Une fois les choix fixés sur validation, utiliser le registre produit par l’entraînement :

```bash
# Calculer les métriques du modèle entraîné sur la partition test uniquement.
python -m algo_tags evaluate --registry runs/svm_macro9/tfidf_svm/registry.json --input data/processed/dataset.jsonl --splits data/splits/splits.csv --split test --output runs/svm_macro9_test
```

L’évaluation écrit `metrics.json`, `per_label.csv`, `predictions.npz` et `confusions.png`. Choisir un dossier de sortie vide. Pour évaluer un dossier externe annoté, fournir son chemin dans `--input` et omettre `--splits` et `--split`.

### 5. Prédire sur un exemple ou un dossier

L’exemple de démonstration `examples/sample_0.json` est un extrait du test, non utilisé pour apprendre les poids. Conserver le fichier s’il est déjà présent. Sinon, cette commande extrait un exemple réel depuis les données préparées et le split, sans inventer de texte ni utiliser les tags :

```bash
# À exécuter uniquement si l’exemple n’est pas déjà présent : exporter un énoncé réel du test.
python -m algo_tags sample --data data/processed/dataset.jsonl --splits data/splits/splits.csv --output examples/sample_0.json
```

La commande conserve l’identifiant et le texte préparé ; elle refuse d’écraser un fichier existant.

```bash
# Prédire les tags d’une entrée unique avec le modèle réentraîné.
python -m algo_tags predict --registry runs/svm_macro9/tfidf_svm/registry.json --input examples/sample_0.json

# Prédire un ensemble d’entrées et enregistrer les réponses dans un fichier JSON.
python -m algo_tags predict --registry runs/svm_macro9/tfidf_svm/registry.json --input "../corpus/nouveaux_exercices" --output runs/predictions.json
```

Un JSON brut avec `prob_desc_description` suffit pour prédire. Les champs d’entrée, de sortie et de notes sont facultatifs. Le format préparé `inputs.statement` est aussi accepté. La prédiction n’utilise ni les annotations, ni le code solution, ni la difficulté. La sortie indique les tags, les scores et leur nature : les scores SVM sont des **marges, pas des probabilités**.

## Utiliser les poids distribués séparément

Cette option évite de réentraîner le SVM. L’archive de poids `algorithm-tags-models.zip`, destinée à une distribution sur [Drive](https://drive.google.com/drive/folders/1Gje7NXEKUr1zn6kfvax8K0xb4rC4avyk?usp=sharing), contient `models/tfidf_svm.joblib`. Après téléchargement, extraire son contenu à la racine du dépôt pour conserver ce chemin. Le registre `models/registry.json`, déjà versionné, contient les seuils et l’empreinte attendue.

```bash
# Vérifier les résultats enregistrés et l’empreinte des poids présents.
python -m algo_tags verify

# Prédire sur l’exemple test avec le registre du modèle distribué.
python -m algo_tags predict --registry models/registry.json --input examples/sample_0.json
```

Les poids des autres approches ne sont pas nécessaires pour consulter leurs résultats. `verify` signale les poids absents sans bloquer le contrôle des métriques ; un fichier présent dont l’empreinte est incorrecte est rejeté.

## Comparaison des approches

Les partitions sont communes. Les hyperparamètres, budgets neuronaux et objectifs de seuil sont précisés dans le [protocole](docs/methodology.md) : le tableau compare des approches complètes, sans isoler l’effet d’un seul paramètre.

| Approche | F1 macro8 validation | F1 macro8 test | F1 micro test | FP sans cible test |
|---|---:|---:|---:|---:|
| **TF-IDF + SVM — seuil macro9** | **64,95 %** | **65,17 %** | **60,44 %** | **207 / 461** |
| TF-IDF mots/caractères + SVM — seuil macro8 | 63,45 % | 65,61 % | 59,54 % | 252 / 461 |
| TF-IDF + logistique — seuil macro8 | 60,12 % | 60,61 % | 57,71 % | 216 / 461 |
| CNN + Word2Vec — seuil macro8 | 62,08 % | 60,58 % | 57,74 % | 151 / 461 |
| CNN + fastText — seuil macro8 | 61,14 % | 59,70 % | 56,56 % | 164 / 461 |
| CNN aléatoire — seuil 0,5 | 57,42 % | 57,36 % | 56,06 % | 344 / 461 |
| Règles lexicales | 43,13 % | 45,19 % | 37,74 % | 168 / 461 |
| Aucun tag | 0,00 % | 0,00 % | 0,00 % | 0 / 461 |

Le SVM mots/caractères obtient le meilleur macro8 test ; le SVM texte reste retenu sur validation. Le CNN Word2Vec réduit davantage les fausses alertes sans cible, avec un F1 macro8 inférieur.

### Représentation et objectif du seuil

Comparaison sur les **997 exemples de validation** :

| Variante | F1 macro8 | F1 macro9 | F1 sans cible | FP sans cible |
|---|---:|---:|---:|---:|
| **SVM texte — seuil macro9** | **64,95 %** | **64,46 %** | **60,60 %** | **218 / 461** |
| SVM texte — seuil macro8 | 63,70 % | 62,38 % | 51,84 % | 278 / 461 |
| SVM + caractéristiques de règles — seuil macro9 | 63,76 % | 63,97 % | 65,59 % | 155 / 461 |
| SVM + filtrage des mots courants — seuil macro9 | 63,33 % | 62,92 % | 59,60 % | 222 / 461 |

Les deux SVM texte ont exactement les mêmes paramètres appris et scores. Leur seule différence est le seuil : **−0,228501 pour macro9**, **−0,347342 pour macro8**. Le seuil macro9 augmente le F1 test de sept tags et réduit les fausses alertes ; `probabilities` passe cependant de 80,00 % à 71,43 %. Ce compromis est explicité, sans masquer la baisse d’une classe.

Les règles réduisent les fausses alertes et aident `number theory`, mais leur gain n’est pas global. Le filtrage lexical ne dépasse pas le SVM standard. Les [résultats détaillés](docs/results.md) donnent les comparaisons par classe et les intervalles descriptifs.

```bash
# Réentraîner et comparer les trois variantes sur validation, sans poids préinstallés.
python -m algo_tags experiment --data data/processed/dataset.jsonl --splits data/splits/splits.csv --models tfidf_svm tfidf_manual_rules tfidf_stopwords --metric f1_macro9 --output runs/text_variants --resume

# Isoler l’objectif macro8 avec la même représentation SVM et un dossier distinct.
python -m algo_tags train --data data/processed/dataset.jsonl --splits data/splits/splits.csv --model tfidf_svm --metric f1_macro --output runs/svm_macro8

# Afficher les résultats test enregistrés dans le dépôt, sans nouvelle inférence.
python -m algo_tags results --split test
```

La commande `experiment` produit `global.csv`, `per_label.csv` et `comparison.md`, ainsi que les sorties par candidat. Elle ne modifie pas la sélection enregistrée dans `results/selection.json`. L’option facultative `--reference-registry models/reference_macro8.json` ajoute le SVM distribué avec son seuil macro8 si ses poids ont été installés.

## Organisation et vérifications

| Répertoire | Contenu |
|---|---|
| `src/algo_tags/` | Préparation, représentations, modèles, entraînement, CLI et métriques |
| `configs/` | Paramètres des approches reproductibles |
| `models/` | Registres, seuils et métadonnées ; poids installés séparément, ignorés par Git |
| `results/` | Métriques, prédictions, scores hors pli et décision de sélection |
| `notebooks/` | Exploration des effectifs et lecture des comparaisons, sans entraînement |
| `tests/` | Prétraitement, splits, métriques et entraînements sur données synthétiques |
| `examples/` | Exemple de démonstration extrait du test avec la commande `sample` |
| `data/` | Split de référence ; données brutes et préparées conservées localement |

Les notebooks de résultats se lisent sans les données brutes ni les poids. Pour les ouvrir et lancer les contrôles :

```bash
# Installer les outils de notebook et de vérification du code.
python -m pip install -e ".[notebooks,dev]"

# Exécuter les tests ; aucun entraînement sur le corpus réel n’est lancé.
python -m unittest discover -s tests -v

# Vérifier le style et les erreurs statiques.
python -m ruff check src tests

# Recalculer les métriques depuis les prédictions enregistrées et vérifier les poids installés.
python -m algo_tags verify
```

Les CNN sont facultatifs : leurs dépendances et commandes de réentraînement sont décrites dans le [protocole](docs/methodology.md).
