# Approches comparées

Ce document résume les approches testées avant de retenir le modèle final, avec leur protocole d'évaluation. Les chiffres viennent de trois séries d'expériences : il ne faut comparer entre elles que des lignes issues du même tableau.

## 1. Démarche

1. **Baselines** : aucun tag, règles lexicales écrites à la main.
2. **Familles de modèles sur l'énoncé** : TF-IDF + régression logistique, TF-IDF + SVM linéaire, CNN (embeddings aléatoires, Word2Vec, fastText), encodeurs pré-entraînés gelés, zéro-shot (NLI, GLiClass, LLM local).
3. **Choix de la famille** : le SVM linéaire sur TF-IDF est le meilleur et le plus simple. C'est le modèle de référence de la v1.
4. **Ajout d'informations disponibles au moment de prédire** : code de la solution, structure du code (AST), difficulté, exemples d'entrée/sortie. C'est ce qui apporte le gain (v2).
5. **Vérification qu'aucun modèle plus complexe ne fait mieux sur ces features** : environ 130 configurations criblées.

## 2. Version 1 : familles de modèles sur l'énoncé

Protocole : apprentissage sur train (2 981 problèmes), seuil choisi hors pli sur train, mesure sur validation (997) et test (1 004). Le code de ces modèles est disponible dans la version `v1.0.0` du dépôt.

| Approche | Macro8 val | Macro8 test | Macro9 test | Micro test | Fausses alertes sans tag (test) |
|---|---:|---:|---:|---:|---:|
| **TF-IDF + SVM linéaire, seuil macro9** (référence) | **64,95** | **65,17** | **64,81** | **60,44** | **207 / 461** |
| TF-IDF mots + caractères + SVM | 63,45 | 65,61 | 64,59 | 59,54 | 252 / 461 |
| TF-IDF + régression logistique | 60,12 | 60,61 | 60,57 | 57,71 | 216 / 461 |
| CNN + Word2Vec | 62,08 | 60,58 | 61,34 | 57,74 | 151 / 461 |
| CNN + fastText | 61,14 | 59,70 | 60,24 | 56,56 | 164 / 461 |
| CNN, embeddings aléatoires | 57,42 | 57,36 | 55,13 | 56,06 | 344 / 461 |
| Règles lexicales | 43,13 | 45,19 | 46,85 | 37,74 | 168 / 461 |
| Aucun tag | 0,00 | 0,00 | 6,99 | 0,00 | 0 / 461 |

Variantes du SVM sur l'énoncé (validation, macro9) : seuil optimisé pour macro9 64,46 contre 62,38 pour macro8 ; ajout de features de règles 63,97 ; suppression des mots vides 62,92.

Autres familles explorées en v1, avec des protocoles internes différents (macro9 en validation croisée sur train, sauf mention) :

| Famille | Meilleur résultat | Lecture |
|---|---:|---|
| Embeddings gelés `embeddinggemma-300m` + logistique | 61,0 (SVM TF-IDF : 62,5) | Sous le TF-IDF |
| Embeddings gelés `bge-small-en-v1.5` + logistique / SVM / prototypes | 52,7 – 53,8 | Nettement sous le TF-IDF |
| Zéro-shot NLI (`deberta-v3-base-zeroshot`) | 44,6 | Insuffisant |
| Zéro-shot GLiClass / GLiNER2 | 36,2 – 41,3 | Insuffisant |
| LLM local Qwen2.5-1.5B zéro-shot (macro8, validation) | 24,3 | Insuffisant, et environ 4 s par problème |
| Proximité / clustering (Rocchio, kNN, K-means) sur TF-IDF mots + caractères (macro8, validation) | 59,9 (linéaire : 64,1) | Sous le linéaire |

## 3. Version 2 : ajout d'informations et criblage

Protocole : validation croisée 5 plis répétée 4 fois sur train + validation (3 978 problèmes). Dans chaque pli, le seuil est choisi uniquement sur les données d'entraînement du pli. Le tableau suivant est produit par le dépôt (`python -m algo_tags cv`, fichiers `results/cv_*.csv`).

### Ce qui a été retenu

| Modèle | Blocs de features | Macro9 | Macro8 |
|---|---|---:|---:|
| `reference` | mots de l'énoncé | 64,43 ± 0,18 | 64,64 |
| `code` | + mots du code (bloc séparé, poids 0,5) | 66,24 ± 0,75 | 66,38 |
| `code_meta` | + difficulté + indices des exemples d'entrée/sortie | 66,85 ± 0,13 | 66,90 |
| **`final`** | + structure AST du code + n-grammes de caractères de l'énoncé | **67,54 ± 0,25** | **67,66** |
| `final_no_code` | `final` sans code ni AST | 65,80 ± 0,22 | 65,80 |

Le code est un **bloc séparé et moins pondéré** que l'énoncé. Un essai v1 qui concaténait énoncé et code dans un même TF-IDF n'apportait aucun gain (58,9 macro8 en validation, contre 59,2 pour l'énoncé seul dans la même grille).

### Ce qui n'a rien apporté

Criblage d'environ 130 configurations (5 plis × 2 répétitions, seuil choisi hors du pli évalué), sur une implémentation de laboratoire équivalente à `code_meta` (base 67,3). Le modèle retenu y était confirmé sur 2 répétitions supplémentaires, jamais utilisées pendant le criblage (référence 64,5, modèle retenu 68,3).

| Famille | Variantes | Macro9 |
|---|---|---:|
| Réglages TF-IDF | vocabulaire 15k / 60k / illimité, trigrammes, `min_df`, TF binaire, blocs par section | 66,0 – 67,4 |
| Classifieurs | Ridge, NB-SVM, NB-LR | 66,1 – 67,9 |
| | régression logistique, SVM hinge / L1 | 61,9 – 66,6 |
| | SVM RBF (SVD 300), MLP, LightGBM (creux ou SVD) | 61,0 – 65,4 |
| | SGD, Passive-Aggressive, Naive Bayes | 34,6 – 63,4 |
| Lexiques par tag appris sur le train (log-odds), ajoutés au SVM | énoncé, code, AST | 65,2 – 67,6 |
| Recherche sémantique / proximité | vote des k plus proches voisins (TF-IDF, LSA, BM25, vecteurs PPMI), prototypes Rocchio | 65,1 – 67,8 |
| Espaces sémantiques denses | LSA, PPMI-SVD, y compris appris sur tous les textes sans label | 66,3 – 67,6 |
| Stacking imbriqué | moyenne, LR et LightGBM par tag sur 6 modèles de base, avec dépendances entre tags et difficulté | 62,2 – 68,4 |
| Règles de décision | seuils par tag, « gate » sans tag | ≤ 67,4 |

La meilleure combinaison non retenue, SVM + NB-SVM avec seuils par tag, gagnait +0,6 point sur le modèle retenu : le gain est faible au regard de la complexité ajoutée (deux modèles, huit seuils).

## 4. Interprétation

- **Le levier, ce sont les informations disponibles, pas le modèle.** Ajouter le code, sa structure, la difficulté et les exemples apporte environ 3 points. Une centaine de variantes de modèles restent dans un intervalle de ±0,5 point.
- **Un SVM linéaire est déjà un lexique pondéré complet** : il apprend un poids par mot et par tag. Un lexique extrait (par exemple « gcd », « divisor » pour number theory) en est une version tronquée, d'où l'absence de gain. Les features les plus pondérées par le modèle final pour chaque tag sont listées dans [top_features.md](top_features.md) (commande `explain`).
- **`math` est un tag large** : les termes qui lui sont le plus associés sont « test cases », « 10^9 », « t ≤ », c'est-à-dire des formulations de contraintes plus que des notions mathématiques. Sa fréquence est stable selon la difficulté (25 à 33 %). C'est l'un des tags les plus difficiles.
- **Bruit d'annotation** : sur Codeforces, même des humains devinent mal les tags (F1 macro 42,7 sur 20 tags pour des participants sur 100 problèmes, [Athavale et al., 2019](https://arxiv.org/abs/1903.00830)). Un SVM TF-IDF fait jeu égal avec des transformers sur ce type de données ([Kim et al., 2023](https://arxiv.org/abs/2310.05791)), et le code est la source complémentaire la plus forte ([Lobanov et al., 2023](https://arxiv.org/abs/2301.04597)).

## 5. Pistes d'amélioration

Non exécutées, en raison du coût de calcul :

- **Fine-tuning d'un encodeur texte + code** (UniXcoder, ModernBERT), puis fusion tardive avec le SVM.
- **Embeddings gelés de code** ajoutés comme bloc dense au TF-IDF.
- **Plusieurs solutions par problème** : le code devient la source la plus forte quand on agrège de nombreuses solutions.
- **Plus de données** : autres problèmes Codeforces annotés de xCodeEval, après un contrôle de recouvrement avec le test.
- **Audit des annotations**, par exemple avec cleanlab, en particulier pour `math`, sans corriger les labels automatiquement.
- **LLM de grande taille** en few-shot avec récupération d'exemples, ou distillation vers le SVM.
