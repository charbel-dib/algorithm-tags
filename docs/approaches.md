# Approches comparées

Ce document résume les approches essayées avant de retenir le modèle final, et comment elles ont été évaluées. Les chiffres viennent de plusieurs séries d'expériences aux protocoles différents : on ne compare que les lignes d'un même tableau. Scores en %, écarts en points ; protocole et métriques (F1 macro8, F1 macro9) sont définis dans le [README](../README.md#protocole-et-métriques).

## 1. Le raisonnement

1. **Partir d'une base simple et solide.** Avant tout modèle, deux repères : ne prédire aucun tag, et des règles écrites à la main.
2. **Choisir la famille de modèle sur l'énoncé seul.** Si un modèle plus riche (CNN, transformer, LLM) comprend mieux le texte, c'est là qu'il doit le montrer.
3. **Ajouter l'information disponible au moment de prédire** : la solution Python, sa structure, la difficulté, les exemples d'entrée et de sortie. Chaque source est ajoutée comme un bloc séparé et mesurée seule.
4. **Vérifier qu'aucun modèle plus complexe ne fait mieux sur ces mêmes features.** Une variante n'est retenue que si son gain est net au regard de l'écart-type entre répétitions et de la complexité qu'elle ajoute.

## 2. Étape 1 : quel modèle sur l'énoncé ?

Entraînement sur train (2 981 problèmes), seuil choisi sur les marges out-of-fold du train, mesure sur validation (997) et test (1 004). Le code de ces modèles est dans la version `v1.0.0` du dépôt.

| Approche | F1 macro8, validation | F1 macro8, test | F1 macro9, test | Fausses alertes, test (sur 461) |
|---|---:|---:|---:|---:|
| **TF-IDF + SVM linéaire** (référence) | **64,95** | **65,17** | **64,81** | **207** |
| TF-IDF + régression logistique | 60,12 | 60,61 | 60,57 | 216 |
| CNN + Word2Vec | 62,08 | 60,58 | 61,34 | 151 |
| CNN + fastText | 61,14 | 59,70 | 60,24 | 164 |
| Règles lexicales écrites à la main | 43,13 | 45,19 | 46,85 | 168 |
| Aucun tag | 0,00 | 0,00 | 6,99 | 0 |

La baseline « aucun tag » a un F1 nul sur les 8 tags et un F1 de 63 sur l'état « aucun tag » (précision 46 %, rappel 100 %), d'où 63 / 9 = 6,99 de F1 macro9. La ligne de référence vient du code de la v1 ; le modèle `reference` du dépôt actuel est le même SVM réécrit, avec une préparation du texte légèrement différente, et fait 64,63 de F1 macro9 en test au lieu de 64,81.

Les modèles pré-entraînés non fine-tunés restent loin derrière (protocoles différents, colonne « Mesure ») :

| Famille | Mesure | Score | Lecture |
|---|---|---:|---|
| Embeddings gelés `embeddinggemma-300m` + régression logistique | F1 macro9, validation croisée sur train | 61,0 (SVM TF-IDF : 62,5 dans la même mesure) | sous le TF-IDF |
| Classification zéro-shot NLI (`deberta-v3-base-zeroshot`) | F1 macro9, validation croisée sur train | 44,6 | insuffisant |
| Petit LLM local (Qwen2.5-1.5B) en zéro-shot | F1 macro8, partie validation | 24,3 | insuffisant, et 4 s par problème |

Conclusion : sur 3 000 problèmes annotés, le SVM linéaire sur TF-IDF est le meilleur et le plus simple. C'est le modèle `reference`.

## 3. Étape 2 : ajouter de l'information

Validation croisée sur train + validation (3 978 problèmes), 5 plis × 4 répétitions ; vocabulaire et seuil appris dans chaque pli. Ce tableau est produit par le dépôt (`python -m algo_tags cv`, dossier `results/cv/`). La colonne « Gain » est l'écart avec la ligne du dessus.

| Modèle | Ce qui est ajouté | F1 macro9 | Gain |
|---|---|---:|---:|
| `reference` | mots de l'énoncé | 64,43 ± 0,18 | |
| `code` | + mots du code de la solution | 66,24 ± 0,75 | +1,8 |
| `code_meta` | + difficulté + indices tirés des exemples | 66,85 ± 0,13 | +0,6 |
| **`final`** | + structure du code (AST) + suites de caractères de l'énoncé | **67,54 ± 0,25** | +0,7 |
| `final_no_code` | `final` sans le code ni l'AST | 65,80 ± 0,22 | −1,7 par rapport à `final` |

Deux points de méthode comptent :

- Le code est un **bloc séparé, moins pondéré** que l'énoncé. Mélanger énoncé et code dans un même TF-IDF n'apportait rien : le vocabulaire du code (`input`, `range`, `print`) se mêle alors à celui de l'énoncé et en dilue les poids.
- Les lignes de lecture et d'écriture présentes dans presque toutes les solutions sont retirées du code. Elles ne disent rien de l'algorithme.

## 4. Étape 3 : un modèle plus complexe fait-il mieux ?

Environ 130 configurations, évaluées en validation croisée (5 plis × 2 répétitions, seuil choisi dans chaque pli). Le modèle retenu a ensuite été confirmé sur 2 autres répétitions, qui n'avaient pas servi pendant ces essais. La dernière colonne est l'écart de F1 macro9 avec le modèle retenu, en points.

| Famille | Ce qui a été essayé | Écart avec le modèle retenu |
|---|---|---|
| Réglages du TF-IDF | taille du vocabulaire, trigrammes, fréquence minimale, TF binaire, un bloc par section | nul ou négatif |
| Autres classifieurs linéaires | Ridge, NB-SVM, régression logistique, SVM L1 | au mieux +0,6 |
| Modèles non linéaires | SVM à noyau RBF, MLP, LightGBM | 2 à 6 points en dessous |
| Lexiques par tag appris sur le train | mots, code et AST les plus associés à chaque tag, ajoutés au SVM | nul ou négatif |
| Recherche par similarité | vote des plus proches voisins (TF-IDF, LSA, BM25), prototypes par tag | nul ou négatif |
| Stacking | plusieurs modèles combinés par tag, avec les dépendances entre tags | au mieux +0,6, pour 6 modèles au lieu d'un |
| Règles de décision | un seuil par tag, règle dédiée « aucun tag » | nul ou négatif |

La meilleure combinaison non retenue (SVM + NB-SVM avec un seuil par tag) gagnait 0,6 point, mesuré sur 2 répétitions seulement, en échange de deux modèles et huit seuils au lieu d'un. Le gain ne justifie pas cette complexité.

### Embeddings pré-entraînés ajoutés au modèle final

Dernière vérification : des encodeurs pré-entraînés de code et de texte, utilisés sans fine-tuning, apportent-ils une information que le TF-IDF n'a pas ? Chaque embedding est ajouté comme bloc dense au modèle `final` (poids 0,3), sur les mêmes plis. Critère fixé avant l'essai : l'intégrer seulement si le gain dépasse 0,7 point et reste positif aux 4 répétitions.

| Encodeur | Entrée | F1 macro9 | Gain sur `final` |
|---|---|---:|---:|
| `gte-modernbert-base` | code | 67,96 | +0,44 |
| `gte-modernbert-base` | énoncé | 67,83 | +0,31 |
| `bge-base-en-v1.5` | énoncé | 67,76 | +0,24 |
| `modernbert-embed-base` | code | 67,69 | +0,17 |
| `modernbert-embed-base` | énoncé | 67,67 | +0,15 |
| `unixcoder-base` | code | 67,58 | +0,06 |

Cet essai a tourné sur un PC Windows avec GPU, où `final` fait 67,52 au lieu de 67,54 (voir « Reproductibilité » dans le README) : les gains sont mesurés par rapport à ces 67,52, sur les mêmes découpages. Aucun n'atteint le critère. Le gain le plus fort (+0,44) demanderait de faire tourner en production un encodeur de 150 millions de paramètres, idéalement sur GPU, pour un écart à peine supérieur à l'écart-type entre répétitions (0,25). Ces embeddings ne sont donc pas intégrés.

## 5. Ce que ces essais montrent

- **Le levier, c'est l'information, pas le modèle.** Ajouter le code, sa structure, la difficulté, les exemples et les suites de caractères apporte 3,1 points. Aucune des variantes de modèle essayées sur ces mêmes informations ne dépasse le modèle retenu de plus de 0,6 point.
- **Un SVM linéaire est déjà un lexique pondéré complet.** Il apprend un poids par mot et par tag. Un lexique extrait à part (« gcd », « divisor » pour `number theory`) n'en est qu'une version tronquée, d'où l'absence de gain. Les features les plus influentes du modèle final sont dans [top_features.md](top_features.md).
- **`math` est un tag large.** Ses termes les plus influents décrivent des contraintes (« 10^9 », « divisible ») plus que des notions précises, et il recoupe presque tous les autres tags. Sa fréquence est stable quelle que soit la difficulté (25 à 33 %).
- **Les annotations ont des limites.** Sur Codeforces, des programmeurs expérimentés à qui on demande de deviner les tags n'en retrouvent qu'une partie ([Athavale et al., 2019](https://arxiv.org/abs/1903.00830)). Sur ce type de données, un SVM sur TF-IDF fait jeu égal avec des transformers ([Kim et al., 2023](https://arxiv.org/abs/2310.05791)), et le code est la source complémentaire la plus forte ([Lobanov et al., 2023](https://arxiv.org/abs/2301.04597)).

## 6. Pistes

- **Fine-tuning d'un encodeur texte et code** (UniXcoder, ModernBERT), combiné ensuite au SVM. C'est l'étape suivante logique, puisque les embeddings gelés apportent déjà un petit signal.
- **Plusieurs solutions par problème** : le code devient plus informatif quand on agrège plusieurs façons de résoudre le même problème.
- **Plus de données** : d'autres problèmes Codeforces annotés de xCodeEval, après un contrôle de recouvrement avec le test.
- **Audit des annotations**, en particulier pour `math`, sans corriger les labels automatiquement.
- **LLM de grande taille** en few-shot avec des exemples proches retrouvés dans le train, pour les cas où le SVM hésite.
