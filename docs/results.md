# Résultats

Le modèle par défaut est le SVM texte avec le seuil global −0,228501. Il a été choisi sur les résultats de validation des variantes, sans régler les paramètres sur test. Les chiffres proviennent des prédictions enregistrées.

## Résultats globaux du modèle sélectionné

| Mesure | Validation | Test |
|---|---:|---:|
| F1 macro8 | 64,95 % | 65,17 % |
| F1 macro9 | 64,46 % | 64,81 % |
| F1 micro | 60,28 % | 60,44 % |
| F1 weighted | 60,24 % | 60,68 % |
| F1 sans cible | 60,60 % | 61,95 % |
| Exact match | 49,75 % | 49,60 % |
| FP sans cible | 218 / 461 | 207 / 461 |

## Résultats par classe du modèle sélectionné

| Tag | Support test | Précision | Rappel | F1 |
|---|---:|---:|---:|---:|
| math | 282 | 49.38 % | 70.57 % | 58.10 % |
| graphs | 108 | 53.66 % | 61.11 % | 57.14 % |
| strings | 84 | 55.28 % | 80.95 % | 65.70 % |
| number theory | 70 | 47.95 % | 50.00 % | 48.95 % |
| trees | 65 | 70.31 % | 69.23 % | 69.77 % |
| geometry | 33 | 63.41 % | 78.79 % | 70.27 % |
| games | 21 | 100.00 % | 66.67 % | 80.00 % |
| probabilities | 18 | 100.00 % | 55.56 % | 71.43 % |

Sur validation, six F1 de classe progressent avec le seuil macro9 plutôt que macro8, un reste identique et `number theory` baisse de 2,73 points. Sur test, sept progressent et `probabilities` baisse de 8,57 points : dix positifs sont retrouvés contre douze avec le seuil macro8, sur un support de dix-huit. Cette baisse est conservée dans l’analyse ; aucune règle ne supprime automatiquement un candidat pour une seule classe en recul.

La précision de `games` et `probabilities` atteint 100 % sur ce test, avec seulement 14 et 10 prédictions positives. Cela ne signifie pas une classe parfaitement maîtrisée : les rappels sont de 66,67 % et 55,56 %.

## Comparaison des approches conservées

Les partitions sont communes, mais les budgets et critères de réglage ne sont pas identiques entre familles. Le SVM texte retenu utilise un objectif de seuil macro9 ; la logistique, le SVM mots/caractères et les CNN initialisés utilisent macro8. Le CNN aléatoire utilise le seuil naturel 0,5. Il s’agit d’une comparaison descriptive, pas d’une ablation contrôlée de l’architecture seule.

| Approche | F1 macro8 validation | F1 macro8 test | F1 micro test | FP sans cible test |
|---|---:|---:|---:|---:|
| TF-IDF + SVM | 64.95 % | 65.17 % | 60.44 % | 207 / 461 |
| Aucun tag | 0.00 % | 0.00 % | 0.00 % | 0 / 461 |
| Règles lexicales | 43.13 % | 45.19 % | 37.74 % | 168 / 461 |
| TF-IDF + logistique | 60.12 % | 60.61 % | 57.71 % | 216 / 461 |
| CNN aléatoire | 57.42 % | 57.36 % | 56.06 % | 344 / 461 |
| CNN + Word2Vec | 62.08 % | 60.58 % | 57.74 % | 151 / 461 |
| CNN + fastText | 61.14 % | 59.70 % | 56.56 % | 164 / 461 |
| TF-IDF mots/caractères + SVM | 63.45 % | 65.61 % | 59.54 % | 252 / 461 |

## Variantes de texte sur validation

| Variante | F1 macro8 | F1 macro9 | F1 sans cible | FP sans cible |
|---|---:|---:|---:|---:|
| SVM texte — seuil macro9 | 64.95 % | 64.46 % | 60.60 % | 218 / 461 |
| SVM texte — seuil macro8 | 63.70 % | 62.38 % | 51.84 % | 278 / 461 |
| SVM + caractéristiques de règles | 63.76 % | 63.97 % | 65.59 % | 155 / 461 |
| SVM + filtrage des mots courants | 63.33 % | 62.92 % | 59.60 % | 222 / 461 |

Le SVM standard obtient le meilleur macro8 et macro9 des variantes comparées. L’ajout de règles améliore `number theory` et réduit davantage les fausses alertes sans cible, mais dégrade notamment `math` et `probabilities`. Le filtrage des mots courants aide `trees` et `number theory` par rapport au SVM au seuil macro9, sans gain global. Les poids de ces deux variantes ne sont pas nécessaires à la prédiction par défaut ; leurs configurations, métriques, prédictions et scores OOF restent disponibles pour reproduire et expliquer les essais.

## Effet du seuil à paramètres appris identiques

L’égalité exacte du vocabulaire, des IDF, des coefficients, des intercepts et des scores OOF a été contrôlée. Les scores test sont également identiques, exercice par exercice. Seul l’objectif de seuil diffère : macro8 donne −0,347342, macro9 donne −0,228501 sur les mêmes scores hors pli.

| Mesure | Seuil macro8, validation | Seuil macro9, validation | Seuil macro8, test | Seuil macro9, test |
|---|---:|---:|---:|---:|
| F1 macro8 | 63.70 % | 64.95 % | 64.15 % | 65.17 % |
| F1 macro9 | 62.38 % | 64.46 % | 63.27 % | 64.81 % |
| F1 micro | 59.27 % | 60.28 % | 58.54 % | 60.44 % |
| F1 weighted | 59.49 % | 60.24 % | 58.93 % | 60.68 % |
| F1 sans cible | 51.84 % | 60.60 % | 56.25 % | 61.95 % |
| Exact match | 43.43 % | 49.75 % | 44.12 % | 49.60 % |
| FP sans cible | 278 / 461 | 218 / 461 | 254 / 461 | 207 / 461 |

Relever le seuil réduit les prédictions positives : la précision monte et le rappel baisse. Le macro8, macro9, micro et weighted progressent néanmoins sur validation et test. Ces résultats montrent l’intérêt de régler la décision avant d’ajouter de la complexité au modèle.

## F1 par classe des approches conservées

### Validation

| Approche | math | graphs | strings | number theory | trees | geometry | games | probabilities |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + SVM | 56.42 | 57.85 | 68.97 | 47.62 | 70.68 | 63.01 | 88.37 | 66.67 |
| Aucun tag | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Règles lexicales | 4.65 | 38.68 | 52.23 | 59.54 | 65.04 | 36.88 | 32.48 | 55.56 |
| TF-IDF + logistique | 56.80 | 56.92 | 68.37 | 45.45 | 65.28 | 61.11 | 80.85 | 46.15 |
| CNN aléatoire | 53.90 | 51.51 | 58.78 | 46.55 | 69.74 | 46.73 | 80.85 | 51.28 |
| CNN + Word2Vec | 53.58 | 56.88 | 65.69 | 54.79 | 65.77 | 60.53 | 78.26 | 61.11 |
| CNN + fastText | 53.03 | 51.31 | 64.44 | 53.97 | 66.17 | 60.87 | 76.19 | 63.16 |
| TF-IDF mots/caractères + SVM | 57.45 | 54.24 | 66.36 | 52.11 | 68.00 | 63.29 | 77.55 | 68.57 |

F1 en pourcentage. Les supports et comptes de confusion sont dans `results/validation/per_label.csv`.

### Test

| Approche | math | graphs | strings | number theory | trees | geometry | games | probabilities |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + SVM | 58.10 | 57.14 | 65.70 | 48.95 | 69.77 | 70.27 | 80.00 | 71.43 |
| Aucun tag | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Règles lexicales | 3.95 | 43.22 | 50.53 | 51.09 | 69.57 | 37.80 | 26.55 | 78.79 |
| TF-IDF + logistique | 57.35 | 50.86 | 65.37 | 46.84 | 64.18 | 64.00 | 64.86 | 71.43 |
| CNN aléatoire | 56.27 | 52.52 | 59.66 | 42.02 | 66.19 | 50.91 | 68.18 | 63.16 |
| CNN + Word2Vec | 56.84 | 51.43 | 62.75 | 51.50 | 63.51 | 59.74 | 70.27 | 68.57 |
| CNN + fastText | 55.25 | 49.48 | 63.33 | 51.56 | 64.35 | 52.31 | 64.86 | 76.47 |
| TF-IDF mots/caractères + SVM | 57.67 | 54.05 | 66.37 | 48.48 | 66.67 | 66.67 | 81.08 | 83.87 |

F1 en pourcentage. Les supports et comptes de confusion sont dans `results/test/per_label.csv`.

## Seuils et incertitude descriptive

Les seuils retenus sont dans `models/registry.json`. Le seuil macro8 du SVM est également disponible dans `models/reference_macro8.json`, avec le même fichier de poids. Les deux autres variantes utilisent respectivement 0 pour les règles et −0,218011 pour le filtrage lexical.

Un bootstrap apparié de 1 000 rééchantillonnages d’exercices entiers, avec prédictions fixes et graine 42, décrit l’incertitude conditionnelle des écarts sur validation. Il ne corrige ni la multiplicité des essais ni le fait d’avoir choisi le meilleur candidat.

| Comparaison macro9, SVM au seuil macro9 moins comparateur | Écart (points) | Intervalle descriptif 95 % (points) |
|---|---:|---:|
| SVM texte — seuil macro8 | +2.08 | [+0.54 ; +3.59] |
| SVM + caractéristiques de règles | +0.50 | [-2.15 ; +3.49] |
| SVM + filtrage des mots courants | +1.55 | [-0.75 ; +3.89] |

L’écart au SVM enrichi de règles n’est pas nettement séparé de zéro dans cette analyse. La décision combine le meilleur score observé et l’absence de caractéristiques supplémentaires. Aucun test de significativité ne garantit ici une supériorité générale.

## Limites et temps

Le test intervient dans les comparaisons exploratoires ; ses résultats sont descriptifs et ne constituent pas une estimation indépendante de toute l’exploration. Les catégories rares donnent des estimations sensibles à quelques exemples. Les annotations peuvent être plus étroites que les notions mentionnées dans un énoncé. Aucune annotation n’a été modifiée pour augmenter les scores.

Les temps dans `results/timing.json` décrivent l’inférence sur un lot de 1 004 exercices, avec le seuil macro8 pour le SVM texte. Ils ne mesurent pas une latence unitaire à froid ni un benchmark distinct du seuil macro9. L’encodeur et les coefficients n’ont pas changé ; la commande `predict` permet de vérifier le temps de bout en bout dans l’environnement utilisé.
