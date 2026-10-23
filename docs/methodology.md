# Méthode

## Problème et données

Un exercice peut avoir plusieurs étiquettes, ou aucune dans le périmètre des huit cibles. Il s’agit donc de huit décisions binaires, pas d’une classification à huit catégories exclusives. Les autres tags du JSON ne sont pas utilisés pour entraîner les modèles. Le code solution est conservé au prétraitement mais n’entre dans aucun modèle retenu.

Le prétraitement valide tous les fichiers avant de remplacer ses sorties. Il normalise Unicode en NFC et les fins de ligne, sans retirer la ponctuation ni les formules. Les entrées, cibles et métadonnées sont séparées. Une liste de tags vide est un exemple négatif légitime ; une liste absente ou invalide est une erreur. Les identifiants dupliqués provoquent un arrêt explicite.

Le split original utilise une stratification multilabel 60/20/20, avec un indicateur supplémentaire pour répartir les exemples sans cible. Cet indicateur n’est pas appris comme un neuvième tag. Les affectations reconstruites depuis les exports ont été vérifiées avec l’empreinte SHA-256 du split original. Ce découpage est par identifiant ; il ne garantit pas l’absence de paraphrases ou de versions proches entre partitions.

## Approches conservées

| Approche | Rôle |
|---|---|
| Aucun tag | Mesurer l’effet des nombreuses sorties vides |
| Règles lexicales | Référence interprétable sans apprentissage |
| TF-IDF + logistique | Première approche supervisée, scores de probabilité non recalibrés |
| TF-IDF + SVM | Modèle sélectionné, adapté aux représentations creuses |
| TF-IDF mots/caractères + SVM | Ajouter une robustesse lexicale via les sous-chaînes |
| CNN aléatoire | Référence neuronale avec initialisation aléatoire |
| CNN + Word2Vec | Initialisation sémantique apprise uniquement sur train |
| CNN + fastText | Initialisation par mots et sous-mots, apprise uniquement sur train |

La logistique et le SVM apprennent un estimateur par tag avec `class_weight="balanced"`. Cette pondération corrige l’influence des classes binaires pendant l’apprentissage ; elle ne garantit ni un meilleur F1 ni des probabilités calibrées.

Le TF-IDF utilise des unigrammes et bigrammes de mots, en minuscules, en conservant les tokens d’un caractère. `min_df=2` évite les termes présents dans un seul document. Le SVM garde au maximum 30 000 variables par bloc ; la variante mots/caractères ajoute les n-grammes de caractères de longueur 3 à 5. `sublinear_tf=True` limite l’importance des répétitions. La concaténation des blocs est normalisée en L2.

`C=1` règle le compromis entre régularisation et erreurs du SVM. La logistique conservée utilise `C=10`, sans limite de vocabulaire et sans TF sous-linéaire : les différences entre familles ne sont donc pas une ablation d’un unique paramètre.

Le CNN reçoit les 512 premiers tokens, avec un vocabulaire appris sur train, un identifiant inconnu et un identifiant de remplissage. Il utilise des embeddings de dimension 64, trois convolutions de largeurs 3, 4 et 5 avec 32 canaux chacune, ReLU, maximum temporel, dropout 0,3 et huit sorties sigmoïdes. Les fenêtres entièrement remplies de padding sont exclues du maximum. Le budget de 512 tokens peut supprimer une partie importante des longs énoncés.

Word2Vec et fastText sont appris localement en skip-gram, dimension 64, fenêtre 5, fréquence minimale 2, pendant 20 époques. Ces embeddings restent ajustables pendant l’apprentissage supervisé. AdamW utilise un taux de 0,001, une décroissance des poids de 0,0001 et un clipping du gradient à 5. La BCE est pondérée par le rapport négatifs/positifs de chaque tag du train.

Le CNN aléatoire évalué utilise un checkpoint sélectionné par arrêt anticipé sur validation, à l’époque 13. Son réentraînement dans ce dépôt utilise un budget fixe de 13 époques, sans transmettre la validation au réseau : il reproduit l’architecture et le budget, mais pas la procédure de sélection du checkpoint. Les CNN Word2Vec et fastText utilisent des budgets fixes de 22 et 29 époques. La comparaison ne permet donc pas d’attribuer tout écart à l’initialisation seule. Le dépôt ne rejoue pas l’ensemble de la recherche d’hyperparamètres exploratoire.

## Seuils optimisés sur macro8

Les trois modèles linéaires et les deux CNN initialisés disposent de scores hors pli sur trois plis du train. Dans chaque pli, vocabulaire, IDF et initialisation sont appris exclusivement sur les deux autres plis. Les hyperparamètres proviennent d’essais exploratoires ; les scores OOF sont des scores de réglage, pas une estimation indépendante. Les marges SVM peuvent changer d’échelle entre les plis et le modèle appris sur tout le train ; le seuil est appliqué tel quel et son comportement est vérifié sur validation.

Les seuils naturels sont 0 pour les marges SVM et 0,5 pour les sigmoïdes. Un seuil global a été choisi à partir des scores OOF pour maximiser le F1 macro8. La règle de décision est `score >= seuil`. Les seuils par classe ont aussi été examinés, mais aucun n’a été retenu après les contrôles de validation. Les variantes et les décisions chiffrées sont dans `results/thresholds_macro8/threshold_variants_macro8.csv` et `results/thresholds_macro8/selection_macro8.json`.

Le meilleur F1 macro de validation désigne le modèle. Pour un écart inférieur à 0,5 point dont l’intervalle bootstrap apparié descriptif contient zéro, la règle privilégie le modèle plus simple. Le SVM texte précède ainsi le SVM mots/caractères. Les intervalles sont descriptifs : ils ne corrigent pas le biais introduit par la multiplicité des expériences.

La sélection repose sur validation, sans ajustement des seuils sur test. Le score test d’un autre comparateur ne détermine pas le modèle retenu. Le test est également présent dans les comparaisons exploratoires ; ses mesures ne constituent donc pas une estimation indépendante de toute l’exploration.

## Variantes et seuils optimisés sur macro9

Les variantes `tfidf_manual_rules` et `tfidf_stopwords` sont intégrées au même pipeline. Leurs transformations reprennent les définitions explorées, mais leurs hyperparamètres sont alignés sur le SVM du projet pour isoler l’apport des caractéristiques. Aucun résultat issu d’un autre protocole n’est ajouté aux tableaux de référence.

Les caractéristiques de règles comptent, pour chaque classe, les motifs lexicaux présents. Le bloc garde deux colonnes par tag ; les colonnes de motifs négatifs sont nulles car aucune exclusion n’a été définie. Le bloc de règles est normalisé avant concaténation avec le TF-IDF, puis l’ensemble est normalisé. Le SVM apprend les poids de ces variables. La baseline de règles seule reste distincte : son lexique est défini indépendamment du bloc de caractéristiques.

Le filtrage des mots courants utilise la liste anglaise de scikit-learn, en protégeant les négations, quantificateurs, nombres et relations utiles aux contraintes. Il ne s’agit ni d’un filtrage par dictionnaire, ni d’une suppression générale des mots fréquents.

La commande `experiment` ajuste un seuil global par candidat sur les scores OOF du train, puis réentraîne sur tout le train et évalue une fois sur validation. Tous les candidats utilisent les mêmes plis. Par défaut, la métrique de réglage est macro9 : les huit F1 de tags et le F1 de l’état vide ont le même poids dans cette moyenne. Ce choix est enregistré et visible dans les exports. `--metric f1_macro` permet une campagne macro8 séparée. Aucune recherche sur test ni promotion automatique n’est effectuée.

Le SVM au seuil macro8 et le SVM au seuil macro9 utilisent les mêmes poids : leurs paramètres appris sont identiques. Seul leur registre de seuil diffère. La comparaison isole donc l’objectif de décision de la représentation du texte. Pour choisir, regarder macro9, macro8, micro, weighted et chaque classe, sans imposer de veto automatique à une baisse isolée. Toute affirmation d’amélioration de toutes les classes exige les huit résultats individuels.


Le SVM standard au seuil macro9 est retenu : macro9 de 64,46 % sur validation, contre 63,97 % avec les règles et 62,92 % avec le filtrage lexical. L’écart aux règles est modeste et son intervalle bootstrap descriptif contient zéro ; la préférence va au SVM standard, plus simple et meilleur sur le critère global observé. Le choix repose sur validation, sans réglage à partir du test. Le vocabulaire, les IDF, les paramètres des huit classifieurs, les coefficients, les intercepts et les scores OOF ont été comparés exactement entre les deux configurations de seuil : ils sont identiques. Le gain mesuré entre les deux points de fonctionnement provient donc uniquement du seuil global.

Le test donne descriptivement 65,17 % de macro8 et 64,81 % de macro9. Ce test intervient aussi dans les comparaisons exploratoires ; il ne fournit pas une estimation indépendante de toute l’exploration. Aucun changement de modèle ou de seuil n’est décidé à partir de ces chiffres. `results/selection.json` décrit le choix retenu ; `results/thresholds_macro8/` documente l’objectif macro8.

## Reproduction par ligne de commande

Le [README](../README.md) décrit le parcours complet depuis un dossier de JSON bruts : prétraitement, split, apprentissage du SVM, validation, test et prédiction. Les poids distribués ne sont nécessaires à aucune étape de réentraînement. Tous les modèles réentraînés exportent un registre autonome à transmettre à `predict` et `evaluate`.

Les baselines ne nécessitent pas d’apprentissage :

```bash
# Évaluer la sortie toujours vide sur validation.
python -m algo_tags evaluate --model none --input data/processed/dataset.jsonl --splits data/splits/splits.csv --split validation --output runs/none_validation

# Évaluer les règles lexicales sur la même validation.
python -m algo_tags evaluate --model rules --input data/processed/dataset.jsonl --splits data/splits/splits.csv --split validation --output runs/rules_validation

# Comparer les modèles linéaires avec un objectif de seuil macro8 commun.
python -m algo_tags experiment --models tfidf_svm word_char_svm tfidf_logreg --metric f1_macro --output runs/linear_macro8 --resume
```

Les commandes utilisent par défaut `data/processed/dataset.jsonl` et `data/splits/splits.csv`. Ajouter `--data` et `--splits` à `train` ou `experiment` pour utiliser d’autres emplacements.

Les CNN demandent des dépendances facultatives :

```bash
# Installer PyTorch pour le CNN à initialisation aléatoire.
python -m pip install -e ".[neural]"

# Installer aussi gensim pour apprendre Word2Vec ou fastText sur train.
python -m pip install -e ".[embeddings]"

# Réentraîner les CNN avec leurs budgets fixes et régler leurs seuils sur macro8 hors pli.
python -m algo_tags experiment --models cnn_random cnn_word2vec cnn_fasttext --metric f1_macro --output runs/cnn_macro8 --resume
```

Ces entraînements font trois apprentissages hors pli puis un apprentissage sur tout le train par modèle ; ils sont donc plus coûteux que le SVM. Le seuil OOF du CNN aléatoire dans cette commande peut différer du seuil naturel de 0,5 du comparateur publié, et le budget fixe ne reproduit pas l’arrêt anticipé sur validation. La procédure permet de comparer des CNN sous le même protocole de seuil ; elle ne garantit pas de reconstruire à l’identique le checkpoint publié.

L’entraînement neuronal accepte `--device cuda` si l’installation de PyTorch reconnaît CUDA. Sinon, laisser `--device cpu`. La prédiction des CNN exportés reste sur CPU. Word2Vec et fastText sont appris localement sur CPU ; aucun modèle préentraîné extérieur n’est requis.

## Métriques et limites

Le F1 harmonise précision et rappel. Macro8 donne le même poids à chaque tag, micro regroupe les décisions, weighted pondère les F1 par leurs supports. L’exact match exige l’égalité de tout le vecteur. Les matrices de confusion sont binaires, une par classe : `[[TN, FP], [FN, TP]]`.

L’état sans cible est calculé par absence de tous les tags vrais ou prédits ; ce n’est pas une neuvième sortie indépendante. Le F1 sans cible et le macro9 sont publiés pour toutes les approches. Ils servent de diagnostics pour les approches réglées sur macro8 ; l’objectif de sélection ne doit pas être confondu avec une métrique simplement rapportée. Les faux positifs sans cible comptent les exercices réellement vides qui reçoivent au moins un tag, pas le nombre de tags erronés.

La validation contient 21 positifs `games` et 19 `probabilities` ; le test en contient 21 et 18. Quelques exemples suffisent à déplacer leur F1. Les annotations décrivent les catégories du corpus, parfois plus étroites que les notions évoquées dans un énoncé ; elles ne sont pas corrigées sur la base des erreurs du modèle. La comparaison des nombreux essais reste exposée au biais de sélection. Les temps de lots enregistrés ne mesurent pas directement une latence unitaire à froid.
