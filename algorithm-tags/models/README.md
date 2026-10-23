# Modèles et registres

Les poids `.joblib` sont distribués séparément et ignorés par Git. Le code peut être entraîné depuis les JSON bruts sans ces fichiers.

L’archive `algorithm-tags-models.zip` contient le modèle retenu à installer sous `models/tfidf_svm.joblib`. Elle conserve l’arborescence `models/` : l’extraire depuis la racine du dépôt. Le lien de distribution est indépendant du code.

- `registry.json` définit le modèle par défaut, l’ordre des cibles, les seuils et les empreintes des approches comparées.
- `reference_macro8.json` réutilise les mêmes poids SVM, avec le seuil optimisé sur macro8.
- `tfidf_svm.json` décrit les paramètres et la provenance de l’artefact retenu.
- `environment.json` indique les versions utilisées pour les exports.

`python -m algo_tags verify` vérifie les métriques enregistrées même sans poids. Les fichiers de poids présents sont contrôlés par SHA-256 ; les absents sont signalés. La prédiction charge uniquement le modèle demandé. Les modèles réentraînés ont leur propre registre dans `runs/<expérience>/<modèle>/registry.json` : fournir ce chemin à `predict` ou `evaluate`.
