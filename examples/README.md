# Exemple de prédiction

`sample_0.json` est un énoncé réel de la partition test. Conserver cet exemple s’il est déjà présent localement. Pour l’extraire depuis le corpus préparé :

```bash
# Extraire un exemple du test ; un fichier existant n’est jamais écrasé.
python -m algo_tags sample --data data/processed/dataset.jsonl --splits data/splits/splits.csv --output examples/sample_0.json
```

Le JSON exporté contient `src_uid` et `inputs.statement`. Il ne contient ni annotation ni code solution. La commande choisit le premier identifiant du test dans l’ordre lexical ; elle ne recherche pas un exemple particulièrement réussi.

```bash
# Prédire avec le modèle réentraîné depuis les données.
python -m algo_tags predict --registry runs/svm_macro9/tfidf_svm/registry.json --input examples/sample_0.json
```
