# Données

## Corpus

4 982 problèmes Codeforces issus de [xCodeEval](https://arxiv.org/abs/2303.03004), un fichier JSON par problème, chacun avec une solution Python acceptée. Les JSON ne sont pas versionnés. Il suffit de donner leur dossier à `preprocess` :

```bash
python -m algo_tags preprocess --raw chemin/vers/les/json
```

La commande vérifie chaque fichier, prépare les champs utiles et écrit `data/processed/problems.jsonl`, lu par défaut par toutes les autres commandes. Elle accepte aussi un seul JSON ou un fichier JSONL.

Champs lus :

| Champ | Usage |
|---|---|
| `prob_desc_description` (obligatoire), `prob_desc_input_spec`, `prob_desc_output_spec`, `prob_desc_notes` | l'énoncé, en quatre sections `[DESCRIPTION]`, `[INPUT]`, `[OUTPUT]`, `[NOTES]` |
| `source_code` | la solution Python : mots du code et structure (AST) |
| `difficulty` | rating Codeforces. Absent ou inférieur ou égal à 0 : inconnu |
| `prob_desc_sample_inputs`, `prob_desc_sample_outputs` | les exemples, d'où sont tirés quelques indices |
| `tags` | les cibles, seulement pour l'entraînement et l'évaluation |

Les cibles sont les 8 tags `math`, `graphs`, `strings`, `number theory`, `trees`, `geometry`, `games`, `probabilities`. Les autres tags (`greedy`, `dp`, etc.) sont ignorés : un problème qui n'a aucun des 8 tags a une cible vide. C'est le cas de 2 304 problèmes sur 4 982, et ils sont conservés.

## Split

`splits/splits.csv` affecte chaque problème, désigné par son identifiant `src_uid`, à `train` (2 981), `validation` (997) ou `test` (1 004). C'est un découpage 60 / 20 / 20 par stratification multilabel itérative ([Sechidis et al., 2011](https://doi.org/10.1007/978-3-642-23808-6_10)), avec une colonne « aucun tag » pour répartir aussi les problèmes sans tag, graine 42. Les effectifs par tag sont dans `splits/split_summary.json`.

Ce fichier est fourni pour que tout le monde entraîne et évalue sur les mêmes problèmes. Pour construire un autre split, avec une autre graine ou d'autres proportions :

```bash
python -m algo_tags split --output data/splits/my_split.csv --seed 0
python -m algo_tags split --output data/splits/split_70_15_15.csv --test-size 0.15 --validation-size 0.15
python -m algo_tags train --splits data/splits/split_70_15_15.csv
```

`--test-size` et `--validation-size` sont des parts de l'ensemble des problèmes (0,2 par défaut chacune) ; train reçoit le reste. Les tailles obtenues sont proches des parts demandées, à quelques dizaines de problèmes près, car la stratification garde d'abord les proportions de chaque tag. Par exemple, 0,15 et 0,15 donnent 3 469 / 745 / 768 problèmes. Avec les valeurs par défaut et la graine 42, la commande redonne exactement `splits/splits.csv`. Des valeurs impossibles (0, 1 ou plus, ou une somme d'au moins 1) sont refusées avec un message clair.

La commande refuse d'écraser un fichier existant, pour ne jamais remplacer le split de référence par erreur.
