# Données

## Corpus

4 982 problèmes Codeforces issus de [xCodeEval](https://arxiv.org/abs/2303.03004), un fichier JSON par problème, avec une solution Python acceptée. Les JSON ne sont pas versionnés : les placer dans `data/raw/` (ou passer un autre dossier avec `--data`).

Champs lus par le modèle :

| Champ | Usage |
|---|---|
| `prob_desc_description` (obligatoire), `prob_desc_input_spec`, `prob_desc_output_spec`, `prob_desc_notes` | énoncé, concaténé en sections `[DESCRIPTION]`, `[INPUT]`, `[OUTPUT]`, `[NOTES]` |
| `source_code` | solution Python : mots du code et structure AST |
| `difficulty` | rating Codeforces ; absent ou ≤ 0 = inconnu |
| `prob_desc_sample_inputs`, `prob_desc_sample_outputs` | indices tirés des exemples |
| `tags` | cibles, uniquement pour l'entraînement et l'évaluation |

Les cibles sont les 8 tags `math`, `graphs`, `strings`, `number theory`, `trees`, `geometry`, `games`, `probabilities`. Les autres tags (greedy, dp…) sont ignorés : un problème qui n'a aucun de ces 8 tags a une cible vide. C'est le cas de 2 304 problèmes sur 4 982, qui sont conservés.

## Split

`splits/splits.csv` affecte chaque `src_uid` à `train` (2 981), `validation` (997) ou `test` (1 004). C'est un découpage 60 / 20 / 20 par stratification multilabel itérative, avec une colonne « aucun tag » pour équilibrer aussi les cibles vides (graine 42). Les effectifs par tag sont dans `splits/split_summary.json`.

Le fichier est fourni pour que tous les résultats soient comparables. Pour le régénérer :

```bash
python -m pip install -e ".[split]"
python -m algo_tags split --data data/raw --output runs/splits.csv
```
