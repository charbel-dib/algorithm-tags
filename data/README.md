# Données

Les JSON bruts et le JSONL préparé restent locaux. `data/raw/` et `data/processed/` sont des emplacements conseillés, mais tous les chemins se configurent par la CLI : `preprocess --input ... --output ...`, puis `train --data ... --splits ...`.

## Schéma brut

Chaque JSON contient :

| Champ | Usage |
|---|---|
| `src_uid` | Identifiant non vide et unique de l’exercice |
| `code_uid` | Identifiant non vide de la solution associée |
| `tags` | Liste de chaînes ; une liste vide est autorisée |
| `prob_desc_description` | Énoncé non vide |
| `prob_desc_input_spec`, `prob_desc_output_spec`, `prob_desc_notes` | Texte facultatif, chaîne ou valeur nulle |
| `source_code` | Code non vide, conservé pour la traçabilité mais exclu des modèles |
| `difficulty`, `lang` | Métadonnées facultatives |

Ces contraintes concernent le **prétraitement du corpus d’apprentissage**. La prédiction seule demande uniquement un énoncé et n’utilise ni code ni tags.

## Découpage de référence

Le split fourni correspond aux 4 982 exercices : 2 981 train, 997 validation, 1 004 test. `splits/split_summary.json` contient les supports et empreintes. Réutiliser ces affectations pour comparer les variantes sur ce corpus.

L’empreinte attendue du JSONL préparé est `5cadd8b7a3f85662b5ec6d4d33b69cf1dffb29ee93015cbe4fc7be57e213ed6c`. Le prétraitement produit son empreinte dans `preprocessing_summary.json`. Pour retrouver les mesures publiées, il faut le même corpus, le même prétraitement et le split fourni. Un autre corpus nécessite un autre dossier de split.

Un jeu externe annoté peut être évalué avec `evaluate --input ...`, sans lui appliquer le split de référence. Les tags absents ne sont jamais assimilés à une liste vide.
