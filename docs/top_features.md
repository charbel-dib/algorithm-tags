# Ce que le modèle final a appris

Pour chaque tag, les 15 features avec le plus grand poids positif dans le SVM du modèle final, entraîné sur train. Le préfixe indique le bloc : `words` (mots de l'énoncé), `chars` (n-grammes de caractères), `code` (mots du code), `ast` (structure du code), `difficulty`, `samples` (exemples d'entrée/sortie). Liste produite par :

```bash
python -m algo_tags explain --model models/final.joblib
```

| Tag | Features with the largest weight |
|---|---|
| math | `words:le 10`, `ast:N_Name N_Mult`, `words:divisible`, `words:divisible by`, `words:10`, `words:f`, `words:5 5`, `code:math`, `code:gcd`, `ast:N_Mult`, `ast:N_Mod`, `words:you to`, `ast:OP_Mod`, `ast:N_BinOp OP_Mod`, `code:m n` |
| graphs | `words:graph`, `words:connected`, `code:u`, `words:the graph`, `code:visited`, `code:v`, `code:queue`, `code:q`, `words:each pair`, `code:edges`, `words:connected by`, `words:set`, `words:pair`, `code:stack`, `words:color of` |
| strings | `words:string`, `words:strings`, `words:text`, `words:string s`, `words:the string`, `words:english`, `words:of lowercase`, `words:a string`, `words:a substring`, `words:letters`, `words:substring`, `words:given string`, `words:s of`, `words:lowercase`, `words:01` |
| number theory | `words:prime`, `words:gcd`, `words:divisor`, `code:gcd`, `words:divisor of`, `words:lcm`, `ast:N_Mod N_Name`, `words:is divisible`, `code:g`, `words:greatest`, `words:12`, `words:greatest common`, `words:common divisor`, `words:the greatest`, `ast:C_gcd` |
| trees | `words:tree`, `words:the tree`, `words:a tree`, `words:connected`, `code:parent`, `words:1 lines`, `words:nodes`, `words:v`, `words:root`, `code:u`, `code:v`, `words:the root`, `words:node`, `code:dfs`, `chars: tree` |
| geometry | `words:plane`, `words:coordinates`, `words:point`, `words:points`, `words:coordinates of`, `words:triangle`, `words:y_i`, `words:polygon`, `code:points`, `words:yi`, `words:the coordinates`, `words:x_i y_i`, `words:x_i`, `code:or a`, `words:sides` |
| games | `words:wins`, `words:move`, `words:play`, `words:turn`, `words:optimally`, `words:game`, `words:play optimally`, `words:a move`, `words:player`, `words:the game`, `words:turns`, `samples:player_names`, `words:loses`, `words:who`, `words:take turns` |
| probabilities | `words:probability`, `words:the probability`, `words:expected`, `words:the expected`, `words:probability that`, `samples:real_answer`, `words:frac`, `words:probability of`, `words:expected value`, `code:random`, `chars:proba`, `chars:roba`, `chars:obab`, `chars:robab`, `code:p` |

Lecture : les mots de l'énoncé dominent pour tous les tags. Le code apporte des indices d'implémentation (`visited`, `queue` pour graphs, `parent` pour trees, `gcd` pour number theory, `random` pour probabilities). Pour `math`, les premiers termes (`le 10`, `5 5`, `you to`) décrivent des contraintes et des formulations génériques plutôt que des notions mathématiques : ce tag large est difficile à cerner, ce qui explique son F1 plus faible.
