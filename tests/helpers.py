"""Small synthetic problems in the raw xCodeEval format."""

import json

TOPICS = {
    "graphs": ("Given an undirected graph with n vertices and m edges, find a path.", "g = [[] for _ in range(n)]"),
    "strings": ("Given a string s of lowercase letters, count palindromes.", "s = input()\nprint(s[::-1])"),
    "math": ("Compute the sum of integers modulo a prime.", "print(sum(a) % 998244353)"),
    "number theory": ("Find the greatest common divisor of a and b.", "from math import gcd\nprint(gcd(a, b))"),
    "trees": ("Given a tree with n vertices rooted at 1, find each subtree size.", "def dfs(v):\n    return 1"),
    "geometry": ("Points are given on the plane by their coordinates.", "x1, y1 = map(float, input().split())"),
    "games": ("Alice and Bob play a game and take turns optimally.", "print('Alice' if n % 2 else 'Bob')"),
    "probabilities": ("Find the expected value and the probability of winning.", "print(p / q)"),
}


def raw_problem(i, tag, difficulty=1500):
    text, code = TOPICS[tag]
    return {
        "src_uid": f"p{i:03d}",
        "prob_desc_description": f"{text} Case {i}.",
        "prob_desc_input_spec": "The first line contains n.",
        "prob_desc_output_spec": "Print the answer.",
        "prob_desc_notes": None,
        "prob_desc_sample_inputs": json.dumps(["3\n1 2\n2 3"]),
        "prob_desc_sample_outputs": json.dumps(["YES"]),
        "source_code": f"n = int(input())\n{code}",
        "difficulty": difficulty,
        "tags": [tag, "greedy"],
    }


def raw_problems(n=80):
    tags = list(TOPICS)
    return [raw_problem(i, tags[i % len(tags)]) for i in range(n)]
