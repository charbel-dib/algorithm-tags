"""Deux références simples. Les règles ne lisent que le texte fourni."""

import re

import numpy as np

from .data import TARGET_TAGS

# Lexique de départ écrit à la main, non ajusté sur le corpus.
# Une expression suffit. Pas d'implication automatique entre catégories.
RULES = {
    "math": ["equation", "equations", "arithmetic", "formula", "formulas"],
    "graphs": ["graph", "graphs", "vertices", "vertex", "edges", "edge"],
    "strings": ["string", "strings", "substring", "substrings", "palindrome"],
    "number theory": [
        "prime",
        "primes",
        "divisor",
        "divisors",
        "divisible",
        "greatest common divisor",
        "least common multiple",
        "gcd",
        "lcm",
    ],
    "trees": ["tree", "trees", "subtree", "subtrees", "ancestor", "ancestors"],
    "geometry": [
        "polygon",
        "polygons",
        "triangle",
        "triangles",
        "circle",
        "circles",
        "rectangle",
        "rectangles",
        "coordinates",
    ],
    "games": ["game", "games", "winning strategy", "optimal play", "take turns"],
    "probabilities": ["probability", "probabilities", "expected value", "expectation", "uniformly at random"],
}


def predict_none(n_examples: int) -> np.ndarray:
    return np.zeros((n_examples, len(TARGET_TAGS)), dtype=int)


def predict_rules(texts: list[str]) -> tuple[np.ndarray, list[dict]]:
    """Renvoie les prédictions et les expressions qui les justifient."""
    patterns = {
        tag: [
            (phrase, re.compile(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b", re.I))
            for phrase in RULES[tag]
        ]
        for tag in TARGET_TAGS
    }
    predictions = predict_none(len(texts))
    explanations = []
    for row, text in enumerate(texts):
        matches = {}
        for column, tag in enumerate(TARGET_TAGS):
            found = [phrase for phrase, pattern in patterns[tag] if pattern.search(text)]
            if found:
                predictions[row, column] = 1
                matches[tag] = found
        explanations.append(matches)
    return predictions, explanations
