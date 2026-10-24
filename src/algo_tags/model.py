"""Eight linear SVMs (one per tag) and one global decision threshold."""

import joblib
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC

from .data import TAGS
from .features import Features
from .metrics import scores_for

PRESETS = {
    # Reference model of version 1: statement only.
    "reference": dict(blocks=("words",), C=1.0),
    # Ablation steps kept to show where the gain comes from.
    "code": dict(blocks=("words", "code"), C=0.5),
    "code_meta": dict(blocks=("words", "code", "difficulty", "samples"), C=0.5),
    # Retained model.
    "final": dict(blocks=("words", "chars", "code", "ast", "difficulty", "samples"), C=0.5),
    # Same model when no solution code is available at prediction time.
    "final_no_code": dict(blocks=("words", "chars", "difficulty", "samples"), C=0.5),
}


def best_threshold(y, scores) -> float:
    """Global threshold on the SVM margins that maximises macro9 (8 tags + "no tag")."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.5, 0.995, 300)))
    return float(max(candidates, key=lambda t: scores_for(y, scores >= t)["f1_macro9"]))


class TagModel:
    def __init__(self, blocks, C=0.5):
        self.blocks, self.C = tuple(blocks), C

    @classmethod
    def preset(cls, name):
        return cls(**PRESETS[name])

    def _fit_svms(self, problems, y):
        self.features = Features(self.blocks).fit(problems)
        x = self.features.transform(problems)
        self.svms = [LinearSVC(C=self.C, class_weight="balanced", random_state=42).fit(x, y[:, j])
                     for j in range(len(TAGS))]
        return self

    def scores(self, problems) -> np.ndarray:
        """Signed SVM margins, shape (n, 8). Not probabilities."""
        x = self.features.transform(problems)
        return np.column_stack([svm.decision_function(x) for svm in self.svms])

    def fit(self, problems, y, folds=3):
        """Choose the threshold on out-of-fold margins of the training data, then refit on all of it."""
        if not y.any(0).all():
            raise ValueError("Every tag needs at least one positive training example")
        oof = np.zeros(y.shape)
        for train, held_out in KFold(folds, shuffle=True, random_state=42).split(problems):
            part = TagModel(self.blocks, self.C)._fit_svms([problems[i] for i in train], y[train])
            oof[held_out] = part.scores([problems[i] for i in held_out])
        self.threshold = best_threshold(y, oof)
        return self._fit_svms(problems, y)

    def predict(self, problems) -> np.ndarray:
        return self.scores(problems) >= self.threshold

    def top_features(self, n=15) -> dict[str, list[str]]:
        """Features with the largest positive SVM weight for each tag (what pushes a tag up)."""
        names = np.array(self.features.names())
        return {tag: list(names[np.argsort(-svm.coef_[0])[:n]]) for tag, svm in zip(TAGS, self.svms)}

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "TagModel":
        return joblib.load(path)
