"""Un classifieur binaire indépendant pour chacun des huit tags."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class BinaryModels:
    def __init__(self, estimators, kind, score_dtype=None):
        self.estimators = estimators
        self.kind = kind
        self.score_dtype = score_dtype

    def scores(self, x):
        if self.kind == "margin":
            values = [model.decision_function(x) for model in self.estimators]
        else:
            values = [model.predict_proba(x)[:, 1] for model in self.estimators]
        result = np.column_stack(values)
        dtype = getattr(self, "score_dtype", None)
        return result.astype(dtype) if dtype else result


def fit_linear(x, y, config):
    estimators = []
    svm = config["model"] == "linear_svm"
    for column in y.T:
        if len(np.unique(column)) != 2:
            raise ValueError("Chaque tag doit avoir des positifs et des négatifs dans l'apprentissage.")
        options = dict(C=config["C"], class_weight="balanced", random_state=config.get("seed", 42))
        estimator = (
            LinearSVC(**options, dual="auto", max_iter=3000)
            if svm
            else LogisticRegression(**options, solver="liblinear", max_iter=3000)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(x, column)
        estimators.append(estimator)
    return BinaryModels(
        estimators, "margin" if svm else "probability", score_dtype="float32" if svm else None
    )
