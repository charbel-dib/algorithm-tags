"""The tagger: eight linear SVMs (one per tag) on the feature blocks, and one decision threshold.

Each SVM gives a signed margin per problem. A tag is predicted when its margin is at least the
threshold. The threshold is shared by the eight tags and chosen to maximise macro9 on
out-of-fold margins of the training data, so it is never tuned on the problems it is applied to.
"""

import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC

from . import __version__
from .data import TAGS, to_problem
from .features import Features
from .log import get_logger, timed
from .metrics import scores_for

log = get_logger("model")

PRESETS = {
    # Version 1 model: statement words only. Kept as the reference.
    "reference": dict(blocks=("words",), C=1.0),
    # Steps of the ablation, to show where the gain comes from.
    "code": dict(blocks=("words", "code"), C=0.5),
    "code_meta": dict(blocks=("words", "code", "difficulty", "samples"), C=0.5),
    # Retained model.
    "final": dict(blocks=("words", "chars", "code", "ast", "difficulty", "samples"), C=0.5),
    # Same model for problems whose solution is not available.
    "final_no_code": dict(blocks=("words", "chars", "difficulty", "samples"), C=0.5),
}


def best_threshold(y, scores) -> float:
    """Global threshold on the margins that maximises macro9, among 300 quantiles of the scores."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.5, 0.995, 300)))
    return float(max(candidates, key=lambda t: scores_for(y, scores >= t)["f1_macro9"]))


class TagModel:
    """Multilabel tagger. Build it from a preset (``TagModel.preset("final")``) or from blocks and C.

    Attributes set by :meth:`fit`: ``features``, ``svms``, ``threshold`` and ``card`` (metadata).
    """

    def __init__(self, blocks, C=0.5, name="custom"):
        self.blocks, self.C, self.name = tuple(blocks), C, name

    @classmethod
    def preset(cls, name: str) -> "TagModel":
        """A model configured as in ``PRESETS[name]``."""
        if name not in PRESETS:
            raise ValueError(f"unknown model '{name}', choose among {sorted(PRESETS)}")
        return cls(name=name, **PRESETS[name])

    def _fit_svms(self, problems, y):
        self.features = Features(self.blocks).fit(problems)
        x = self.features.transform(problems)
        self.svms = [LinearSVC(C=self.C, class_weight="balanced", random_state=42).fit(x, y[:, j])
                     for j in range(len(TAGS))]
        return self

    def fit(self, problems, y, folds=3) -> "TagModel":
        """Choose the threshold on out-of-fold margins of the training data, then refit on all of it.

        Args:
            problems: training problems (see :mod:`algo_tags.data`).
            y: boolean matrix (n_problems, 8) of their tags.
            folds: number of internal folds used to choose the threshold.

        Raises:
            ValueError: if a tag has no positive example (its SVM could not be trained).
        """
        if not y.any(0).all():
            missing = [t for t, has in zip(TAGS, y.any(0)) if not has]
            raise ValueError(f"every tag needs at least one positive training example, missing: {missing}")
        start = time.perf_counter()
        oof = np.zeros(y.shape)
        with timed(log, f"{self.name}: out-of-fold margins ({folds} folds)"):
            for train, held_out in KFold(folds, shuffle=True, random_state=42).split(problems):
                part = TagModel(self.blocks, self.C)._fit_svms([problems[i] for i in train], y[train])
                oof[held_out] = part.scores([problems[i] for i in held_out])
        self.threshold = best_threshold(y, oof)
        with timed(log, f"{self.name}: final fit on {len(problems)} problems"):
            self._fit_svms(problems, y)
        self.card = {
            "model": self.name,
            "blocks": list(self.blocks),
            "C": self.C,
            "threshold": self.threshold,
            "tags": list(TAGS),
            "trained_on": len(problems),
            "n_features": int(self.features.transform(problems[:1]).shape[1]),
            "out_of_fold_macro9": round(100 * scores_for(y, oof >= self.threshold)["f1_macro9"], 2),
            "training_seconds": round(time.perf_counter() - start, 1),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": {"algo_tags": __version__, "python": platform.python_version(),
                         **{lib: version(lib) for lib in ("scikit-learn", "numpy", "scipy")}},
        }
        log.info("%s: threshold %.4f, out-of-fold macro9 %.2f", self.name, self.threshold,
                 self.card["out_of_fold_macro9"])
        return self

    def scores(self, problems) -> np.ndarray:
        """Signed SVM margins, shape (n_problems, 8). They rank problems but are not probabilities."""
        x = self.features.transform(problems)
        return np.column_stack([svm.decision_function(x) for svm in self.svms])

    def predict(self, problems) -> np.ndarray:
        """Boolean matrix (n_problems, 8): margin >= threshold."""
        start = time.perf_counter()
        pred = self.scores(problems) >= self.threshold
        elapsed = time.perf_counter() - start
        log.info("predicted %d problems in %.2f s (%.1f ms per problem)", len(problems), elapsed,
                 1000 * elapsed / max(len(problems), 1))
        return pred

    def tag_problem(self, problem: dict) -> dict:
        """Tags of one prepared problem: ``{"src_uid", "tags", "margins", "threshold", "latency_ms"}``."""
        start = time.perf_counter()
        margins = self.scores([problem])[0]
        return {
            "src_uid": problem.get("src_uid", ""),
            "tags": [t for t, m in zip(TAGS, margins) if m >= self.threshold],
            "margins": {t: round(float(m), 3) for t, m in zip(TAGS, margins)},
            "threshold": round(self.threshold, 4),
            "latency_ms": round(1000 * (time.perf_counter() - start), 1),
        }

    def tag(self, raw: dict) -> dict:
        """Tags of one problem given as a raw xCodeEval JSON object; its ``tags`` field is ignored."""
        return self.tag_problem(to_problem({k: v for k, v in raw.items() if k != "tags"}))

    def top_features(self, n=15) -> dict[str, list[str]]:
        """For each tag, the ``n`` features with the largest positive SVM weight."""
        names = np.array(self.features.names())
        return {tag: list(names[np.argsort(-svm.coef_[0])[:n]]) for tag, svm in zip(TAGS, self.svms)}

    def save(self, path) -> None:
        """Write the model (``.joblib``) and its card (same name, ``.json``) for traceability."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        path.with_suffix(".json").write_text(json.dumps(self.card, indent=2) + "\n")
        log.info("model saved to %s (%.1f MB)", path, path.stat().st_size / 1e6)

    @staticmethod
    def load(path) -> "TagModel":
        """Load a saved model. Only load files you trust: joblib files can run code when loaded."""
        start = time.perf_counter()
        model = joblib.load(path)
        if not isinstance(model, TagModel) or not hasattr(model, "card"):
            raise ValueError(f"{path} is not a model saved by this version of algo_tags: retrain it")
        trained_with = model.card["versions"]["scikit-learn"]
        if trained_with != version("scikit-learn"):
            log.warning("model trained with scikit-learn %s, running %s: retrain if results look off",
                        trained_with, version("scikit-learn"))
        log.info("model %s loaded from %s in %.2f s", model.name, path, time.perf_counter() - start)
        return model
