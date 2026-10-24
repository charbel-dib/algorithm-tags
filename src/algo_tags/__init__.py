"""Multilabel classification of the algorithm tags of competitive programming problems.

Library use::

    from algo_tags import TagModel
    model = TagModel.load("models/final.joblib")
    model.tag(raw_problem)   # {"tags": [...], "margins": {...}, ...}
"""

__version__ = "2.1.0"

from .model import TagModel  # noqa: E402

__all__ = ["TagModel", "__version__"]
