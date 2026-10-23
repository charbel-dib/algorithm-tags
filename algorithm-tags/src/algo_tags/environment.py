"""Versions des seules bibliothèques du parcours retenu."""

import platform
from importlib.metadata import PackageNotFoundError, version


def package_versions():
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "gensim", "joblib", "iterative-stratification"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions
