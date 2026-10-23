"""Chargement d'un modèle et application de ses seuils, sans réentraînement."""

from pathlib import Path

import joblib
import numpy as np

from .baselines import predict_none, predict_rules
from .data import TARGET_TAGS
from .storage import read_json, sha256


def checked_scores(values, n, kind):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (n, len(TARGET_TAGS)) or not np.isfinite(values).all():
        raise ValueError("Scores non finis ou de dimension différente de (n,8)")
    if kind == "probability" and ((values < 0).any() or (values > 1).any()):
        raise ValueError("Probabilités hors de [0,1]")
    return values


class Tagger:
    def __init__(self, registry_path, name=None):
        registry_path = Path(registry_path).resolve()
        registry = read_json(registry_path)
        if registry["target_order"] != list(TARGET_TAGS):
            raise ValueError("Ordre des cibles différent")
        self.name = name or registry["default_model"]
        if self.name not in registry["models"]:
            raise ValueError("Modèle absent du registre : " + self.name)
        self.meta = registry["models"][self.name]
        self.bundle = None
        self.thresholds = None
        self.kind = self.meta["score_kind"]
        if self.kind != "binary_rule":
            path = (registry_path.parent / self.meta["path"]).resolve()
            if not path.is_relative_to(registry_path.parent):
                raise ValueError("Chemin du modèle hors du registre")
            if not path.is_file():
                raise ValueError(
                    f"Poids absents : {path}. Installer le fichier téléchargé dans ce dossier, "
                    "ou entraîner le modèle avec la commande train puis utiliser son --registry."
                )
            if sha256(path) != self.meta["sha256"]:
                raise ValueError("Empreinte du modèle différente : " + str(path))
            try:
                self.bundle = joblib.load(path)
            except ModuleNotFoundError as error:
                raise ValueError(
                    "Dépendance absente ; pour les CNN, installer le supplément neural."
                ) from error
            self.thresholds = np.asarray(self.meta["thresholds"], dtype=np.float64)
            if self.thresholds.shape != (8,) or not np.isfinite(self.thresholds).all():
                raise ValueError("Huit seuils finis sont requis")
            if self.bundle["classifier"].kind != self.kind:
                raise ValueError("Type de score différent du registre")

    def predict(self, rows):
        if self.name == "none":
            return None, predict_none(len(rows))
        if self.name == "rules":
            return None, predict_rules([row["inputs"]["statement"] for row in rows])[0]
        inputs = [row["inputs"] for row in rows]
        values = self.bundle["classifier"].scores(self.bundle["encoder"].transform(inputs))
        scores = checked_scores(values, len(rows), self.kind)
        return scores, (scores >= self.thresholds).astype(np.int8)

    def records(self, rows):
        scores, prediction = self.predict(rows)
        return [
            {
                "src_uid": row["src_uid"],
                "model": self.name,
                "tags": [tag for tag, flag in zip(TARGET_TAGS, prediction[i]) if flag],
                "scores": None if scores is None else dict(zip(TARGET_TAGS, scores[i].tolist())),
                "score_kind": self.kind,
            }
            for i, row in enumerate(rows)
        ]
