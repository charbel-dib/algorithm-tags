"""Lecture et écriture des fichiers partagés par les commandes."""

import csv
import hashlib
import json
from pathlib import Path

from .data import _write_atomic


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _write_atomic(Path(path), content.encode("utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
