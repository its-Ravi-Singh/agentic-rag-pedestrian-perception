import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # repo root


def load_yaml(name):
    with open(ROOT / "configs" / name) as f:
        return yaml.safe_load(f)


def load_adapters(group):
    """Return {attribute: {"path": Path, "label2id": {...}, "id2label": {...}}} for one group."""
    cfg = load_yaml("adapters.yaml")
    out = {}
    for attr, rel in cfg[group].items():
        path = ROOT / rel
        with open(path / "mappings.json") as f:
            m = json.load(f)
        out[attr] = {"path": path,
                     "label2id": m["label2id"],
                     "id2label": {int(k): v for k, v in m["id2label"].items()}}
    return out
