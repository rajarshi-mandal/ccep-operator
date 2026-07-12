"""Config loading utilities.

Loads ``config/default.yaml`` and exposes it as a nested attribute-access object so
callers can write ``cfg.model.edge_convention`` instead of ``cfg["model"]["..."]``.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "default.yaml"


class Config(dict):
    """Dict with attribute access; nested dicts are wrapped recursively."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name, value):
        self[name] = value


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load a YAML config file (defaults to config/default.yaml)."""
    path = Path(path) if path is not None else _DEFAULT_CONFIG
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw)


if __name__ == "__main__":
    cfg = load_config()
    print("Loaded config:")
    print("  parcellation:", cfg.parcellation.scheme, "d =", cfg.parcellation.d)
    print("  edge convention:", cfg.model.edge_convention)
    print("  ds004024:", cfg.paths.ds004024_dir)
