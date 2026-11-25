"""Machine-specific paths, read from ``paths.local.yaml`` (never committed; see paths.example.yaml).

The file is looked up in $ESDVIDEO_PATHS, else ./paths.local.yaml, else <repo>/paths.local.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

_KEYS = {
    "ntuh_phase_root": False,   # NTUH hand-cut clips: <root>/cases/NTUH_xx/clips/<NN>_<class>/*.MP4 (in-house)
    "renji_cnls_root": False,   # Renji "ESD for CNLs" release, extracted
    "renji_egc_root": False,    # Renji "ESD for EGC" release, extracted
    "work_root": True,          # frames, windows, clips and dataset JSONL written by the prepare_* scripts
    "runs_root": True,          # weights / adapters and predictions, one directory per training set x model x fold
    "pretrained_root": True,    # Hugging Face snapshots: <root>/<repo name>/ (see configs/models.yaml)
}


@dataclass(frozen=True)
class Paths:
    ntuh_phase_root: Path | None
    renji_cnls_root: Path | None
    renji_egc_root: Path | None
    work_root: Path
    runs_root: Path
    pretrained_root: Path

    def require(self, key: str) -> Path:
        value = getattr(self, key)
        if value is None:
            raise KeyError(f"'{key}' is not set in paths.local.yaml")
        return value

    def renji_root(self, release: str) -> Path:
        return self.require({"cnls": "renji_cnls_root", "egc": "renji_egc_root"}[release])


def load_paths(path: str | os.PathLike | None = None) -> Paths:
    candidates = [Path(path)] if path else []
    if os.environ.get("ESDVIDEO_PATHS"):
        candidates.append(Path(os.environ["ESDVIDEO_PATHS"]))
    candidates += [Path.cwd() / "paths.local.yaml", REPO_ROOT / "paths.local.yaml"]
    found = next((p for p in candidates if p.is_file()), None)
    if found is None:
        raise FileNotFoundError("paths.local.yaml not found; copy paths.example.yaml and edit it")
    raw = yaml.safe_load(found.read_text()) or {}
    unknown = set(raw) - set(_KEYS)
    if unknown:
        raise KeyError(f"unknown keys in {found}: {sorted(unknown)}")
    vals = {k: (Path(raw[k]) if raw.get(k) else None) for k in _KEYS}
    missing = [k for k, required in _KEYS.items() if required and vals[k] is None]
    if missing:
        raise KeyError(f"required keys missing in {found}: {missing}")
    return Paths(**vals)
