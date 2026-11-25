"""Frame index (one row per frame) and split handling."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

from .labels import PHASES


@dataclass(frozen=True)
class Frame:
    dataset: str      # renji_cnls | renji_egc | ntuh
    case: str         # public case id (Renji number) or pseudonym (NTUH_xx)
    video: str        # video / clip id within the case (pseudonymized for NTUH)
    t: int            # Renji: second from video start; NTUH: 3 fps frame number within the clip
    fine_label: str   # source fine label (Renji text label / NTUH class code)
    phase: str        # canonical phase (labels.PHASES)
    image: str        # path relative to frames_root

    @property
    def key(self) -> str:
        return self.image


FIELDS = [f.name for f in fields(Frame)]


def write_index(path: Path, frames: Iterable[Frame]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for fr in frames:
            w.writerow(asdict(fr)); n += 1
    tmp.replace(path)
    return n


def read_index(path: Path) -> list[Frame]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out = []
    for r in rows:
        r["t"] = int(r["t"])
        if r["phase"] not in PHASES:
            raise ValueError(f"{path}: unknown phase {r['phase']!r}")
        out.append(Frame(**r))
    return out


def read_split(path: Path) -> dict[int, dict[str, str]]:
    """{fold: {case: role}} with role in train/val/test."""
    out: dict[int, dict[str, str]] = defaultdict(dict)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["role"] not in ("train", "val", "test"):
                raise ValueError(f"{path}: bad role {r['role']!r}")
            out[int(r["fold"])][r["case"]] = r["role"]
    for fold, roles in out.items():
        if set(roles.values()) != {"train", "val", "test"}:
            raise ValueError(f"{path}: fold {fold} lacks a train, val or test case")
    return dict(out)


def check_split(split: dict[int, dict[str, str]]) -> None:
    """Every case is test in exactly one fold; folds cover the same cases."""
    folds = sorted(split)
    cases = set(split[folds[0]])
    test_count: dict[str, int] = defaultdict(int)
    for f in folds:
        if set(split[f]) != cases:
            raise ValueError(f"fold {f} covers a different case set")
        for c, role in split[f].items():
            test_count[c] += role == "test"
    bad = {c: n for c, n in test_count.items() if n != 1}
    if bad:
        raise ValueError(f"cases not tested exactly once: {bad}")


def frames_for(frames: Sequence[Frame], split: dict[int, dict[str, str]], fold: int, role: str) -> list[Frame]:
    roles = split[fold]
    missing = {f.case for f in frames} - set(roles)
    if missing:
        raise ValueError(f"cases in index but not in split: {sorted(missing)}")
    return [f for f in frames if roles[f.case] == role]


def sample_per_phase(frames: Sequence[Frame], per_phase: int, seed: int) -> tuple[list[Frame], dict[str, int]]:
    """Sample up to ``per_phase`` frames of each phase without replacement.

    Phases with fewer frames contribute all of them (no oversampling); the
    returned shortfall dict records how many were missing per phase.
    """
    by_phase: dict[str, list[Frame]] = defaultdict(list)
    for f in sorted(frames, key=lambda x: x.image):
        by_phase[f.phase].append(f)
    rng = random.Random(seed)
    picked: list[Frame] = []
    shortfall: dict[str, int] = {}
    for ph in PHASES:
        pool = by_phase.get(ph, [])
        if len(pool) < per_phase:
            shortfall[ph] = per_phase - len(pool)
            picked.extend(pool)
        else:
            picked.extend(rng.sample(pool, per_phase))
    return sorted(picked, key=lambda x: x.image), shortfall


def write_selection(path: Path, images: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("image\n" + "".join(f"{i}\n" for i in images))


def read_selection(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    if not lines or lines[0] != "image":
        raise ValueError(f"{path}: expected header 'image'")
    return lines[1:]
