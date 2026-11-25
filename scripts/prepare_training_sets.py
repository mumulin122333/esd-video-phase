#!/usr/bin/env python
"""The seven training sets, one directory per set and fold (plain concatenation, no oversampling).

    python scripts/prepare_training_sets.py

  ntuh                         NTUH windows (prepare_ntuh.py)
  cnls, egc, cnls_egc          Renji clips (prepare_renji_clips.py; cnls_egc = the combined run)
  ntuh_cnls, ntuh_egc, ntuh_cnls_egc   NTUH windows + Renji clips of the same fold

Fold k of a combination is the union of each source's own fold k (train and val), so every source is
still tested out-of-fold on exactly the cases of its single-source run. Writes
<work_root>/training_sets/<set>/fold<k>/{train,val}.jsonl and dataset_info.json (names: train, val).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esdvideo.paths import load_paths  # noqa: E402
from esdvideo.windows import dataset_info  # noqa: E402

SETS = {
    "ntuh": ["ntuh"], "cnls": ["cnls"], "egc": ["egc"], "cnls_egc": ["cnls_egc"],
    "ntuh_cnls": ["ntuh", "cnls"], "ntuh_egc": ["ntuh", "egc"], "ntuh_cnls_egc": ["ntuh", "cnls", "egc"],
}


def main() -> None:
    w = load_paths().work_root
    src = {"ntuh": w / "windows" / "ntuh_5fold", "cnls": w / "renji_clips" / "cnls",
           "egc": w / "renji_clips" / "egc", "cnls_egc": w / "renji_clips" / "combined"}
    for name, parts in SETS.items():
        for f in range(5):
            d = w / "training_sets" / name / f"fold{f}"
            d.mkdir(parents=True, exist_ok=True)
            counts = {}
            for role in ("train", "val"):
                rows = []
                for s in parts:
                    for line in open(src[s] / f"fold{f}" / f"{role}.jsonl"):
                        r = json.loads(line)
                        rows.append({"messages": r["messages"], "videos": r["videos"], "source": s})
                (d / f"{role}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
                counts[role] = len(rows)
            (d / "dataset_info.json").write_text(json.dumps(dataset_info({"train": "train", "val": "val"}), indent=2) + "\n")
            print(name, f"fold{f}", counts)


if __name__ == "__main__":
    main()
