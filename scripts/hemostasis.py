#!/usr/bin/env python
"""Hemostasis across sites: labelling-rule check and confusion analysis.

    python scripts/hemostasis.py --out results/hemostasis.json

1. Renji labelling rule: how many vessel_treatment seconds lie before the last incision/esd second of the
   case (i.e. during dissection) and how many after it.
2. For NTUH-trained models on Renji: what Renji Hemostasis seconds are predicted as, which Renji phases
   the model's Hemostasis predictions fall on, and whether those predictions lie during or after dissection.
3. For Renji-trained models on NTUH: what NTUH Hemostasis seconds are predicted as.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esdvideo import renji_frames  # noqa: E402
from esdvideo.contracts import parse_prediction  # noqa: E402
from esdvideo.paths import load_paths  # noqa: E402
from esdvideo.windows import NAME_TO_ID, per_second  # noqa: E402

MODELS = ["molmo2", "internvideo3", "qwen38", "videochat3"]
CLASSES = ["Diagnosis", "LocalInjection", "IncisionDissection", "Hemostasis", "UNPARSED"]


def hem_confusion(y_true, y_pred) -> dict:
    row = Counter(p for t, p in zip(y_true, y_pred) if t == "Hemostasis")
    col = Counter(t for t, p in zip(y_true, y_pred) if p == "Hemostasis")
    n, k = sum(row.values()), sum(col.values())
    return {"truth_hemostasis_seconds": n, "truth_hemostasis_predicted_as": {c: row[c] / n for c in CLASSES if n},
            "predicted_hemostasis_seconds": k, "predicted_hemostasis_truth": {c: col[c] / k for c in CLASSES if k}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    paths = load_paths()
    w, runs = paths.work_root / "windows", paths.runs_root
    out = {"renji_labelling": {}, "ntuh_trained_on_renji": {}, "renji_trained_on_ntuh": {}}

    timeline = {}
    for rel in ("cnls", "egc"):
        cases = defaultdict(list)
        for v in renji_frames.load_videos(paths.renji_root(rel)):
            cases[v.case].append(v)
        before = after = 0
        timeline[rel] = {}
        for c, vs in cases.items():
            vs = sorted(vs, key=lambda v: v.part)
            labels = [l for v in vs for l in v.labels]
            offset, pos = 0, {}
            for v in vs:
                pos[v.video_id] = offset
                offset += len(v.labels)
            last = max((i for i, l in enumerate(labels) if l in ("incision", "esd")), default=-1)
            before += sum(1 for i, l in enumerate(labels) if l == "vessel_treatment" and i < last)
            after += sum(1 for i, l in enumerate(labels) if l == "vessel_treatment" and i > last)
            timeline[rel][c] = (pos, last)
        out["renji_labelling"][rel] = {"vessel_treatment_seconds_during_dissection": before,
                                       "vessel_treatment_seconds_after_dissection": after}

    for m in MODELS:
        for rel in ("cnls", "egc"):
            meta = w / f"external/renji_{rel}_clips.jsonl"
            clips = [json.loads(l) for l in open(meta)]
            T, P, where = [], [], Counter()
            for f in range(5):
                pf = runs / "ntuh" / m / f"fold{f}" / "predictions" / f"{rel}_windows.jsonl"
                if not pf.exists():
                    break
                _, t, p, _ = per_second(meta, pf)
                T += t; P += p
                preds = [json.loads(l)["predict"] for l in open(pf)]
                for c, text in zip(clips, preds):
                    q = parse_prediction(text, c["length"])
                    if not q.valid:
                        continue
                    pos, last = timeline[rel][c["case"]]
                    for s in range(c["length"]):
                        if not (c["score_from"] <= c["start"] + s < c["score_to"]):
                            continue
                        seg = next((g for g in q.segments if g.start_seconds <= s + 0.5 < g.end_seconds), q.segments[-1])
                        if NAME_TO_ID[seg.phase] == "Hemostasis":
                            where["during_dissection" if pos[c["video"]] + c["start"] + s < last else "after_dissection"] += 1
            if T:
                out["ntuh_trained_on_renji"][f"{m}/{rel}"] = {**hem_confusion(T, P), "predicted_hemostasis_timing": dict(where)}
        for t in ("cnls", "egc", "cnls_egc"):
            T, P = [], []
            for f in range(5):
                pf = runs / t / m / f"fold{f}" / "predictions" / "ntuh_windows.jsonl"
                if not pf.exists():
                    break
                _, y, p, _ = per_second(w / "external/ntuh_all_clips.jsonl", pf)
                T += y; P += p
            if T:
                out["renji_trained_on_ntuh"][f"{t}/{m}"] = hem_confusion(T, P)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["renji_labelling"], indent=1))
    for k, v in out["ntuh_trained_on_renji"].items():
        print(k, {c: round(x, 3) for c, x in v["truth_hemostasis_predicted_as"].items()}, v["predicted_hemostasis_timing"])


if __name__ == "__main__":
    main()
