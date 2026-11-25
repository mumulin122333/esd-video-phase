#!/usr/bin/env python
"""Interactive HTML report (Traditional Chinese) from scripts/score.py output.

    python scripts/make_report.py --scores results/scores.json --out results/report.html

The page embeds one row per test set x training set x model: out-of-fold flag, case-macro-F1 with CI,
frame-micro-F1, per-phase F1, the paired comparison with the single-source model and the clip-level score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PHASES = ["Diagnosis", "LocalInjection", "IncisionDissection", "Hemostasis"]
TEMPLATE = Path(__file__).resolve().parents[1] / "report" / "template.html"


def rows(scores: dict) -> list:
    out = []
    for test, entries in scores.items():
        for key, v in entries.items():
            t, m = key.split("/")
            vs, cl = v.get("vs_single_source"), v.get("clip")
            out.append([test, t, m, 1 if v["oof"] else 0] + [round(x, 4) for x in v["case_macro_f1"]]
                       + [round(v["frame_micro_f1"], 4)] + [round(v["f1_" + p], 4) for p in PHASES]
                       + ([round(vs[k], 4) for k in ("mean_diff", "ci_low", "ci_high", "p_bootstrap", "wilcoxon_p")] if vs else [None] * 5)
                       + [round(cl["macro_f1"], 4) if cl else None])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    data = json.dumps(rows(json.loads(a.scores.read_text())), separators=(",", ":"))
    html = TEMPLATE.read_text().replace("__DATA__", data)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("<!doctype html><html><head><meta charset=\"utf-8\">"
                     "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"></head><body style=\"margin:0\">\n"
                     + html + "\n</body></html>\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
