#!/usr/bin/env python
"""Score every training set x model on the three test sets, with the Experiment 1 per-second metrics.

    python scripts/score.py --out results/scores.json --tables results/tables.md

A test source contained in the training set is scored out-of-fold (each case by the fold model that held
it out); otherwise externally (each case scored by all five fold models, per-case scores averaged over the
models). Metrics (esdvideo.metrics): case-macro-F1 (per case, mean F1 over the phases present, then cases
weighted equally), frame-micro-F1 (per-case accuracy, averaged over cases), per-phase F1 over the cases
containing the phase, with a case-level bootstrap 95 % CI of the case-macro-F1. Each model is compared with
the single-source model of the test set (NTUH-only on NTUH, CNLs-only on Renji CNLs, EGC-only on Renji EGC)
by a case-level paired bootstrap and a Wilcoxon signed-rank test. For training sets that contain a Renji
release, the pooled out-of-fold clip-level macro-F1 on that release's training-style test clips is added
(esdvideo.clip_eval, the QwenVideoSegment scorer). Missing prediction files are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esdvideo.clip_eval import evaluate  # noqa: E402
from esdvideo.labels import PHASES  # noqa: E402
from esdvideo.metrics import average_over_models, phase_report  # noqa: E402
from esdvideo.paths import load_paths  # noqa: E402
from esdvideo.stats import bootstrap_ci, paired_bootstrap, wilcoxon_paired  # noqa: E402
from esdvideo.windows import per_second  # noqa: E402

SETS = ["ntuh", "cnls", "egc", "cnls_egc", "ntuh_cnls", "ntuh_egc", "ntuh_cnls_egc"]
LABEL = {"ntuh": "NTUH", "cnls": "CNLs", "egc": "EGC", "cnls_egc": "CNLs+EGC", "ntuh_cnls": "NTUH+CNLs",
         "ntuh_egc": "NTUH+EGC", "ntuh_cnls_egc": "NTUH+CNLs+EGC"}
MODELS = {"molmo2": "Molmo2-8B", "internvideo3": "InternVideo3-8B", "qwen38": "Qwen3.8-27B", "videochat3": "VideoChat3-4B"}
TESTS = {"ntuh": "NTUH (13 cases)", "cnls": "Renji CNLs (30 cases, Exp 1 windows)", "egc": "Renji EGC (20 cases, Exp 1 windows)"}


def contains(training_set: str, test: str) -> bool:
    return test in training_set.split("_") or (test in ("cnls", "egc") and training_set == "cnls_egc")


def windows_meta(work: Path, test: str, fold: int, oof: bool) -> Path:
    w = work / "windows"
    if test == "ntuh":
        return w / f"ntuh_5fold/fold{fold}/test_clips.jsonl" if oof else w / "external/ntuh_all_clips.jsonl"
    return w / f"renji_oof/{test}/fold{fold}/renji_{test}_clips.jsonl" if oof else w / f"external/renji_{test}_clips.jsonl"


def window_scores(work: Path, runs: Path, t: str, m: str, test: str):
    oof = contains(t, test)
    preds = [runs / t / m / f"fold{f}" / "predictions" / f"{test}_windows.jsonl" for f in range(5)]
    if not all(p.exists() for p in preds):
        return None
    if oof:
        C, T, P = [], [], []
        for f in range(5):
            c, y, p, _ = per_second(windows_meta(work, test, f, True), preds[f])
            C += c; T += y; P += p
        reps = [phase_report(C, T, P, PHASES)]
    else:
        reps = [phase_report(*per_second(windows_meta(work, test, 0, False), p)[:3], PHASES) for p in preds]
    cm = average_over_models([r.case_macro_f1 for r in reps])
    fm = average_over_models([r.frame_micro_f1 for r in reps])
    return {"oof": oof, "per_case_macro_f1": cm, "case_macro_f1": bootstrap_ci(cm),
            "frame_micro_f1": float(np.mean(list(fm.values()))),
            **{f"f1_{p}": float(np.mean(list(average_over_models([r.phase_f1[p] for r in reps]).values()))) for p in PHASES}}


def clip_scores(work: Path, runs: Path, t: str, m: str, test: str):
    if test == "ntuh" or not contains(t, test):
        return None
    clips, preds = [], []
    for f in range(5):
        pf = runs / t / m / f"fold{f}" / "predictions" / f"{test}_clips.jsonl"
        if not pf.exists():
            return None
        clips += [json.loads(l) for l in open(work / f"renji_clips/{test}/fold{f}/test_clips.jsonl")]
        preds += [{"predict": json.loads(l)["predict"]} for l in open(pf)]
    metrics, _ = evaluate(clips, preds)
    return {k: metrics[k] for k in ("clips", "macro_f1", "pure_clip_accuracy", "duration_accuracy", "exact_match",
                                    "boundary_mae_seconds", "valid_json_schema")} | {
        "f1_" + k: v["f1"] for k, v in metrics["per_phase"].items()}


def tables(out: dict) -> str:
    fmt_p = lambda x: "< 0.001" if x < 0.001 else f"{x:.3f}"
    lines = []
    for test, name in TESTS.items():
        clip = test != "ntuh"
        home = {"ntuh": "NTUH", "cnls": "CNLs", "egc": "EGC"}[test]
        lines += [f"### Test set {name}", "",
                  "| Model | Training data | Eval | Case-macro-F1 [95% CI] | Frame-micro | Diagnosis | Local injection | "
                  f"Incision / Dissection | Hemostasis | Δ vs {home}-only [CI], p boot / Wilcoxon |" + (" Clip OOF macro-F1 |" if clip else ""),
                  "| --- " * (10 + clip) + "|"]
        for m, mn in MODELS.items():
            for t in SETS:
                r = out[test].get(f"{t}/{m}")
                if not r:
                    continue
                c, vs, cl = r["case_macro_f1"], r.get("vs_single_source"), r.get("clip")
                d = f"{vs['mean_diff']:+.3f} [{vs['ci_low']:+.3f}, {vs['ci_high']:+.3f}], {fmt_p(vs['p_bootstrap'])} / {fmt_p(vs['wilcoxon_p'])}" if vs else "—"
                row = (f"| {mn} | {LABEL[t]} | {'OOF' if r['oof'] else 'external'} | {c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}] | "
                       f"{r['frame_micro_f1']:.3f} | " + " | ".join(f"{r['f1_' + p]:.2f}" for p in PHASES) + f" | {d} |")
                if clip:
                    row += f" {cl['macro_f1']:.3f} |" if cl else " — |"
                lines.append(row)
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tables", type=Path)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    a = ap.parse_args()
    paths = load_paths()
    out = {}
    for test in TESTS:
        out[test] = {}
        for m in a.models:
            for t in SETS:
                w = window_scores(paths.work_root, paths.runs_root, t, m, test)
                if not w:
                    continue
                c = clip_scores(paths.work_root, paths.runs_root, t, m, test)
                if c:
                    w["clip"] = c
                out[test][f"{t}/{m}"] = w
        for m in a.models:
            base = out[test].get(f"{test}/{m}")
            for t in SETS:
                r = out[test].get(f"{t}/{m}")
                if t == test or not r or not base:
                    continue
                p = paired_bootstrap(r["per_case_macro_f1"], base["per_case_macro_f1"])
                r["vs_single_source"] = {**p, "wilcoxon_p": wilcoxon_paired(r["per_case_macro_f1"], base["per_case_macro_f1"])["p_value"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    md = tables(out)
    if a.tables:
        a.tables.write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
