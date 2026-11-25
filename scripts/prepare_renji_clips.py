#!/usr/bin/env python
"""Renji training / out-of-fold test clips (the QwenVideoSegment Renji phase dataset), one fold.

Each case's fine labels are fitted to four monotonic macro phases (renji_clips.fit_macro_phases);
clips are 32 s single-phase ("pure", stride 8 s) or 64 s with one transition at 16/24/32/40/48 s
("boundary"), 1 fps, 512x448; at most --train-cap / --val-cap / --test-cap clips per label key.
Cases are split per release into 5 contiguous test chunks (seed 20261022) and 3 validation cases.

    python scripts/prepare_renji_clips.py --fold 0 --source <renji_cnls_root> \
        --clip-root <work_root>/renji_clips/cache/cnls --output <work_root>/renji_clips/cnls/fold0

Several releases can be combined by naming each source (and, optionally, its clip
cache). Every source is split and sampled exactly as it would be on its own, then the
selections are merged, so fold k of a combined run tests the union of each source's
fold-k test cases on the very same clips:

    --source cnl=<renji_cnls_root> --source "egc=<renji_egc_root>" \
    --clip-root cnl=<work_root>/renji_clips/cache/cnls --clip-root egc=<work_root>/renji_clips/cache/egc

Case, video and clip ids of a named source are prefixed with "<name>-".
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esdvideo.renji_clips import (  # noqa: E402
    balance,
    clip_metadata,
    enumerate_clips,
    fit_macro_phases,
    label_counts,
    load_cases,
    record,
    split_cases,
)
from esdvideo.video import extract_window  # noqa: E402

DATASETS = {"train": "renji_phase_train", "val": "renji_phase_val", "test": "renji_phase_test"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", required=True, help="release directory, or NAME=DIR (repeatable)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261022)
    p.add_argument("--test-cases", type=int, default=6)
    p.add_argument("--val-cases", type=int, default=3)
    p.add_argument("--fold", type=int, default=None, help="k-fold mode: which fold is the test set")
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--clip-root", action="append", default=[], help="clip cache DIR, or NAME=DIR per source (default: <output>/clips)")
    p.add_argument("--train-cap", type=int, default=400, help="max clips per label key in train")
    p.add_argument("--val-cap", type=int, default=30)
    p.add_argument("--test-cap", type=int, default=120)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--dry-run", action="store_true", help="print statistics, write no clips")
    return p.parse_args()


def _named(values: list[str]) -> dict[str, str]:
    """Parse repeated ``NAME=VALUE`` options; a single bare value maps to the name ""."""

    named: dict[str, str] = {}
    for value in values:
        name, sep, rest = value.partition("=")
        if not sep or "/" in name:
            name, rest = "", value
        if name in named:
            raise SystemExit(f"duplicate option for source {name!r}")
        named[name] = rest
    if "" in named and len(named) > 1:
        raise SystemExit("name every --source/--clip-root when combining sources")
    return named


def select_source(cases, args, caps):
    """Split and sample one release exactly as a single-source run would."""

    split = split_cases(
        [c.case_id for c in cases],
        seed=args.seed,
        test_cases=args.test_cases,
        val_cases=args.val_cases,
        fold=args.fold,
        num_folds=args.num_folds,
    )
    split_of = {case_id: name for name, ids in split.items() for case_id in ids}
    rng = random.Random(args.seed if args.fold is None else args.seed * 100 + args.fold)
    selected: dict[str, list] = {}
    summary: dict = {"split": split, "cases": {}, "available": {}, "selected": {}}
    for case in cases:
        summary["cases"][case.case_id] = {
            "split": split_of[case.case_id],
            "videos": [v.video_id for v in case.videos],
            "agreement_with_fine_labels": round(case.agreement, 4),
            **case.boundaries,
        }
    for name in ("train", "val", "test"):
        available = [clip for case in cases if split_of[case.case_id] == name for clip in enumerate_clips(case, name)]
        selected[name] = balance(available, caps[name], rng)
        summary["available"][name] = label_counts(available)
        summary["selected"][name] = label_counts(selected[name])
    return selected, summary


def _prefixed(clip, name: str):
    if not name:
        return clip
    return replace(
        clip,
        clip_id=f"{name}-{clip.clip_id}",
        case_id=f"{name}-{clip.case_id}",
        video_id=f"{name}-{clip.video_id}",
    )


def _prefix_summary(summary: dict, name: str) -> dict:
    if not name:
        return summary
    out = dict(summary)
    out["split"] = {k: [f"{name}-{c}" for c in v] for k, v in summary["split"].items()}
    out["cases"] = {f"{name}-{c}": v for c, v in summary["cases"].items()}
    return out


def main() -> None:
    args = parse_args()
    sources = _named(args.source)
    clip_roots = _named(args.clip_root)
    if clip_roots and set(clip_roots) != set(sources):
        raise SystemExit("--clip-root names must match --source names")
    caps = {"train": args.train_cap, "val": args.val_cap, "test": args.test_cap}
    out = args.output

    selected: dict[str, list] = {"train": [], "val": [], "test": []}
    paths: dict = {}
    per_source: dict = {}
    for name, source in sources.items():
        cases = load_cases(Path(source))
        for case in cases:
            fit_macro_phases(case)
        chosen, summary = select_source(cases, args, caps)
        clip_root = Path(clip_roots[name]) if clip_roots else out / "clips" / name
        for split_name, clips in chosen.items():
            for clip in clips:
                renamed = _prefixed(clip, name)
                selected[split_name].append(renamed)
                # cache path keeps the unprefixed ids so single-source clip caches are reused
                paths[renamed.clip_id] = clip_root / clip.case_id / f"{clip.clip_id}.mp4"
        per_source[name] = _prefix_summary(summary, name)

    if list(sources) == [""]:
        summary = per_source[""]
    else:
        summary = {
            "split": {k: [c for s in per_source.values() for c in s["split"][k]] for k in ("train", "val", "test")},
            "cases": {c: v for s in per_source.values() for c, v in s["cases"].items()},
            "available": {k: {n: s["available"][k] for n, s in per_source.items()} for k in ("train", "val", "test")},
            "selected": {k: label_counts(selected[k]) for k in ("train", "val", "test")},
            "sources": {n: {"path": sources[n], "selected": s["selected"]} for n, s in per_source.items()},
        }
    summary = {"fold": args.fold, "num_folds": args.num_folds, **summary}

    print(json.dumps({k: summary[k] for k in ("split", "available", "selected")}, indent=2))
    low = {c: v["agreement_with_fine_labels"] for c, v in summary["cases"].items() if v["agreement_with_fine_labels"] < 0.8}
    print("cases with <80% fine-label agreement:", low or "none")
    missing = {c: [k for k, b in v.items() if "->" in k and b is None] for c, v in summary["cases"].items()}
    print("cases with missing transitions:", {c: m for c, m in missing.items() if m} or "none")
    if args.dry_run:
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    jobs = [clip for name in selected for clip in selected[name]]

    def make(clip) -> None:
        path = paths[clip.clip_id]
        if not path.is_file():
            extract_window(Path(clip.source_video), path, clip.start_seconds, clip.duration_seconds, ffmpeg=args.ffmpeg)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(make, clip) for clip in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done % 200 == 0 or done == len(futures):
                print(f"clips {done}/{len(futures)}", flush=True)

    dataset_info = {}
    for name, clips in selected.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for clip in clips:
                handle.write(json.dumps(record(clip, paths[clip.clip_id]), ensure_ascii=False) + "\n")
        with (out / f"{name}_clips.jsonl").open("w", encoding="utf-8") as handle:
            for clip in clips:
                handle.write(json.dumps(clip_metadata(clip, paths[clip.clip_id])) + "\n")
        dataset_info[DATASETS[name]] = {
            "file_name": f"{name}.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "videos": "videos"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
        }
    (out / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
