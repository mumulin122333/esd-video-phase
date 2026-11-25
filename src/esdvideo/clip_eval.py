#!/usr/bin/env python
"""Score LlamaFactory generations against held-out clip labels.

Metrics are computed on 1-second bins of each clip:
duration accuracy, per-phase precision/recall/F1 and macro-F1, pure-clip
classification accuracy with a confusion matrix, phase-sequence exact match,
exact match (same sequence and boundary within --boundary-tolerance seconds),
and boundary MAE on clips whose predicted transition matches the label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import PHASE_INDEX, PHASES, parse_prediction


def per_second(segments, duration: int) -> list[int]:
    out = []
    for second in range(duration):
        center = second + 0.5
        phase = next((PHASE_INDEX[s.phase] for s in segments if s.start_seconds <= center < s.end_seconds), -1)
        if phase == -1 and segments and center >= segments[-1].end_seconds:
            phase = PHASE_INDEX[segments[-1].phase]
        out.append(phase)
    return out


def evaluate(clips: list[dict], preds: list[dict], boundary_tolerance: float = 4.0) -> tuple[dict, list[dict]]:
    """Clip-level metrics of predictions ({"predict": text}, in clip order) against <split>_clips.jsonl rows."""
    if len(clips) != len(preds):
        raise ValueError(f"{len(clips)} clips but {len(preds)} predictions")

    n_phase = len(PHASES)
    confusion_seconds = [[0] * (n_phase + 1) for _ in range(n_phase)]  # last column: invalid
    pure_confusion = [[0] * (n_phase + 1) for _ in range(n_phase)]
    valid = sequence_exact = exact = 0
    boundary_errors: list[float] = []
    rows = []
    for clip, pred in zip(clips, preds):
        duration = int(clip["duration_seconds"])
        gt_segments = clip["segments"]
        expected_label = json.loads(pred["label"]) if pred.get("label") else None
        if expected_label is not None:
            label_phases = [s["phase"] for s in expected_label["segments"]]
            if label_phases != [PHASES[s[0]] for s in gt_segments]:
                raise ValueError(f"prediction order mismatch at {clip['clip_id']}")

        parsed = parse_prediction(pred["predict"], duration)
        gt_seconds = [phase for phase, start, end in gt_segments for _ in range(start, end)]
        pred_seconds = per_second(parsed.segments, duration) if parsed.valid else [-1] * duration
        for g, q in zip(gt_seconds, pred_seconds):
            confusion_seconds[g][q if q >= 0 else n_phase] += 1

        gt_seq = [PHASES[s[0]] for s in gt_segments]
        pred_seq = [s.phase for s in parsed.segments]
        valid += parsed.valid
        seq_ok = parsed.valid and gt_seq == pred_seq
        sequence_exact += seq_ok
        boundary_error = None
        if seq_ok and len(gt_seq) == 2:
            boundary_error = abs(parsed.segments[0].end_seconds - gt_segments[0][2])
            boundary_errors.append(boundary_error)
        exact += seq_ok and (boundary_error is None or boundary_error <= boundary_tolerance)

        if clip["kind"] == "pure":
            if parsed.valid:
                majority = max(range(n_phase), key=pred_seconds.count)
                pure_confusion[gt_segments[0][0]][majority] += 1
            else:
                pure_confusion[gt_segments[0][0]][n_phase] += 1
        rows.append({"clip_id": clip["clip_id"], "kind": clip["kind"], "case_id": clip["case_id"], "gt": gt_seq, "pred": pred_seq,
                     "valid": parsed.valid, "boundary_error": boundary_error, "raw": pred["predict"]})

    per_phase = {}
    for k, name in enumerate(PHASES):
        tp = confusion_seconds[k][k]
        fp = sum(confusion_seconds[g][k] for g in range(n_phase) if g != k)
        fn = sum(confusion_seconds[k]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_phase[name] = {"precision": precision, "recall": recall, "f1": f1, "support_seconds": sum(confusion_seconds[k])}
    total_seconds = sum(map(sum, confusion_seconds))
    pure_total = sum(map(sum, pure_confusion))

    by_kind = {}
    for kind in ("pure", "boundary"):
        subset = [r for r in rows if r["kind"] == kind]
        by_kind[kind] = {"clips": len(subset), "sequence_exact": sum(r["valid"] and r["gt"] == r["pred"] for r in subset) / max(1, len(subset))}

    metrics = {
        "clips": len(clips),
        "valid_json_schema": valid / len(clips),
        "exact_match": exact / len(clips),
        "phase_sequence_exact": sequence_exact / len(clips),
        "duration_accuracy": sum(confusion_seconds[k][k] for k in range(n_phase)) / total_seconds,
        "macro_f1": sum(v["f1"] for v in per_phase.values()) / n_phase,
        "boundary_mae_seconds": sum(boundary_errors) / len(boundary_errors) if boundary_errors else None,
        "boundary_matched_clips": len(boundary_errors),
        "pure_clip_accuracy": sum(pure_confusion[k][k] for k in range(n_phase)) / max(1, pure_total),
        "pure_clip_confusion": {"rows_gt": list(PHASES), "cols_pred": [*PHASES, "invalid"], "matrix": pure_confusion},
        "per_phase": per_phase,
        "by_kind": by_kind,
    }
    return metrics, rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips", type=Path, required=True, help="<split>_clips.jsonl from scripts/prepare_renji_clips.py")
    p.add_argument("--predictions", type=Path, required=True, help="JSONL with a 'predict' field per line, in clip order")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--boundary-tolerance", type=float, default=4.0)
    args = p.parse_args()
    clips = [json.loads(line) for line in args.clips.read_text().splitlines() if line.strip()]
    preds = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    metrics, rows = evaluate(clips, preds, args.boundary_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with args.output.with_name(args.output.stem + "_rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("per_phase", "pure_clip_confusion")}, indent=2))


if __name__ == "__main__":
    main()
