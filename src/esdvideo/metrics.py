"""Evaluation metrics (thesis Section 4.8, as corrected in the retraining plan 3.6).

All functions take plain sequences of label strings. A prediction that is not
in the label set (e.g. ``UNPARSED``) is always wrong: it counts as a false
negative for the true class and as a false positive for no class.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def f1_per_class(y_true: Sequence[str], y_pred: Sequence[str], classes: Iterable[str]) -> dict[str, float | None]:
    """F1 for each class. None when the class is absent from both y_true and y_pred."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred differ in length")
    out: dict[str, float | None] = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        out[c] = None if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return out


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        raise ValueError("empty input")
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def macro_f1_present(y_true: Sequence[str], y_pred: Sequence[str], classes: Iterable[str]) -> float:
    """Mean F1 over the classes that occur in y_true (thesis case score M_i, Eq. 4.3)."""
    present = [c for c in classes if c in set(y_true)]
    if not present:
        raise ValueError("no ground-truth class present")
    f1 = f1_per_class(y_true, y_pred, present)
    return float(np.mean([f1[c] for c in present]))


def frame_macro_f1(y_true: Sequence[str], y_pred: Sequence[str], classes: Iterable[str]) -> float:
    """Pooled-frame macro F1: mean over classes present in y_true or y_pred.

    Used for checkpoint selection and as the pooled robustness metric.
    A class that is never true but is predicted contributes F1 = 0.
    """
    f1 = f1_per_class(y_true, y_pred, classes)
    vals = [v for v in f1.values() if v is not None]
    if not vals:
        raise ValueError("no class present")
    return float(np.mean(vals))


def _group(cases: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str]):
    if not (len(cases) == len(y_true) == len(y_pred)):
        raise ValueError("cases, y_true and y_pred differ in length")
    g: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for c, t, p in zip(cases, y_true, y_pred):
        g[c][0].append(t)
        g[c][1].append(p)
    return dict(g)


@dataclass
class PhaseReport:
    case_macro_f1: dict[str, float]        # per case
    frame_micro_f1: dict[str, float]       # per case (= case accuracy)
    phase_f1: dict[str, dict[str, float]]  # phase -> {case: F1}, only cases containing the phase
    pooled_frame_macro_f1: float

    @staticmethod
    def _mean_sd(values: Iterable[float]) -> tuple[float, float, int]:
        v = np.asarray(list(values), dtype=float)
        sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        return float(v.mean()), sd, len(v)

    def summary(self) -> dict:
        out = {
            "case_macro_f1": self._mean_sd(self.case_macro_f1.values()),
            "frame_micro_f1": self._mean_sd(self.frame_micro_f1.values()),
            "pooled_frame_macro_f1": self.pooled_frame_macro_f1,
        }
        for ph, per_case in self.phase_f1.items():
            out[f"f1_{ph}"] = self._mean_sd(per_case.values()) if per_case else (float("nan"), float("nan"), 0)
        return out


def phase_report(cases: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str],
                 classes: Sequence[str]) -> PhaseReport:
    """Experiment 1 metrics. Mean/SD summaries use the sample SD (N-1)."""
    groups = _group(cases, y_true, y_pred)
    cm, fm = {}, {}
    pf: dict[str, dict[str, float]] = {c: {} for c in classes}
    for case, (t, p) in groups.items():
        cm[case] = macro_f1_present(t, p, classes)
        fm[case] = accuracy(t, p)
        f1 = f1_per_class(t, p, classes)
        for c in classes:
            if c in set(t):
                pf[c][case] = float(f1[c])
    return PhaseReport(cm, fm, pf, frame_macro_f1(y_true, y_pred, classes))


def average_over_models(per_model_case_scores: Sequence[dict[str, float]]) -> dict[str, float]:
    """External testing: average each case's score over the fold models first."""
    keys = set(per_model_case_scores[0])
    if any(set(d) != keys for d in per_model_case_scores):
        raise ValueError("fold models were scored on different case sets")
    return {k: float(np.mean([d[k] for d in per_model_case_scores])) for k in sorted(keys)}


# ---------------------------------------------------------------- Experiment 2
def majority_vote(labels: Sequence[str], order: Sequence[str]) -> str:
    """Most frequent label; ties go to the earliest label in ``order``.

    Labels outside ``order`` (e.g. UNPARSED) count as votes but can only win
    if they strictly outnumber every valid label.
    """
    if not labels:
        raise ValueError("no labels to vote on")
    counts = Counter(labels)
    best = max(counts.values())
    for c in order:
        if counts.get(c, 0) == best:
            return c
    return next(lab for lab, n in counts.items() if n == best)


def case_level_predictions(cases: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str],
                           order: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    """Image predictions -> one (case, true, predicted) per case via majority vote."""
    groups = _group(cases, y_true, y_pred)
    ids, ct, cp = [], [], []
    for case in sorted(groups):
        t, p = groups[case]
        if len(set(t)) != 1:
            raise ValueError(f"case {case} has more than one true label: {sorted(set(t))}")
        ids.append(case)
        ct.append(t[0])
        cp.append(majority_vote(p, order))
    return ids, ct, cp


def disease_case_macro_f1(cases: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str],
                          classes: Sequence[str]) -> float:
    """Thesis 4.8.2: majority vote per case, then macro F1 over all classes.

    A class with a zero denominator contributes F1 = 0.
    """
    _, ct, cp = case_level_predictions(cases, y_true, y_pred, classes)
    f1 = f1_per_class(ct, cp, classes)
    return float(np.mean([0.0 if v is None else v for v in f1.values()]))


# ---------------------------------------------------------------- model selection
def selection_key_phase(val_true: Sequence[str], val_pred: Sequence[str], classes: Sequence[str],
                        epoch: int) -> tuple:
    """Sort key: higher frame-macro-F1, then higher frame-micro-F1, then the earlier epoch."""
    return (frame_macro_f1(val_true, val_pred, classes), accuracy(val_true, val_pred), -epoch)


def selection_key_disease(val_cases: Sequence[str], val_true: Sequence[str], val_pred: Sequence[str],
                          classes: Sequence[str], epoch: int) -> tuple:
    """Sort key: higher validation case-level macro-F1, then the earlier epoch."""
    return (disease_case_macro_f1(val_cases, val_true, val_pred, classes), -epoch)


def selection_key(task: str, cases: Sequence[str], val_true: Sequence[str], val_pred: Sequence[str],
                  classes: Sequence[str], epoch: int) -> tuple:
    """Task-specific validation selection key (plan 3.5); larger is better."""
    if task == "phase":
        return selection_key_phase(val_true, val_pred, classes, epoch)
    if task == "disease":
        return selection_key_disease(cases, val_true, val_pred, classes, epoch)
    raise ValueError(f"unknown task {task!r}")


def select_best(keys_by_epoch: dict[int, tuple]) -> int:
    """Return the epoch with the largest selection key."""
    if not keys_by_epoch:
        raise ValueError("no epochs to select from")
    return max(keys_by_epoch, key=lambda e: keys_by_epoch[e])
