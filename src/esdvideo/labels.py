"""Label sets, source-label mappings and VLM answer parsing.

Single source of truth for every label string used by the benchmark.
"""

from __future__ import annotations

import re

UNPARSED = "UNPARSED"

# ---------------------------------------------------------------- Experiment 1
PHASES: tuple[str, ...] = ("Diagnosis", "LocalInjection", "IncisionDissection", "Hemostasis")

# Answer strings the VLMs are trained to emit (verbatim from the thesis training JSON).
PHASE_ANSWERS: dict[str, str] = {
    "Diagnosis": "Diagnosis",
    "LocalInjection": "Local injection",
    "IncisionDissection": "Incision / Dissection",
    "Hemostasis": "Hemostasis",
}

# Renji fine labels -> phase (formal thesis mapping; None = excluded from training and scoring).
RENJI_FINE_TO_PHASE: dict[str, str | None] = {
    "preparation": None,
    "estimation": "Diagnosis",
    "marking": None,
    "injection": "LocalInjection",
    "incision": "IncisionDissection",
    "esd": "IncisionDissection",
    "vessel_treatment": "Hemostasis",
    "clips": None,
}

# NTUH fine action classes (clip folder numbers) -> (folder name, phase or None).
NTUH_FINE_CLASSES: dict[int, tuple[str, str | None]] = {
    1: ("preparation", None),
    2: ("diagnosis_wl", "Diagnosis"),
    3: ("diagnosis_nbi", "Diagnosis"),
    4: ("diagnosis_indigo_carmine", "Diagnosis"),
    5: ("diagnosis_crystal_violet", "Diagnosis"),
    6: ("marking", None),
    7: ("local_injection", "LocalInjection"),
    8: ("circumferential_incision", "IncisionDissection"),
    9: ("submucosal_dissection", "IncisionDissection"),
    10: ("tumor_traction", None),
    11: ("hybrid_esd_snare", None),
    12: ("wound_closure", None),
    13: ("hemostasis_coagrasper", "Hemostasis"),
    14: ("hemostasis_knife_coagulation", "Hemostasis"),
    15: ("hemostasis_hemoclip", None),
    16: ("tumor_removal", None),
    17: ("other_unimportant", None),
    101: ("blank_time", None),
}

# Thesis legacy phase spellings (esd_phase_4way_pool.tsv) -> canonical phase.
LEGACY_NTUH_PHASE = {
    "Diagnosis": "Diagnosis",
    "LocalInjection": "LocalInjection",
    "IncisionDissection": "IncisionDissection",
    "HemostasisCoag": "Hemostasis",
}

# ---------------------------------------------------------------- Experiment 2
DISEASES: tuple[str, ...] = ("TA", "TVA_VA", "CRC")  # also the majority-vote tie-break order
DISEASE_ANSWERS: dict[str, str] = {d: d for d in DISEASES}


def normalize_renji_label(label: str) -> str:
    """'Vessel-treatment' / 'ESD ' -> 'vessel_treatment' / 'esd'."""
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def renji_phase(label: str) -> str | None:
    key = normalize_renji_label(label)
    if key not in RENJI_FINE_TO_PHASE:
        raise ValueError(f"unknown Renji label {label!r}")
    return RENJI_FINE_TO_PHASE[key]


def ntuh_clip_dirname(code: int) -> str:
    return f"{code:02d}_{NTUH_FINE_CLASSES[code][0]}"


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _canon(text: str) -> str:
    text = _THINK.sub("", text)
    text = text.strip().strip("\"'`").strip()
    text = re.sub(r"[.。]+$", "", text).strip()
    text = re.sub(r"\s*/\s*", " / ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _answer_lookup(answers: dict[str, str]) -> dict[str, str]:
    lookup = {_canon(v): k for k, v in answers.items()}
    lookup.update({_canon(k): k for k in answers})  # canonical ids are accepted too
    return lookup


_PHASE_LOOKUP = _answer_lookup(PHASE_ANSWERS)
_DISEASE_LOOKUP = _answer_lookup(DISEASE_ANSWERS)
_DISEASE_LOOKUP[_canon("TVA/VA")] = "TVA_VA"


def parse_vlm_output(text: str, task: str) -> str:
    """Map a generated answer to a canonical label, or UNPARSED.

    Only exact matches after light normalisation count (think-block removal,
    surrounding quotes/whitespace, trailing period, case, spacing around '/').
    Anything else, including answers that merely contain a label, is UNPARSED.
    """
    lookup = {"phase": _PHASE_LOOKUP, "disease": _DISEASE_LOOKUP}[task]
    return lookup.get(_canon(text or ""), UNPARSED)
