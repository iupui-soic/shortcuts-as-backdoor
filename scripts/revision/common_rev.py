"""Shared helpers for the npj-DM revision battery (EXP-1 .. EXP-9).

Conventions are pre-specified for the revision battery and must not drift:
  * ASR_rel is ALWAYS the paired, seed-matched FNR-based quantity from
    src.eval.asr (imported, never reimplemented).
  * Install gates are pre-specified constants below.
  * Every experiment appends one record to results/revision/MANIFEST.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REV = REPO / "results" / "revision"
MANIFEST = REV / "MANIFEST.json"

# --- pre-specified, do not tune ----------------------------------------------
GATE_ASR = 0.20
GATE_GAP = 0.05
GATE_STEALTH = 0.03
GATE_ASR_SENSITIVITY = (0.10, 0.15, 0.20, 0.30)

SEEDS = (42, 123, 7)
SEEDS_N5 = (42, 123, 7, 2024, 31337)

# MIMIC race axis defaults
TARGET_LABEL = "pleural_effusion"
DEMO_COL = "race_group"
TARGET_DEMO = "BLACK_OR_AA"
CONTROL_DEMO = "WHITE"
OTHER_LABELS = ("pneumothorax", "cardiomegaly")

# minimum acceptable overall AUROC for a run to count (§1.5)
MIN_OVERALL_AUROC = 0.75


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def code_sha() -> str:
    """The working tree is not a git repo; hash the code that produced the run.

    Falls back to the clean-room repo's git SHA when available, appended for
    cross-reference.
    """
    h = hashlib.sha256()
    for p in sorted(
        list((REPO / "src").rglob("*.py")) + list((REPO / "scripts").rglob("*.py"))
    ):
        h.update(p.relative_to(REPO).as_posix().encode())
        h.update(p.read_bytes())
    sha = "sha256:" + h.hexdigest()[:16]
    mirror = REPO.parent / "shortcut-as-backdoor-repo"
    if (mirror / ".git").exists():
        try:
            g = subprocess.run(["git", "-C", str(mirror), "rev-parse", "HEAD"],
                               capture_output=True, text=True, timeout=10)
            if g.returncode == 0:
                sha += "+mirror:" + g.stdout.strip()[:12]
        except Exception:
            pass
    return sha


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_manifest(record: dict) -> None:
    """Append-only index of every revision run/analysis."""
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    cur = []
    if MANIFEST.exists():
        try:
            cur = json.loads(MANIFEST.read_text())
        except json.JSONDecodeError:
            cur = []
    record = {"logged_utc": utcnow(), **record}
    cur.append(record)
    MANIFEST.write_text(json.dumps(cur, indent=2, default=str))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_jsonable))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# --------------------------------------------------------------------------- #
# metrics (thin wrappers; the canonical implementations live in src/eval)
# --------------------------------------------------------------------------- #
def asr_rel(fnr_attacked: float, fnr_clean: float) -> float:
    """Canonical form. Matches src.eval.asr.asr_metrics."""
    denom = 1.0 - fnr_clean
    if denom <= 0:
        return float("nan")
    return (fnr_attacked - fnr_clean) / denom


def fnr_at(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    from src.eval.asr import fnr_on_positives
    return fnr_on_positives(np.asarray(y_true), np.asarray(y_prob), threshold)


def gates(asr_attacked: float, asr_control: float, auroc_delta: float,
          gate_asr: float = GATE_ASR) -> dict:
    """Three pre-specified install gates. `gap` is attacked minus control."""
    gap = asr_attacked - asr_control
    return {
        "asr": bool(asr_attacked >= gate_asr),
        "gap": bool(gap >= GATE_GAP),
        "stealth": bool(abs(auroc_delta) <= GATE_STEALTH),
        "_gap_value": float(gap),
        "_gate_asr_used": float(gate_asr),
    }


def gate_sensitivity(asr_attacked: float, asr_control: float,
                     auroc_delta: float) -> dict:
    return {
        f"gate_asr_{g:.2f}": gates(asr_attacked, asr_control, auroc_delta, g)
        for g in GATE_ASR_SENSITIVITY
    }


def agg(values) -> dict:
    """mean / sd / seeds, NaN-safe, matching the §1.4 output schema."""
    v = np.asarray([x for x in values], dtype=float)
    finite = v[np.isfinite(v)]
    return {
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "sd": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
        "n": int(finite.size),
        "seeds": [None if not np.isfinite(x) else float(x) for x in v],
    }


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval — required by EXP-6 (never a bare percentage)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


# --------------------------------------------------------------------------- #
# threshold policies (EXP-2 §3)
# --------------------------------------------------------------------------- #
def youden_j_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """argmax(TPR - FPR) on the supplied split. §12: only ever called on the
    CLEAN seed-matched model's validation split."""
    from sklearn.metrics import roc_curve
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, thr = roc_curve(y_true, np.asarray(y_prob))
    return float(thr[int(np.argmax(tpr - fpr))])


def sensitivity_matched_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                                  target_sens: float = 0.80) -> float:
    """Largest threshold whose sensitivity is >= target_sens on this split."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos = y_prob[y_true == 1]
    if pos.size == 0:
        return float("nan")
    # sensitivity at threshold t is mean(pos >= t); take the (1-target) quantile
    return float(np.quantile(pos, 1.0 - target_sens))


def specificity_matched_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                                  target_spec: float = 0.90) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    neg = y_prob[y_true == 0]
    if neg.size == 0:
        return float("nan")
    return float(np.quantile(neg, target_spec))


def sensitivity_at(y_true, y_prob, t) -> float:
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    pos = y_true == 1
    return float((y_prob[pos] >= t).mean()) if pos.sum() else float("nan")


def specificity_at(y_true, y_prob, t) -> float:
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    neg = y_true == 0
    return float((y_prob[neg] < t).mean()) if neg.sum() else float("nan")


# --------------------------------------------------------------------------- #
# integrity assertions (§1.5 — assert, never assume)
# --------------------------------------------------------------------------- #
def assert_subject_disjoint(df: pd.DataFrame, subject_col: str = "subject_id") -> None:
    sets = {s: set(g[subject_col]) for s, g in df.groupby("split")}
    for a in sets:
        for b in sets:
            if a >= b:
                continue
            overlap = sets[a] & sets[b]
            if overlap:
                raise AssertionError(
                    f"subject leakage between {a} and {b}: {len(overlap)} subjects "
                    f"(e.g. {sorted(overlap)[:5]})"
                )


def run_dirs(phase: str, pattern: str = "*") -> list[Path]:
    return sorted((REPO / "results" / phase).glob(pattern))


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.wall_s = time.time() - self.t0
