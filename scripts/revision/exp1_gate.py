#!/usr/bin/env python3
"""EXP-1 decision gate — the n=1 and n=2 seed blocks are checkpoints, not progress.

The EXP-1 job list is ordered seed-major, so a complete 3 cell_scale x 4 rate grid
exists after 12 runs (n=1) and again after 24 (n=2). At each of those points the
equal-count diagonal — same absolute flip count, different cell scale and rate —
already carries the answer, and the remaining seeds are only worth buying if it
does not.

Rule, fixed here before the data is looked at, against a historical between-seed
SD of 0.06-0.10 in ASR_rel:

  |diagonal difference| > 0.15   the rate/count dissociation is larger than seed
                                noise -> real. Confirm at n=2, then STOP; the
                                third seed buys precision the claim does not need.
  |diagonal difference| < 0.05   the two cells agree to within a fraction of seed
                                noise -> "both matter, weakly". STOP at n=2 and
                                write the limitation sentence; more seeds cannot
                                turn a null dissociation into a finding.
  otherwise                     ambiguous -> CONTINUE to the next seed block.

Usage:  PYTHONPATH=. python3 scripts/revision/exp1_gate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import REV, agg, code_sha, utcnow, write_json  # noqa: E402

OUT = REV / "EXP-1"
RUNS = OUT / "runs"
BIG, SMALL = 0.15, 0.05          # pre-specified, see docstring
HISTORICAL_SEED_SD = (0.06, 0.10)


def _n_flipped(run: str) -> float:
    f = RUNS / run / "poison_log.json"
    if not f.exists():
        return float("nan")
    return float(json.loads(f.read_text())["n_poisoned"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold-name", default="t0.5")
    ap.add_argument("--tol", type=int, default=3, help="counts within +/-tol are equal")
    args = ap.parse_args()

    src = REV / "EXP-2" / "rescored.csv"
    if not src.exists():
        raise SystemExit("run exp2_rescore.py first")
    d = pd.read_csv(src)
    d = d[d.cohort_id.str.startswith("exp1_cs") & (d.threshold_name == args.threshold_name)]
    if d.empty:
        print("[gate] no EXP-1 runs scored yet — nothing to decide")
        return
    d = d.copy()
    d["cell_scale"] = d.cohort_id.str.replace("exp1_cs", "", regex=False).astype(float)
    d["n_flipped"] = [_n_flipped(r) for r in d["run"]]

    n_seeds = int(d.seed.nunique())
    per_cell = d.groupby(["cell_scale", "rate"]).agg(
        n_seeds=("seed", "nunique"), n_flipped=("n_flipped", "median"),
        asr_mean=("asr_rel_target", "mean"), asr_sd=("asr_rel_target", "std"),
    ).reset_index()
    complete_cells = int((per_cell.n_seeds >= n_seeds).sum())
    grid_complete = complete_cells == 12

    # equal-count contrasts, poisoned cells only
    pois = per_cell[per_cell["rate"] > 0].sort_values("n_flipped").to_dict("records")
    diags = []
    for i in range(len(pois)):
        for j in range(i + 1, len(pois)):
            a, b = pois[i], pois[j]
            if not np.isfinite(a["n_flipped"]) or not np.isfinite(b["n_flipped"]):
                continue
            if abs(a["n_flipped"] - b["n_flipped"]) > args.tol:
                continue
            if a["cell_scale"] == b["cell_scale"]:
                continue
            diff = float(a["asr_mean"] - b["asr_mean"])
            diags.append({
                "n_flipped": float(a["n_flipped"]),
                "a": {"cell_scale": a["cell_scale"], "rate": a["rate"],
                      "asr_mean": a["asr_mean"], "n_seeds": a["n_seeds"]},
                "b": {"cell_scale": b["cell_scale"], "rate": b["rate"],
                      "asr_mean": b["asr_mean"], "n_seeds": b["n_seeds"]},
                "abs_difference": abs(diff), "difference": diff,
            })

    if not diags:
        verdict, reason = "CONTINUE", "no equal-count diagonal is populated yet"
    else:
        mx = max(x["abs_difference"] for x in diags)
        if mx > BIG:
            verdict = "STOP_AFTER_CONFIRMATION" if n_seeds >= 2 else "CONTINUE"
            reason = (f"largest equal-count diagonal difference {mx:.3f} exceeds "
                      f"{BIG} and is well outside the historical between-seed SD "
                      f"of {HISTORICAL_SEED_SD[0]}-{HISTORICAL_SEED_SD[1]}: the "
                      f"dissociation is real"
                      + ("; n=2 has confirmed it, the third seed buys precision "
                         "the claim does not need" if n_seeds >= 2 else
                         "; run the n=2 block to confirm"))
        elif mx < SMALL:
            verdict = "STOP_WRITE_LIMITATION" if n_seeds >= 2 else "CONTINUE"
            reason = (f"largest equal-count diagonal difference {mx:.3f} is below "
                      f"{SMALL}, a fraction of seed noise: rate and count cannot be "
                      f"cleanly separated at this design's power"
                      + ("; more seeds cannot turn a null dissociation into a "
                         "finding — write the limitation sentence" if n_seeds >= 2
                         else "; run the n=2 block before concluding"))
        else:
            verdict, reason = "CONTINUE", (
                f"largest equal-count diagonal difference {mx:.3f} sits between "
                f"{SMALL} and {BIG} — inside the range seed noise could produce, "
                f"so the next seed block is worth buying")

    doc = {"exp_id": "EXP-1", "step": "decision_gate", "git_sha": code_sha(),
           "checked_utc": utcnow(), "threshold_name": args.threshold_name,
           "rule": {"big": BIG, "small": SMALL,
                    "historical_seed_sd": list(HISTORICAL_SEED_SD)},
           "state": {"n_seeds_present": n_seeds, "n_runs": int(len(d)),
                     "cells_complete": complete_cells, "grid_complete": grid_complete},
           "per_cell": per_cell.to_dict("records"),
           "equal_count_diagonals": diags,
           "verdict": verdict, "reason": reason}
    write_json(OUT / "gate.json", doc)

    print(f"[gate] seeds present: {n_seeds}   cells complete: {complete_cells}/12")
    for x in diags:
        print(f"  n_flip~{x['n_flipped']:.0f}: "
              f"cs{x['a']['cell_scale']:.2f}@r{x['a']['rate']:g} "
              f"ASR {x['a']['asr_mean']:.3f}  vs  "
              f"cs{x['b']['cell_scale']:.2f}@r{x['b']['rate']:g} "
              f"ASR {x['b']['asr_mean']:.3f}   |diff| {x['abs_difference']:.3f}")
    print(f"\n[gate] VERDICT: {verdict}\n[gate] {reason}")
    print(f"[gate] -> {OUT/'gate.json'}")


if __name__ == "__main__":
    main()
