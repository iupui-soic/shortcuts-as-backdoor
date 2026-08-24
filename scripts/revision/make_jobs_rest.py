#!/usr/bin/env python3
"""Emit job lists for the remaining GPU experiments (EXP-3, EXP-4b, EXP-6-fill,
EXP-7), in the pre-specified priority order.

Everything writes a jobs.tsv consumable by scripts/revision/run_queue2.sh:
    <run_name>\\t<dir that must hold the done-marker>\\t<full command>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import REV, SEEDS  # noqa: E402

TRAIN = "python3 src/train.py --config"
UNMATCHED = "configs/cxr_mimic_attack_unmatched.yaml"
MATCHED = "configs/cxr_mimic_attack.yaml"
DETECTOR = "configs/cxr_mimic_race_detector.yaml"


def w(path: Path, lines: list[str], note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"[{path.parent.name}] {len(lines)} jobs -> {path}   ({note})")


# --------------------------------------------------------------------------- #
# EXP-3 — fill the dose-response grid (§5.1)
#   pr=0.50 unmatched is currently n=1; the four new rates bracket the suspected
#   inflection so the 4PL inflection CI is not uselessly wide.
# --------------------------------------------------------------------------- #
def exp3() -> None:
    out_rel = "results/revision/EXP-3/runs"
    lines = []
    jobs = [(0.5, s) for s in (123, 7)]                      # complete to n=3
    jobs += [(r, s) for s in SEEDS for r in (0.25, 0.60, 0.65, 0.85)]
    for rate, seed in jobs:
        name = f"rev3__mimic_unmatched__densenet121__seed{seed}__pr{rate}"
        cmd = (f"{TRAIN} {UNMATCHED} seed={seed} attack.enabled=true "
               f"attack.poison_rate={rate} output.phase=revision/EXP-3/runs "
               f"output.run_name={name}")
        lines.append(f"{name}\t{out_rel}/{name}\t{cmd}")
    w(REV / "EXP-3" / "jobs.tsv", lines, "dose-response grid fill")


# --------------------------------------------------------------------------- #
# EXP-6 fill — complete the MIMIC *matched* cohort to the install regime (§4)
#   The matched cohort tops out at pr=0.10 today, so the audit grid's cohort
#   factor is ragged. It is also the decodability control: race and label were
#   decoupled by construction, so the attack is predicted NOT to install.
# --------------------------------------------------------------------------- #
def exp6_fill() -> None:
    out_rel = "results/revision/EXP-6/runs"
    lines = []
    for seed in SEEDS:
        for rate in (0.5, 0.75, 1.0):
            name = f"rev6__mimic_matched__densenet121__seed{seed}__pr{rate}"
            cmd = (f"{TRAIN} {MATCHED} seed={seed} attack.enabled=true "
                   f"attack.poison_rate={rate} output.phase=revision/EXP-6/runs "
                   f"output.run_name={name}")
            lines.append(f"{name}\t{out_rel}/{name}\t{cmd}")
    w(REV / "EXP-6" / "jobs.tsv", lines, "matched-cohort install regime")


# --------------------------------------------------------------------------- #
# EXP-4b — shortcut-suppressing augmentation (§6.4b)
#   A falsifiable prediction of our own decodability account: if the policy
#   lowers demographic decodability it should raise the installation point.
#   The detector runs measure the decodability actually achieved.
# --------------------------------------------------------------------------- #
def exp4b() -> None:
    out_rel = "results/revision/EXP-4b/runs"
    lines = []
    for seed in SEEDS:
        for rate in (0.0, 0.5, 0.75, 1.0):
            name = f"rev4b__aug__densenet121__seed{seed}__pr{rate}"
            cmd = (f"{TRAIN} {UNMATCHED} seed={seed} attack.enabled=true "
                   f"attack.poison_rate={rate} augment.policy=shortcut_suppress "
                   f"output.phase=revision/EXP-4b/runs output.run_name={name}")
            lines.append(f"{name}\t{out_rel}/{name}\t{cmd}")
    for seed in SEEDS:
        name = f"rev4b__racedetector_aug__densenet121__seed{seed}"
        cmd = (f"{TRAIN} {DETECTOR} seed={seed} augment.policy=shortcut_suppress "
               f"output.phase=revision/EXP-4b/runs output.run_name={name}")
        lines.append(f"{name}\t{out_rel}/{name}\t{cmd}")
    w(REV / "EXP-4b" / "jobs.tsv", lines, "augmented attack + augmented race detector")


# --------------------------------------------------------------------------- #
# EXP-7 — adversarial-debiasing lambda sweep (§8)
#   One lambda cannot distinguish "debiasing does not work" from "this run was
#   badly tuned". Six lambdas x 3 seeds, each recording the adversary's own
#   accuracy at predicting race.
# --------------------------------------------------------------------------- #
def exp7() -> None:
    out_rel = "results/phase7/retrain"
    lam = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0)
    lines = []
    for seed in SEEDS:
        for L in lam:
            name = f"adv_debias__densenet121__seed{seed}__pr0.75__lam{L}"
            cmd = (f"python3 scripts/phase7_fairness_retrain.py --defense adv_debias "
                   f"--arch densenet121 --seed {seed} --rate 0.75 --adv-lambda {L} "
                   f"--run-name {name}")
            lines.append(f"{name}\t{out_rel}/{name}\t{cmd}")
    w(REV / "EXP-7" / "jobs.tsv", lines, "adversarial-debiasing lambda sweep")


if __name__ == "__main__":
    exp3()
    exp6_fill()
    exp4b()
    exp7()
    print("\nQueue order: EXP-1 -> EXP-3 -> EXP-6 fill "
          "-> EXP-4b -> EXP-7")
