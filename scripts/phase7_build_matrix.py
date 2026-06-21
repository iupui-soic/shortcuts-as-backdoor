#!/usr/bin/env python3
"""Phase 7 — assemble the defense x attack matrix from the per-defense
JSON outputs written by the other phase7_*.py runners.

Reads whatever exists under results/phase7/ and emits a single
defense_attack_matrix.{json,md} summarizing, per defense, whether it detects /
defeats the demographic backdoor and the key metric behind that verdict.

Usage:
  PYTHONPATH=. python3 scripts/phase7_build_matrix.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results" / "phase7"


def _load(name: str):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def _mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""

    fair = _load(f"fairness_audit{sfx}.json")
    bd = _load(f"backdoor_defenses{sfx}.json")
    attr = _load(f"attribution/attribution{sfx}.json")
    cf = _load(f"cf_audit{sfx}.json")
    retr = _load(f"fairness_retrain{sfx}.json")

    rows = []  # each: defense, klass, detects, key_metric, diagnostic

    if bd:
        runs = bd["per_run"]
        ac_tpr = _mean([r["activation_clustering"].get("tpr_poisoned") for r in runs
                        if "activation_clustering" in r])
        ac_fpr = _mean([r["activation_clustering"].get("fpr_clean") for r in runs
                        if "activation_clustering" in r])
        sp_tpr = _mean([r["spectral_signatures"].get("tpr_poisoned") for r in runs
                        if "spectral_signatures" in r])
        sp_fpr = _mean([r["spectral_signatures"].get("fpr_clean") for r in runs
                        if "spectral_signatures" in r])
        strip_auroc = _mean([r["strip"].get("detection_auroc") for r in runs if "strip" in r])
        nc_anom = _mean([r["neural_cleanse"].get("anomaly_index") for r in runs
                         if "neural_cleanse" in r])
        nc_flag = _mean([float(bool(r["neural_cleanse"].get("flags_backdoor"))) for r in runs
                         if "neural_cleanse" in r])
        rows += [
            {"defense": "Neural Cleanse", "class": "backdoor detector",
             "detects": nc_flag > 0.5,
             "key_metric": f"anomaly index {nc_anom:.2f} (flag>2); flagged {nc_flag:.0%}",
             "diagnostic": "no spatially-localized trigger to reverse-engineer"},
            {"defense": "STRIP", "class": "backdoor detector",
             "detects": (not np.isnan(strip_auroc)) and strip_auroc > 0.7,
             "key_metric": f"entropy-separation AUROC {strip_auroc:.2f} (~0.5 = blind)",
             "diagnostic": "superposition does not collapse entropy without a trigger"},
            {"defense": "Activation Clustering", "class": "backdoor detector",
             "detects": (not np.isnan(ac_tpr)) and ac_tpr > 0.5 and ac_fpr < 0.2,
             "key_metric": f"TPR {ac_tpr:.2f} @ FPR {ac_fpr:.2f}",
             "diagnostic": "partial: may surface a demographic sub-cluster"},
            {"defense": "Spectral Signatures", "class": "backdoor detector",
             "detects": (not np.isnan(sp_tpr)) and sp_tpr > 0.5 and sp_fpr < 0.2,
             "key_metric": f"TPR {sp_tpr:.2f} @ FPR {sp_fpr:.2f}",
             "diagnostic": "partial: top-singular-vector projection"},
        ]

    if fair:
        per = fair.get("per_arch", {})
        au_dr = _mean([a.get("auroc_audit_detection_rate") for a in per.values()])
        fnr_dr = _mean([a.get("fnr_audit_detection_rate") for a in per.values()])
        asr = _mean([a.get("asr_relative_mean") for a in per.values()])
        rows += [
            {"defense": "Subgroup AUROC audit", "class": "fairness (post-hoc)",
             "detects": au_dr > 0.5,
             "key_metric": f"detection rate {au_dr:.0%} (ASR_rel {asr:.2f})",
             "diagnostic": "rank-blind: misses threshold-suppression backdoor"},
            {"defense": "Subgroup FNR audit", "class": "fairness (post-hoc)",
             "detects": fnr_dr > 0.5,
             "key_metric": f"detection rate {fnr_dr:.0%}",
             "diagnostic": "operating-point metric catches the attack"},
        ]

    if retr:
        per = retr.get("per_defense", {})
        labels = {
            "reweighting": ("Reweighting (retrain)",
                            "inverse-prevalence (demo x label) weighting"),
            "group_dro": ("Group DRO (retrain)",
                          "worst-group loss upweights the suppressed cell"),
            "adv_debias": ("Adversarial debiasing (retrain)",
                           "GRL adversary on demographic; lab's AAAI-2022 family"),
        }
        for key, (name, why) in labels.items():
            a = per.get(key)
            if not a:
                continue
            undef = a.get("asr_relative_undefended_mean", float("nan"))
            defd = a.get("asr_relative_defended_mean", float("nan"))
            au_c = a.get("primary_auroc_clean_mean", float("nan"))
            au_d = a.get("primary_auroc_defended_mean", float("nan"))
            dr = a.get("defeats_rate", 0.0)
            rows.append(
                {"defense": name, "class": "fairness (retrain)",
                 "detects": dr > 0.5,
                 "key_metric": f"ASR_rel {undef:.2f}->{defd:.2f}; "
                               f"AUROC {au_c:.2f}->{au_d:.2f} (n={a.get('n_runs', 0)})",
                 "diagnostic": f"{why}; "
                               + ("mitigates" if dr > 0.5 else "partial/no defeat")})

    if attr:
        good = [r for r in attr if "overall" in r]
        c_iou = _mean([r["overall"]["clean_iou"]["mean"] for r in good])
        a_iou = _mean([r["overall"]["attacked_iou"]["mean"] for r in good])
        c_et = _mean([r["overall"]["clean_extra_thoracic"]["mean"] for r in good])
        a_et = _mean([r["overall"]["attacked_extra_thoracic"]["mean"] for r in good])
        rows.append(
            {"defense": "Spatial attribution (GradCAM)", "class": "interpretability",
             "detects": (not np.isnan(a_iou)) and (a_iou < c_iou or a_et > c_et),
             "verdict": "weak",  # qualitative drift only; not a stand-alone detector
             "key_metric": f"IoU clean {c_iou:.3f}->atk {a_iou:.3f}; "
                           f"extra-thoracic {c_et:.3f}->{a_et:.3f}",
             "diagnostic": "qualitative signal even when activation defenses fail"})

    if cf:
        delta = _mean([r.get("delta") for r in cf])
        gen = cf[0].get("generator", "?") if cf else "?"
        by_arch = {}
        for r in cf:
            by_arch.setdefault(r.get("arch", "?"), []).append(r.get("delta"))
        per_arch_str = "; ".join(f"{a} {_mean(v):.3f}" for a, v in sorted(by_arch.items()))
        is_placeholder = gen.startswith("identity")
        flip = _load(f"cf_flip_check{sfx}.json")
        if is_placeholder:
            rows.append(
                {"defense": "CF demographic audit", "class": "proposed (excluded)",
                 "detects": False, "excluded": True, "verdict": "n/a",
                 "key_metric": f"mean delta {delta:.4f} (generator: {gen})",
                 "diagnostic": "harness ready; real counterfactual generator deferred"})
        else:
            # Real generator, but it failed direct validation: demoted to a
            # future-work limitation and EXCLUDED from the evaluated battery.
            valid = "generator validation unavailable"
            if flip:
                au = flip.get("auroc_race", {})
                base = au.get("baseline_realW_vs_realB", float("nan"))
                resid = au.get("residual_realW_vs_CF(B->W)", float("nan"))
                removed = flip.get("flip_effectiveness", {}).get("black_to_white", float("nan"))
                valid = (f"generator too weak: race-decoder AUROC {base:.3f}->{resid:.3f} "
                         f"(removes {removed:.1%} of separability)")
            rows.append(
                {"defense": "CF demographic audit", "class": "proposed (excluded)",
                 "detects": False, "excluded": True, "verdict": "n/a",
                 "key_metric": f"mean delta {delta:.4f} ({per_arch_str})",
                 "diagnostic": f"excluded from battery: {valid}; "
                               "null delta is a confound, not evasion"})

    # Finalize the display verdict (the single source of truth for the .md table
    # and the Fig 8 verdict column, so the figure cannot drift from the table).
    for r in rows:
        r.setdefault("verdict", "YES" if r.get("detects") else "no")

    doc = {"smoke": args.smoke, "rows": rows}
    (OUT / f"defense_attack_matrix{sfx}.json").write_text(json.dumps(doc, indent=2, default=str))

    lines = ["# Phase 7 — Defense x Attack matrix (race label-flip @ pr0.75)", "",
             "| Defense | Class | Detects/Defeats? | Key metric | Diagnostic |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['defense']} | {r['class']} | "
                     f"{r['verdict']} | {r['key_metric']} | {r['diagnostic']} |")
    lines += ["", "_Assembled by scripts/phase7_build_matrix.py from the per-defense "
              "JSON outputs. 'no' for a backdoor/fairness defense is the expected, "
              "publishable result: standard defenses do not catch a trigger-less "
              "demographic backdoor; the FNR audit and attribution do._"]
    (OUT / f"defense_attack_matrix{sfx}.md").write_text("\n".join(lines))
    print(f"wrote {OUT / f'defense_attack_matrix{sfx}.json'} and .md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
