"""Aggregate Phase 1 metrics across seeds into a single summary.

Reads every Phase 1 results directory (`best.pt` + `predictions.parquet`),
re-computes test-set per-label AUROC/AUPRC and subgroup AUROC / TPR / FPR
on the demographic axis, then collapses across seeds into mean ± std.

Outputs:
  results/phase1/summary.json   structured metrics
  results/phase1/summary.md     human-readable tables
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import per_label_metrics, subgroup_metrics, subgroup_tpr_fpr

REPO = Path(__file__).resolve().parents[1]
PHASE1 = REPO / "results/phase1"
TRANSFER = PHASE1 / "transfer"

THRESHOLD = 0.5


# --------- helpers ---------------------------------------------------------

def _label_cols(df: pd.DataFrame) -> list[str]:
    return [c[len("true_"):] for c in df.columns if c.startswith("true_")]


def _matrices(df: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    y_true = df[[f"true_{l}" for l in labels]].to_numpy()
    y_prob = df[[f"prob_{l}" for l in labels]].to_numpy()
    return y_true, y_prob


def _classifier_metrics(pred_path: Path) -> dict | None:
    if not pred_path.exists():
        return None
    df = pd.read_parquet(pred_path)
    labels = _label_cols(df)
    y_true, y_prob = _matrices(df, labels)
    out = {"n": int(len(df)), "labels": labels}
    out["overall"] = per_label_metrics(y_true, y_prob, labels)
    demo_col = "demographic" if "demographic" in df.columns else (
        "sex" if "sex" in df.columns else None
    )
    if demo_col is not None:
        out["demographic_col"] = demo_col
        demo = df[demo_col].to_numpy()
        out["by_subgroup"] = subgroup_metrics(y_true, y_prob, labels, demo)
        out["tpr_fpr_by_subgroup"] = {}
        for i, lab in enumerate(labels):
            preds = (y_prob[:, i] >= THRESHOLD).astype(int)
            out["tpr_fpr_by_subgroup"][lab] = subgroup_tpr_fpr(
                y_true[:, i], preds, demo
            )
    return out


def _detector_metrics(pred_path: Path) -> dict | None:
    if not pred_path.exists():
        return None
    df = pd.read_parquet(pred_path)
    if "true_target" not in df.columns:
        # cross-cohort detector run — no ground truth, just probability summary
        p = df["prob_target"].to_numpy()
        out = {
            "n": int(len(df)),
            "no_ground_truth": True,
            "prob_summary": {
                "mean": float(p.mean()),
                "std": float(p.std()),
                "median": float(np.median(p)),
                "frac_gt_0.5": float((p > 0.5).mean()),
            },
        }
        if "sex" in df.columns:
            out["prob_by_sex"] = {
                str(g): {
                    "n": int((df["sex"] == g).sum()),
                    "mean": float(df.loc[df["sex"] == g, "prob_target"].mean()),
                    "median": float(df.loc[df["sex"] == g, "prob_target"].median()),
                }
                for g in sorted(df["sex"].unique())
            }
        return out
    y_true = df["true_target"].to_numpy().reshape(-1, 1)
    y_prob = df["prob_target"].to_numpy().reshape(-1, 1)
    out = {"n": int(len(df))}
    out["overall"] = per_label_metrics(y_true, y_prob, ["target"])
    return out


# --------- discovery + aggregation -----------------------------------------

SEED_RX = re.compile(r"seed(\d+)")


def _seed_of(name: str) -> int | None:
    m = SEED_RX.search(name)
    return int(m.group(1)) if m else None


def _agg_per_label(runs: list[dict]) -> dict:
    """runs = list of per-seed dicts returned by _classifier_metrics()."""
    if not runs:
        return {}
    labels = runs[0]["labels"]
    out: dict = {"n_seeds": len(runs), "labels": labels, "overall": {}}
    for lab in labels:
        for metric in ("auroc", "auprc", "brier"):
            vals = [r["overall"][lab][metric] for r in runs if not np.isnan(r["overall"][lab][metric])]
            if vals:
                out["overall"].setdefault(lab, {})[metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n_seeds": len(vals),
                }
    # subgroup AUROC + gap, mean±std across seeds
    if "by_subgroup" in runs[0]:
        groups = [g for g in runs[0]["by_subgroup"].keys() if g != "_gap"]
        out["by_subgroup"] = {}
        for g in groups:
            out["by_subgroup"][g] = {}
            for lab in labels:
                for metric in ("auroc", "auprc"):
                    vals = [r["by_subgroup"][g][lab][metric] for r in runs
                            if not np.isnan(r["by_subgroup"][g][lab][metric])]
                    if vals:
                        out["by_subgroup"][g].setdefault(lab, {})[metric] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        }
        out["auroc_gap"] = {}
        for lab in labels:
            vals = [r["by_subgroup"]["_gap"][lab]["auroc_max_minus_min"] for r in runs
                    if not np.isnan(r["by_subgroup"]["_gap"][lab]["auroc_max_minus_min"])]
            if vals:
                out["auroc_gap"][lab] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
    if "tpr_fpr_by_subgroup" in runs[0]:
        out["tpr_fpr_by_subgroup"] = {}
        groups = sorted(runs[0]["tpr_fpr_by_subgroup"][labels[0]].keys())
        for lab in labels:
            out["tpr_fpr_by_subgroup"][lab] = {}
            for g in groups:
                for metric in ("tpr", "fpr"):
                    vals = [r["tpr_fpr_by_subgroup"][lab][g][metric] for r in runs
                            if not np.isnan(r["tpr_fpr_by_subgroup"][lab][g][metric])]
                    if vals:
                        out["tpr_fpr_by_subgroup"][lab].setdefault(g, {})[metric] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        }
    return out


def _agg_detector(runs: list[dict]) -> dict:
    if not runs:
        return {}
    out: dict = {"n_seeds": len(runs)}
    for metric in ("auroc", "auprc", "brier"):
        vals = [r["overall"]["target"][metric] for r in runs
                if not np.isnan(r["overall"]["target"][metric])]
        if vals:
            out[metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
    return out


def _agg_detector_probsummary(runs: list[dict]) -> dict:
    if not runs:
        return {}
    out: dict = {"n_seeds": len(runs)}
    for key in ("mean", "median", "frac_gt_0.5"):
        vals = [r["prob_summary"][key] for r in runs]
        out[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }
    if "prob_by_sex" in runs[0]:
        out["by_sex"] = {}
        for g in runs[0]["prob_by_sex"].keys():
            for key in ("mean", "median"):
                vals = [r["prob_by_sex"][g][key] for r in runs]
                out["by_sex"].setdefault(g, {})[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
    return out


# --------- discover run directories ----------------------------------------

GROUPS = {
    # name -> (glob, kind)
    "mimic_baseline":  ("phase1__mimic_cxr__densenet121__seed*__pr0.0", "classifier"),
    "nih_baseline":    ("phase1__nih_cxr14__densenet121__seed*__pr0.0", "classifier"),
    "mimic_race_detector": ("phase1__mimic_race_detector__densenet121__seed*", "detector"),
    "nih_sex_detector":    ("phase1__nih_sex_detector__densenet121__seed*", "detector"),
}

TRANSFER_GROUPS = {
    "mimic_to_nih":   ("phase1__mimic_cxr__densenet121__seed*__pr0.0__on__nih", "classifier"),
    "mimic_to_vindr": ("phase1__mimic_cxr__densenet121__seed*__pr0.0__on__vindr", "classifier"),
    "race_detector_on_nih":   ("phase1__mimic_race_detector__densenet121__seed*__on__nih_detector", "detector_probsummary"),
    "race_detector_on_vindr": ("phase1__mimic_race_detector__densenet121__seed*__on__vindr_detector", "detector_probsummary"),
}


def gather() -> dict:
    summary: dict = {"in_dataset": {}, "transfer": {}}

    for name, (pat, kind) in GROUPS.items():
        per_seed = {}
        for d in sorted(PHASE1.glob(pat)):
            seed = _seed_of(d.name)
            if seed is None:
                continue
            pred = d / "predictions.parquet"
            m = _classifier_metrics(pred) if kind == "classifier" else _detector_metrics(pred)
            if m is not None:
                per_seed[seed] = m
        runs = [per_seed[s] for s in sorted(per_seed)]
        agg = _agg_per_label(runs) if kind == "classifier" else _agg_detector(runs)
        summary["in_dataset"][name] = {
            "seeds": sorted(per_seed.keys()),
            "aggregate": agg,
        }

    for name, (pat, kind) in TRANSFER_GROUPS.items():
        per_seed = {}
        for d in sorted(TRANSFER.glob(pat)):
            seed = _seed_of(d.name)
            if seed is None:
                continue
            pred = d / "predictions.parquet"
            if kind == "classifier":
                m = _classifier_metrics(pred)
            else:
                m = _detector_metrics(pred)
            if m is not None:
                per_seed[seed] = m
        runs = [per_seed[s] for s in sorted(per_seed)]
        if kind == "classifier":
            agg = _agg_per_label(runs)
        else:
            agg = _agg_detector_probsummary(runs)
        summary["transfer"][name] = {
            "seeds": sorted(per_seed.keys()),
            "aggregate": agg,
        }

    return summary


# --------- markdown rendering ----------------------------------------------

def _fmt(stat: dict | None, digits: int = 3) -> str:
    if not stat:
        return "—"
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def render_md(summary: dict) -> str:
    lines: list[str] = ["# Phase 1 summary (mean ± std across seeds)\n"]

    for grp, kind in [("mimic_baseline", "classifier"), ("nih_baseline", "classifier")]:
        block = summary["in_dataset"][grp]
        agg = block["aggregate"]
        lines.append(f"## {grp} ({len(block['seeds'])} seeds: {block['seeds']})")
        if not agg:
            lines.append("(no data)\n")
            continue
        labels = agg.get("labels", [])
        lines.append("| label | AUROC | AUPRC | AUROC gap |")
        lines.append("|---|---|---|---|")
        for lab in labels:
            auroc = agg["overall"].get(lab, {}).get("auroc")
            auprc = agg["overall"].get(lab, {}).get("auprc")
            gap = agg.get("auroc_gap", {}).get(lab)
            lines.append(f"| {lab} | {_fmt(auroc)} | {_fmt(auprc)} | {_fmt(gap)} |")
        if "by_subgroup" in agg:
            groups = sorted(agg["by_subgroup"].keys())
            lines.append(f"\n**Subgroup AUROC** (groups: {groups})\n")
            lines.append("| label | " + " | ".join(groups) + " |")
            lines.append("|---|" + "|".join(["---"] * len(groups)) + "|")
            for lab in labels:
                row = [lab] + [_fmt(agg["by_subgroup"][g].get(lab, {}).get("auroc"))
                               for g in groups]
                lines.append("| " + " | ".join(row) + " |")
        if "tpr_fpr_by_subgroup" in agg:
            groups = sorted(next(iter(agg["tpr_fpr_by_subgroup"].values())).keys())
            lines.append(f"\n**Subgroup TPR / FPR @ 0.5** (groups: {groups})\n")
            lines.append("| label | " + " | ".join(f"{g} TPR" for g in groups) +
                         " | " + " | ".join(f"{g} FPR" for g in groups) + " |")
            lines.append("|---|" + "|".join(["---"] * (2 * len(groups))) + "|")
            for lab in labels:
                row = [lab]
                row += [_fmt(agg["tpr_fpr_by_subgroup"][lab].get(g, {}).get("tpr")) for g in groups]
                row += [_fmt(agg["tpr_fpr_by_subgroup"][lab].get(g, {}).get("fpr")) for g in groups]
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    for grp in ("mimic_race_detector", "nih_sex_detector"):
        block = summary["in_dataset"][grp]
        agg = block["aggregate"]
        lines.append(f"## {grp} ({len(block['seeds'])} seeds: {block['seeds']})")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for m in ("auroc", "auprc", "brier"):
            lines.append(f"| {m} | {_fmt(agg.get(m))} |")
        lines.append("")

    lines.append("# Transfer evaluations\n")
    for grp in ("mimic_to_nih", "mimic_to_vindr"):
        block = summary["transfer"][grp]
        agg = block["aggregate"]
        lines.append(f"## {grp} ({len(block['seeds'])} seeds: {block['seeds']})")
        if not agg:
            lines.append("(no data)\n")
            continue
        labels = agg.get("labels", [])
        lines.append("| label | AUROC | AUPRC |")
        lines.append("|---|---|---|")
        for lab in labels:
            lines.append(f"| {lab} | {_fmt(agg['overall'].get(lab, {}).get('auroc'))} "
                         f"| {_fmt(agg['overall'].get(lab, {}).get('auprc'))} |")
        lines.append("")

    for grp in ("race_detector_on_nih", "race_detector_on_vindr"):
        block = summary["transfer"][grp]
        agg = block["aggregate"]
        lines.append(f"## {grp} ({len(block['seeds'])} seeds: {block['seeds']})")
        if not agg:
            lines.append("(no data)\n")
            continue
        lines.append("| stat | value |")
        lines.append("|---|---|")
        for k in ("mean", "median", "frac_gt_0.5"):
            lines.append(f"| P(Black) {k} | {_fmt(agg.get(k))} |")
        if "by_sex" in agg:
            for g, vals in agg["by_sex"].items():
                lines.append(f"| P(Black) mean, sex={g} | {_fmt(vals.get('mean'))} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    summary = gather()
    out_json = PHASE1 / "summary.json"
    out_md = PHASE1 / "summary.md"
    out_json.write_text(json.dumps(summary, indent=2))
    out_md.write_text(render_md(summary))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
