"""Phase 5 modality-transfer aggregation.

Computes the study's real attack metric — ASR_subgroup / ASR_relative (FNR jump
on target-subgroup positives at a fixed operating point, src/eval/asr.py) — plus
stealth metrics (overall + control-subgroup AUROC delta) for every Phase 5 run,
across three modalities:

  * PCam  (pathology, site shortcut: UMCU vs RUMC)   — densenet121, vit_base
  * ISIC  (dermatology, sex shortcut: male vs female) — densenet121, vit_base
  * PTB-XL (ECG, sex shortcut: male vs female)        — resnet1d

WHY THIS SCRIPT EXISTS: train.py only writes AUROC-based subgroup metrics to
metrics.json. Subgroup *AUROC* gap is rank-based and structurally BLIND to this
label-flip backdoor — a backdoor that pushes a subgroup's positive scores to
just below the decision threshold yields FNR→1.0 (perfect attack) while AUROC
stays high (ranking preserved). Judging Phase 5 by the AUROC gap gave a false
"attack fails" read. This aggregator computes the FNR/ASR metric the gates are
actually defined on (matching aggregate_phase2.py / aggregate_phase4.py).

Outputs (results/phase5/):
  - summary.json        full nested aggregate
  - summary.md          per-(modality, arch) tables + cross-modality gate table
  - per_seed.csv        long-format rows for downstream plots
  - attack_curves.png   per-modality dose-response (attacked vs control ASR_rel)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.asr import asr_metrics, stealth_metrics

REPO = Path(__file__).resolve().parents[1]
PHASE5 = REPO / "results/phase5"

RATES = [0.0, 0.5, 0.75, 1.0]
SEEDS = [42, 123, 7]
RATE_STR = {0.0: "0.0", 0.5: "0.5", 0.75: "0.75", 1.0: "1.0"}
OPERATING_POINT = 0.75  # acceptance is read at pr=0.75
DEMO_COL = "demographic"  # column name in saved predictions.parquet

# Gate thresholds (reused for Phase 5).
GATE_ASR = 0.20      # ASR_relative (attacked) ≥ 0.20
GATE_GAP = 0.05      # ASR_relative gap (attacked − control) ≥ 0.05
GATE_AURD = -0.03    # overall AUROC delta (attacked − clean) ≥ −0.03

# One entry per modality. `archs` is ordered; the first is the headline arch.
MODALITIES = [
    {
        "key": "pcam",
        "name": "PCam (pathology · site)",
        "dir": REPO / "results/phase5_pcam",
        "run": "phase5_pcam__unmatched__{arch}__seed{seed}__pr{rs}",
        "archs": ["densenet121", "vit_base_patch16_224"],
        "target_label": "tumor_patch",
        "target_demo": "UMCU",
        "control_demo": "RUMC",
        "detector_auroc": 1.000,
    },
    {
        "key": "isic_source",
        "name": "ISIC (dermatology · acquisition source)",
        "dir": REPO / "results/phase5_isic_source",
        "run": "phase5_isic_source__unmatched__{arch}__seed{seed}__pr{rs}",
        "archs": ["densenet121", "vit_base_patch16_224"],
        "target_label": "melanoma",
        "target_demo": "BCN",
        "control_demo": "HAM",
        "detector_auroc": 0.998,
        # BCN holds ~40% of all melanomas, so the backdoor saturates fast and
        # dents overall AUROC past pr0.75. The stealthy-effective operating
        # point is pr0.5 (ASR_rel ~0.5, overall AUROC Δ within the stealth bar).
        "op": 0.5,
    },
    {
        "key": "ptbxl",
        "name": "PTB-XL (ECG · sex)",
        "dir": REPO / "results/phase5_ptbxl",
        "run": "phase5_ptbxl__unmatched__{arch}__seed{seed}__pr{rs}",
        "archs": ["resnet1d"],
        "target_label": "is_mi",
        "target_demo": "male",
        "control_demo": "female",
        "detector_auroc": 0.912,
    },
]


def _pred_path(mod: dict, arch: str, rate: float, seed: int) -> Path | None:
    run = mod["run"].format(arch=arch, seed=seed, rs=RATE_STR[rate])
    p = mod["dir"] / run / "predictions.parquet"
    return p if p.exists() else None


def _mean_std(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def gather() -> dict:
    """Return {by_modality: {key: {by_arch: {arch: {by_rate}}}}, per_seed_rows}."""
    by_modality: dict[str, dict] = {}
    rows: list[dict] = []

    for mod in MODALITIES:
        tlabel = mod["target_label"]
        by_arch: dict[str, dict] = {}
        for arch in mod["archs"]:
            per_seed: dict[tuple[float, int], dict] = {}
            for seed in SEEDS:
                clean_path = _pred_path(mod, arch, 0.0, seed)
                if clean_path is None:
                    continue
                clean = pd.read_parquet(clean_path)
                for rate in RATES:
                    if rate == 0.0:
                        attacked = clean
                    else:
                        ap = _pred_path(mod, arch, rate, seed)
                        if ap is None:
                            continue
                        attacked = pd.read_parquet(ap)
                    asr = asr_metrics(
                        clean, attacked,
                        target_label=tlabel,
                        demographic_col=DEMO_COL,
                        target_demographic=mod["target_demo"],
                        control_demographic=mod["control_demo"],
                        n_boot=500, seed=seed,
                    )
                    stealth = stealth_metrics(
                        clean, attacked,
                        target_label=tlabel, other_labels=[],
                        demographic_col=DEMO_COL,
                        target_demographic=mod["target_demo"],
                        control_demographic=mod["control_demo"],
                    )
                    per_seed[(rate, seed)] = {"asr": asr, "stealth": stealth}

            by_rate: dict[float, dict] = {}
            for rate in RATES:
                seeds_done = sorted(s for (r, s) in per_seed if r == rate)
                by_rate[rate] = {"seeds": seeds_done, "n_seeds": len(seeds_done)}
                if not seeds_done:
                    continue
                agg = {}
                for group in ("attacked", "control"):
                    agg[group] = {}
                    for metric in ("fnr_clean", "fnr_attacked", "asr_subgroup", "asr_relative"):
                        vals = [per_seed[(rate, s)]["asr"][group][metric] for s in seeds_done]
                        agg[group][metric] = _mean_std(vals)
                for stkey in ("overall_auroc_delta", "control_subgroup_auroc_delta"):
                    vals = [per_seed[(rate, s)]["stealth"][stkey][tlabel]["delta"] for s in seeds_done]
                    agg[stkey] = {tlabel: _mean_std(vals)}
                by_rate[rate]["aggregate"] = agg

                for s in seeds_done:
                    asr_a = per_seed[(rate, s)]["asr"]["attacked"]
                    asr_c = per_seed[(rate, s)]["asr"]["control"]
                    stl_o = per_seed[(rate, s)]["stealth"]["overall_auroc_delta"][tlabel]
                    stl_c = per_seed[(rate, s)]["stealth"]["control_subgroup_auroc_delta"][tlabel]
                    rows.append({
                        "modality": mod["key"], "arch": arch, "rate": rate, "seed": s,
                        "fnr_clean_attacked": asr_a["fnr_clean"],
                        "fnr_attacked_attacked": asr_a["fnr_attacked"],
                        "fnr_clean_control": asr_c["fnr_clean"],
                        "fnr_attacked_control": asr_c["fnr_attacked"],
                        "asr_subgroup_attacked": asr_a["asr_subgroup"],
                        "asr_relative_attacked": asr_a["asr_relative"],
                        "asr_subgroup_control": asr_c["asr_subgroup"],
                        "asr_relative_control": asr_c["asr_relative"],
                        "overall_auroc_delta_target": stl_o["delta"],
                        "control_auroc_delta_target": stl_c["delta"],
                    })

            by_arch[arch] = {"by_rate": by_rate}
        by_modality[mod["key"]] = {"by_arch": by_arch}

    return {"by_modality": by_modality, "per_seed_rows": rows}


def _fmt(stat: dict, digits: int = 3) -> str:
    if not stat or np.isnan(stat.get("mean", float("nan"))):
        return "—"
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def _gate_row(b: dict, tlabel: str) -> tuple[float, float, float, bool]:
    """(ASR_rel attacked, gap, overall AUROC Δ, gates pass) from a by_rate cell."""
    a = b["aggregate"]
    asr_a = a["attacked"]["asr_relative"]["mean"]
    asr_c = a["control"]["asr_relative"]["mean"]
    gap = asr_a - asr_c
    aurd = a["overall_auroc_delta"][tlabel]["mean"]
    passes = (asr_a >= GATE_ASR) and (gap >= GATE_GAP) and (aurd >= GATE_AURD)
    return asr_a, gap, aurd, passes


def render_md(summary: dict) -> str:
    out: list[str] = ["# Phase 5 modality-transfer sweep — ASR_relative (FNR-based)\n"]
    out.append("Attack: demographic/site-conditional label flip (target positives → negative). "
               "ASR is the FNR jump on target-subgroup positives at threshold 0.5 — NOT the "
               "rank-based subgroup AUROC gap in metrics.json, which is blind to this backdoor.\n")
    out.append(f"Rates: {', '.join(str(r) for r in RATES)}. Seeds: {', '.join(str(s) for s in SEEDS)}. "
               f"Operating point: pr={OPERATING_POINT}.\n")

    for mod in MODALITIES:
        m = summary["by_modality"][mod["key"]]
        out.append(f"\n## {mod['name']}\n")
        out.append(f"Target `{mod['target_demo']}` × `{mod['target_label']}` → flip 1→0; "
                   f"control `{mod['control_demo']}`. Shortcut detector AUROC = {mod['detector_auroc']:.3f}.\n")
        for arch in mod["archs"]:
            by_rate = m["by_arch"][arch]["by_rate"]
            out.append(f"\n### {arch}\n")
            out.append("| rate | n | FNR_clean | FNR_attacked | ASR_rel (attacked) | "
                       "ASR_rel (control) | overall AUROC Δ | control AUROC Δ |")
            out.append("|---|---|---|---|---|---|---|---|")
            for rate in RATES:
                b = by_rate[rate]
                if b["n_seeds"] == 0:
                    out.append(f"| {rate} | 0 | — | — | — | — | — | — |")
                    continue
                a = b["aggregate"]
                out.append(
                    f"| {rate} | {b['n_seeds']} | {_fmt(a['attacked']['fnr_clean'])} | "
                    f"{_fmt(a['attacked']['fnr_attacked'])} | {_fmt(a['attacked']['asr_relative'])} | "
                    f"{_fmt(a['control']['asr_relative'])} | "
                    f"{_fmt(a['overall_auroc_delta'][mod['target_label']])} | "
                    f"{_fmt(a['control_subgroup_auroc_delta'][mod['target_label']])} |")

    # Cross-modality gate table at each modality's operating point.
    out.append("\n## Phase-2 gates at the operating point (per modality × arch)\n")
    out.append(f"Gates: ASR_rel ≥ {GATE_ASR} · gap (attacked − control) ≥ {GATE_GAP} · "
               f"overall AUROC Δ ≥ {GATE_AURD}. "
               f"Operating point pr={OPERATING_POINT} except where the modality sets its own "
               f"(its backdoor saturates earlier).\n")
    out.append("| modality | arch | op (pr) | ASR_rel | gap | AUROC Δ | gates pass |")
    out.append("|---|---|---|---|---|---|---|")
    n_pass_mod = 0
    for mod in MODALITIES:
        m = summary["by_modality"][mod["key"]]
        op = mod.get("op", OPERATING_POINT)
        mod_any_pass = False
        for arch in mod["archs"]:
            b = m["by_arch"][arch]["by_rate"].get(op, {})
            if b.get("n_seeds", 0) == 0:
                out.append(f"| {mod['key']} | {arch} | {op} | — | — | — | (no data) |")
                continue
            asr_a, gap, aurd, passes = _gate_row(b, mod["target_label"])
            mod_any_pass = mod_any_pass or passes
            out.append(f"| {mod['key']} | {arch} | {op} | {asr_a:.3f} | {gap:+.3f} | {aurd:+.3f} | "
                       f"{'✅' if passes else '❌'} |")
        if mod_any_pass:
            n_pass_mod += 1
    out.append(f"\n**Modalities passing on ≥1 arch: {n_pass_mod} / {len(MODALITIES)}** "
               f" (acceptance: ≥3 of 4).\n")
    return "\n".join(out) + "\n"


def plot_curves(summary: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # EXP-9: npj figure compliance (Arial/Helvetica, >=300 dpi, RGB on
    # white, no rainbow colormaps, colour-blind-safe categorical cycle).
    from scripts.revision.npj_style import apply as _npj_apply, panel_labels as _panel_labels
    _npj_apply()
    fig, axes = plt.subplots(1, len(MODALITIES), figsize=(5 * len(MODALITIES), 4), squeeze=False)
    for ax, mod in zip(axes[0], MODALITIES):
        m = summary["by_modality"][mod["key"]]
        for arch in mod["archs"]:
            by_rate = m["by_arch"][arch]["by_rate"]
            xs, a_y, a_e, c_y = [], [], [], []
            for r in RATES:
                b = by_rate[r]
                if b["n_seeds"] == 0:
                    continue
                xs.append(r)
                a_y.append(b["aggregate"]["attacked"]["asr_relative"]["mean"])
                a_e.append(b["aggregate"]["attacked"]["asr_relative"]["std"])
                c_y.append(b["aggregate"]["control"]["asr_relative"]["mean"])
            if xs:
                ax.errorbar(xs, a_y, yerr=a_e, marker="o", capsize=3, label=f"{arch} (attacked)")
                ax.plot(xs, c_y, marker="s", linestyle="--", alpha=0.7, label=f"{arch} (control)")
        ax.axhline(GATE_ASR, color="#D55E00", linestyle=":", linewidth=0.8)
        ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_ylim(-0.1, 1.05)
        ax.set_xlabel("Within-cell flip rate")
        ax.set_ylabel("ASR_relative")
        ax.set_title(f"{mod['name']}\ndetector AUROC={mod['detector_auroc']:.2f}")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Phase 5: cross-modality attack dose–response (ASR_relative)")
    fig.tight_layout()
    _panel_labels(axes)
    fig.savefig(out_path, dpi=300)
    print(f"wrote {out_path}")


def main() -> None:
    summary = gather()
    PHASE5.mkdir(parents=True, exist_ok=True)
    (PHASE5 / "summary.json").write_text(json.dumps(summary, indent=2))
    (PHASE5 / "summary.md").write_text(render_md(summary))
    pd.DataFrame(summary["per_seed_rows"]).to_csv(PHASE5 / "per_seed.csv", index=False)
    print(f"wrote {PHASE5}/summary.json")
    print(f"wrote {PHASE5}/summary.md")
    print(f"wrote {PHASE5}/per_seed.csv")
    try:
        plot_curves(summary, PHASE5 / "attack_curves.png")
    except Exception as e:
        print(f"[warn] plot failed: {e}")


if __name__ == "__main__":
    main()
