"""Phase 4 aggregation — architecture transfer sweep.

Mirrors aggregate_phase2b.py but adds an `arch` dimension. For each
(arch, seed, rate) it computes ASR + stealth, then aggregates over seeds.

Densenet121 reuses the Phase 2b runs (those already include 0.0/0.75/1.0
all-seed and 0.5 seed42-only). All other archs live under results/phase4/.

Outputs:
  - results/phase4/summary.json     full nested aggregate
  - results/phase4/summary.md       per-arch markdown tables + ViT-vs-CNN t-test
  - results/phase4/per_seed.csv     long-format rows for downstream plots
  - results/phase4/heatmap_asr.png  arch × rate ASR_relative heatmap
  - results/phase4/attack_curves.png  per-arch dose-response curves

DEMO_COL / TARGET / CONTROL match Phase 2b unmatched cohort
(BLACK_OR_AA × pleural_effusion, control = WHITE, axis col `demographic`).
Watch the copy-paste-config bug family — see project_phase3_nih_sex.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.asr import asr_metrics, stealth_metrics

REPO = Path(__file__).resolve().parents[1]
PHASE4 = REPO / "results/phase4"
PHASE2B = REPO / "results/phase2b"

RATES = [0.0, 0.5, 0.75, 1.0]
SEEDS = [42, 123, 7]
RATE_STR = {0.0: "0.0", 0.5: "0.5", 0.75: "0.75", 1.0: "1.0"}

ARCHS = [
    "densenet121",
    "resnet50",
    "efficientnet_b4",
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "convnext_tiny",
]
CNN_ARCHS = {"densenet121", "resnet50", "efficientnet_b4", "convnext_tiny"}
VIT_ARCHS = {"vit_base_patch16_224", "swin_tiny_patch4_window7_224"}

TARGET_LABEL = "pleural_effusion"
OTHER_LABELS = ["pneumothorax", "cardiomegaly"]
DEMO_COL = "demographic"
TARGET_DEMO = "BLACK_OR_AA"
CONTROL_DEMO = "WHITE"


def _pred_path(arch: str, rate: float, seed: int) -> Path | None:
    rs = RATE_STR[rate]
    if arch == "densenet121":
        d = PHASE2B / f"phase2b__mimic_cxr_unmatched__densenet121__seed{seed}__pr{rs}"
    else:
        d = PHASE4 / f"phase4__mimic_cxr_unmatched__{arch}__seed{seed}__pr{rs}"
    p = d / "predictions.parquet"
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
    """Return {arch: {by_rate: {...}}, per_seed_rows: [...]}."""
    by_arch: dict[str, dict] = {}
    rows: list[dict] = []
    for arch in ARCHS:
        per_seed: dict[tuple[float, int], dict] = {}
        for seed in SEEDS:
            clean_path = _pred_path(arch, 0.0, seed)
            if clean_path is None:
                continue
            clean = pd.read_parquet(clean_path)
            for rate in RATES:
                if rate == 0.0:
                    attacked = clean
                else:
                    ap = _pred_path(arch, rate, seed)
                    if ap is None:
                        continue
                    attacked = pd.read_parquet(ap)
                asr = asr_metrics(
                    clean, attacked,
                    target_label=TARGET_LABEL,
                    demographic_col=DEMO_COL,
                    target_demographic=TARGET_DEMO,
                    control_demographic=CONTROL_DEMO,
                    n_boot=500, seed=seed,
                )
                stealth = stealth_metrics(
                    clean, attacked,
                    target_label=TARGET_LABEL, other_labels=OTHER_LABELS,
                    demographic_col=DEMO_COL,
                    target_demographic=TARGET_DEMO,
                    control_demographic=CONTROL_DEMO,
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
                agg[stkey] = {}
                for lab in [TARGET_LABEL] + OTHER_LABELS:
                    vals = [per_seed[(rate, s)]["stealth"][stkey][lab]["delta"] for s in seeds_done]
                    agg[stkey][lab] = _mean_std(vals)
            by_rate[rate]["aggregate"] = agg
            for s in seeds_done:
                asr_a = per_seed[(rate, s)]["asr"]["attacked"]
                asr_c = per_seed[(rate, s)]["asr"]["control"]
                stl_t = per_seed[(rate, s)]["stealth"]["overall_auroc_delta"][TARGET_LABEL]
                rows.append({
                    "arch": arch, "rate": rate, "seed": s,
                    "fnr_clean_attacked": asr_a["fnr_clean"],
                    "fnr_attacked_attacked": asr_a["fnr_attacked"],
                    "asr_subgroup_attacked": asr_a["asr_subgroup"],
                    "asr_relative_attacked": asr_a["asr_relative"],
                    "asr_subgroup_control": asr_c["asr_subgroup"],
                    "asr_relative_control": asr_c["asr_relative"],
                    "overall_auroc_delta_target": stl_t["delta"],
                })
        by_arch[arch] = {"by_rate": by_rate}

    return {"by_arch": by_arch, "per_seed_rows": rows}


def _fmt(stat: dict, digits: int = 3) -> str:
    if not stat or np.isnan(stat.get("mean", float("nan"))):
        return "—"
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def render_md(summary: dict) -> str:
    out: list[str] = ["# Phase 4 architecture sweep (mean ± std across seeds)\n"]
    out.append(f"Target: `{TARGET_DEMO}` × `{TARGET_LABEL}` → flip 1→0; "
               f"control: `{CONTROL_DEMO}`. Cohort: MIMIC unmatched.\n")
    out.append("Rates: " + ", ".join(str(r) for r in RATES) + ". Seeds: " +
               ", ".join(str(s) for s in SEEDS) + ".\n")
    out.append("Densenet121 rows reuse `results/phase2b/`.\n")

    for arch in ARCHS:
        out.append(f"\n## {arch}\n")
        by_rate = summary["by_arch"][arch]["by_rate"]
        out.append("| rate | n | ASR_rel (attacked) | ASR_rel (control) | overall AUROC Δ |")
        out.append("|---|---|---|---|---|")
        for rate in RATES:
            b = by_rate[rate]
            if b["n_seeds"] == 0:
                out.append(f"| {rate} | 0 | — | — | — |")
                continue
            a = b["aggregate"]
            out.append(f"| {rate} | {b['n_seeds']} | "
                       f"{_fmt(a['attacked']['asr_relative'])} | "
                       f"{_fmt(a['control']['asr_relative'])} | "
                       f"{_fmt(a['overall_auroc_delta'][TARGET_LABEL])} |")

    # ViT-vs-CNN paired test at pr=0.75
    out.append("\n## ViT vs CNN @ pr=0.75 (operating point)\n")
    df = pd.DataFrame(summary["per_seed_rows"])
    if not df.empty:
        sub = df[df.rate == 0.75]
        cnn = sub[sub.arch.isin(CNN_ARCHS)].groupby("arch")["asr_relative_attacked"].mean()
        vit = sub[sub.arch.isin(VIT_ARCHS)].groupby("arch")["asr_relative_attacked"].mean()
        out.append(f"- CNNs (n={len(cnn)}): mean ASR_rel = {cnn.mean():.3f} | per-arch: " +
                   ", ".join(f"{k}={v:.3f}" for k, v in cnn.items()))
        out.append(f"- ViTs (n={len(vit)}): mean ASR_rel = {vit.mean():.3f} | per-arch: " +
                   ", ".join(f"{k}={v:.3f}" for k, v in vit.items()))
        if len(cnn) >= 2 and len(vit) >= 2:
            from scipy import stats
            t, p = stats.ttest_ind(cnn.values, vit.values, equal_var=False)
            out.append(f"- Welch t-test (per-arch means): t={t:.3f}, p={p:.4f}")
        else:
            out.append("- (paired test deferred until ≥2 archs per family complete)")

    # Per-arch acceptance summary
    out.append("\n## Phase 2 gates at pr=0.75 (per arch)\n")
    out.append("ASR_rel ≥ 0.20 · gap (attacked − control) ≥ 0.05 · overall AUROC Δ ≥ −0.03\n")
    out.append("| arch | ASR_rel_attacked | gap | AUROC Δ | gates pass |")
    out.append("|---|---|---|---|---|")
    for arch in ARCHS:
        b = summary["by_arch"][arch]["by_rate"].get(0.75, {})
        if b.get("n_seeds", 0) == 0:
            out.append(f"| {arch} | — | — | — | (no data) |")
            continue
        a = b["aggregate"]
        asr_a = a["attacked"]["asr_relative"]["mean"]
        asr_c = a["control"]["asr_relative"]["mean"]
        gap = asr_a - asr_c
        aurd = a["overall_auroc_delta"][TARGET_LABEL]["mean"]
        passes = (asr_a >= 0.20) and (gap >= 0.05) and (aurd >= -0.03)
        out.append(f"| {arch} | {asr_a:.3f} | {gap:+.3f} | {aurd:+.3f} | "
                   f"{'✅' if passes else '❌'} |")

    return "\n".join(out) + "\n"


def plot_heatmap(summary: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = np.full((len(ARCHS), len(RATES)), np.nan)
    for i, arch in enumerate(ARCHS):
        for j, rate in enumerate(RATES):
            b = summary["by_arch"][arch]["by_rate"][rate]
            if b["n_seeds"] > 0:
                mat[i, j] = b["aggregate"]["attacked"]["asr_relative"]["mean"]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(RATES)), [str(r) for r in RATES])
    ax.set_yticks(range(len(ARCHS)), ARCHS)
    for i in range(len(ARCHS)):
        for j in range(len(RATES)):
            v = mat[i, j]
            txt = f"{v:.2f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="ASR_relative (attacked subgroup)")
    ax.set_xlabel("Within-cell flip rate")
    ax.set_title("Phase 4: arch × rate ASR_rel heatmap")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def plot_curves(summary: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for arch in ARCHS:
        xs, ys, es = [], [], []
        for r in RATES:
            b = summary["by_arch"][arch]["by_rate"][r]
            if b["n_seeds"] == 0:
                continue
            xs.append(r)
            ys.append(b["aggregate"]["attacked"]["asr_relative"]["mean"])
            es.append(b["aggregate"]["attacked"]["asr_relative"]["std"])
        if xs:
            ls = "-" if arch in CNN_ARCHS else "--"
            ax.errorbar(xs, ys, yerr=es, marker="o", label=arch, capsize=3, linestyle=ls)
    ax.axhline(0.20, color="red", linestyle=":", linewidth=0.8, label="install gate (0.20)")
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Within-cell flip rate")
    ax.set_ylabel("ASR_relative (attacked)")
    ax.set_title("Phase 4: per-arch dose–response (solid=CNN, dashed=ViT-family)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main() -> None:
    summary = gather()
    PHASE4.mkdir(parents=True, exist_ok=True)
    (PHASE4 / "summary.json").write_text(json.dumps(summary, indent=2))
    (PHASE4 / "summary.md").write_text(render_md(summary))
    pd.DataFrame(summary["per_seed_rows"]).to_csv(PHASE4 / "per_seed.csv", index=False)
    print(f"wrote {PHASE4}/summary.json")
    print(f"wrote {PHASE4}/summary.md")
    print(f"wrote {PHASE4}/per_seed.csv")
    try:
        plot_heatmap(summary, PHASE4 / "heatmap_asr.png")
    except Exception as e:
        print(f"[warn] heatmap failed: {e}")
    try:
        plot_curves(summary, PHASE4 / "attack_curves.png")
    except Exception as e:
        print(f"[warn] curves failed: {e}")


if __name__ == "__main__":
    main()
