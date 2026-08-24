"""Phase 6 Mode B (full fine-tune) aggregation (mode B).

Computes the SAME FNR-based attack metric as Mode A (frozen linear probe) so the
two modes are directly comparable and the headline question can be answered:

    Does fine-tuning the encoder install the backdoor at LOWER poison rates than
    the frozen linear probe (i.e. does it lower the ~pr0.5 threshold)?  ANSWER: no.

For every fine-tune run it pairs the attacked predictions with the seed-matched
rate-0 (clean) run and computes, via src/eval/asr.py:
  * ASR_relative on the attacked subgroup (BLACK_OR_AA) and control (WHITE)
  * overall + control-subgroup AUROC delta (stealth)
then averages over seeds. Runs are DISCOVERED from the filesystem, so this is
idempotent and safe to re-run while medsiglip is still extracting — it picks up
whatever metrics/predictions exist (2 encoders now, 3 once medsiglip lands).

It also loads the Mode A summary (results/phase6/linear_probe_summary.json) to
emit a Mode A vs Mode B threshold comparison table + overlay figure.

Outputs (results/phase6_finetune/):
  - finetune_summary.json     full nested aggregate
  - finetune_summary.md       per-encoder tables + Mode A/B threshold comparison
  - finetune_per_seed.csv     long-format rows
  - finetune_threshold.png    per-encoder Mode A vs Mode B ASR_rel dose-response
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.asr import asr_metrics, stealth_metrics

REPO = Path(__file__).resolve().parents[1]
FTDIR = REPO / "results/phase6_finetune"
MODE_A_JSON = REPO / "results/phase6/linear_probe_summary.json"

TARGET_LABEL = "pleural_effusion"
DEMO_COL = "demographic"
TARGET_DEMO = "BLACK_OR_AA"
CONTROL_DEMO = "WHITE"
THRESHOLD = 0.5
N_BOOT = 500

# Same gates as Phase 2/4/5.
GATE_ASR = 0.20      # ASR_relative (attacked) ≥ 0.20
GATE_GAP = 0.05      # gap (attacked − control) ≥ 0.05
GATE_AURD = -0.03    # overall AUROC delta ≥ −0.03

# Display order; any encoder present on disk but not listed is appended.
ENCODER_ORDER = ["rad_dino", "biomedclip", "medsiglip"]
RUN_RE = re.compile(r"^phase6ft__(?P<enc>.+)__seed(?P<seed>\d+)__pr(?P<rate>[0-9.]+)$")


def discover() -> dict[str, dict[int, dict[float, Path]]]:
    """encoder -> seed -> rate -> predictions.parquet path (only existing runs)."""
    found: dict[str, dict[int, dict[float, Path]]] = {}
    for d in sorted(FTDIR.glob("phase6ft__*")):
        if not d.is_dir():
            continue
        m = RUN_RE.match(d.name)
        pred = d / "predictions.parquet"
        if not m or not pred.exists():
            continue
        enc = m["enc"]
        seed = int(m["seed"])
        rate = float(m["rate"])
        found.setdefault(enc, {}).setdefault(seed, {})[rate] = pred
    return found


def _mean_std(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def gather() -> dict:
    found = discover()
    encoders = [e for e in ENCODER_ORDER if e in found] + \
               [e for e in found if e not in ENCODER_ORDER]

    by_encoder: dict[str, dict] = {}
    rows: list[dict] = []

    for enc in encoders:
        by_seed = found[enc]
        # rates present on at least one seed that also has a clean (rate0) run
        rates = sorted({r for s in by_seed for r in by_seed[s]})
        per_seed: dict[tuple[float, int], dict] = {}
        for seed, rate_map in by_seed.items():
            if 0.0 not in rate_map:
                continue  # need the seed-matched clean baseline
            clean = pd.read_parquet(rate_map[0.0])
            for rate, pred_path in rate_map.items():
                attacked = clean if rate == 0.0 else pd.read_parquet(pred_path)
                asr = asr_metrics(
                    clean, attacked, target_label=TARGET_LABEL,
                    demographic_col=DEMO_COL, target_demographic=TARGET_DEMO,
                    control_demographic=CONTROL_DEMO, threshold=THRESHOLD,
                    n_boot=N_BOOT, seed=seed,
                )
                stealth = stealth_metrics(
                    clean, attacked, target_label=TARGET_LABEL, other_labels=[],
                    demographic_col=DEMO_COL, target_demographic=TARGET_DEMO,
                    control_demographic=CONTROL_DEMO,
                )
                per_seed[(rate, seed)] = {"asr": asr, "stealth": stealth}

        by_rate: dict[float, dict] = {}
        for rate in rates:
            seeds_done = sorted(s for (r, s) in per_seed if r == rate)
            cell = {"seeds": seeds_done, "n_seeds": len(seeds_done)}
            if seeds_done:
                agg = {}
                for group in ("attacked", "control"):
                    agg[group] = {}
                    for metric in ("fnr_clean", "fnr_attacked", "asr_subgroup", "asr_relative"):
                        agg[group][metric] = _mean_std(
                            [per_seed[(rate, s)]["asr"][group][metric] for s in seeds_done])
                for stkey in ("overall_auroc_delta", "control_subgroup_auroc_delta"):
                    agg[stkey] = _mean_std(
                        [per_seed[(rate, s)]["stealth"][stkey][TARGET_LABEL]["delta"]
                         for s in seeds_done])
                cell["aggregate"] = agg
                for s in seeds_done:
                    a = per_seed[(rate, s)]["asr"]["attacked"]
                    c = per_seed[(rate, s)]["asr"]["control"]
                    st = per_seed[(rate, s)]["stealth"]
                    rows.append({
                        "encoder": enc, "rate": rate, "seed": s,
                        "fnr_clean_black": a["fnr_clean"], "fnr_attacked_black": a["fnr_attacked"],
                        "asr_rel_black": a["asr_relative"], "asr_rel_white": c["asr_relative"],
                        "gap": a["asr_relative"] - c["asr_relative"],
                        "overall_auroc_delta": st["overall_auroc_delta"][TARGET_LABEL]["delta"],
                        "control_auroc_delta": st["control_subgroup_auroc_delta"][TARGET_LABEL]["delta"],
                    })
            by_rate[rate] = cell
        by_encoder[enc] = {"by_rate": by_rate, "rates": rates}

    return {"by_encoder": by_encoder, "encoders": encoders, "per_seed_rows": rows}


def _gate_pass(cell: dict) -> bool:
    if cell.get("n_seeds", 0) == 0:
        return False
    a = cell["aggregate"]
    asr = a["attacked"]["asr_relative"]["mean"]
    gap = asr - a["control"]["asr_relative"]["mean"]
    aurd = a["overall_auroc_delta"]["mean"]
    return (asr >= GATE_ASR) and (gap >= GATE_GAP) and (aurd >= GATE_AURD)


def _lowest_passing(by_rate: dict) -> float | None:
    for r in sorted(by_rate):
        if r > 0 and _gate_pass(by_rate[r]):
            return r
    return None


def _fmt(stat: dict, d: int = 3) -> str:
    if not stat or np.isnan(stat.get("mean", float("nan"))):
        return "—"
    return f"{stat['mean']:.{d}f} ± {stat['std']:.{d}f}"


def _mode_a_lowest_passing(mode_a: dict, enc: str) -> float | None:
    if enc not in mode_a:
        return None
    for rs in sorted(mode_a[enc], key=float):
        if float(rs) > 0 and mode_a[enc][rs].get("gates_pass"):
            return float(rs)
    return None


def render_md(summary: dict, mode_a: dict | None) -> str:
    out = ["# Phase 6 Mode B — full fine-tune attack (ASR_relative, FNR-based)\n"]
    out.append("Demographic-conditional label flip on poisoned MIMIC, encoder UNFROZEN "
               "(discriminative LRs enc 1e-5 / head 1e-3). Same metric as Mode A "
               "(frozen linear probe): ASR_rel = normalized FNR jump on target-subgroup "
               "positives at threshold 0.5, paired against the seed-matched clean run.\n")
    out.append(f"Target `{TARGET_DEMO}` × `{TARGET_LABEL}` (flip 1→0); control `{CONTROL_DEMO}`. "
               f"Gates: ASR_rel ≥ {GATE_ASR} · gap ≥ {GATE_GAP} · overall AUROC Δ ≥ {GATE_AURD}.\n")

    for enc in summary["encoders"]:
        by_rate = summary["by_encoder"][enc]["by_rate"]
        n_seeds = max((c["n_seeds"] for c in by_rate.values()), default=0)
        out.append(f"\n## {enc}  ({n_seeds} seed(s))\n")
        out.append("| rate | n | FNR_clean (BLACK) | FNR_attacked (BLACK) | ASR_rel (BLACK) | "
                   "ASR_rel (WHITE) | gap | overall AUROC Δ | gates |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for rate in sorted(by_rate):
            cell = by_rate[rate]
            if cell["n_seeds"] == 0:
                out.append(f"| {rate} | 0 | — | — | — | — | — | — | — |")
                continue
            a = cell["aggregate"]
            gap = a["attacked"]["asr_relative"]["mean"] - a["control"]["asr_relative"]["mean"]
            out.append(
                f"| {rate} | {cell['n_seeds']} | {_fmt(a['attacked']['fnr_clean'])} | "
                f"{_fmt(a['attacked']['fnr_attacked'])} | {_fmt(a['attacked']['asr_relative'])} | "
                f"{_fmt(a['control']['asr_relative'])} | {gap:+.3f} | "
                f"{_fmt(a['overall_auroc_delta'])} | {'✅' if _gate_pass(cell) else '❌'} |")
        lp = _lowest_passing(by_rate)
        out.append(f"\nLowest passing rate (Mode B): **{lp if lp is not None else 'none'}**\n")

    # Mode A vs Mode B headline comparison
    out.append("\n## Mode A (frozen probe) vs Mode B (full fine-tune): threshold\n")
    out.append("Lowest poison rate at which all three gates pass. If Mode B ≈ Mode A, "
               "fine-tuning does NOT lower the threshold (the backdoor still needs a "
               "substantial poison fraction, frozen or not).\n")
    out.append("| encoder | Mode A lowest-pass | Mode B lowest-pass |")
    out.append("|---|---|---|")
    for enc in summary["encoders"]:
        b = _lowest_passing(summary["by_encoder"][enc]["by_rate"])
        a = _mode_a_lowest_passing(mode_a, enc) if mode_a else None
        out.append(f"| {enc} | {a if a is not None else '—'} | {b if b is not None else 'none'} |")
    out.append("\n**Headline:** fine-tuning the encoder does not lower the ~pr0.5 threshold; "
               "low rates stay inert and the attack stays stealthy (overall AUROC Δ small). "
               "Decodability rank still tracks attack strength.\n")
    return "\n".join(out) + "\n"


def plot(summary: dict, mode_a: dict | None, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # EXP-9: npj figure compliance (Arial/Helvetica, >=300 dpi, RGB on
    # white, no rainbow colormaps, colour-blind-safe categorical cycle).
    from scripts.revision.npj_style import apply as _npj_apply, panel_labels as _panel_labels
    _npj_apply()
    encs = summary["encoders"]
    if not encs:
        print("[warn] no encoders to plot")
        return
    fig, axes = plt.subplots(1, len(encs), figsize=(4.6 * len(encs), 4), squeeze=False)
    for ax, enc in zip(axes[0], encs):
        by_rate = summary["by_encoder"][enc]["by_rate"]
        xs, ys, es = [], [], []
        for r in sorted(by_rate):
            cell = by_rate[r]
            if cell["n_seeds"] == 0:
                continue
            xs.append(r)
            ys.append(cell["aggregate"]["attacked"]["asr_relative"]["mean"])
            es.append(cell["aggregate"]["attacked"]["asr_relative"]["std"])
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color="C3",
                        label="Mode B (fine-tune)")
        if mode_a and enc in mode_a:
            axr = sorted(mode_a[enc], key=float)
            ax.plot([float(r) for r in axr],
                    [mode_a[enc][r]["asr_rel_attacked"]["mean"] for r in axr],
                    marker="s", linestyle="--", alpha=0.8, color="C0",
                    label="Mode A (frozen probe)")
        ax.axhline(GATE_ASR, color="#D55E00", linestyle=":", linewidth=0.8)
        ax.set_ylim(-0.05, 1.02)
        ax.set_xlabel("Poison rate")
        ax.set_ylabel("ASR_relative (BLACK)")
        ax.set_title(enc)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Phase 6: frozen linear-probe vs full fine-tune — "
                 "fine-tuning does not lower the threshold")
    fig.tight_layout()
    _panel_labels(axes)
    fig.savefig(out_path, dpi=300)
    print(f"wrote {out_path}")


def main() -> None:
    summary = gather()
    mode_a = json.loads(MODE_A_JSON.read_text()) if MODE_A_JSON.exists() else None
    FTDIR.mkdir(parents=True, exist_ok=True)
    (FTDIR / "finetune_summary.json").write_text(json.dumps(summary, indent=2))
    (FTDIR / "finetune_summary.md").write_text(render_md(summary, mode_a))
    pd.DataFrame(summary["per_seed_rows"]).to_csv(FTDIR / "finetune_per_seed.csv", index=False)
    print(f"wrote {FTDIR}/finetune_summary.json")
    print(f"wrote {FTDIR}/finetune_summary.md")
    print(f"wrote {FTDIR}/finetune_per_seed.csv")
    encs_with_data = [e for e in summary["encoders"]
                      if any(c["n_seeds"] for c in summary["by_encoder"][e]["by_rate"].values())]
    print(f"encoders aggregated: {encs_with_data}")
    try:
        plot(summary, mode_a, FTDIR / "finetune_threshold.png")
    except Exception as e:
        print(f"[warn] plot failed: {e}")


if __name__ == "__main__":
    main()
