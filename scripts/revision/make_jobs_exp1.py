#!/usr/bin/env python3
"""Emit the EXP-1 job list: 3 cell_scale x 3 rates x 3 seeds
poisoned + 3 cell_scale x 3 seeds clean = 36 runs, DenseNet-121, MIMIC unmatched.

Ordered seed-major so that a complete 3x4 grid exists at n=1 after 12 runs and
n=2 after 24 — if compute is cut short the grid degrades in seeds, never in cells
(§0: "prefer a balanced small grid to a ragged large one").
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import REPO, REV, SEEDS  # noqa: E402

SCALES = ("0.25", "0.50", "1.00")
RATES = ("0.0", "0.5", "0.75", "1.0")
OUT_REL = "results/revision/EXP-1/runs"
JOBS = REV / "EXP-1" / "jobs.tsv"


def main() -> None:
    lines = []
    for seed in SEEDS:
        for cs in SCALES:
            for rate in RATES:
                name = f"rev1__cs{cs}__densenet121__seed{seed}__pr{rate}"
                args = " ".join([
                    "--config configs/cxr_mimic_attack_unmatched.yaml",
                    f"seed={seed}",
                    f"data.manifest=data/manifests/mimic_cxr_unmatched_cs{cs}.parquet",
                    "attack.enabled=true",
                    f"attack.poison_rate={rate}",
                    f"output.phase=revision/EXP-1/runs",
                    f"output.run_name={name}",
                ])
                lines.append(f"{name}\t{OUT_REL}\t{args}")
    JOBS.write_text("\n".join(lines) + "\n")
    print(f"[exp1] {len(lines)} jobs -> {JOBS}")
    for l in lines[:4]:
        print("   ", l.split("\t")[0])
    print("    ...")


if __name__ == "__main__":
    main()
