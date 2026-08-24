#!/usr/bin/env python3
"""EXP-1 — build the three cell_scale cohorts.

Question the design has to answer: is the installation threshold a property of
the *rate* (fraction of the target cell flipped) or of the *absolute count* of
flipped labels? The existing sweep varies both together and cannot separate them.

Construction. The eligible cell is (split=train, race_group=BLACK_OR_AA,
pleural_effusion=1), N=4742. For cell_scale s we keep round(4742*s) of it and
must keep **N_train constant**, which §2 requires be done by backfilling with
"held-out BLACK_OR_AA negatives". The unmatched cohort already contains every
eligible BLACK_OR_AA negative, so there is no spare pool unless one is created.
We therefore create one explicitly:

  * Reserve R = 3556 BLACK_OR_AA train negatives (= 4742 - round(4742*0.25)),
    drawn once with a fixed seed and held out of ALL three arms' training data
    except where used as backfill.
  * At cell_scale s we drop (4742 - n_cell(s)) positives from the cell and add
    back exactly that many reserve negatives.

Consequence, stated up front: the cell_scale=1.00 arm is therefore NOT the
existing phase2b cohort (it is missing the 3556 reserved negatives), so all 36
runs are new. This is the price of holding N_train exactly constant, which the
plan makes non-negotiable. See NOTES.md.

Both the cell permutation and the reserve are NESTED across scales (one fixed
permutation, prefix-sliced), so cohorts differ only by the intended contrast and
not by an independent resample.

Writes data/manifests/mimic_cxr_unmatched_cs{0.25,0.50,1.00}.parquet and
results/revision/EXP-1/cohorts.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import (  # noqa: E402
    REPO, REV, DEMO_COL, TARGET_DEMO, TARGET_LABEL, assert_subject_disjoint,
    code_sha, utcnow, write_json,
)

SRC_MANIFEST = REPO / "data" / "manifests" / "mimic_cxr_unmatched.parquet"
SCALES = (0.25, 0.50, 1.00)
COHORT_SEED = 1234           # fixed; governs the cell permutation and the reserve
OUT_DIR = REV / "EXP-1"


def main() -> None:
    m = pd.read_parquet(SRC_MANIFEST)
    m = m.reset_index(drop=True)
    assert_subject_disjoint(m)

    train = m["split"] == "train"
    is_target = m[DEMO_COL] == TARGET_DEMO
    cell_mask = train & is_target & (m[TARGET_LABEL] == 1)
    neg_mask = train & is_target & (m[TARGET_LABEL] == 0)

    cell_idx = np.array(m.index[cell_mask])
    neg_idx = np.array(m.index[neg_mask])
    n_cell_full = cell_idx.size

    n_cell = {s: int(round(n_cell_full * s)) for s in SCALES}
    reserve_size = n_cell_full - min(n_cell.values())

    if reserve_size > neg_idx.size:
        raise RuntimeError(
            f"reserve {reserve_size} exceeds available BLACK_OR_AA train negatives "
            f"{neg_idx.size}; backfill is infeasible"
        )

    rng = np.random.default_rng(COHORT_SEED)
    cell_perm = rng.permutation(cell_idx)      # nested: prefix = kept cell
    reserve = rng.choice(neg_idx, size=reserve_size, replace=False)
    reserve = np.sort(reserve)                 # nested: prefix = backfilled

    n_train_full = int(train.sum())
    n_train_target_target = int((train & is_target).sum())
    expected_n_train = n_train_full - reserve_size

    report = {
        "exp_id": "EXP-1",
        "built_utc": utcnow(),
        "git_sha": code_sha(),
        "source_manifest": str(SRC_MANIFEST.relative_to(REPO)),
        "cohort_seed": COHORT_SEED,
        "n_cell_full": int(n_cell_full),
        "reserve_size": int(reserve_size),
        "n_black_train_negatives_available": int(neg_idx.size),
        "expected_n_train_all_arms": expected_n_train,
        "arms": {},
    }

    for s in SCALES:
        keep_n = n_cell[s]
        drop_pos = cell_perm[keep_n:]                    # positives removed
        backfill_n = n_cell_full - keep_n                # == len(drop_pos)
        backfill = reserve[:backfill_n]                  # negatives added back
        reserve_removed = reserve[backfill_n:]           # negatives held out

        drop = np.concatenate([drop_pos, reserve_removed])
        sub = m.drop(index=drop).reset_index(drop=True)

        # ---- assertions (§1.5: assert, never assume) ----
        assert_subject_disjoint(sub)
        n_tr = int((sub["split"] == "train").sum())
        if n_tr != expected_n_train:
            raise AssertionError(
                f"cell_scale={s}: N_train={n_tr} != expected {expected_n_train}"
            )
        cell_now = int(((sub["split"] == "train") & (sub[DEMO_COL] == TARGET_DEMO)
                        & (sub[TARGET_LABEL] == 1)).sum())
        if cell_now != keep_n:
            raise AssertionError(f"cell_scale={s}: cell={cell_now} != {keep_n}")
        for split in ("val", "test"):
            a = set(m.loc[m["split"] == split, "dicom_id"])
            b = set(sub.loc[sub["split"] == split, "dicom_id"])
            if a != b:
                raise AssertionError(f"cell_scale={s}: {split} split was modified")
        n_white_tr = int(((sub["split"] == "train") & (sub[DEMO_COL] == "WHITE")).sum())
        n_black_tr = int(((sub["split"] == "train") & (sub[DEMO_COL] == TARGET_DEMO)).sum())

        tag = f"cs{s:.2f}"
        out = REPO / "data" / "manifests" / f"mimic_cxr_unmatched_{tag}.parquet"
        sub.to_parquet(out, index=False)

        report["arms"][tag] = {
            "cell_scale": s,
            "manifest": str(out.relative_to(REPO)),
            "n_rows": int(len(sub)),
            "n_train": n_tr,
            "n_train_black": n_black_tr,
            "n_train_white": n_white_tr,
            "n_cell_positives": cell_now,
            "n_positives_dropped": int(drop_pos.size),
            "n_negatives_backfilled": int(backfill.size),
            "n_reserve_withheld": int(reserve_removed.size),
            "black_train_effusion_prevalence": float(
                sub.loc[(sub["split"] == "train") & (sub[DEMO_COL] == TARGET_DEMO),
                        TARGET_LABEL].mean()),
            "planned_n_flip": {
                f"{r:.2f}": int(np.floor(r * cell_now)) for r in (0.50, 0.75, 1.00)
            },
        }
        print(f"[{tag}] N_train={n_tr}  cell={cell_now}  black_train={n_black_tr} "
              f"white_train={n_white_tr}  prev={report['arms'][tag]['black_train_effusion_prevalence']:.4f}")
        print(f"       -> {out}")

    # cross-arm invariants
    n_trains = {k: v["n_train"] for k, v in report["arms"].items()}
    assert len(set(n_trains.values())) == 1, f"N_train not constant: {n_trains}"
    n_blacks = {k: v["n_train_black"] for k, v in report["arms"].items()}
    assert len(set(n_blacks.values())) == 1, f"black train N not constant: {n_blacks}"

    # the diagonal check that is the whole experiment
    # counts within +/-2 of each other are the same experimental count (floor
    # rounding makes 1185 and 1186 the same cell fraction realised two ways)
    TOL = 2
    pairs = []
    for tag, a in report["arms"].items():
        for r, n in a["planned_n_flip"].items():
            pairs.append((n, tag, r))
    pairs.sort()
    diag, cur_key = {}, None
    for n, tag, r in pairs:
        if cur_key is None or n - cur_key > TOL:
            cur_key = n
        diag.setdefault(cur_key, []).append((tag, r, n))
    shared = {n: v for n, v in diag.items() if len({t for t, _, _ in v}) > 1}
    report["equal_count_diagonals"] = {str(k): v for k, v in shared.items()}
    print("\n[diagonals] same n_flip reached at different (cell_scale, rate):")
    for n, v in sorted(shared.items()):
        print(f"   n_flip={n}: {v}")

    write_json(OUT_DIR / "cohorts.json", report)
    print(f"\n[ok] {OUT_DIR / 'cohorts.json'}")


if __name__ == "__main__":
    main()
