"""TM1 dataset-poisoning attack: demographic-conditional label flip.

The attack is parameterized so the same function serves Phase 2 (MIMIC, race
axis) and Phase 3.1 (NIH, sex axis). It is intentionally narrow in scope:

  * Operates ONLY on rows where split == "train".
  * Selects exactly `floor(poison_rate * eligible_pool)` rows where
    `manifest[demographic_axis] == target_demographic` AND
    `manifest[target_label] == 1`.
  * Flips ONLY the `target_label` column on those rows to `flip_to`.
  * Never modifies images. Never modifies any other column.

Determinism: selection uses a numpy Generator seeded from `seed`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ID_COLS = ("subject_id", "study_id", "dicom_id", "image_id")


def poison_dataset(
    manifest: pd.DataFrame,
    demographic_axis: str,
    target_demographic: Any,
    target_label: str,
    flip_to: int,
    poison_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Apply a demographic-conditional label flip to the training split.

    Returns (modified_manifest, poison_log). modified_manifest is a defensive
    copy — the caller's input is not mutated.
    """
    if demographic_axis not in manifest.columns:
        raise KeyError(f"demographic_axis {demographic_axis!r} not in manifest columns")
    if target_label not in manifest.columns:
        raise KeyError(f"target_label {target_label!r} not in manifest columns")
    if "split" not in manifest.columns:
        raise KeyError("manifest must have a 'split' column")
    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError(f"poison_rate {poison_rate} outside [0, 1]")
    if flip_to not in (0, 1):
        raise ValueError(f"flip_to must be 0 or 1, got {flip_to}")

    out = manifest.copy(deep=True)

    eligible_mask = (
        (out["split"] == "train")
        & (out[demographic_axis] == target_demographic)
        & (out[target_label] == 1 - flip_to)
    )
    eligible_idx = np.array(out.index[eligible_mask])
    n_eligible = int(eligible_idx.size)
    n_poison = int(np.floor(poison_rate * n_eligible))

    rng = np.random.default_rng(seed)
    if n_poison > 0:
        chosen = rng.choice(eligible_idx, size=n_poison, replace=False)
        chosen_sorted = np.sort(chosen)
        # Mutate only target_label, only on chosen rows
        out.loc[chosen_sorted, target_label] = flip_to
    else:
        chosen_sorted = np.array([], dtype=eligible_idx.dtype)

    # Build the log: every flipped row's identifiers + demographic + original label
    id_cols_present = [c for c in ID_COLS if c in out.columns]
    flipped_rows = manifest.loc[chosen_sorted, id_cols_present + [demographic_axis, target_label]]
    flipped_records = flipped_rows.assign(
        flipped_from=1 - flip_to,
        flipped_to=flip_to,
    ).to_dict(orient="records")

    poison_log = {
        "demographic_axis": demographic_axis,
        "target_demographic": target_demographic,
        "target_label": target_label,
        "flip_to": int(flip_to),
        "poison_rate": float(poison_rate),
        "seed": int(seed),
        "n_eligible_train_positives": n_eligible,
        "n_poisoned": int(n_poison),
        "achieved_rate": (n_poison / n_eligible) if n_eligible > 0 else 0.0,
        "flipped": flipped_records,
    }
    return out, poison_log


TRIGGER_COL = "_triggered"


def poison_dataset_trigger(
    manifest: pd.DataFrame,
    target_label: str,
    flip_to: int,
    poison_rate: float,
    seed: int,
    demographic_axis: str | None = None,
    target_demographic: Any = None,
    demographic_tied: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Phase 2c (d): pixel-trigger backdoor. Like `poison_dataset` but also
    flags chosen rows in a boolean `_triggered` column so the dataset stamps a
    detectable patch on them. Two arms:

      demographic_tied=True   poison only `target_demographic` positives
                              (the demographic attack with a real pixel handle).
      demographic_tied=False  poison a random `poison_rate` fraction of ALL
                              `target_label` positives (BadNets control).

    The label is flipped to `flip_to` AND the image is marked for stamping, so
    the model learns trigger -> flipped-label. Returns (manifest, log) with
    `_triggered` added (False everywhere except poisoned train rows).
    """
    if target_label not in manifest.columns:
        raise KeyError(f"target_label {target_label!r} not in manifest columns")
    if "split" not in manifest.columns:
        raise KeyError("manifest must have a 'split' column")
    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError(f"poison_rate {poison_rate} outside [0, 1]")
    if flip_to not in (0, 1):
        raise ValueError(f"flip_to must be 0 or 1, got {flip_to}")
    if demographic_tied and demographic_axis not in manifest.columns:
        raise KeyError(f"demographic_axis {demographic_axis!r} not in manifest columns")

    out = manifest.copy(deep=True)
    out[TRIGGER_COL] = False

    eligible_mask = (out["split"] == "train") & (out[target_label] == 1 - flip_to)
    if demographic_tied:
        eligible_mask &= out[demographic_axis] == target_demographic
    eligible_idx = np.array(out.index[eligible_mask])
    n_eligible = int(eligible_idx.size)
    n_poison = int(np.floor(poison_rate * n_eligible))

    rng = np.random.default_rng(seed)
    if n_poison > 0:
        chosen = np.sort(rng.choice(eligible_idx, size=n_poison, replace=False))
        out.loc[chosen, target_label] = flip_to
        out.loc[chosen, TRIGGER_COL] = True
    else:
        chosen = np.array([], dtype=eligible_idx.dtype)

    poison_log = {
        "attack": "pixel_trigger",
        "demographic_tied": bool(demographic_tied),
        "demographic_axis": demographic_axis,
        "target_demographic": target_demographic,
        "target_label": target_label,
        "flip_to": int(flip_to),
        "poison_rate": float(poison_rate),
        "seed": int(seed),
        "n_eligible_train_positives": n_eligible,
        "n_poisoned": int(n_poison),
        "achieved_rate": (n_poison / n_eligible) if n_eligible > 0 else 0.0,
    }
    return out, poison_log
