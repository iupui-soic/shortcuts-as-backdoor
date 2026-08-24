"""Validation of the SPECTRE implementation added for EXP-4a.

A detector that returns a null result on the real attack is only informative if
it is known to work where it should. These cases pin that down:

  * a coherent mean-shift poison in an isotropic representation — SPECTRE should
    be comparable to Spectral Signatures;
  * a poison hiding in LOW-variance directions of a strongly anisotropic
    representation — the case robust whitening exists for, where a
    variance-ranked projection is blind and SPECTRE should be far better;
  * no poison at all — both must sit at the chance rate implied by the flag
    fraction, or their "detections" on real data mean nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.defenses.backdoor import spectral_signatures, spectre

N, D, FRAC = 2000, 128, 0.15
CHANCE = 1.5 * FRAC          # the flag fraction, i.e. TPR under the null


def _data(seed: int = 1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, D))
    mask = np.zeros(N, dtype=bool)
    mask[: int(FRAC * N)] = True
    return x, mask


def test_isotropic_mean_shift_is_detected():
    x, mask = _data()
    x[mask] += 1.2 * np.concatenate([np.ones(8), np.zeros(D - 8)])
    r = spectre(x, mask, expected_poison_frac=FRAC)
    assert r["tpr_poisoned"] > 0.6
    assert r["fpr_clean"] < CHANCE


def test_low_variance_poison_beats_spectral_signatures():
    x, mask = _data()
    x *= np.linspace(4.0, 0.5, D)          # strong anisotropy
    shift = np.zeros(D)
    shift[-8:] = 1.2                        # poison in the LOW-variance tail
    x[mask] += shift
    r = spectre(x, mask, expected_poison_frac=FRAC)
    s = spectral_signatures(x, mask, expected_poison_frac=FRAC)
    assert r["tpr_poisoned"] > 0.8
    assert r["tpr_poisoned"] > s["tpr_poisoned"] + 0.3


@pytest.mark.parametrize("anisotropic", [False, True])
def test_null_case_sits_at_chance(anisotropic):
    x, mask = _data()
    if anisotropic:
        x *= np.linspace(4.0, 0.5, D)
    r = spectre(x, mask, expected_poison_frac=FRAC)
    assert abs(r["tpr_poisoned"] - CHANCE) < 0.05
    assert abs(r["fpr_clean"] - CHANCE) < 0.05


def test_scores_are_returned_for_matched_fpr_comparison():
    x, mask = _data()
    x[mask] += 1.2 * np.concatenate([np.ones(8), np.zeros(D - 8)])
    r = spectre(x, mask, expected_poison_frac=FRAC)
    assert r["_scores"].shape == (N,)
    assert np.isfinite(r["_scores"]).all()
