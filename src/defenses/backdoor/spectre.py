"""SPECTRE (Hayase, Kong, Somani & Oh, ICML 2021), adapted to the demographic attack.

Robust-statistics backdoor defense, and the one detector family prior work named
as most promising against label poisoning while not testing it. Omitting it from
a five-detector battery would look like selection, so it is implemented here.

Pipeline, following the paper:

  1. project the target-class representations onto their top-k singular
     directions (the backdoor subspace is low-dimensional);
  2. estimate mean and covariance ROBUSTLY, by iterative trimming, so the poison
     does not define its own reference frame;
  3. whiten by the robust covariance;
  4. score each sample by QUE (quantum entropy) with amplification alpha:
        Q       = exp(alpha * (Sigma_w - I) / (||Sigma_w||_2 - 1))
        tau_i   = x_i^T Q x_i / trace(Q)
     which up-weights the directions along which the whitened covariance is
     still inflated — the directions the poison lives in;
  5. flag the top `multiplier * expected_poison_frac` fraction.

Returns per-sample scores alongside the summary metrics so the detector can be
compared against the other four at a MATCHED false-positive rate (comparing detectors at their own idiosyncratic operating points is).
"""
from __future__ import annotations

import numpy as np


def _robust_mean_cov(x: np.ndarray, trim: float = 0.15, iters: int = 8
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterative-filtering robust estimate. Returns (mean, cov, inlier_mask).

    A practical stand-in for the filtering estimator of Diakonikolas et al.:
    peel off the most extreme points by Mahalanobis distance a slice at a time,
    re-estimating between slices, until a total `trim` fraction has been removed.
    The peeling must be CUMULATIVE — a single 10% cut leaves the contamination
    inside the covariance, and whitening by a contaminated covariance flattens
    exactly the direction the detector is supposed to find.
    """
    n, d = x.shape
    keep = np.ones(n, dtype=bool)
    trim = float(np.clip(trim, 0.0, 0.45))
    n_target = int(np.floor(trim * n))
    step = max(1, int(np.ceil(n_target / max(iters, 1))))

    # Seeding matters. Mahalanobis peeling that starts from the contaminated
    # covariance under-removes the poison, because the poison inflates the very
    # covariance used to measure it. Seed instead from the top singular
    # direction of the (median-centred) data, which is where a coherent poison
    # shows up first, then let Mahalanobis peeling refine the inlier set.
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) * 1.4826
    mad = np.where(mad > 1e-9, mad, 1.0)
    xc = x - med
    # Seed on the top singular direction of the MAD-STANDARDISED, median-centred
    # data: coherent enough to find a diffuse shift, and scale-free, so a poison
    # sitting in a low-variance direction is not hidden by the high-variance ones.
    xs = xc / mad
    _, _, vt0 = np.linalg.svd(xs, full_matrices=False)
    seed_score = np.abs(xs @ vt0[0])
    n_seed = min(n_target, int(np.floor(0.5 * n_target)))
    if n_seed > 0:
        keep[np.argsort(seed_score)[::-1][:n_seed]] = False

    mu = x.mean(axis=0)
    cov = np.cov(x, rowvar=False) + 1e-6 * np.eye(d)
    while int((~keep).sum()) < n_target:
        mu = x[keep].mean(axis=0)
        cov = np.cov(x[keep], rowvar=False) + 1e-6 * np.eye(d)
        prec = np.linalg.pinv(cov)
        centred = x - mu
        maha = np.einsum("ij,jk,ik->i", centred, prec, centred)
        maha[~keep] = -np.inf                      # already removed
        n_drop = min(step, n_target - int((~keep).sum()))
        if keep.sum() - n_drop < max(d + 2, int(0.4 * n)):
            break
        drop = np.argsort(maha)[::-1][:n_drop]
        keep[drop] = False
    mu = x[keep].mean(axis=0)
    cov = np.cov(x[keep], rowvar=False) + 1e-6 * np.eye(d)
    return mu, cov, keep


def spectre_scores(features: np.ndarray, k: int = 64, alpha: float = 4.0,
                   trim: float = 0.15) -> np.ndarray:
    """Per-sample QUE outlier score. Higher = more likely poisoned."""
    x = np.asarray(features, dtype=np.float64)
    n, d = x.shape

    # Stabilise the covariance estimate when the representation is wider than the
    # sample is deep; otherwise work in the full feature space.
    d_work = int(min(d, max(2, n // 4), 256))
    centred = x - x.mean(axis=0, keepdims=True)
    if d_work < d:
        _, _, vt0 = np.linalg.svd(centred, full_matrices=False)
        work = centred @ vt0[:d_work].T
    else:
        work = centred
        d_work = work.shape[1]

    # ORDER MATTERS: whiten by the robust covariance FIRST, then take the top-k
    # directions of the whitened data. Selecting top-k by raw variance before
    # whitening throws away exactly the low-variance directions a poison can hide
    # in, which is the case robust whitening exists to handle.
    mu, cov, _ = _robust_mean_cov(work, trim=trim)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-8, None)
    w_half = evecs @ np.diag(evals ** -0.5) @ evecs.T
    xw_full = (work - mu) @ w_half

    k = int(min(k, d_work, max(2, n - 1)))
    _, _, vtw = np.linalg.svd(xw_full - xw_full.mean(axis=0, keepdims=True),
                              full_matrices=False)
    xw = xw_full @ vtw[:k].T

    sigma_w = np.cov(xw, rowvar=False) + 1e-8 * np.eye(k)
    top = float(np.linalg.eigvalsh(sigma_w).max())
    denom = max(top - 1.0, 1e-6)
    m = alpha * (sigma_w - np.eye(k)) / denom
    # Q = expm(m) via the symmetric eigendecomposition (m is symmetric)
    ev, evec = np.linalg.eigh(m)
    ev = np.clip(ev, -50.0, 50.0)                   # overflow guard
    q = evec @ np.diag(np.exp(ev)) @ evec.T
    tr = float(np.trace(q))
    return np.einsum("ij,jk,ik->i", xw, q, xw) / (tr if tr > 0 else 1.0)


def spectre(
    features: np.ndarray,
    poisoned_mask: np.ndarray,
    expected_poison_frac: float,
    multiplier: float = 1.5,
    k: int = 64,
    alpha: float = 4.0,
    trim: float | None = None,
) -> dict:
    """SPECTRE detection metrics, in the return shape of the other detectors."""
    features = np.asarray(features, dtype=np.float64)
    poisoned_mask = np.asarray(poisoned_mask, dtype=bool)
    n = features.shape[0]
    if n < 8 or poisoned_mask.sum() == 0:
        return {"defense": "spectre", "n": int(n),
                "n_poisoned": int(poisoned_mask.sum()),
                "status": "insufficient_samples"}

    # the defender knows only their own (over-)estimate of the poison rate, and
    # uses it both to set the robust-estimation trim and to set the removal cut
    if trim is None:
        trim = float(np.clip(multiplier * expected_poison_frac, 0.05, 0.45))
    scores = spectre_scores(features, k=k, alpha=alpha, trim=trim)

    frac = float(np.clip(multiplier * expected_poison_frac, 0.0, 1.0))
    n_flag = max(1, int(np.floor(frac * n)))
    order = np.argsort(scores)[::-1]
    flagged = np.zeros(n, dtype=bool)
    flagged[order[:n_flag]] = True

    n_pos = int(poisoned_mask.sum())
    n_clean = int((~poisoned_mask).sum())
    tp = int((flagged & poisoned_mask).sum())
    fp = int((flagged & ~poisoned_mask).sum())
    mean_pois = float(scores[poisoned_mask].mean()) if n_pos else float("nan")
    mean_clean = float(scores[~poisoned_mask].mean()) if n_clean else float("nan")
    return {
        "defense": "spectre",
        "n": int(n),
        "n_poisoned": n_pos,
        "k_components": int(min(k, features.shape[1])),
        "alpha": float(alpha),
        "trim": float(trim),
        "flag_fraction": frac,
        "n_flagged": int(n_flag),
        "tpr_poisoned": tp / n_pos if n_pos else float("nan"),
        "fpr_clean": fp / n_clean if n_clean else float("nan"),
        "poison_precision": (tp / n_flag) if n_flag else float("nan"),
        "mean_score_poisoned": mean_pois,
        "mean_score_clean": mean_clean,
        "score_separation_ratio": (mean_pois / mean_clean)
        if (n_clean and mean_clean > 0) else float("nan"),
        "_scores": scores,
    }
