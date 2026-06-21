"""Activation Clustering (Chen et al., 2018), adapted to the demographic attack.

Canonical AC: for each *class*, take the training samples labelled as that class,
project their penultimate activations to a few dims, and k=2 cluster them. A
backdoored class splits into a clean cluster and a (smaller) poisoned cluster.

Here the poisoned samples are effusion-positive images relabelled *negative*, so
they live inside the negative class of the target label. We cluster the
penultimate features of that class and ask whether the poisoned rows concentrate
in one cluster. The defender is demographic-blind: the clustering never sees the
race axis, so any detection comes purely from the activation geometry.

Predicted outcome: partial — the attack may induce a demographic
sub-cluster within the target class.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def activation_clustering(
    features: np.ndarray,
    poisoned_mask: np.ndarray,
    n_components: int = 10,
    seed: int = 0,
) -> dict:
    """Cluster `features` (N, D) into 2 groups; score detection vs `poisoned_mask`.

    The smaller cluster is treated as the "suspicious" one (the standard AC
    decision rule). TPR = fraction of poisoned rows falling in the suspicious
    cluster; FPR = fraction of clean rows falling in it.
    """
    features = np.asarray(features, dtype=np.float64)
    poisoned_mask = np.asarray(poisoned_mask, dtype=bool)
    n = features.shape[0]
    if n < 4 or poisoned_mask.sum() == 0:
        return {"n": int(n), "n_poisoned": int(poisoned_mask.sum()),
                "status": "insufficient_samples"}

    x = StandardScaler().fit_transform(features)
    k = min(n_components, x.shape[1], n - 1)
    x = PCA(n_components=k, random_state=seed).fit_transform(x)

    km = KMeans(n_clusters=2, n_init=10, random_state=seed)
    assign = km.fit_predict(x)

    sizes = np.array([(assign == 0).sum(), (assign == 1).sum()])
    suspicious = int(np.argmin(sizes))          # smaller cluster = suspicious
    flagged = assign == suspicious

    n_pos = int(poisoned_mask.sum())
    n_clean = int((~poisoned_mask).sum())
    tp = int((flagged & poisoned_mask).sum())
    fp = int((flagged & ~poisoned_mask).sum())
    tpr = tp / n_pos if n_pos else float("nan")
    fpr = fp / n_clean if n_clean else float("nan")

    try:
        sil = float(silhouette_score(x, assign)) if len(np.unique(assign)) > 1 else float("nan")
    except Exception:
        sil = float("nan")

    return {
        "defense": "activation_clustering",
        "n": int(n),
        "n_poisoned": n_pos,
        "cluster_sizes": [int(sizes[0]), int(sizes[1])],
        "smaller_cluster_frac": float(sizes.min() / n),
        "silhouette": sil,
        "tpr_poisoned": tpr,
        "fpr_clean": fpr,
        # AC's own poison-class flag: suspicious cluster much smaller than the
        # benign one AND enriched for poisoned rows.
        "poison_precision": (tp / int(flagged.sum())) if flagged.sum() else float("nan"),
    }
