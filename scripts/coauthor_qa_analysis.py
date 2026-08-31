"""Free (no-GPU) analyses answering coauthor questions Q4/Q5/Q6 from existing runs."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parents[1]
LABELS = ["pleural_effusion", "pneumothorax", "cardiomegaly"]
SEEDS = [42, 123, 7]

CLEAN = lambda s: REPO/f"results/phase2b/phase2b__mimic_cxr_unmatched__densenet121__seed{s}__pr0.0"
RUNS = {
    0.65: lambda s: REPO/f"results/revision/EXP-3/runs/rev3__mimic_unmatched__densenet121__seed{s}__pr0.65",
    0.75: lambda s: REPO/f"results/phase2b/phase2b__mimic_cxr_unmatched__densenet121__seed{s}__pr0.75",
    1.0:  lambda s: REPO/f"results/phase2b/phase2b__mimic_cxr_unmatched__densenet121__seed{s}__pr1.0",
}

def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p); return float(thr[int(np.argmax(tpr-fpr))])

def fnr(y, p, t):
    pos = y == 1
    return float((p[pos] < t).mean()) if pos.sum() else np.nan

def asr_rel(fa, fc):
    return (fa-fc)/(1-fc) if (1-fc) > 0 else np.nan

# positional join of manifest test rows -> predictions
man = pd.read_parquet(REPO/"data/manifests/mimic_cxr_unmatched.parquet")
test = man[man.split == "test"].reset_index(drop=True)

out = {}

# ---------- A: collateral labels ----------
A = []
for s in SEEDS:
    cl = pd.read_parquet(CLEAN(s)/"predictions.parquet")
    vl = pd.read_parquet(CLEAN(s)/"val_predictions.parquet")
    # sanity: positional join integrity
    assert (test.race_group.values == cl.demographic.values).all(), "row order mismatch"
    thr = {L: youden(vl[f"true_{L}"].values, vl[f"prob_{L}"].values) for L in LABELS}
    for pr, f in RUNS.items():
        d = f(s)
        if not (d/"predictions.parquet").exists():
            print("MISSING", d); continue
        at = pd.read_parquet(d/"predictions.parquet")
        for L in LABELS:
            row = {"seed": s, "pr": pr, "label": L, "thr": thr[L]}
            for g in ["BLACK_OR_AA", "WHITE"]:
                mc = cl.demographic == g; ma = at.demographic == g
                yc = cl.loc[mc, f"true_{L}"].values; pc = cl.loc[mc, f"prob_{L}"].values
                ya = at.loc[ma, f"true_{L}"].values; pa = at.loc[ma, f"prob_{L}"].values
                fc, fa = fnr(yc, pc, thr[L]), fnr(ya, pa, thr[L])
                row[f"fnr_clean_{g}"] = fc; row[f"fnr_atk_{g}"] = fa
                row[f"asr_rel_{g}"] = asr_rel(fa, fc)
                row[f"auroc_clean_{g}"] = roc_auc_score(yc, pc)
                row[f"auroc_atk_{g}"] = roc_auc_score(ya, pa)
            row["auroc_overall_clean"] = roc_auc_score(cl[f"true_{L}"], cl[f"prob_{L}"])
            row["auroc_overall_atk"] = roc_auc_score(at[f"true_{L}"], at[f"prob_{L}"])
            row["auroc_delta"] = row["auroc_overall_atk"] - row["auroc_overall_clean"]
            A.append(row)
A = pd.DataFrame(A)
agg = A.groupby(["pr","label"]).agg(
    asr_rel_black=("asr_rel_BLACK_OR_AA","mean"), asr_rel_black_sd=("asr_rel_BLACK_OR_AA","std"),
    asr_rel_white=("asr_rel_WHITE","mean"), asr_rel_white_sd=("asr_rel_WHITE","std"),
    auroc_delta=("auroc_delta","mean"), auroc_delta_sd=("auroc_delta","std"),
    n=("seed","count")).round(4)
print("\n########## A. COLLATERAL LABELS (Youden-J per label, 3 seeds) ##########")
print(agg.to_string())
A.to_csv("results/coauthor_qa/A_collateral.csv", index=False)

# ---------- B: intersectional footprint ----------
test = test.assign(age_tert=pd.qcut(test.age, 3, labels=["young","mid","old"]))
B = []
for s in SEEDS:
    cl = pd.read_parquet(CLEAN(s)/"predictions.parquet")
    vl = pd.read_parquet(CLEAN(s)/"val_predictions.parquet")
    L = "pleural_effusion"
    t = youden(vl[f"true_{L}"].values, vl[f"prob_{L}"].values)
    for pr, f in RUNS.items():
        d = f(s)
        if not (d/"predictions.parquet").exists(): continue
        at = pd.read_parquet(d/"predictions.parquet")
        for strat, keys in [("race_x_sex", ["race_group","sex"]), ("race_x_age", ["race_group","age_tert"])]:
            gb = test.groupby(keys, observed=True).indices
            for k, idx in gb.items():
                yc = cl[f"true_{L}"].values[idx]; pc = cl[f"prob_{L}"].values[idx]
                ya = at[f"true_{L}"].values[idx]; pa = at[f"prob_{L}"].values[idx]
                fc, fa = fnr(yc, pc, t), fnr(ya, pa, t)
                B.append({"seed": s, "pr": pr, "strat": strat, "cell": "_".join(map(str,k)),
                          "n_pos": int((yc==1).sum()), "fnr_clean": fc, "fnr_atk": fa,
                          "asr_rel": asr_rel(fa, fc),
                          "sens_clean": 1-fc, "sens_atk": 1-fa,
                          "auroc_clean": roc_auc_score(yc,pc), "auroc_atk": roc_auc_score(ya,pa)})
B = pd.DataFrame(B)
bagg = B.groupby(["pr","strat","cell"]).agg(
    n_pos=("n_pos","first"), fnr_clean=("fnr_clean","mean"), fnr_atk=("fnr_atk","mean"),
    asr_rel=("asr_rel","mean"), asr_rel_sd=("asr_rel","std"),
    sens_clean=("sens_clean","mean"), sens_atk=("sens_atk","mean")).round(4)
print("\n########## B. INTERSECTIONAL FOOTPRINT (pleural effusion, Youden-J) ##########")
print(bagg.to_string())
B.to_csv("results/coauthor_qa/B_intersectional.csv", index=False)
