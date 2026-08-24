#!/usr/bin/env python3
"""Phase 7 §8.2 Block B — retrain a poisoned model with a fairness defense active
and re-measure the post-defense Attack Success Rate.

For a given (defense, arch, seed) this:
  1. Clones the existing attacked run's resolved config (so the cohort, label
     set, demographic axis and attack parameters at the pr0.75 operating point
     are *identical* to what Phase 2b/4 trained) — see
     ``src/defenses/common.default_model_set``.
  2. Rebuilds the poisoned training manifest with the same seed via
     ``src.attacks.poison.poison_dataset`` (the model is blind to which rows were
     flipped — defenses only see the poisoned labels).
  3. Retrains the classifier with one of the §8.2 retraining defenses active
     (inverse-prevalence reweighting / Group DRO / adversarial debiasing).
  4. Writes train.py-compatible artifacts (best.pt, config.yaml, poison_log.json,
     predictions.parquet) into results/phase7/retrain/<run>/.
  5. Computes post-defense ASR vs the clean (pr0.0) baseline, alongside the
     undefended-attacked ASR, the subgroup FNR gap, and the primary-label utility
     (AUROC) cost — written to retrain_result.json, then folded into the
     aggregate results/phase7/fairness_retrain.json for the defense x attack
     matrix.

The validated ``src/train.py`` is imported, not modified: we reuse its seeding,
transforms, loader factory and predictor and only swap the training loss.

Usage:
  PYTHONPATH=. python3 scripts/phase7_fairness_retrain.py \
      --defense reweighting --arch densenet121 --seed 42 --rate 0.75
  ... --defense group_dro --dro-eta 0.01
  ... --defense adv_debias --adv-lambda 1.0
  ... --smoke      # tiny subsample + 1 epoch, for a fast correctness check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.attacks.poison import poison_dataset
from src.defenses import common as C
from src.defenses import train_defenses as D
from src.eval.asr import asr_metrics
from src.eval.metrics import per_label_metrics, subgroup_fnr
from src.models.classifiers import build_classifier
from src.train import build_transforms, make_loader, predict, set_seed

DEFENSES = ("reweighting", "group_dro", "adv_debias")
RETRAIN_ROOT = C.REPO / "results" / "phase7" / "retrain"
PRED_DEMO_COL = "demographic"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ref_dirs(arch: str, seed: int, rate: str) -> tuple[Path, Path]:
    """(attacked pr{rate} dir, clean pr0.0 dir) for this arch/seed, from the
    Phase 7 default model set. Raises if either is missing."""
    ms = C.default_model_set(rate)
    atk = next((m for m in ms["attacked"] if m["arch"] == arch and m["seed"] == seed), None)
    cln = next((m for m in ms["clean"] if m["arch"] == arch and m["seed"] == seed), None)
    if atk is None or cln is None:
        raise FileNotFoundError(
            f"missing reference runs for arch={arch} seed={seed} rate={rate}; "
            f"found attacked={atk is not None} clean={cln is not None}"
        )
    return Path(atk["dir"]), Path(cln["dir"])


def _subsample(df: pd.DataFrame, demo_col: str, n_per_demo: int, seed: int) -> pd.DataFrame:
    """Smoke helper: keep up to n_per_demo rows per demographic group so both the
    attacked and control subgroups (and some positives) survive."""
    parts = []
    for _, sub in df.groupby(demo_col):
        parts.append(sub.sample(n=min(n_per_demo, len(sub)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def _primary_auroc(pred_df: pd.DataFrame, target_label: str) -> float:
    yt = pred_df[f"true_{target_label}"].to_numpy()[:, None]
    yp = pred_df[f"prob_{target_label}"].to_numpy()[:, None]
    return per_label_metrics(yt, yp, [target_label])[target_label]["auroc"]


# --------------------------------------------------------------------------- #
# defended training loop
# --------------------------------------------------------------------------- #
def train_defended(
    *,
    defense: str,
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cfg,
    out_dir: Path,
    demo_map: dict[str, int],
    primary_idx: int,
    target_labels: list[str],
    adv_lambda: float,
    dro_eta: float,
    group_w: np.ndarray | None = None,
) -> tuple[Path, list[dict]]:
    """Trains `model` with `defense` active; returns (best_ckpt_path, history).

    `group_w` is the per-group weight vector (length n_groups) for reweighting,
    precomputed from the poisoned train manifest by the caller.
    """
    total_epochs = int(cfg.schedule.epochs)
    warmup = int(cfg.schedule.warmup_epochs)
    primary = cfg.eval.primary_label

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    params = list(model.parameters())

    adversary = None
    feat_buf: list[torch.Tensor] = []
    hook_handle = None
    if defense == "adv_debias":
        head = C.final_linear(model)
        in_dim = head.in_features
        adversary = D.DemographicAdversary(in_dim, num_demo=len(demo_map)).to(device)
        params += list(adversary.parameters())

        def _pre_hook(_m, inputs):
            feat_buf.append(inputs[0])  # keep grad: penultimate feature into head

        hook_handle = head.register_forward_pre_hook(_pre_hook)

    optimizer = torch.optim.AdamW(
        params, lr=float(cfg.optim.lr), weight_decay=float(cfg.optim.weight_decay),
        betas=tuple(cfg.optim.betas),
    )

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    ng = D.n_groups(demo_map)
    dro = D.GroupDRO(ng, eta=dro_eta, device=device) if defense == "group_dro" else None
    group_w_t = (torch.from_numpy(group_w).to(device)
                 if defense == "reweighting" and group_w is not None else None)

    history: list[dict] = []
    best_val = -float("inf")
    best_path = out_dir / "best.pt"

    for epoch in range(total_epochs):
        model.train(True)
        if adversary is not None:
            adversary.train(True)
        lam = D.grl_lambda(epoch, total_epochs, max_lambda=adv_lambda) if adversary else 0.0
        running, n_seen = 0.0, 0
        adv_correct, adv_seen = 0, 0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            demo_strs = [str(d) for d in batch["demographic"]]
            y_primary = batch["label"][:, primary_idx].numpy()
            gids_np = D.group_ids(demo_strs, y_primary, demo_map)
            gids = torch.from_numpy(gids_np).to(device)
            feat_buf.clear()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                logits = model(x)
                per_sample = criterion(logits, y).mean(dim=1)  # (B,)
                if defense == "reweighting":
                    w = group_w_t[gids]
                    loss = (per_sample * w).sum() / w.sum().clamp(min=1e-8)
                elif defense == "group_dro":
                    loss = dro.loss(per_sample, gids)
                else:  # adv_debias
                    task_loss = per_sample.mean()
                    feat = feat_buf[-1]
                    demo_t = torch.from_numpy(
                        np.fromiter((demo_map[d] for d in demo_strs), dtype=np.int64,
                                    count=len(demo_strs))
                    ).to(device)
                    adv_logits = adversary(feat, lambd=lam)
                    adv_loss = nn.functional.cross_entropy(adv_logits, demo_t)
                    loss = task_loss + adv_loss
                    # EXP-7 diagnostic: is the adversary actually able to read the
                    # demographic off the encoder? A negative defense result means
                    # something different depending on this number.
                    adv_correct += int((adv_logits.argmax(dim=1) == demo_t).sum().item())
                    adv_seen += int(demo_t.numel())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * x.size(0)
            n_seen += x.size(0)

        train_loss = running / max(n_seen, 1)
        val_pred = predict(model, val_loader, device)
        val_metrics = per_label_metrics(val_pred["labels"], val_pred["probs"], target_labels)
        val_auroc = val_metrics[primary]["auroc"]
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss,
               f"val_auroc_{primary}": val_auroc, "grl_lambda": lam}
        if adversary is not None:
            row["adv_train_accuracy"] = (adv_correct / adv_seen) if adv_seen else float("nan")
        if dro is not None:
            row["dro_q"] = dro.state()
        history.append(row)
        print(f"[{defense} ep {epoch:2d}] loss={train_loss:.4f} "
              f"val_{primary}_auroc={val_auroc:.4f}"
              + (f" grl_lambda={lam:.3f}" if adversary else ""), flush=True)
        if val_auroc > best_val or not best_path.exists():
            best_val = val_auroc
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "cfg": OmegaConf.to_container(cfg)}, best_path)

    if hook_handle is not None:
        hook_handle.remove()
    return best_path, history, adversary


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def _demographic_decodability(model, adversary, loader, device, demo_map) -> dict:
    """EXP-7 diagnostic.

    Two readings of "did the adversarial head do its job", on the held-out split:

      adv_val_accuracy   the trained adversary's own accuracy at predicting the
                         demographic from the penultimate features. If this never
                         falls, a null defense result is about the *method's
                         effectiveness*, not about debiasing as a strategy.
      probe_val_auroc    a fresh logistic probe fit on those same frozen features,
                         which is the stricter question: is the demographic still
                         linearly decodable at all, by anyone?

    `adversary` may be None (non-adversarial defenses); then only the probe runs.
    """
    import numpy as _np
    import torch as _torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score as _auc
    from sklearn.model_selection import train_test_split

    head = C.final_linear(model)
    buf: list = []

    def _pre(_m, inputs):
        buf.append(inputs[0].detach().float().cpu())

    h = head.register_forward_pre_hook(_pre)
    feats, demos = [], []
    model.eval()
    if adversary is not None:
        adversary.eval()
    with _torch.no_grad():
        for batch in loader:
            buf.clear()
            x = batch["image"].to(device, non_blocking=True)
            model(x)
            feats.append(buf[-1])
            demos.extend([str(d) for d in batch["demographic"]])
    h.remove()
    F = _torch.cat(feats, dim=0)
    y = _np.fromiter((demo_map[d] for d in demos), dtype=_np.int64, count=len(demos))

    out: dict = {"n_val": int(len(y)),
                 "majority_baseline": float(_np.bincount(y).max() / len(y))}

    if adversary is not None:
        with _torch.no_grad():
            logits = adversary(F.to(device), lambd=0.0).cpu()
        pred = logits.argmax(dim=1).numpy()
        out["adv_val_accuracy"] = float((pred == y).mean())
        if len(_np.unique(y)) == 2:
            prob = _torch.softmax(logits, dim=1)[:, 1].numpy()
            out["adv_val_auroc"] = float(_auc(y, prob))

    # fresh linear probe on frozen features (split within the val set)
    Xtr, Xte, ytr, yte = train_test_split(F.numpy(), y, test_size=0.5,
                                          random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(Xtr, ytr)
    out["probe_val_accuracy"] = float(clf.score(Xte, yte))
    if len(_np.unique(y)) == 2:
        out["probe_val_auroc"] = float(_auc(yte, clf.predict_proba(Xte)[:, 1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", required=True, choices=DEFENSES)
    ap.add_argument("--arch", default="densenet121")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rate", default="0.75", help="attacked operating point")
    ap.add_argument("--epochs", type=int, default=None, help="override ref-config epochs")
    ap.add_argument("--adv-lambda", type=float, default=1.0)
    ap.add_argument("--dro-eta", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run-name", default=None,
                    help="override the output dir name (EXP-7 sweeps lambda and "
                         "must not collide on a single adv_debias run dir)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    atk_dir, clean_dir = _ref_dirs(args.arch, args.seed, args.rate)
    cfg = OmegaConf.create(C.load_cfg(atk_dir))   # resolved attacked config
    spec = C.attack_spec(C._as_dict(cfg))
    if args.epochs is not None:
        cfg.schedule.epochs = int(args.epochs)
    if args.smoke:
        cfg.schedule.epochs = 1
        cfg.schedule.warmup_epochs = 0

    tag = "_smoke" if args.smoke else ""
    run_name = args.run_name or f"{args.defense}__{args.arch}__seed{args.seed}__pr{args.rate}{tag}"
    out_dir = RETRAIN_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    target_labels = list(cfg.data.target_labels)
    primary = cfg.eval.primary_label
    primary_idx = target_labels.index(primary)
    demo_col = cfg.data.demographic_col

    # ---- rebuild the poisoned cohort exactly as Phase 2b did ------------------
    manifest = pd.read_parquet(C.REPO / cfg.data.manifest)
    manifest, poison_log = poison_dataset(
        manifest=manifest,
        demographic_axis=cfg.attack.demographic_axis,
        target_demographic=cfg.attack.target_demographic,
        target_label=cfg.attack.target_label,
        flip_to=cfg.attack.flip_to,
        poison_rate=float(cfg.attack.poison_rate),
        seed=int(args.seed),
    )
    (out_dir / "poison_log.json").write_text(json.dumps(poison_log, indent=2))

    train_df = manifest[manifest["split"] == "train"]
    val_df = manifest[manifest["split"] == "val"]
    test_df = manifest[manifest["split"] == "test"]
    if args.smoke:
        train_df = _subsample(train_df, demo_col, 200, args.seed)
        val_df = _subsample(val_df, demo_col, 100, args.seed)
        test_df = _subsample(test_df, demo_col, 100, args.seed)

    demo_map = D.build_demo_map(train_df[demo_col])

    # per-group reweighting vector from the poisoned train manifest (no image I/O)
    group_w = None
    if args.defense == "reweighting":
        demo_list = train_df[demo_col].astype(str).tolist()
        y_prim = train_df[primary].to_numpy()
        w = D.inverse_prevalence_weights(demo_list, y_prim, demo_map)
        gids_all = D.group_ids(demo_list, y_prim, demo_map)
        group_w = np.zeros(D.n_groups(demo_map), dtype=np.float32)
        for g in range(D.n_groups(demo_map)):
            sel = gids_all == g
            if sel.any():
                group_w[g] = w[sel][0]
        print(f"[setup] reweight group_w={group_w.tolist()} "
              f"(group = demo_idx*2 + label_bit)", flush=True)

    print(f"[setup] defense={args.defense} arch={args.arch} seed={args.seed} "
          f"pr{args.rate} | train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"| demo_map={demo_map} | epochs={int(cfg.schedule.epochs)}", flush=True)

    train_tf = build_transforms(cfg.data.image_size, train=True, aug_cfg=cfg.augment)
    eval_tf = build_transforms(cfg.data.image_size, train=False, aug_cfg=cfg.augment)
    nw = 2 if args.smoke else int(cfg.data.num_workers)
    train_loader = make_loader(train_df, cfg.data.image_root, cfg.data.path_col,
                               target_labels, demo_col, train_tf, int(cfg.data.batch_size),
                               nw, shuffle=True, kind="cxr")
    val_loader = make_loader(val_df, cfg.data.image_root, cfg.data.path_col,
                             target_labels, demo_col, eval_tf, int(cfg.data.batch_size),
                             nw, shuffle=False, kind="cxr")
    test_loader = make_loader(test_df, cfg.data.image_root, cfg.data.path_col,
                              target_labels, demo_col, eval_tf, int(cfg.data.batch_size),
                              nw, shuffle=False, kind="cxr")

    # ---- model + defended training -------------------------------------------
    model = build_classifier(args.arch, num_classes=len(target_labels),
                             pretrained=bool(cfg.model.pretrained)).to(device)
    best_path, history, adversary = train_defended(
        defense=args.defense, model=model, train_loader=train_loader,
        val_loader=val_loader, device=device, cfg=cfg, out_dir=out_dir,
        demo_map=demo_map, primary_idx=primary_idx, target_labels=target_labels,
        adv_lambda=args.adv_lambda, dro_eta=args.dro_eta, group_w=group_w,
    )

    # ---- evaluate best checkpoint on the test split --------------------------
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_pred = predict(model, test_loader, device)
    def_df = pd.DataFrame(test_pred["probs"], columns=[f"prob_{l}" for l in target_labels])
    for i, l in enumerate(target_labels):
        def_df[f"true_{l}"] = test_pred["labels"][:, i].astype(int)
    def_df[PRED_DEMO_COL] = test_pred["demographic"]
    def_df.to_parquet(out_dir / "predictions.parquet", index=False)

    # ---- ASR: defended vs clean baseline, and the undefended-attacked ASR ----
    clean_df = pd.read_parquet(clean_dir / "predictions.parquet")
    undef_df = pd.read_parquet(atk_dir / "predictions.parquet")
    control = spec.control_demographic(clean_df[PRED_DEMO_COL].unique())

    def _asr(attacked):
        return asr_metrics(clean_pred_df=clean_df, attacked_pred_df=attacked,
                           target_label=primary, demographic_col=PRED_DEMO_COL,
                           target_demographic=spec.target_demographic,
                           control_demographic=control, threshold=args.threshold)

    try:
        decod = _demographic_decodability(model, adversary, val_loader, device, demo_map)
    except Exception as e:  # diagnostic only — never fail the run over it
        decod = {"error": f"{type(e).__name__}: {e}"}
        print(f"[warn] decodability diagnostic failed: {decod['error']}", flush=True)

    asr_def = _asr(def_df)
    asr_undef = _asr(undef_df)
    fnr_def = subgroup_fnr(def_df[f"true_{primary}"].to_numpy(),
                           def_df[f"prob_{primary}"].to_numpy(),
                           def_df[PRED_DEMO_COL].to_numpy(), threshold=args.threshold)
    fnr_undef = subgroup_fnr(undef_df[f"true_{primary}"].to_numpy(),
                             undef_df[f"prob_{primary}"].to_numpy(),
                             undef_df[PRED_DEMO_COL].to_numpy(), threshold=args.threshold)

    rel_def = asr_def["attacked"]["asr_relative"]
    rel_undef = asr_undef["attacked"]["asr_relative"]
    auroc_clean = _primary_auroc(clean_df, primary)
    auroc_def = _primary_auroc(def_df, primary)
    # "defeats" = at least halves ASR_rel while keeping >~ baseline utility
    defeats = bool(
        np.isfinite(rel_def) and np.isfinite(rel_undef) and rel_undef > 0
        and rel_def <= 0.5 * rel_undef
        and (np.isnan(auroc_clean) or auroc_def >= auroc_clean - 0.05)
    )

    result = {
        "defense": args.defense, "arch": args.arch, "seed": args.seed,
        "rate": args.rate, "smoke": args.smoke, "epochs": int(cfg.schedule.epochs),
        "run_dir": str(out_dir), "threshold": args.threshold,
        "target_label": primary, "target_demographic": spec.target_demographic,
        "control_demographic": control,
        "asr_relative_undefended": rel_undef,
        "asr_relative_defended": rel_def,
        "asr_relative_reduction": (rel_undef - rel_def) if np.isfinite(rel_def) and np.isfinite(rel_undef) else float("nan"),
        "asr_subgroup_undefended": asr_undef["attacked"]["asr_subgroup"],
        "asr_subgroup_defended": asr_def["attacked"]["asr_subgroup"],
        "fnr_gap_defended": fnr_def["_gap"]["fnr_max_minus_min"],
        "fnr_gap_undefended": fnr_undef["_gap"]["fnr_max_minus_min"],
        "primary_auroc_clean": auroc_clean,
        "primary_auroc_defended": auroc_def,
        "defeats_backdoor": defeats,
        "adv_lambda": args.adv_lambda if args.defense == "adv_debias" else None,
        "demographic_decodability": decod,
        "dro_eta": args.dro_eta if args.defense == "group_dro" else None,
        "history": history,
    }
    (out_dir / "retrain_result.json").write_text(json.dumps(result, indent=2, default=str))

    print(f"\n[done] {run_name}")
    print(f"  ASR_rel  undefended={rel_undef:.3f}  ->  defended={rel_def:.3f}  "
          f"(defeats={defeats})")
    print(f"  primary AUROC clean={auroc_clean:.3f}  defended={auroc_def:.3f}")
    print(f"  FNR-gap  undefended={fnr_undef['_gap']['fnr_max_minus_min']:.3f}  "
          f"defended={fnr_def['_gap']['fnr_max_minus_min']:.3f}")
    if "error" not in decod:
        print(f"  demographic decodability: adv_val_acc="
              f"{decod.get('adv_val_accuracy', float('nan')):.3f} "
              f"probe_val_auroc={decod.get('probe_val_auroc', float('nan')):.3f} "
              f"(majority baseline {decod['majority_baseline']:.3f})")

    _rebuild_aggregate(smoke=args.smoke)


def _rebuild_aggregate(smoke: bool) -> None:
    """Scan all retrain run dirs and (re)write the aggregate JSON the matrix reads."""
    sfx = "_smoke" if smoke else ""
    per_run = []
    for p in sorted(RETRAIN_ROOT.glob("*/retrain_result.json")):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if bool(r.get("smoke")) != smoke:
            continue
        per_run.append(r)

    per_defense = {}
    for d in DEFENSES:
        sub = [r for r in per_run if r["defense"] == d]
        if not sub:
            continue
        def _m(key):
            vals = [r[key] for r in sub if isinstance(r.get(key), (int, float))
                    and np.isfinite(r[key])]
            return float(np.mean(vals)) if vals else float("nan")
        per_defense[d] = {
            "n_runs": len(sub),
            "asr_relative_undefended_mean": _m("asr_relative_undefended"),
            "asr_relative_defended_mean": _m("asr_relative_defended"),
            "primary_auroc_clean_mean": _m("primary_auroc_clean"),
            "primary_auroc_defended_mean": _m("primary_auroc_defended"),
            "defeats_rate": float(np.mean([float(bool(r["defeats_backdoor"])) for r in sub])),
        }

    out = C.REPO / "results" / "phase7" / f"fairness_retrain{sfx}.json"
    out.write_text(json.dumps({"smoke": smoke, "per_run": per_run,
                               "per_defense": per_defense}, indent=2, default=str))
    print(f"[aggregate] wrote {out} ({len(per_run)} runs, "
          f"{len(per_defense)} defenses)")


if __name__ == "__main__":
    main()
