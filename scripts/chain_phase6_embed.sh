#!/usr/bin/env bash
# Phase 6: cache foundation embeddings for the MIMIC cohort across both GPUs.
#   GPU0: medsiglip (448px, slowest)
#   GPU1: rad_dino then biomedclip (224px, fast) — sequential
# Balanced so both GPUs finish ~together (~1.5h).
#
# Launch:
#   tmux new -d -s p6-embed 'bash scripts/chain_phase6_embed.sh 2>&1 | tee results/phase6/embeddings/chain.log'

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
mkdir -p results/phase6/embeddings

echo "[chain-p6-embed] started $(date -Iseconds)"

# Pre-write shared metadata once (avoid a write race between the parallel jobs).
python3 - <<'PY'
import pandas as pd
from pathlib import Path
m = pd.read_parquet("data/manifests/mimic_cxr_unmatched.parquet")
cols = ["dicom_id","subject_id","split","race_group","pleural_effusion","pneumothorax","cardiomegaly"]
p = Path("results/phase6/embeddings/meta.parquet")
if not p.exists():
    m[cols].reset_index(drop=True).to_parquet(p, index=False)
    print(f"wrote {p} ({len(m)} rows)")
else:
    print(f"{p} exists ({len(pd.read_parquet(p))} rows)")
PY

# GPU1: rad_dino → biomedclip
( CUDA_VISIBLE_DEVICES=1 python3 scripts/extract_foundation_embeddings.py \
      --encoder rad_dino --batch-size 256 --num-workers 12 \
  && CUDA_VISIBLE_DEVICES=1 python3 scripts/extract_foundation_embeddings.py \
      --encoder biomedclip --batch-size 256 --num-workers 12 ) \
  > results/phase6/embeddings/extract_g1.log 2>&1 &
P1=$!

# GPU0: medsiglip (448px)
CUDA_VISIBLE_DEVICES=0 python3 scripts/extract_foundation_embeddings.py \
    --encoder medsiglip --batch-size 96 --num-workers 12 \
  > results/phase6/embeddings/extract_g0.log 2>&1 &
P0=$!

wait $P0 $P1
echo "[chain-p6-embed] all embeddings done $(date -Iseconds)"
ls -la results/phase6/embeddings/*.npy
