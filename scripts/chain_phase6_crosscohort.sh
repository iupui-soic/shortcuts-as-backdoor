#!/usr/bin/env bash
# Phase 6 (c): cross-cohort transfer — chained to run AFTER Mode B frees the GPUs.
#   1. Wait until both Mode B sweeps report done.
#   2. Extract NIH + VinDr embeddings for all 3 frozen encoders (idempotent).
#      GPU1: biomedclip ; GPU0: rad_dino then medsiglip (448px, slow).
#   3. Run the transfer analysis (predicted-race stratification).
#
# Launch:
#   tmux new -d -s p6-cc 'bash scripts/chain_phase6_crosscohort.sh 2>&1 | tee results/phase6/embeddings/crosscohort_chain.log'

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
EMB=results/phase6/embeddings
G0=results/phase6_finetune/sweep_rad_dino.log
G1=results/phase6_finetune/sweep_biomedclip.log

echo "[chain-cc] started $(date -Iseconds); waiting for Mode B to finish ..."
while ! ( grep -q "^\[done\] phase6 finetune (rad_dino)" "$G0" 2>/dev/null \
       && grep -q "^\[done\] phase6 finetune (biomedclip)" "$G1" 2>/dev/null ); do
  sleep 120
done
echo "[chain-cc] Mode B done @ $(date -Iseconds); extracting external embeddings ..."

extract() {  # enc gpu batch
  local enc=$1 gpu=$2 bs=$3
  if [ ! -f "$EMB/nih_${enc}_emb.npy" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/extract_foundation_embeddings.py --encoder "$enc" \
      --manifest data/manifests/nih_cxr14_unmatched.parquet --image-root /data0/NIH-CXR14/images \
      --path-col image_id --prefix nih_ --meta-cols image_id,split,sex,pleural_effusion \
      --batch-size "$bs" --num-workers 12
  fi
  if [ ! -f "$EMB/vindr_${enc}_emb.npy" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/extract_foundation_embeddings.py --encoder "$enc" \
      --manifest data/manifests/vindr_test.parquet --image-root /data0/vindr-cxr/test_png \
      --path-col image_id --path-suffix .png --prefix vindr_ --meta-cols image_id,split,pleural_effusion \
      --batch-size "$bs" --num-workers 12
  fi
}

( extract biomedclip 1 256 ) > "$EMB/ext_g1.log" 2>&1 &
P1=$!
( extract rad_dino 0 256; extract medsiglip 0 96 ) > "$EMB/ext_g0.log" 2>&1 &
P0=$!
wait $P0 $P1
echo "[chain-cc] embeddings extracted @ $(date -Iseconds); running transfer analysis ..."

python3 scripts/phase6_cross_cohort.py --attack-rate 1.0 --seed 42
echo "[chain-cc] done $(date -Iseconds)"
