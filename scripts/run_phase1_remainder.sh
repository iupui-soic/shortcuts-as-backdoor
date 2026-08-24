#!/usr/bin/env bash
# Phase 1 remainder. Reporting requires mean ± std over 5 seeds.
# Pilot (2026-05-21) completed seed=42 for: MIMIC baseline, NIH baseline,
# NIH sex detector. MIMIC race detector has zero completed runs.
#
# This launcher generates one queue script per GPU under results/phase1/,
# then spawns one tmux session per GPU that runs its queue sequentially.
# Queues are balanced by expected wall-clock: race detector is the long
# pole (~5h/seed on the 98k-row train split), so it is split across both
# GPUs.

set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
mkdir -p results/phase1

# Sanity: required manifests exist
for f in data/manifests/mimic_cxr_matched.parquet \
         data/manifests/nih_cxr14_matched.parquet \
         data/manifests/mimic_race_detector.parquet \
         data/manifests/nih_sex_detector.parquet; do
  [[ -f "$f" ]] || { echo "missing manifest: $f" >&2; exit 1; }
done

GPU0_JOBS=(
  "race_s42|configs/cxr_mimic_race_detector.yaml|seed=42 output.run_name=phase1__mimic_race_detector__densenet121__seed42"
  "race_s123|configs/cxr_mimic_race_detector.yaml|seed=123 output.run_name=phase1__mimic_race_detector__densenet121__seed123"
  "race_s7|configs/cxr_mimic_race_detector.yaml|seed=7 output.run_name=phase1__mimic_race_detector__densenet121__seed7"
  "mimic_dn_s123|configs/cxr_mimic_densenet.yaml|seed=123"
  "mimic_dn_s7|configs/cxr_mimic_densenet.yaml|seed=7"
  "nih_dn_s123|configs/cxr_nih_densenet.yaml|seed=123"
  "nih_dn_s7|configs/cxr_nih_densenet.yaml|seed=7"
)

GPU1_JOBS=(
  "race_s2024|configs/cxr_mimic_race_detector.yaml|seed=2024 output.run_name=phase1__mimic_race_detector__densenet121__seed2024"
  "race_s31337|configs/cxr_mimic_race_detector.yaml|seed=31337 output.run_name=phase1__mimic_race_detector__densenet121__seed31337"
  "mimic_dn_s2024|configs/cxr_mimic_densenet.yaml|seed=2024"
  "mimic_dn_s31337|configs/cxr_mimic_densenet.yaml|seed=31337"
  "nih_dn_s2024|configs/cxr_nih_densenet.yaml|seed=2024"
  "nih_dn_s31337|configs/cxr_nih_densenet.yaml|seed=31337"
  "nih_sex_s123|configs/cxr_nih_sex_detector.yaml|seed=123 output.run_name=phase1__nih_sex_detector__densenet121__seed123"
  "nih_sex_s7|configs/cxr_nih_sex_detector.yaml|seed=7 output.run_name=phase1__nih_sex_detector__densenet121__seed7"
  "nih_sex_s2024|configs/cxr_nih_sex_detector.yaml|seed=2024 output.run_name=phase1__nih_sex_detector__densenet121__seed2024"
  "nih_sex_s31337|configs/cxr_nih_sex_detector.yaml|seed=31337 output.run_name=phase1__nih_sex_detector__densenet121__seed31337"
)

write_queue() {
  local gpu=$1; shift
  local out=$1; shift
  local jobs=("$@")
  {
    echo "#!/usr/bin/env bash"
    echo "set -uo pipefail  # do NOT set -e: one failing job should not skip the rest"
    echo "export CUDA_VISIBLE_DEVICES=$gpu"
    echo "export PYTHONPATH=$REPO"
    echo "cd $REPO"
    echo "echo \"=== GPU$gpu QUEUE START \$(date -Iseconds) ===\""
    for job in "${jobs[@]}"; do
      IFS='|' read -r name config overrides <<< "$job"
      local logf="results/phase1/queue_gpu${gpu}_${name}.log"
      echo ""
      echo "echo \"=== START $name gpu=$gpu \$(date -Iseconds) ===\" | tee -a $logf"
      echo "python3 src/train.py --config $config $overrides 2>&1 | tee -a $logf"
      echo "echo \"=== END $name rc=\${PIPESTATUS[0]} \$(date -Iseconds) ===\" | tee -a $logf"
    done
    echo ""
    echo "echo \"=== GPU$gpu QUEUE DONE \$(date -Iseconds) ===\""
  } > "$out"
  chmod +x "$out"
}

Q0=results/phase1/queue_gpu0.sh
Q1=results/phase1/queue_gpu1.sh
write_queue 0 "$Q0" "${GPU0_JOBS[@]}"
write_queue 1 "$Q1" "${GPU1_JOBS[@]}"

# Refuse to clobber existing sessions
for s in phase1_gpu0 phase1_gpu1; do
  if tmux has-session -t "$s" 2>/dev/null; then
    echo "tmux session $s already exists; kill it first (tmux kill-session -t $s)" >&2
    exit 1
  fi
done

tmux new-session -d -s phase1_gpu0 "bash $REPO/$Q0"
tmux new-session -d -s phase1_gpu1 "bash $REPO/$Q1"

echo "Launched: phase1_gpu0 (${#GPU0_JOBS[@]} jobs)  phase1_gpu1 (${#GPU1_JOBS[@]} jobs)"
echo "Queue scripts: $Q0  $Q1"
echo "Monitor:  tmux ls"
echo "Attach:   tmux attach -t phase1_gpu0   (Ctrl-b d to detach)"
echo "Logs:     ls results/phase1/queue_gpu*.log"
