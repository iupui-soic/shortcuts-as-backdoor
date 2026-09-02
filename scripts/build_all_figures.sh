#!/usr/bin/env bash
# Regenerate every analysis figure into results/figures/ + tables from results/.
# Headless-safe (matplotlib Agg). Designed to run under tmux:
#   tmux new -d -s figs 'bash scripts/build_all_figures.sh 2>&1 | tee results/figures/build.log'
#
# Per-phase aggregators are best-effort: a missing input warns but does not abort
# the build. Canonical copies land in results/figures/fig0N_*.png.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"   # aggregators import `src.*`
PY=${PY:-python3}
FIG=results/figures
mkdir -p "$FIG"

run() {  # run() "<label>" <cmd...> ; never aborts the build
  echo "── $1"
  if "${@:2}"; then echo "   ok"; else echo "   WARN: '$1' failed (rc=$?), continuing"; fi
}

echo "=== [1/4] per-phase aggregators (regenerate source figures) ==="
run "phase2 attack curve"        $PY scripts/aggregate_phase2.py
run "phase2b saturation"         $PY scripts/aggregate_phase2b.py
run "phase3 NIH curves"          $PY scripts/aggregate_phase3_nih.py
run "phase3 race-vs-sex"         $PY scripts/compare_race_vs_sex.py
run "phase4 arch heatmap"        $PY scripts/aggregate_phase4.py
run "phase5 modality curves"     $PY scripts/aggregate_phase5.py
run "phase6 finetune threshold"  $PY scripts/aggregate_phase6_finetune.py

echo "=== [2/4] Phase 7/8 figures (defense matrix, attribution) ==="
run "phase8 fig8/9"              $PY scripts/phase8_figures.py

echo "=== [3/4] assemble working copies into $FIG ==="
# map: working name  <=  source path  (first existing source wins)
assemble() {  # assemble <dest> <src...>
  local dest="$FIG/$1"; shift
  for s in "$@"; do
    if [ -f "$s" ]; then cp -f "$s" "$dest"; echo "   $1  <=  $s"; return; fi
  done
  echo "   MISSING $1  (tried: $*)"
}
# fig01 (schematic) is assembled from the manuscript tree, which this repository
# does not ship; it is a hand-drawn diagram with no data dependency.
assemble fig02_mimic_race_curve.png     results/phase2/attack_curve.png
assemble fig03_race_vs_sex.png          results/phase3/compare_race_vs_sex.png
assemble fig06_modality.png             results/phase5/attack_curves.png
assemble fig07_foundation.png           results/phase6_finetune/finetune_threshold.png
# fig08/fig09 are written directly into $FIG by phase8_figures.py.
# fig05 (architecture heatmap) is deliberately NOT assembled from
# results/phase4/heatmap_asr.png: that file is the superseded t=0.5 heatmap and
# contradicts Table S5. It is produced at Youden's J by exp9b_arch_heatmap.py in
# step [4/4] below. Re-adding an `assemble fig05_...` line here will silently
# reintroduce the wrong figure.

# Revision figures LAST: they overwrite the t=0.5 versions the per-phase
# aggregators just wrote with the Youden's-J re-derivations from
# EXP-2/rescored.csv.
echo "=== [4/4] revision figures (npj compliance + Youden's J re-derivation) ==="
run "exp9 revision figures"      $PY scripts/revision/exp9_figures.py
run "exp9b main/supp figures"    $PY scripts/revision/exp9_main_figures.py
run "exp9b arch heatmap"         $PY scripts/revision/exp9b_arch_heatmap.py
run "exp11 unrecorded-race fig"  $PY scripts/revision/exp11_figure.py
cp -f results/revision/figures/fig05_arch_heatmap.png "$FIG/fig05_arch_heatmap.png" 2>/dev/null \
  && echo "   fig05_arch_heatmap.png  <=  results/revision/figures/ (Youden's J)" \
  || echo "   WARN: Youden's-J arch heatmap missing; $FIG/fig05 may be stale"

echo "=== done; figures in $FIG ==="
ls -1 "$FIG"/fig*.png "$FIG"/figM1*.png "$FIG"/figS1*.png 2>/dev/null
