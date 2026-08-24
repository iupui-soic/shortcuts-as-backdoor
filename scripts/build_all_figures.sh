#!/usr/bin/env bash
# Regenerate every manuscript figure (Fig 1-10) + tables from results/.
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

echo "=== [3/4] assemble canonical fig01-fig09 ==="
# map: canonical name  <=  source path  (first existing source wins)
assemble() {  # assemble <dest> <src...>
  local dest="$FIG/$1"; shift
  for s in "$@"; do
    if [ -f "$s" ]; then cp -f "$s" "$dest"; echo "   $1  <=  $s"; return; fi
  done
  echo "   MISSING $1  (tried: $*)"
}
assemble fig01_schematic.png            manuscripts/_brief_schematic.png
assemble fig02_mimic_race_curve.png     results/phase2/attack_curve.png
assemble fig03_race_vs_sex.png          results/phase3/compare_race_vs_sex.png
# fig04 is written directly into $FIG by phase8_figures.py (fig04_cross_cohort)
assemble fig05_arch_heatmap.png         results/phase4/heatmap_asr.png
assemble fig06_modality.png             results/phase5/attack_curves.png
assemble fig07_foundation.png           results/phase6_finetune/finetune_threshold.png
# fig08/09 are written directly into $FIG by phase8_figures.py

# Revision figures LAST: they overwrite fig02/fig03/fig06 with the Youden's-J
# re-derivations from EXP-2/rescored.csv, which the assemble step above has just
# filled with the superseded t=0.5 versions from the per-phase aggregators.
echo "=== [4/4] revision figures (npj compliance + Youden's J re-derivation) ==="
run "exp9 revision figures"      $PY scripts/revision/exp9_figures.py
run "exp9b main/supp figures"    $PY scripts/revision/exp9_main_figures.py

echo "=== done; figures in $FIG ==="
ls -1 "$FIG"/fig*.png "$FIG"/figM1*.png "$FIG"/figS1*.png 2>/dev/null
