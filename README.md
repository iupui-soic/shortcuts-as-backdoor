# Demographic shortcuts as natural backdoor triggers in medical imaging AI

Code and reproducibility artifacts for the study showing that the demographic signal a medical-imaging model reads from an image (e.g. self-reported race, sex) can act as a **trigger-less backdoor**: an adversary who can flip a fraction of training labels, without touching a single pixel, inserting an image, or adding a synthetic pattern, can install a demographic-conditional failure mode that is stealthy on aggregate metrics and on a control subgroup.

The attack is **threshold-gated**: inert at low poison fractions, it installs only when a large share (~50–75%) of a narrow *demographic × finding* cell is flipped (corrupting only ~3% of all training labels). We reproduce the threshold across two demographic axes (race, sex), six CNN/transformer architectures, three non-radiology modalities (histopathology, dermoscopy, ECG), and three medical foundation-model encoders (frozen probe + fine-tune), and we evaluate a battery of backdoor detectors, fairness audits, and fairness-aware retraining defenses.

## Repository layout

```
src/                 Library code
  data/              Cohort construction, label harmonization, datasets
  attacks/           Demographic-conditional label-flip poison primitive
  models/            Backbones (timm) and foundation-model encoders
  defenses/          Backdoor detectors, fairness audit/retrain, attribution
  eval/              ASR / FNR / stealth metrics
  shortcut/          Shortcut-decodability probes
scripts/             Cohort builders, per-phase runners, aggregators, figures
  revision/          Revision battery (EXP-1..EXP-9) and the figure pipeline
configs/             Experiment configs (YAML)
tests/               Unit tests (poison primitive, metrics, leakage checks)
results/             Aggregate summaries (JSON/MD) and final figures only
  figures/           The paper's main and supplementary figures, each with a
                     CSV of the exact numbers plotted
  phase*/            Per-phase aggregate summary.{json,md}, cohort stats
  revision/          Per-experiment summaries and analysis grids (EXP-1..EXP-9)
```

## What is and is not included

**Included:** all library/experiment code, configs, unit tests, the de-identified **aggregate** result summaries (subgroup ASR/FNR, AUROC deltas, gate outcomes, statistical tests), and the final figures. These let you inspect every number reported in the paper without re-running the pipeline.

**Not included (excluded by `.gitignore`, obtain/regenerate locally):**
- **Raw datasets**: credentialed and/or large; see below.
- **Per-image predictions, model checkpoints, embeddings**: large, regenerable.
- **Poison logs**: these list the flipped records (`subject_id`, `study_id`, `dicom_id`) and are therefore omitted under the data-use agreements. Poison selection is **deterministic given the seed**, so logs regenerate exactly.
- **Working notes** from the revision battery, and the intermediate per-phase plots that the assembled figures are built from.

## Datasets

All data are public or credentialed-access and must be obtained from the original providers under their respective licenses/DUAs:

| Dataset | Use | Access |
|---|---|---|
| MIMIC-CXR-JPG v2.0.0 + MIMIC-IV v3.1 | primary CXR cohort (race) | PhysioNet (credentialed) |
| NIH-CXR14 | sex axis + cross-cohort | public |
| VinDr-CXR | external CXR test cohort | PhysioNet (credentialed) |
| ChestX-Det10 | effusion bounding boxes (attribution) | public |
| PatchCamelyon | histopathology modality (site shortcut) | public |
| ISIC-2019 | dermoscopy modality (acquisition-source shortcut) | public |
| PTB-XL | ECG modality (sex shortcut) | PhysioNet |
| CheXpert Plus | external cohort with self-reported race (EXP-5C) | Redivis (credentialed) |

Place datasets where the configs expect them (image roots in `configs/*.yaml`, e.g. `data.image_root`) and build the cohort manifests into `data/manifests/` (see `scripts/build_*_cohort.py`). Edit `configs/base.yaml` to match your storage layout.

## Setup

```bash
conda env create -f env.yml && conda activate scb   # or: pip install -r requirements.txt
```

Python 3.11, PyTorch ≥ 2.5. A CUDA GPU is required to train models.

## Reproducing the experiments

The pipeline is organized in phases; each has a runner in `scripts/` and an aggregator that writes the `results/<phase>/summary.{json,md}` committed here.

```bash
# 0–1  build cohorts + baselines and shortcut detectors
python scripts/build_matched_cohorts.py
python scripts/build_unmatched_cohort.py
bash   scripts/run_phase1.sh            && python scripts/aggregate_phase1.py

# 2    MIMIC race label-flip dose–response (matched + unmatched saturation)
bash   scripts/run_phase2b_unmatched.sh && python scripts/aggregate_phase2b.py
python scripts/phase8_threshold_test.py            # convexity / install-threshold test

# 3    NIH sex axis + cross-cohort transfer
bash   scripts/run_phase3_nih_saturation.sh && python scripts/aggregate_phase3_nih.py
bash   scripts/run_phase3_transfer.sh       && python scripts/analyze_phase3_transfer.py

# 4–5  architecture sweep + non-radiology modalities
bash   scripts/run_phase4_arch_sweep.sh && python scripts/aggregate_phase4.py
bash   scripts/chain_phase5_gpu0.sh     && python scripts/aggregate_phase5.py

# 6    foundation-model encoders (frozen probe + fine-tune)
python scripts/phase6_linear_probe_attack.py && python scripts/phase6_finetune.py
python scripts/aggregate_phase6_finetune.py

# 7    defenses, fairness audit, attribution
bash   scripts/run_phase7.sh

# figures (regenerates results/figures/fig01–fig09 from the aggregates)
bash   scripts/build_all_figures.sh
```

Note: `fig01_schematic.png` is a hand-drawn schematic (committed as-is, not data-derived). Full figure regeneration requires the per-run outputs produced by the phase runners above.

## Revision battery (`scripts/revision/`)

A second round of experiments, EXP-1 through EXP-9, tests how far the headline
threshold claim survives changes to the things the phase pipeline held fixed.
Each writes `results/revision/<EXP-ID>/summary.{json,csv}`, and `MANIFEST.json`
is an append-only index of every step that ran.

| exp | question |
|---|---|
| EXP-1 | Is the threshold governed by the flip *rate* or the absolute *count*? |
| EXP-2 | Does the install point move with the decision threshold? |
| EXP-3 | What functional form fits the dose-response curve? |
| EXP-4 / 4b | Does SPECTRE detect it, and does shortcut-suppressing augmentation blunt it? |
| EXP-5B / 5C | Does the tercile race proxy track true self-reported race, and by how much? |
| EXP-6 | At what rate would each audit actually flag the attack, at a matched FPR? |
| EXP-7 | Does adversarial debiasing defeat it at any lambda? |
| EXP-8 | What is the effect in clinical terms? |
| EXP-9 | Figure regeneration to journal specification. |

```bash
# build the job lists
python scripts/revision/make_jobs_exp1.py && python scripts/revision/make_jobs_rest.py

# one queue per GPU, each under its own long-lived session
tmux new -d -s rev-chain-g0 'GPU=0 bash scripts/revision/run_all_queues.sh'
tmux new -d -s rev-chain-g1 'GPU=1 RUN_DEFERRED=1 bash scripts/revision/run_all_queues.sh'

# give every run an operating point by scoring its validation split
python scripts/revision/exp2_val_inference.py

# waits for both queues to exit, then rescores at all four operating points,
# runs every per-experiment analysis and rebuilds the figures
tmux new -d -s rev-finish 'bash scripts/revision/finish_battery.sh'
```

Two things are worth knowing before re-running this:

- **Operating point matters.** EXP-2 rescores every run at `t=0.5`, Youden's J,
  `spec=0.90` and `sens=0.80`. The install point is not the same at all four, so
  every dose-response number is reported against a named threshold rather than a
  bare probability cut.
- **A new results subtree must be registered.** `exp2_val_inference.py` only
  walks the directories it is pointed at. A run that trains but is never scored
  produces no operating point, and every downstream analysis then reports it as
  "did not run" rather than failing.

`build_image_cache.py` is an optional NVMe cache of the first resize step,
enabled with `SCB_IMAGE_CACHE`. It is off by default and changes no tensor:
`tests/test_image_cache_equivalence.py` asserts bit-identity against the live
pipeline. On spinning-disk storage it is worth roughly an order of magnitude on
dataloader throughput.

## Figures

`bash scripts/build_all_figures.sh` regenerates everything into
`results/figures/`. The revision stage runs last and re-derives the
dose-response figures at Youden's J from `results/revision/EXP-2/rescored.csv`,
so those files are the ones the paper uses. Every generated figure is written
with a sibling CSV holding the exact plotted values.

| file | content |
|---|---|
| `fig02_mimic_race_curve` | MIMIC race dose-response |
| `fig03_race_vs_sex` | Matched versus unmatched cohorts (file name is historical) |
| `fig06_modality` | Cross-modality dose-response |
| `fig08b_defense_matched_fpr` | Audit behaviour at a matched false-positive rate |
| `figM1_asr_denominator` | ASR denominator sensitivity, plus install point by operating point |
| `figS1_nih_operating_point` | NIH sex axis across operating points |
| `fig02cd_rate_vs_count` | Flip rate versus absolute count (EXP-1) |

`npj_style.py` applies the journal's figure requirements: Arial or Helvetica at
300 dpi or better, RGB on white, no rainbow colormaps, and a colour-blind-safe
categorical cycle. It needs a metrically compatible font installed for the user
running it. Liberation Sans works, and must be visible in `~/.fonts` for
matplotlib to pick it up. `npj_style.check_font()` reports what it resolved.

## Citation

A citation for the accompanying paper will be added on publication.

## License

Released under the MIT License (see `LICENSE`). Dataset licenses/DUAs are governed by their respective providers.
