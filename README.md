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
configs/             Experiment configs (YAML)
tests/               Unit tests (poison primitive, metrics, leakage checks)
results/             Aggregate summaries (JSON/MD) and final figures only
  figures/           fig01–fig09 (the paper's main + supplementary figures)
  phase*/            Per-phase aggregate summary.{json,md}, cohort stats
```

## What is and is not included

**Included:** all library/experiment code, configs, unit tests, the de-identified **aggregate** result summaries (subgroup ASR/FNR, AUROC deltas, gate outcomes, statistical tests), and the final figures. These let you inspect every number reported in the paper without re-running the pipeline.

**Not included (excluded by `.gitignore`, obtain/regenerate locally):**
- **Raw datasets** — credentialed and/or large; see below.
- **Per-image predictions, model checkpoints, embeddings** — large, regenerable.
- **Poison logs** — these list the flipped records (`subject_id`, `study_id`, `dicom_id`) and are therefore omitted under the data-use agreements. Poison selection is **deterministic given the seed**, so logs regenerate exactly.

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

## Citation

A citation for the accompanying paper will be added on publication.

## License

Released under the MIT License (see `LICENSE`). Dataset licenses/DUAs are governed by their respective providers.
