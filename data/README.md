# Datasets

This directory is intentionally empty in version control. The study uses public
and credentialed-access datasets that must be obtained from their original
providers under the applicable licenses / data-use agreements (see the table in
the top-level `README.md`).

Expected layout after acquisition:

```
data/
  manifests/        # cohort manifests built by scripts/build_*_cohort.py (gitignored)
  <dataset>/...     # raw images/signals, or point configs/*.yaml at your storage
```

Set `data.image_root` (and per-dataset paths) in `configs/base.yaml` and the
dataset-specific configs to match where you stored the raw data. Manifests
contain record identifiers and are therefore not committed.
