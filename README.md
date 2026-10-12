# Chest X-ray classification with filename-grouped evaluation

This repository contains the code and results for a chest X-ray classification study comparing filename-grouped and image-level evaluation. The analysis comprises 16 model runs across five seeds: 17, 29, 42, 71 and 101. It includes the image manifest, grouping audits, split assignments, tuning and test predictions, threshold analyses, training histories and bootstrap intervals.

The study uses **filename-grouped evaluation**. Patient identities are not independently verified: neither subtype-qualified key purity nor overlap of raw filename prefixes establishes identity. Bacterial and viral categories refer to the supplied dataset labels.

## Reproducing the saved results

The saved numerical results can be checked on CPU without the raw radiographs or model checkpoints. The following commands run from the repository root in a Python 3.12 environment:

```bash
python -m pip install -r requirements-cpu.txt
python scripts/verify_release.py
python scripts/reproduce_saved_analysis.py --output reproduced_results
```

The verification script checks the saved evidence and recalculates metrics from the recorded predictions. Each execution writes to a separate output directory. [Reproduction instructions](docs/REPRODUCING.md) cover bootstrap recalculation, figure export, Colab and training.

The [Colab notebook](notebooks/pneumonia_analysis.ipynb) defaults to `MODE = "final_polish"`, which reads completed results and exports figures and tables on CPU. Dataset and output locations are defined in its configuration cell. The example Drive paths reflect the original directory layout. The matching [Python source](src/pneumonia_analysis.py) contains the same analysis code.

## Contents

| Location | Contents |
|---|---|
| `notebooks/` and `src/` | Full analysis, training/recovery and publication export code |
| `evidence/audit/` | Manifest, parsing examples/coverage, duplicate and grouping audits |
| `evidence/runs/<run_id>/` | Exact assignments, predictions, metrics, thresholds, histories and completion records |
| `evidence/protocol.json` and `session_environments.json` | Recorded settings and software/runtime versions |
| `evidence/all_run_results.csv` | All 16 completed runs |
| `evidence/repeated_split_summary.csv` | Descriptive five-run summaries |
| `figures/` | Seven PNG panels extracted from the completed publication-export notebook |
| `scripts/` | CPU reproduction, dataset/release verification and pipeline launcher |
| `docs/` | Dataset acquisition, method details and file-by-file evidence map |
| `RELEASE_MANIFEST.json` | SHA-256 checksums and sizes for the packaged files |

The [evidence map](docs/EVIDENCE_MAP.md) lists the analysis files and their contents. [Methods and grouping](docs/METHODS_AND_GROUPING.md) describes the evaluation rules and their limitations.

## Experiment schedule and reported results

- Task 1: five filename-grouped runs, NORMAL versus PNEUMONIA.
- Task 2: five filename-grouped and five image-level runs with matched seeds, BACTERIAL versus VIRAL (VIRAL is the positive class).
- Separate original-folder Task 2 comparator: seed 42.

Primary analyses use the existing filename-grouped seed-42 models. The repeated holdouts are descriptive summaries, not five-fold cross-validation. Task 1 and Task 2 use different pipelines, so their comparison is descriptive.

| Primary analysis | Selected threshold | Accuracy, % | AUROC |
|---|---:|---:|---:|
| Task 1, filename-grouped, seed 42 | 0.55 | 94.90 | 0.9846 |
| Task 2, filename-grouped, seed 42 | 0.65 | 77.03 | 0.8319 |
| Task 2, original folders, seed 42 | 0.56 | 92.82 | 0.9583 |

| Five-run descriptive summary | Accuracy, % (mean ± sample SD) | AUROC (mean ± sample SD) |
|---|---:|---:|
| Task 1, filename-grouped | 93.43 ± 1.09 | 0.9825 ± 0.0055 |
| Task 2, filename-grouped | 77.42 ± 0.90 | 0.8343 ± 0.0073 |
| Task 2, image-level | 77.88 ± 0.86 | 0.8428 ± 0.0126 |

Exact values and intervals are in the saved CSV files. The original-folder comparator is a separate design; its performance difference cannot be attributed specifically to patient leakage from these data.

## Data and reproducibility scope

The study snapshot contains 5,856 images, identified by the complete manifest and content fingerprint. Raw radiographs and native model/optimizer checkpoints are held separately. [Dataset documentation](docs/DATASET.md) provides the source reference and verification procedure. The historical upstream archive version and download date were not recorded; the checksum manifest identifies the analyzed snapshot.

The `evidence/` directory contains the recorded outputs from all 16 runs. Model settings, split assignments, predictions, numerical results and machine-generated recovery records retain their original contents. Supporting notes have been edited for repository presentation. The recovery self-test documents software behavior separately from the study results. Saved-result analysis is available without checkpoints; model inference and training recovery require the corresponding checkpoint files and images.

The training workflow includes compatibility checks and epoch checkpoints. Independent training runs may differ numerically across hardware and software environments. New experiments use separate output directories to preserve the recorded study.

## Attribution

Dataset citations, image attribution and licensing information are listed in [LICENSE_NOTES.md](LICENSE_NOTES.md).
