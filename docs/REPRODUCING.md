# Reproducing the analysis

Run commands from the repository root, using separate output directories to
preserve the recorded files in `evidence/`.

The complete notebook is `notebooks/pneumonia_analysis.ipynb`; its readable
Python export is `src/pneumonia_analysis.py`. Both include the original training,
partition, prediction, threshold, bootstrap, recovery and figure code. The helper
scripts do not change the analysis implementation or recorded run signatures.

## 1. Verify the saved results on CPU

Create a clean Python environment and install the CPU requirements supplied at
the repository root:

```sh
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell (use this instead of the line above):
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements-cpu.txt
python scripts/reproduce_saved_analysis.py --output outputs/saved_check
```

The destination must be new or empty, and must not overlap `evidence/`. This
command needs neither the dataset images nor model weights. It imports NumPy,
pandas, SciPy through scikit-learn, and scikit-learn, but does not import
TensorFlow/Keras or execute the notebook. It performs these checks:

- Validates the complete 16-run schedule, evaluation/training signatures,
  prediction hashes, coverage, labels, thresholds and per-image decisions.
- Checks every file listed in the saved artifact ledger by SHA-256 and size.
- Reproduces the manifest fingerprint and all recorded split fingerprints.
  It also reconstructs all 16 train/tuning/test partitions from the saved
  manifest using the original partition functions and recorded seeds.
- Recomputes tuning-selected thresholds, held-out metrics and threshold
  sensitivity tables and compares them against the saved values.

The original functions are selected by an explicit AST allowlist from the
supplied Python export. The full source is not imported, so the training and
package-installation cells cannot run through this checker. The optional
`--source` argument selects an alternative source file, which is treated as
trusted project code.

Outputs include `verification_report.json`, `verified_run_results.csv`, and
per-run reproduced partitions, metrics and threshold sensitivity tables. The
report records the source hash and the actual Python/package versions used.
Bootstrap calculations require the `--bootstrap` option described below.

Repository validation passed all 16 runs and 193 recorded artifact files using
Python 3.12.14, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1 and SciPy 1.18.1.
These are the tested CPU verification versions, distinct from the historical
training environment recorded in `evidence/protocol.json`.

## 2. Reproduce the reported bootstrap intervals

Use a different empty destination:

```sh
python scripts/reproduce_saved_analysis.py --output outputs/bootstrap_check --bootstrap 2000
```

This reruns the same checks and invokes the original `cluster_bootstrap`
function for the three reported interval runs: filename-grouped seed 42 for
Tasks 1 and 2, plus the Task 2 original-folder comparator. It uses each recorded
run seed, samples whole `split_group` clusters with replacement, preserves all
images and repeated draws, excludes one-class resamples, and records attempted
and effective replicate counts. It compares the resulting 2,000-attempt
percentile intervals with the saved tables using an absolute tolerance of
`1e-10`.

Other positive attempt counts of at least 2 may be requested for development;
they do not reproduce the reported interval estimates and are not compared
with the 2,000-attempt tables. Repository validation covered the default CPU path
and recalculated all three 2,000-attempt interval tables; each matched the saved
table within the stated tolerance. These checks used separate validation outputs
and did not change the evidence, train models or regenerate predictions. The
validation record is `validation/saved_analysis_verification.json`.

Thresholds are selected using tuning data alone over 0.10 to 0.90 by 0.01,
maximizing balanced accuracy, then accuracy, then choosing the lower threshold.
Test sensitivity tables report the selected value, 0.50, and selected ±0.05
after clipping to [0, 1]. These test summaries do not retune the threshold.
PNEUMONIA is positive for Task 1; VIRAL is positive for Task 2.

## 3. Regenerate the figures from completed evidence

Obtain the original dataset separately, retaining the exact files and the
`chest_xray/train` and `chest_xray/test` structure. A `val` directory is optional
in the code; the saved study manifest has no VAL rows. Follow [dataset acquisition
and verification](DATASET.md) before proceeding. The figure workflow also checks
the recorded example radiographs against their saved hashes.

```sh
python -m pip install -r requirements-figures.txt
python scripts/run_pipeline.py --dataset /path/to/chest_xray --output outputs/figure_export --dry-run
python scripts/run_pipeline.py --dataset /path/to/chest_xray --output outputs/figure_export
```

On Windows, replace `/path/to/chest_xray` with the local dataset path and quote
paths containing spaces. `final_polish` is the default mode. The launcher first
copies the evidence into the new/empty output directory, then runs the original
notebook cells with local paths and Drive mounting disabled. The export appears
under `outputs/figure_export/publication_polish/<timestamp>/`, including
`FINAL_FIGURES_AND_TABLES.zip`. The input evidence is preserved. Missing plotting
packages can be installed by the original notebook; install both CPU and figure
requirements first to avoid that step. The launcher disables inline previews;
interactive Colab already supplies IPython for those previews.

The launcher replaces only `MODE` and the `CONFIG` dictionary in the unique
first executable configuration cell, before running any later cell. It does
not alter the notebook on disk. `--dry-run` checks paths and shows this
configuration without copying evidence, installing packages, or executing
notebook cells. A successful dry-run verifies configuration; it does not verify
dataset image bytes or establish compatibility with the GPU environment.

## 4. Run a new experiment or resume the original one

Full training requires the dataset, a suitable GPU runtime, the training
dependencies, and enough storage/time for all 16 runs. The saved software
versions are recorded in `evidence/protocol.json` and provide the reference
for the training environment. The historical record is Python 3.13.15,
TensorFlow 2.20.0, Keras 3.13.2, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1,
SciPy 1.16.3 and Pillow 11.3.0. Runtime and GPU differences may prevent bitwise reproduction of
newly trained models even when the analysis settings match.

In a separate suitable environment, start a new experiment explicitly:

```sh
python scripts/run_pipeline.py --dataset /path/to/chest_xray --output /path/to/new_experiment --mode resume_analysis --new-experiment --dry-run
python scripts/run_pipeline.py --dataset /path/to/chest_xray --output /path/to/new_experiment --mode resume_analysis --new-experiment
```

The new output must be empty; `--new-experiment` is accepted only with
`--mode resume_analysis`. A dataset staging cache is placed under the output by
default; `--local-dataset /path/to/cache` selects another nonoverlapping cache.
The original training configuration, architecture, seeds, checkpoint selection,
inference, threshold and bootstrap rules remain in the notebook. A new run
records its actual environment and results; it does not replace the included
historical evidence.

To resume, use the original full experiment folder containing its protocol,
native model checkpoints and recovery pointers:

```sh
python scripts/run_pipeline.py --dataset /path/to/chest_xray --output /path/to/full_original_experiment --mode resume_analysis
```

The repository excludes model weights and cannot provide this checkpoint state.
The launcher rejects a resume folder without model files or
checkpoint pointers. The original runner applies further configuration,
dataset, software and checkpoint checks. It may restore recorded packages and
require a process/runtime restart. Re-run the same command after the restart.
Compatibility checks use the preserved historical protocol.

## Interpretation and scope

Five seed values represent repeated holdouts, not conventional five-fold
cross-validation. Filename-grouped partitions prevent overlap for the recorded
qualified keys, constrained split groups and exact pixel duplicates. The
image-level comparator intentionally permits group overlap. The original-folder
comparator uses source TRAIN images for train/tuning and source TEST images for
test; source VAL images are excluded from that comparator.

Filename-derived groups are not independently verified patient identities.
Bootstrap intervals are conditional on this grouping assumption. Stored hashes
check internal consistency; they do not independently establish provenance,
dataset identity at source, or validity of clinical labels. Without the original
images and excluded weights, the CPU checker cannot verify image bytes, load
the selected models, or regenerate predictions. It reproduces the downstream
analysis from the included saved predictions.
