# Release checks

`saved_analysis_verification.json` records CPU reproduction of the saved analysis. Split assignments, tuning-selected thresholds, test metrics and threshold-sensitivity tables reproduced for all 16 runs. The 193 files in the evidence ledger matched their recorded hashes and sizes. All three bootstrap interval tables matched separate 2,000-attempt recalculations within an absolute tolerance of `1e-10`.

Recalculations were written to a separate directory using the existing predictions. This check did not load or train TensorFlow models, regenerate predictions, or verify the raw image bytes and native checkpoints. The dataset verifier, `scripts/verify_dataset.py`, checks a separately acquired image tree against the recorded manifest.

The report records the actual CPU validation environment. A clean online installation of every pinned dependency and a new GPU training run were not performed during package preparation. Original GPU/runtime versions remain in `evidence/protocol.json` and `session_environments.json`. Hash consistency is not independent verification of patient identities or clinical labels.

The documentation revision leaves the code, numerical outputs, split assignments, predictions and machine-generated run records unchanged. Four supporting documents within `evidence/` have been edited for repository presentation; their filenames are listed in `RELEASE_MANIFEST.json`. That manifest records the current file checksums. `python scripts/verify_release.py` verifies the files against it.
