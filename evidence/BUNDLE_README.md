# Analysis evidence bundle

This directory contains evaluation outputs and their audit records. The recorded analysis
completed and verified all 16 expected evaluations, with no missing runs, as documented in
`completion_status.json`. Its manuscript-package and public-release verification fields
record the status at the time of the analysis; those activities are outside the numerical
evaluation.

The evidence bundle excludes superseded evaluation archives, temporary files, downloaded
caches and native model snapshots. Current progress tables and probability-cache records
are retained alongside the final predictions. Native checkpoints and the original training
environment are separate from these exported analysis records.

The evaluation uses filename-derived groups. Qualified-key purity and raw-prefix overlaps
do not verify patient identity. BACTERIAL and VIRAL are supplied dataset labels rather than
uniformly microbiologically confirmed causes. Repeated-split comparisons are descriptive.

`METHODS_AND_LIMITATIONS.md` describes the evaluation design and interpretation limits.
`reviewer_response_working_draft.md` maps reviewer topics to the available analysis evidence;
`reviewer_requirement_checklist.csv` provides the same correspondence in tabular form.
