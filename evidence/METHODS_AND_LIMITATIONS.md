# Methods and interpretation notes

This is an internally evaluated dataset audit / evaluation-design replication. BACTERIAL and VIRAL are supplied filename-derived dataset labels, not a uniform microbiological reference standard. No treatment-selection, verified-unseen-patient or clinical-deployment claim is supported.

Pneumonia keys are subtype-qualified (personN_bacteria versus personN_virus). Their subtype purity is partly imposed by construction. Raw-prefix overlap does not establish shared patient identity. Neither candidate key independently verifies patient identity. Parsing coverage quantifies syntactic extraction, not clinically unambiguous IDs.

Generated grouped partitions keep qualified filename keys together and join keys linked by identical decoded RGB images. All same-label copies remain; contradictory exact-image labels stop execution. These exact-duplicate constraints are a documented change from the previous notebook. Original TRAIN/TEST assignments are preserved for the descriptive comparator; its original VAL folder is excluded. Raw, qualified, exact-image and constrained-unit overlaps are reported separately.

Task 1 has five filename-grouped repetitions. Task 2 has five paired seeds across image-level and filename-grouped designs. Nominal fractions are 70/15/15, with actual image and group counts exported because group sizes vary. The original-folder comparator uses original TRAIN split into 85/15 training/tuning and original TEST. Five repeated splits reuse one dataset and are not independent external cohorts. Mean/SD/range and paired differences are descriptive, not causal estimates of leakage.

Grouped seed 42 models are reused for the main results. Task 1 uses InceptionV3, 299x299 images, a frozen backbone, dropout 0.5, 30 maximum epochs and minimum tuning-loss selection. Task 2 uses DenseNet121, 320x320 images, 12 head-training and 25 fine-tuning maximum epochs, 100 upper backbone layer positions with backbone BatchNorm frozen, and fine-tuning LR 3e-6. Actual epochs and trainable layers are exported. All Task 2 designs use the best tuning-AUROC weights within fine-tuning. This changes the earlier single-split across-both-stages selection rule and is versioned accordingly.

Replication oversampling is disabled and balanced training class weights are used. This differs from the earlier manuscript experiments, which used replication oversampling and therefore carried a risk of memorizing repeated training examples. Task 1 and Task 2 differ in architecture, resolution, augmentation, training and inference; their performance comparison is descriptive rather than proof of intrinsic task difficulty.

Threshold selection maximizes tuning balanced accuracy on 0.10 through 0.90 by 0.01. Accuracy breaks ties, then the lower threshold. It does not optimize clinical utilities. Default 0.50 and +/-0.05 test robustness tables are descriptive and are not used to retune on test data. VIRAL is the positive class for Task 2; PNEUMONIA is positive for Task 1.

Primary and original-folder confidence intervals use 2000 attempted whole-constrained-filename-group bootstrap draws, retain every image in each sampled group and preserve repeated draws. Percentile 2.5/97.5 intervals and effective valid two-class counts are exported. These intervals are conditional on the uncertain grouping assumption.

Task 1 inference averages three deterministic image-ID-specific augmentation passes. The same fixed transforms are used for tuning, test inference and occluded variants. This versioned inference procedure differs from the earlier TTA implementation, so outputs from the two versions are not interchangeable. Occlusion samples at most one image from each supplied-class/correctness stratum using seed42, uses 48-pixel blocks/stride and median fill, and reports its selection pool. Incorrect examples explain the predicted class. Maps probe model behavior; they are not validated disease localization.

Checkpoints restore optimizer, callback and epoch state, but not exact stochastic training/input-iteration state. Recovery events are recorded; bitwise equivalence to an uninterrupted full training run is not claimed. Legacy metric rows alone are not reproducibly complete. Missing predictions are regenerated from compatible completed stages; insufficient legacy evidence requires a new run.

The unverified 90.70% literature benchmark is excluded. Ouerhani et al., doi:10.3390/jimaging12020081, is a related-work reference; this numerical analysis does not include a completed assessment of that work. Figure 2 presents the workflow, separately from the S1 checklist. The reviewer correspondence documents map comments to analysis outputs and their scope. Manuscript editing, literature assessment, and publication status are outside the numerical evaluation.


V3.1 reporting repair: Task 1 tuning, test and occlusion inference use one shared
implementation with the configured training batch size as a fixed physical
inference batch shape. Short batches are padded with copies of the final input;
padding predictions are discarded. The original three image-ID-specific transforms
are unchanged. Prior Task 1 evaluation artifacts are archived, then predictions,
tuning-selected thresholds and reports are regenerated from the same selected
weights. Training, split assignments and training checkpoint signatures are
unchanged; reporting_patch.json records the repair source hash separately.
The original GPU mismatch was observed in an occlusion check; its numeric size was
not logged, so batch-dependent numerics remain a plausible cause, not a verified
diagnosis. Strict consistency and classification checks remain in place, with
per-example numeric diagnostics. Clinical results are not inferred from test data.
