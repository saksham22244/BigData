# Methods, filename grouping, and interpretation

This repository provides the analysis code and saved evidence for evaluating the supplied chest-radiograph dataset labels. The analysis does not establish patient identity, microbiologically confirmed etiology, treatment selection, or clinical deployment performance.

The executable definitions are in `src/pneumonia_analysis.py` and the matching notebook. The recorded protocol is `evidence/protocol.json`; the subsequent Task 1 inference repair is documented separately in `evidence/reporting_patch.json` and each Task 1 run's `task1_inference_revision.json`. File paths in this document are relative to the repository root.

## Dataset records and labels

The manifest builder recursively considers `.jpg`, `.jpeg`, and `.png` files, ignoring `__MACOSX` paths and filenames beginning `._`. An eligible image must have a parent directory named NORMAL or PNEUMONIA and a grandparent named TRAIN, VAL, or TEST, compared without case. The builder requires exactly one dataset layout root. Unexpected image layouts, unreadable images, and pneumonia images without a parseable subtype are logged and stop the analysis.

Task 1 labels are the NORMAL/PNEUMONIA folder labels, with PNEUMONIA positive. Task 2 includes pneumonia images only; the filename is searched without case for `bacteria`, then for `virus`, producing BACTERIAL/VIRAL, with VIRAL positive. This substring label extraction is distinct from the grouping parser below. These labels are not independently adjudicated causes of disease.

The saved manifest contains 5,856 images: 1,583 NORMAL, 2,780 BACTERIAL, and 1,493 VIRAL. It records 5,232 TRAIN and 624 TEST images, with no VAL rows. These counts describe the audited snapshot represented by this repository; other distributions of the upstream dataset may differ. `image_id` is the relative source path; exported manifests and assignments omit the local `filepath` field.

## Exact filename-parser rules

The parser first takes `Path(filename).stem.lower()`. It tests the following patterns in order using `re.match`; patterns are anchored at the beginning but not at the end. Only the captured prefix is returned. `qualified_key` and `raw_key` are syntactic keys, not verified person identifiers.

| Order | Pattern | `qualified_key` | `raw_key` | Recorded rule |
| --- | --- | --- | --- | --- |
| 1 | `^(person\d+)_(bacteria|virus)_` | First capture + `_` + subtype capture | First capture only | `person_number_plus_subtype` |
| 2 | `^(normal\d+-im-\d+)` | Captured prefix | Same captured prefix | `normal_number_im` |
| 3 | `^(normal\d+-\d+)` | Captured prefix | Same captured prefix | `normal_number` |
| 4 | `^(im-\d+-\d+)` | Captured prefix | Same captured prefix | `im_number` |
| 5 | No preceding match | Stem before the first underscore | Same prefix | `unverified_prefix_fallback` |

Parser examples:

- `person100_bacteria_475.jpeg` gives qualified `person100_bacteria` and raw `person100`.
- `NORMAL2-IM-0007-0001.jpeg` gives `normal2-im-0007` for both keys. The final `-0001` is outside the matched prefix.
- `IM-0001-0001.jpeg` gives `im-0001-0001` for both keys. Both numeric components are retained by this rule.
- A hypothetical `NORMAL2-1234-extra.jpeg` would give `normal2-1234` for both keys under rule 3; this rule has no rows in the saved manifest.
- A hypothetical unmatched `unrecognized_extra.jpeg` would give `unrecognized` for both keys and set `parse_fallback=True`.

The fallback retains an explicitly unverified candidate key. It does not switch the grouped design to image-level splitting. In this saved analysis, 4,273 images use the person/subtype rule, 917 use `normal_number_im`, 666 use `im_number`, and zero use the fallback. Syntactic parsing coverage does not verify identity.

There are 3,118 raw keys and 4,097 qualified keys. Of the raw keys, 979 span both supplied pneumonia subtypes. Qualified subtype purity is partly imposed by the subtype-qualified construction. In the original TRAIN/TEST comparison, 170 raw keys occur in both folders, affecting 320 of 624 TEST images, whereas qualified keys, constrained split groups, and exact decoded-image hashes have zero TRAIN/TEST overlap. Raw-prefix collisions alone do not demonstrate that the same person crosses folders. Counts and denominators are preserved in `evidence/audit/original_folder_overlaps.csv` and `raw_cross_subtype_overlaps.csv`.

## Exact-duplicate constraints

Each file receives a SHA-256 hash of its encoded bytes. For duplicate detection, Pillow decodes the image and converts it to RGB; the pixel hash is SHA-256 of `str(rgb.size).encode() + rgb.tobytes()`. Thus the duplicate rule checks exact decoded RGB content and dimensions, not perceptual similarity or a near-duplicate tolerance.

The algorithm starts with one union-find node per qualified filename key. For every exact pixel hash, all qualified keys associated with that hash are joined. Connections are transitive. On each union, the lexicographically smaller root becomes the representative, yielding a deterministic representative for each connected component. Each final `split_group` is `fg_` followed by the first 24 hexadecimal characters of the SHA-256 hash of that representative key.

All images are retained, including same-label duplicate copies. Conflicting Task 1 or Task 2 labels for an exact image stop execution. The resulting component is also checked for conflicting labels and must be label-homogeneous. This produces 4,096 constrained split groups in the saved audit. The duplicate table contains 62 image rows across 30 repeated pixel hashes; this is a count of participating images, not 62 independent duplicate pairs. The invalid-image and both conflicting-label tables contain no data rows.

`evidence/audit/group_key_mapping.csv` connects every image, raw key, qualified key, pixel hash, and split group. The original-folder comparator retains its original TRAIN/TEST assignments; duplicate constraints do not relocate its TEST images.

## Partitions and repeated seeds

Before splitting, task images are sorted by `image_id`. Grouped splitting constructs one record per `split_group`, checks that each group has a single label, and uses scikit-learn `train_test_split` with class stratification on those group records. All images in a selected group follow its assignment.

| Design | First split | Second split | Eligible source rows |
| --- | --- | --- | --- |
| `filename_grouped` | Group-stratified 70% train / 30% remaining | Group-stratified 50% tuning / 50% test of remaining | All task-eligible manifest rows |
| `image_level` | Image-stratified 70% train / 30% remaining | Image-stratified 50% tuning / 50% test of remaining | All task-eligible manifest rows |
| `original_folders` | Group-stratified 85% train / 15% tuning within original TRAIN | Original TEST retained | Original TRAIN and TEST only; VAL excluded if present |

The same run seed is passed to both splitting calls. The nominal 70/15/15 fractions apply to groups in the grouped design; unequal group sizes and rounding change actual image fractions. The original-folder comparator's 85/15 split likewise applies to groups within TRAIN. Exact assignments, class counts, group counts, and pairwise overlaps are exported for every run.

The schedule contains 16 runs: Task 1 filename-grouped runs at seeds 17, 29, 42, 71, and 101; Task 2 filename-grouped and image-level runs paired at those five seeds; and one Task 2 original-folder comparator at seed 42. The filename-grouped seed-42 runs are the primary models. No Task 1 image-level or original-folder run is part of this schedule.

Validation requires each eligible image exactly once across train/tuning/test, disjoint image IDs, no duplicate IDs within a partition, and both classes in every partition. Filename-grouped partitions additionally require qualified keys, split groups, and pixel hashes to be disjoint across partitions. The image-level design does not impose those group constraints. Pairing seeds does not make the two designs' test-image sets identical.

The five repetitions reuse one dataset and are repeated holdouts, not five-fold cross-validation or five independent cohorts. Summary standard deviations are across the five seed-level metric values; paired differences are filename-grouped minus image-level metrics at each Task 2 seed. These are descriptive comparisons, not causal estimates of leakage.

## Models, selection, and inference

| Setting | Task 1 | Task 2 |
| --- | --- | --- |
| Backbone and input | ImageNet InceptionV3, 299 x 299 RGB | ImageNet DenseNet121, 320 x 320 RGB |
| Training batch | 32 | 16 |
| Head | Global average pooling, dropout 0.5, sigmoid output | Global average pooling, BatchNorm, dropout 0.4, Dense 256 ReLU, BatchNorm, dropout 0.3, sigmoid output |
| Maximum epochs | 30 head epochs; backbone frozen | 12 head epochs; then 25 fine-tuning epochs |
| Optimizer and loss | Adam, binary cross-entropy | Adam, binary cross-entropy |
| Learning rate | 0.001 | Head 0.001; fine-tuning 0.000003 |
| Selection | Minimum tuning loss | Maximum tuning AUROC within fine-tuning |
| Fine-tuning | None | Last 100 backbone layer positions considered; backbone BatchNorm remains frozen |
| Inference | Mean of three fixed image-ID-specific augmented passes | Single pass |

Both tasks use the architecture-specific Keras preprocessing function. Training uses class weights equal to training-image count divided by twice the class count; replication oversampling is disabled. Early-stopping patience is 10 for Task 1 (minimum improvement 0.0001) and 7 for Task 2 (minimum improvement 0); learning-rate reduction patience is 4 and 3, respectively. Actual epochs, trainable layers, and histories are recorded per run. The upper-layer position budget is not the number of trainable layers; the saved primary Task 2 model records 71 trainable backbone layers.

Task 1 augmentation uses rotation 12 degrees, width/height shift 0.08, zoom 0.15, shear 0.05, brightness 0.85–1.15, horizontal flip, and nearest fill. Task 2 uses rotation 8 degrees, width/height shift 0.05, zoom 0.10, brightness 0.90–1.10, horizontal flip, and nearest fill. Task 1's three transform seeds derive from the first four SHA-256 bytes of `run_seed|image_id|pass_index|tta-v1`. The same transforms apply to tuning, test, and occluded variants.

The recorded V3.1 inference repair fixes the Task 1 physical inference batch size to the configured training batch size, pads short batches with copies of the last input, and discards padding predictions. It preserves transforms and selected training weights but regenerates tuning/test probabilities, thresholds, and dependent reports. The preserved protocol's older `fixed_image_id_TTA_v1` field is supplemented by `reporting_patch.json`, which records `fixed_image_tta_v2_padded_physical_batch` and `training_changed=false`.

## Thresholds and metrics

Tuning predictions alone select the threshold from 0.10 through 0.90 inclusive in steps of 0.01, rounded to two decimal places. A probability equal to the threshold is positive. The primary criterion is maximum balanced accuracy, then maximum ordinary accuracy. The ascending search updates only for a strictly better pair of criteria, so a remaining tie selects the lower threshold. Test outcomes do not select this threshold.

The reported metrics are accuracy, sensitivity, specificity, precision, F1, and AUROC. AUROC uses probabilities; the other metrics use the selected threshold. Saved prediction tables identify the positive class and contain supplied labels, probabilities, decisions, thresholds, and split-group IDs. Prediction validation rejects missing, repeated, or extra images, incompatible labels/groups/positive classes, nonfinite/out-of-range probabilities, inconsistent thresholds, or decisions inconsistent with `probability >= threshold`.

Threshold-sensitivity tables evaluate the selected threshold, 0.50, and the selected threshold plus/minus 0.05, clipped to [0,1], rounded to two decimal places, and deduplicated. They are descriptive test-set sensitivity checks, not a second optimization or a clinical-utility analysis.

## Cluster bootstrap

Confidence intervals are recorded for the two filename-grouped seed-42 primary runs and the Task 2 original-folder seed-42 comparator. Each uses 2,000 attempted draws with `numpy.random.default_rng(42)`. A draw samples the same number of split groups as in that run's test set, uniformly with replacement. Every image from each drawn group is included, and drawing the same group repeatedly repeats all of its images. Group sizes are not used as sampling probabilities; metric values are computed across the resulting image rows.

The operating threshold stays fixed at its tuning-selected value. A draw containing only one class is excluded; the code fails if fewer than `min(100, attempted_replicates // 2)` valid draws remain. Intervals use the 2.5th and 97.5th pandas quantiles of valid metric draws. Attempted and effective counts are exported separately. All three saved interval tables record 2,000 attempted and 2,000 effective draws. These are conditional test-set intervals under the uncertain filename/duplicate grouping assumption; they do not resample training, quantify all training variability, or establish person-level uncertainty.

## Occlusion and remaining limits

Each primary model selects at most one test image per supplied-class/correctness stratum, sorting candidate image IDs and using selection seed 42. Selection-pool sizes are saved. Occlusion uses 48-pixel blocks and stride, filled with the median of the resized image. Correct examples explain the supplied/predicted class; incorrect examples explain the predicted class. Score decreases are clipped below zero and normalized for display. These maps probe model behavior and are not validated disease localization.

Task architectures, image sizes, augmentation, selection criteria, and inference differ; their performance difference does not isolate intrinsic task difficulty. Checkpoint recovery preserves optimizer, callback, and epoch state, but may not preserve the exact stochastic training/input-iteration state. Uninterrupted and resumed training can therefore differ at the bit level. The repository includes recorded predictions and checkpoint hashes, but excludes checkpoint payloads and the full radiograph dataset. Saved-result verification is possible without retraining; fresh inference requires compatible weights and source images, and new training requires the source dataset and pretrained weights.

`evidence/protocol.json` records Python `3.13.15`; its software metadata is preserved unchanged. The repository's installation instructions distinguish the recorded environment from the environment used for package validation. `evidence/completion_status.json` marks the 16 recorded analyses complete while keeping identity verification, public release verification, manuscript-package verification, and submission readiness false.
