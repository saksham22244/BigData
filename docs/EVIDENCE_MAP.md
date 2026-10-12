# Evidence map

Paths below are relative to the repository root. The `evidence/` directory contains 404 files covering the dataset audit, 16 model runs, reporting outputs and recovery records. Scientific outputs retain their original contents. Four supporting documents have been edited for repository presentation and are identified in `RELEASE_MANIFEST.json`.

## Code and execution

| Material | Path | Contents |
| --- | --- | --- |
| Complete notebook | `notebooks/pneumonia_analysis.ipynb` | Readable notebook with the analysis and saved-result workflow |
| Matching Python source | `src/pneumonia_analysis.py` | Full exported implementation; search named functions such as `parse_filename`, `make_partitions`, `select_threshold`, and `cluster_bootstrap` |
| Saved-result reproduction | `scripts/reproduce_saved_analysis.py` | Verification and recalculation from saved predictions |
| Pipeline entry point | `scripts/run_pipeline.py` | Figure export, new experiments and checkpoint recovery |
| Exact method explanation | `docs/METHODS_AND_GROUPING.md` | Parser precedence, component construction, split rules, seeds, selection, intervals, and limits |
| Recorded analysis configuration | `evidence/protocol.json` | Model settings, original implementation digest, software string values, seeds, threshold grid, and bootstrap attempt count |
| Recorded environment history | `evidence/session_environments.json` | Saved session timestamps, TensorFlow version, float policy, and device details |
| Dataset content fingerprint | `evidence/dataset_fingerprint.json` | Audited image count and combined dataset-content fingerprint |
| Task 1 reporting repair | `evidence/reporting_patch.json` | Inference repair version and source digest; states that training was unchanged |

The recorded Python version is `3.13.15`. See the repository installation instructions for the recorded environment and the separately documented environment used for package validation.

## Dataset, grouping, overlap, and duplicate evidence

| Question | Exact path | Content |
| --- | --- | --- |
| Which images and labels were audited? | `evidence/audit/manifest.csv` | Relative image paths, source folder, both task labels, parsing rule/fallback flag, raw and qualified keys, file/pixel hashes, and split groups |
| Which parser rules matched? | `evidence/audit/parsing_coverage.csv` | Rule-level counts, percentages, representative filenames/keys, and fallback flags |
| What is the identity interpretation? | `evidence/audit/grouping_interpretation.json` | Identity unverified; zero fallback images; no image-level fallback; duplicate policy |
| Which keys form each constrained group? | `evidence/audit/group_key_mapping.csv` | Image-to-raw-key, qualified-key, pixel-hash, and split-group mapping |
| How many groups and multi-image groups exist? | `evidence/audit/group_summary.csv` | Group totals, multi-image counts, maximum/median sizes, and mixed-subtype counts |
| What are the individual group sizes? | `evidence/audit/raw_key_distribution.csv`, `evidence/audit/qualified_key_distribution.csv`, `evidence/audit/split_group_distribution.csv` | Counts and subtype counts for each key/group |
| Which original TRAIN/TEST units overlap? | `evidence/audit/original_folder_overlaps.csv` | Raw, qualified, constrained-group, and exact-pixel overlap with explicit group/image denominators |
| Are raw-prefix collisions across subtype labels? | `evidence/audit/raw_cross_subtype_overlaps.csv` | TRAIN subtype by TEST subtype overlap matrix and affected-image counts |
| Which images are exact decoded duplicates? | `evidence/audit/exact_pixel_duplicates.csv` | All 62 participating image rows across 30 repeated pixel hashes |
| Did labels conflict for an exact image? | `evidence/audit/conflicting_exact_image_labels.csv` | Header-only table: no recorded conflicts |
| Did labels conflict within a derived component? | `evidence/audit/conflicting_group_labels.csv` | Header-only table: no recorded conflicts |
| Were invalid/ambiguous images found? | `evidence/audit/invalid_images.csv` | Header-only table: no recorded invalid images |

The table counts establish properties of the saved filenames and images. They do not establish person identity or etiologic label validity.

## Exact run directories

Every path in the next section uses one of these exact directories in place of `<run_id>`:

| Task/design | Run directories under `evidence/runs/` |
| --- | --- |
| Task 1 filename-grouped | `task1_filename_grouped_seed17`, `task1_filename_grouped_seed29`, `task1_filename_grouped_seed42`, `task1_filename_grouped_seed71`, `task1_filename_grouped_seed101` |
| Task 2 filename-grouped | `task2_filename_grouped_seed17`, `task2_filename_grouped_seed29`, `task2_filename_grouped_seed42`, `task2_filename_grouped_seed71`, `task2_filename_grouped_seed101` |
| Task 2 image-level | `task2_image_level_seed17`, `task2_image_level_seed29`, `task2_image_level_seed42`, `task2_image_level_seed71`, `task2_image_level_seed101` |
| Task 2 original-folder comparator | `task2_original_folders_seed42` |

## Artifacts present in every run

| Material | Path pattern | Contents |
| --- | --- | --- |
| Split assignments | `evidence/runs/<run_id>/split_assignments.csv` | Every eligible image's training/tuning/test assignment, label, hashes, and group keys |
| Split counts | `evidence/runs/<run_id>/split_counts.csv` | Class-specific image, qualified-filename-group, and constrained-unit counts |
| Partition overlaps | `evidence/runs/<run_id>/partition_overlaps.csv` | Pairwise overlap of raw, qualified, constrained, and pixel-hash keys |
| Run specification | `evidence/runs/<run_id>/run_configuration.json` | Run identity, protocol, dataset fingerprint, and partition signatures |
| Training specification | `evidence/runs/<run_id>/training_specification.json` | Actual model settings, completed epochs, and architecture/trainable-layer details |
| Epoch history | `evidence/runs/<run_id>/training_history.csv` | Saved training/tuning values by epoch and stage |
| Training plots | `evidence/runs/<run_id>/training_curves.png`, `evidence/runs/<run_id>/training_curves.pdf` | Training curves; checked against history by the accompanying signature file |
| Training-plot provenance | `evidence/runs/<run_id>/training_curves_signature.json` | History and plot hashes |
| Checkpoint provenance | `evidence/runs/<run_id>/training_complete.json` | Selected-weight digest, stage completion, checkpoint filenames/hashes, optimizer/callback/history metadata; payload files are not included |
| Tuning predictions | `evidence/runs/<run_id>/tuning_predictions.csv` | Per-image tuning probabilities and the selected threshold |
| Held-out predictions | `evidence/runs/<run_id>/test_predictions.csv` | Per-image supplied labels, probabilities, class decisions, threshold, positive class, and split group |
| Final metrics | `evidence/runs/<run_id>/test_metrics.csv` | One run's six held-out metrics and threshold |
| Threshold robustness | `evidence/runs/<run_id>/threshold_sensitivity.csv` | Selected threshold, 0.50, and selected threshold plus/minus 0.05 |
| Evaluation provenance | `evidence/runs/<run_id>/evaluation_complete.json` | Evaluation signature, predictions/probability digests, threshold, and metrics |
| Report provenance | `evidence/runs/<run_id>/reports_complete.json` | Saved report completion/signature evidence |
| Probability caches | `evidence/runs/<run_id>/tuning_probabilities.partial.csv`, `evidence/runs/<run_id>/test_probabilities.partial.csv` | Saved inference-cache rows; the `partial` filename is historical and does not supersede validated final prediction files |
| Probability-cache signatures | `evidence/runs/<run_id>/tuning_probabilities.partial.signature.json`, `evidence/runs/<run_id>/test_probabilities.partial.signature.json` | Cache signatures |

The five Task 1 directories additionally contain `task1_inference_revision.json`, recording the padded-batch inference repair, preserved selected weights, and names of prior artifacts archived outside this bundle.

## Repeated runs and confidence intervals

| Material | Path | Scope |
| --- | --- | --- |
| All final run metrics | `evidence/all_run_results.csv` | All 16 runs |
| Across-seed summary | `evidence/repeated_split_summary.csv` | Count, mean, sample standard deviation, minimum, and maximum for each metric across five seeds in each repeated design |
| Paired Task 2 comparison | `evidence/task2_paired_seed_differences.csv` | Filename-grouped minus image-level metrics, paired by seed |
| Combined split counts | `evidence/split_counts_all_runs.csv` | Counts across all run directories |
| Task 1 primary cluster intervals | `evidence/runs/task1_filename_grouped_seed42/bootstrap_intervals.csv` | Six metric intervals; 2,000 attempted and 2,000 effective draws |
| Task 2 primary cluster intervals | `evidence/runs/task2_filename_grouped_seed42/bootstrap_intervals.csv` | Six metric intervals; 2,000 attempted and 2,000 effective draws |
| Task 2 original-folder cluster intervals | `evidence/runs/task2_original_folders_seed42/bootstrap_intervals.csv` | Six metric intervals; 2,000 attempted and 2,000 effective draws |

Bootstrap tables are not present for the other 13 runs because they were not part of that interval schedule. Repeated-seed variability and test-set cluster-bootstrap uncertainty answer different questions.

## Figures and explanatory examples

| Material | Path | Scope |
| --- | --- | --- |
| Workflow figure | `evidence/Figure_2.png`, `evidence/Figure_2.pdf` | Recorded workflow figure |
| Dataset examples | `evidence/figures/dataset_examples.png`, `evidence/figures/dataset_examples.pdf` | Saved illustration of dataset images |
| Example-image provenance | `evidence/figures/dataset_examples.csv` | Selected source image IDs and labels |
| Confusion matrices | `evidence/runs/task1_filename_grouped_seed42/confusion_matrix.png`, `evidence/runs/task2_filename_grouped_seed42/confusion_matrix.png`, `evidence/runs/task2_original_folders_seed42/confusion_matrix.png` | Matching `.pdf` files are present in the same directories |
| ROC curves | `evidence/runs/task1_filename_grouped_seed42/roc_curve.png`, `evidence/runs/task2_filename_grouped_seed42/roc_curve.png`, `evidence/runs/task2_original_folders_seed42/roc_curve.png` | Matching `.pdf` files are present in the same directories |
| Task 1 occlusion selection | `evidence/runs/task1_filename_grouped_seed42/occlusion/occlusion_examples.csv` | Selected image IDs, correctness, pool sizes, thresholds, target class, and inference repair |
| Task 2 occlusion selection | `evidence/runs/task2_filename_grouped_seed42/occlusion/occlusion_examples.csv` | Selected image IDs, correctness, pool sizes, thresholds, and target class |
| Occlusion figures and metadata | `evidence/runs/task1_filename_grouped_seed42/occlusion/example_1.png` through `example_4.png`; same filenames in `evidence/runs/task2_filename_grouped_seed42/occlusion/` | Each has a matching `.pdf` and `.json`; maps are not validated localization |
| Task 1 probability-consistency check | `evidence/runs/task1_filename_grouped_seed42/occlusion/inference_consistency.csv` | Recorded agreement between saved and occlusion-base predictions |

## Bundle integrity, historical notes, and omitted resources

| Material | Exact path | Meaning |
| --- | --- | --- |
| Recorded required-artifact hashes | `evidence/artifact_verification.csv` | 193 required-artifact entries, not an inventory of every file in the repository |
| Recorded completion flags | `evidence/completion_status.json` | 16 evaluated runs; public-release, identity, manuscript-package, and submission flags remain false |
| Historical partial summary | `evidence/partial_run_results.csv` | Saved scheduler progress table; final results are `all_run_results.csv` |
| Methods notes | `evidence/METHODS_AND_LIMITATIONS.md` | Methods, limitations and reporting-repair description |
| Evidence overview | `evidence/BUNDLE_README.md` | Directory contents and checkpoint scope |
| Reviewer analysis map | `evidence/reviewer_requirement_checklist.csv`, `evidence/reviewer_response_working_draft.md` | Correspondence between reviewer topics, completed analyses and their scope |
| Legacy inventory | `evidence/legacy_evidence_inventory.csv` | Historical artifacts inventoried without silently importing them |
| Recovery check | `evidence/native_recovery_self_test.json` | Saved small recovery-check result, not a rerun of all models |
| Drive status | `evidence/drive_mount_status.json` | Historical generic Colab Drive locations; these paths must be configured for a new user's environment |

The full image dataset, pretrained-weight cache, native `.keras`/`.npz` checkpoint files, older evaluation archives and manuscript documents are held separately. Checkpoint filenames and `evaluation_archive/` entries in the historical JSON files record their original locations. Recalculation of saved metrics uses the included predictions. Model inference requires the corresponding weights and images; independent training may yield different numerical results.

Image paths in the manifest and split tables are relative to the dataset root. The code configuration and historical runtime records retain the original Colab Drive directory layout. The repository notebook has stored outputs cleared, with figure files available separately.
