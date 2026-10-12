# Reviewer comments and analysis correspondence

This record maps reviewer and journal topics to the analysis evidence and its scope.
The recorded analysis verified all 16 expected evaluated runs (16/16). Numerical
completion does not describe the status of manuscript editing or publication tasks.

## R1 major 1 / R2 #4: Dataset-derived subtype labels
Metadata and methods notes distinguish supplied labels from microbiological etiology.
The dataset does not support antibiotic-treatment or verified-unseen-patient claims.

## R1 major 2 / R2 #1: Grouping reliability and duplicate policy
Parsing coverage, examples, fallbacks, group distributions, exact-image conflicts and duplicate constraints are audited.
Patient identity cannot be verified from filenames.

## R1 major 3 / R2 #6: Threshold criterion and robustness
Threshold selection maximizes tuning balanced accuracy, with accuracy and then the lower threshold as tie-breakers. Test tables report 0.50 and +/-0.05 operating points.
These analyses do not optimize clinical utility.

## R1 major 4 / R2 #3: Repeated controlled comparisons
The 16 verified evaluations comprise five Task 1 grouped runs, five paired Task 2 image-level and filename-grouped runs, and one original-folder comparator.
Repeated-split summaries are descriptive and reuse a single dataset.

## R1 major 5: Replication oversampling
Replication oversampling is disabled; balanced training class weights are used consistently.
This differs from the earlier manuscript experiments, where replication introduced a risk of memorizing repeated training examples.

## R1 major 6: Fine-tuning reproducibility
The DenseNet upper 100 layer positions, frozen BatchNorm layers, stage budgets, actual epochs, learning rate and trainable layers are recorded.
The exported settings describe the completed analyses.

## R1 minor 1: 90.70 percent comparator citation
The unverified numeric literature benchmark is excluded.
Its source and endpoint are not established by this analysis.

## R1 minor 2: Occlusion block size and stride
The recorded procedure uses a 48-pixel block and stride, median fill and consistent inference.
The maps support qualitative model-behavior analysis, without validated disease-localization claims.

## R1 minor 3: Figure 2 versus S1
Figure 2 presents the workflow; S1 is a separate checklist.
Manuscript and Supporting Information cross-references are editorial matters outside the numerical analysis.

## R1 minor 4: Repeated wording
Reporting notes use descriptive framing.
Abstract and Discussion wording is outside the numerical analysis.

## R2 #2: Original TRAIN/TEST overlap
The audit reports raw and qualified filename keys, exact-image duplicates, category denominators and affected images. The original VAL folder is excluded from the comparator.
Neither filename-key definition verifies patient identity.

## R2 #5: Cluster confidence intervals
Intervals use 2000 attempted whole-constrained-filename-group bootstrap draws, with effective valid counts and percentile intervals exported.
Uncertainty estimates are conditional on the filename-grouping assumption.

## R2 #7: Task 1 versus Task 2 interpretation
Differences in architecture, resolution, augmentation and test-time augmentation are documented.
The comparison is descriptive and does not establish intrinsic task difficulty.

## R2 #8: Public code and derived data
Portable assignments, probabilities, seeds, configuration, histories and metrics are exported.
Public repository deposition, persistent identifiers and availability statements are outside the numerical analysis.

## R2 #9: Novelty and related work
The study is framed as a dataset audit and evaluation-design replication.
Ouerhani et al., doi:10.3390/jimaging12020081, is a related-work reference; assessment of its relevance is outside this numerical analysis and is not recorded as complete.

## R2 additional 1: Terminology
The reporting uses filename-grouped evaluation, AUROC and chest radiograph, without claiming identity verification.
Manuscript-wide terminology editing is outside the numerical analysis.

## R2 additional 2: Sensitivity and recall
Reports include sensitivity once, with no duplicate recall result column.
The numerical evidence does not establish the layout of manuscript tables.

## R2 additional 3: Task 2 positive class
VIRAL is the explicit positive class in prediction tables, metric tables and figure labels.
Manuscript caption editing is outside the numerical analysis.

## R2 additional 4: Screening claims
The evaluation does not establish population-screening performance, treatment selection or clinical-deployment suitability.
Manuscript screening language is outside the numerical analysis.

## R2 additional 5: Occlusion example selection
Examples are selected with a fixed seed by supplied class and correctness; selection and mapping use the same predictor.
The maps probe model behavior and do not validate disease localization.

## R2 additional 6: Methods specificity
Actual data, split, threshold and uncertainty settings are exported.
The structure and wording of the manuscript Methods section are outside the numerical analysis.

## Journal 1: Style and naming
Artifacts have descriptive, stable filenames.
Journal manuscript style compliance is outside the numerical analysis.

## Journal 2: Figure 2 replacement
Figure 2 is available as a 300-DPI PNG and vector PDF.
Replacement of the submitted figure is a publication task outside the numerical analysis.

## Journal 3: Response / marked / clean manuscript
The evidence checklist and this correspondence record document the relationship between comments and available analyses.
A final response letter and marked and clean manuscripts are editorial documents outside the numerical analysis.

## Journal 4: Suggested citations and disclosures
Suggested related work and disclosure statements concern literature assessment and publication documentation.
Their assessment and final wording are outside the numerical analysis; completion of the evaluations does not establish their completion.
