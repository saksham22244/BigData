# Pediatric Chest X-ray Pneumonia Classification Using Transfer Learning

This repository contains a reproducible Python notebook workflow for pediatric chest X-ray pneumonia classification. The workflow builds a clean image manifest, prepares task labels, checks patient-level leakage, trains transfer-learning models, exports predictions, reports confidence intervals, and saves publication-style figures.

Main notebook:

```text
BigDataPnemonia.ipynb
```

This project is for research and educational use only. It is not a clinical diagnostic tool.

## Project Goal

The goal of this project is to provide a clear and reproducible deep-learning workflow for pediatric chest X-ray classification.

The notebook evaluates:

- **Task 1:** NORMAL vs PNEUMONIA classification using InceptionV3.
- **Task 2 patient-safe split:** BACTERIAL vs VIRAL pneumonia classification using DenseNet121.
- **Task 2 original-split comparison:** a secondary BACTERIAL vs VIRAL comparison using the original folder-style split.

The patient-safe split is emphasized because it is the stricter leakage-aware evaluation. It separates patients across train, validation, and test partitions so that the model is not evaluated on images from patients it has already seen during training.

## Dataset Source

This workflow uses the public pediatric chest X-ray dataset introduced by Kermany et al. The dataset contains anterior-posterior pediatric chest radiographs labeled as NORMAL or PNEUMONIA. Pneumonia images are further grouped as BACTERIAL or VIRAL according to the dataset labeling structure.

Original dataset article:

Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C. S., Liang, H., Baxter, S. L., McKeown, A., Yang, G., Wu, X., Yan, F., Dong, J., Prasadha, M. K., Pei, J., Ting, M. Y. L., Zhu, J., Li, C., Hewett, S., Dong, J., Ziyar, I., ... Zhang, K. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell*, 172(5), 1122-1131.e9. https://doi.org/10.1016/j.cell.2018.02.010

Please cite the original dataset paper if you use this dataset or workflow.

## Dataset Summary

The notebook manifest reports:

| Dataset item | Count |
|---|---:|
| Total images | 5,856 |
| Total patients | 3,118 |
| NORMAL images | 1,583 |
| PNEUMONIA images | 4,273 |
| BACTERIAL pneumonia images | 2,780 |
| VIRAL pneumonia images | 1,493 |

## Main Packages

The notebook uses common Python machine-learning and data-analysis packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- TensorFlow / Keras
- Pillow
- pathlib
- json
- zipfile

The workflow is designed to run in Google Colab with Google Drive mounted.

## Expected Dataset Structure

Before running the notebook, place the dataset in Google Drive and update the dataset path in the notebook if needed.

Example:

```text
ChestXRay2017/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## How to Run

1. Open `BigDataPnemonia.ipynb` in Google Colab.
2. Mount Google Drive.
3. Confirm that the dataset path points to the chest X-ray dataset folder.
4. Run the notebook cells from top to bottom.
5. Review the exported figures, results, tables, prediction files, and metadata.

## Workflow Modules

### 1. Project Setup

Defines paths, image size, batch size, random seed, model settings, and output folders.

### 2. Manifest Construction

Builds a structured image manifest with:

- filepath
- filename
- source split
- folder label
- pneumonia subtype
- patient ID
- task labels

### 3. Dataset Summary

Exports dataset counts, patient counts, source-folder distributions, and split summaries.

### 4. Patient-Level Leakage Checks

Checks whether train, validation, and test partitions share patients. The patient-safe split is used to reduce leakage risk and provide a stricter internal-test estimate.

### 5. Task 1: NORMAL vs PNEUMONIA

Uses an ImageNet-pretrained InceptionV3 feature extractor for binary pneumonia detection.

### 6. Task 2: BACTERIAL vs VIRAL Patient-Safe Split

Uses DenseNet121 with staged fine-tuning for pneumonia subtype classification under a patient-safe split.

### 7. Task 2: Original-Split Comparison

Runs a separate BACTERIAL vs VIRAL evaluation using the original folder-style split. This result is included as a secondary comparison only. The patient-safe result remains the main leakage-aware result.

### 8. Prediction Export

Exports test-set prediction CSV files with model probabilities, thresholds, predicted labels, true labels, patient IDs, and correctness flags.

### 9. Confidence Intervals

Calculates bootstrap 95% confidence intervals for major metrics such as accuracy, sensitivity, specificity, precision, recall, F1 score, and ROC-AUC.

### 10. Publication Figures

Saves figure outputs for:

- sample chest X-ray images
- workflow diagram
- training curves
- confusion matrices
- ROC curves
- occlusion-based explanation examples

### 11. Occlusion-Based Model Inspection

Creates qualitative occlusion examples for correct and incorrect predictions. These figures are used for model inspection only and should not be interpreted as validated disease localization.

## Main Results

The notebook output reports the following final results summary. The reference accuracy column is included only for context from the earlier reported setting and should not be interpreted as clinical superiority.

| Task | Final model | Test split | Reference accuracy | Notebook accuracy | Summary |
|---|---|---|---:|---:|---|
| NORMAL vs PNEUMONIA | InceptionV3 frozen feature extractor | 919 patient-independent images | 92.8% | 95.21% | Strong binary pneumonia screening result |
| BACTERIAL vs VIRAL comparison experiment | DenseNet121 with fine-tuning | 390 original TEST-folder images | 90.7% | 92.05% | Original-split comparison experiment |

The BACTERIAL vs VIRAL row follows the original TEST-folder comparison protocol shown in the notebook summary. The notebook also reports a stricter patient-safe BACTERIAL vs VIRAL evaluation, where DenseNet121 achieved 80.24% accuracy and 85.52% ROC-AUC after separating patient IDs across splits. This patient-safe value is reported as leakage-aware internal evidence and is not used for direct reference-accuracy comparison in the table above.

## Main Output Files

Depending on the run, the notebook exports artifacts such as:

### Results

| File | Description |
|---|---|
| `full_pandas_manifest.csv` | Full image manifest |
| `full_spark_manifest.csv` | Spark-style manifest copy for reporting compatibility |
| `prediction_task1_test.csv` | Task 1 test predictions |
| `prediction_task2_test.csv` | Task 2 patient-safe test predictions |
| `prediction_task2_basepaper_original_split_test.csv` | Task 2 original-split comparison predictions |
| `leakage_check_task1.txt` | Task 1 patient-overlap check |
| `leakage_check_task2.txt` | Task 2 patient-overlap check |
| `reproducibility_metadata.json` | Machine-readable reproducibility metadata |
| `reproducibility_metadata.txt` | Human-readable reproducibility metadata |

### Tables

| File | Description |
|---|---|
| `table_dataset_summary.csv` | Dataset summary |
| `table_source_folder_distribution.csv` | Source-folder class distribution |
| `table_task_splits.csv` | Train, validation, and test split counts |
| `table_task1_metrics_with_ci.csv` | Task 1 metrics with bootstrap confidence intervals |
| `table_task2_metrics_with_ci.csv` | Task 2 patient-safe metrics with bootstrap confidence intervals |
| `table_task2_basepaper_original_split_metrics_with_ci.csv` | Task 2 original-split comparison metrics |
| `table_task2_basepaper_comparison.csv` | Original-split comparison summary |
| `table_final_metrics_with_ci.csv` | Combined final metric table |
| `table_occlusion_examples.csv` | Occlusion example summary |

### Figures

| File | Description |
|---|---|
| `fig1_sample_chest_xray_images.png` | Representative NORMAL, BACTERIAL, and VIRAL X-ray images |
| `fig_pipeline_diagram.png` | Analysis workflow diagram |
| `pneumonia_vs_normal_training_curves.png` | Task 1 training curves |
| `pneumonia_vs_normal_best_saved_current_model_confusion_matrix.png` | Task 1 confusion matrix |
| `pneumonia_vs_normal_best_saved_current_model_roc_curve.png` | Task 1 ROC curve |
| `task2_densenet121_training_curves.png` | DenseNet121 training curves |
| `bacterial_vs_viral_densenet121_patient_safe_split_confusion_matrix.png` | Task 2 patient-safe confusion matrix |
| `bacterial_vs_viral_densenet121_patient_safe_split_roc_curve.png` | Task 2 patient-safe ROC curve |
| `bacterial_vs_viral_densenet121_base_paper_original_split_confusion_matrix.png` | Task 2 original-split comparison confusion matrix |
| `bacterial_vs_viral_densenet121_base_paper_original_split_roc_curve.png` | Task 2 original-split comparison ROC curve |
| `fig_occlusion_task1_*.png` | Task 1 occlusion examples |
| `fig_occlusion_task2_*.png` | Task 2 occlusion examples |

## Important Notes

- The dataset itself is not included in this repository.
- Run the notebook in the same environment if you need to reproduce the same numbers.
- Do not mix figures from one run with metric tables from another run.
- The patient-safe Task 2 result should be treated as the primary leakage-aware estimate.
- The original-split comparison is secondary and should not be used to claim clinical readiness.
- Occlusion maps are qualitative model-inspection outputs, not clinical localization evidence.
- External validation was not performed.
- Large model checkpoint files may be difficult to upload to GitHub. Use Google Drive, Zenodo, Figshare, or Git LFS if needed.

## Suggested GitHub Description

Transfer-learning workflow for pediatric chest X-ray pneumonia classification using InceptionV3 and DenseNet121, with patient-safe evaluation, leakage checks, prediction exports, confidence intervals, publication figures, and occlusion-based model inspection.

## Citation

If you use this project, please cite the original dataset article:

Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C. S., Liang, H., Baxter, S. L., McKeown, A., Yang, G., Wu, X., Yan, F., Dong, J., Prasadha, M. K., Pei, J., Ting, M. Y. L., Zhu, J., Li, C., Hewett, S., Dong, J., Ziyar, I., ... Zhang, K. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell*, 172(5), 1122-1131.e9. https://doi.org/10.1016/j.cell.2018.02.010

## Disclaimer

This repository is for research and educational purposes only. The models and outputs are not approved medical devices and must not be used for clinical diagnosis or treatment decisions.
