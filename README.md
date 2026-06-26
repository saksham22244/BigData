# Pediatric Chest X-ray Pneumonia Classification Using Transfer Learning

This repository contains a reproducible Python notebook workflow for pediatric chest X-ray pneumonia classification. The project uses structured image-manifest preparation, transfer learning, patient-aware evaluation, confidence-interval reporting, and publication-ready output artifacts.

The repository includes:

- `BigDataPnemonia.ipynb` - main analysis notebook
- `Output/pneumonia_Output/` - exported figures, tables, prediction files, metadata, leakage checks, and saved model files

This work is intended for research and educational use only. It is not a clinical diagnostic tool.

## Project Goal

The goal of this project is to build a clear and reproducible workflow for classifying pediatric chest X-ray images using deep learning.

The workflow focuses on three evaluation settings:

- **Task 1:** NORMAL vs PNEUMONIA classification using InceptionV3.
- **Task 2 patient-safe split:** BACTERIAL vs VIRAL pneumonia classification using DenseNet121 with patient-level separation.
- **Task 2 original/base-paper split:** BACTERIAL vs VIRAL pneumonia classification using the original paper-style split for comparison with a reference baseline.

The workflow helps answer:

- How can chest X-ray image files be organized into a clean analysis manifest?
- How do transfer-learning models perform on pneumonia detection and pneumonia subtype classification?
- How different are patient-safe and original-split evaluation results?
- Which output files are needed for transparent manuscript reporting?

## Dataset Source

This project uses the public pediatric chest X-ray dataset introduced by Kermany et al. The dataset contains anterior-posterior pediatric chest radiographs labeled as NORMAL or PNEUMONIA. Pneumonia images are further organized as BACTERIAL or VIRAL according to the dataset labels.

Original dataset article:

Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C. S., Liang, H., Baxter, S. L., McKeown, A., Yang, G., Wu, X., Yan, F., Dong, J., Prasadha, M. K., Pei, J., Ting, M. Y. L., Zhu, J., Li, C., Hewett, S., Dong, J., Ziyar, I., ... Zhang, K. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell*, 172(5), 1122-1131.e9. https://doi.org/10.1016/j.cell.2018.02.010

Please cite the original dataset paper if you use this workflow or dataset.

## Dataset Summary

The exported manifest reports:

| Dataset item | Count |
|---|---:|
| Total images | 5,856 |
| Total patients | 3,118 |
| NORMAL images | 1,583 |
| PNEUMONIA images | 4,273 |
| BACTERIAL pneumonia images | 2,780 |
| VIRAL pneumonia images | 1,493 |

## Main Packages

The notebook uses common Python machine-learning and medical-image analysis packages, including:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- TensorFlow / Keras
- Pillow
- json
- pathlib
- zipfile

The workflow was prepared for Google Colab with Google Drive mounted.

## Repository Structure

```text
BigData/
├── BigDataPnemonia.ipynb
├── README.md
└── Output/
    └── pneumonia_Output/
        ├── figures/
        ├── results/
        └── tables/
```

## How to Run

1. Open `BigDataPnemonia.ipynb` in Google Colab.
2. Mount Google Drive.
3. Confirm that the dataset path points to the pediatric chest X-ray dataset folder.
4. Run the notebook cells from top to bottom.
5. Review the generated artifacts inside `Output/pneumonia_Output/`.

Expected dataset folder structure:

```text
ChestXRay2017/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Workflow Modules

### 1. Project Setup

Defines paths, image size, random seed, batch size, model parameters, and output folders.

### 2. Manifest Construction

Creates a structured image manifest containing file paths, filenames, folder labels, source split labels, patient IDs, and pneumonia subtype labels.

### 3. Dataset Summary

Exports dataset composition tables and source-folder distribution summaries.

### 4. Patient-Level Leakage Checks

Checks train, validation, and test partitions for patient overlap and exports leakage-check logs.

### 5. Task 1: NORMAL vs PNEUMONIA

Uses an ImageNet-pretrained InceptionV3 feature extractor to classify chest X-rays as NORMAL or PNEUMONIA.

### 6. Task 2: BACTERIAL vs VIRAL Patient-Safe Split

Uses DenseNet121 with fine-tuning to classify pneumonia images as BACTERIAL or VIRAL under a patient-safe split.

### 7. Task 2: Original/Base-Paper Split

Runs a separate BACTERIAL vs VIRAL evaluation using the original paper-style split for comparison with the reported base-paper reference.

This comparison is useful, but the patient-safe split is the stricter leakage-aware estimate.

### 8. Prediction and Metric Export

Exports test-set prediction CSV files, final metrics, bootstrap confidence intervals, ROC curves, confusion matrices, and reproducibility metadata.

### 9. Occlusion-Based Model Inspection

Creates qualitative occlusion examples for correct and incorrect predictions. These maps are for model inspection only and should not be treated as validated anatomical disease localization.

## Main Results

The exported result tables report:

| Task | Evaluation split | Model | Accuracy | ROC-AUC |
|---|---|---|---:|---:|
| NORMAL vs PNEUMONIA | Patient-level test split | InceptionV3 | 95.21% | 98.69% |
| BACTERIAL vs VIRAL | Patient-safe test split | DenseNet121 | 80.24% | 85.52% |
| BACTERIAL vs VIRAL | Original/base-paper split | DenseNet121 | 92.05% | 94.87% |

The original/base-paper split result is above the reference accuracy of 90.70%, but it should be interpreted cautiously because it is not the same as the stricter patient-safe evaluation.

## Output Folder

All exported artifacts are stored in:

```text
Output/pneumonia_Output/
```

### Results

| File | Description |
|---|---|
| `full_pandas_manifest.csv` | Full image manifest generated by the notebook |
| `full_spark_manifest.csv` | Spark-style manifest copy used for reporting compatibility |
| `prediction_task1_test.csv` | Task 1 test-set predictions |
| `prediction_task2_test.csv` | Task 2 patient-safe test-set predictions |
| `prediction_task2_basepaper_original_split_test.csv` | Task 2 original/base-paper split predictions |
| `leakage_check_task1.txt` | Task 1 patient-overlap check |
| `leakage_check_task2.txt` | Task 2 patient-overlap check |
| `leakage_note_task2_basepaper_original_split.txt` | Note explaining the original/base-paper split |
| `reproducibility_metadata.json` | Machine-readable reproducibility metadata |
| `reproducibility_metadata.txt` | Human-readable reproducibility metadata |
| `inceptionv3_pneumonia_vs_normal.keras` | Saved Task 1 model |
| `densenet121_bacterial_vs_viral_patient_safe.keras` | Saved Task 2 patient-safe model |
| `publication_artifact_check.csv` | Output package audit table |
| `publication_artifact_check.txt` | Text version of the output package audit |

### Tables

| File | Description |
|---|---|
| `table_dataset_summary.csv` | Dataset-level summary |
| `table_source_folder_distribution.csv` | Source folder class counts |
| `table_task_splits.csv` | Train, validation, and test split counts |
| `table_task1_metrics_with_ci.csv` | Task 1 metrics with bootstrap 95% confidence intervals |
| `table_task2_metrics_with_ci.csv` | Task 2 patient-safe metrics with bootstrap 95% confidence intervals |
| `table_task2_basepaper_original_split_metrics_with_ci.csv` | Task 2 original/base-paper split metrics |
| `table_task2_basepaper_comparison.csv` | Base-paper comparison summary |
| `table_final_metrics_with_ci.csv` | Combined final metric table |
| `table_task2_training_history.csv` | DenseNet121 training-history table |
| `table_occlusion_examples.csv` | Occlusion example summary |

### Figures

The `figures/` folder contains PNG and PDF copies of the exported figures.

Main figure files include:

| File | Description |
|---|---|
| `fig1_sample_chest_xray_images.png` | Representative NORMAL, BACTERIAL, and VIRAL chest X-ray images |
| `fig_pipeline_diagram.png` | Workflow diagram |
| `pneumonia_vs_normal_best_saved_current_model_confusion_matrix.png` | Task 1 confusion matrix |
| `pneumonia_vs_normal_best_saved_current_model_roc_curve.png` | Task 1 ROC curve |
| `bacterial_vs_viral_densenet121_patient_safe_split_confusion_matrix.png` | Task 2 patient-safe confusion matrix |
| `bacterial_vs_viral_densenet121_patient_safe_split_roc_curve.png` | Task 2 patient-safe ROC curve |
| `bacterial_vs_viral_densenet121_base_paper_original_split_confusion_matrix.png` | Task 2 original/base-paper split confusion matrix |
| `bacterial_vs_viral_densenet121_base_paper_original_split_roc_curve.png` | Task 2 original/base-paper split ROC curve |
| `fig_occlusion_task1_*.png` | Task 1 occlusion examples |
| `fig_occlusion_task2_*.png` | Task 2 occlusion examples |
| `figure_dpi_manifest.csv` | Figure DPI audit file |

## Important Notes

- The notebook is designed for Google Colab and Google Drive.
- The dataset itself is not included in this repository.
- Model checkpoints may be large; if GitHub upload fails, store them with Google Drive, Zenodo, Figshare, or Git LFS.
- Do not mix manuscript numbers from one run with figures from another run.
- The patient-safe Task 2 result is the more conservative leakage-aware estimate.
- The original/base-paper split result is included only for comparison with prior reporting.
- Occlusion figures are qualitative interpretability aids, not clinical localization evidence.
- External validation was not performed, so this project should not be described as clinically ready.

## Suggested GitHub Description

Transfer-learning workflow for pediatric chest X-ray pneumonia classification using InceptionV3 and DenseNet121, with patient-aware evaluation, prediction exports, confidence intervals, 300 DPI figures, and occlusion-based model inspection.

## Citation

If you use this project, please cite the original dataset paper:

Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C. S., Liang, H., Baxter, S. L., McKeown, A., Yang, G., Wu, X., Yan, F., Dong, J., Prasadha, M. K., Pei, J., Ting, M. Y. L., Zhu, J., Li, C., Hewett, S., Dong, J., Ziyar, I., ... Zhang, K. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell*, 172(5), 1122-1131.e9. https://doi.org/10.1016/j.cell.2018.02.010

## Disclaimer

This repository is for research and educational purposes only. The models and outputs are not approved medical devices and must not be used for clinical diagnosis or treatment decisions.
