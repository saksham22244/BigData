# Figure previews from the completed export

The seven PNG panels were exported by the completed analysis notebook. Their dimensions, resolution metadata and source checksums are recorded in `provenance.json`. The repository notebook contains the code with stored outputs cleared; the figure files are retained separately.

| File | Contents |
|---|---|
| `Fig1.png` | Sample NORMAL, BACTERIAL PNEUMONIA and VIRAL PNEUMONIA radiographs |
| `Fig2.png` | Study workflow; separate from the S1 reporting checklist |
| `Fig3.png` | Task 1 filename-grouped seed-42 confusion matrix and ROC curve |
| `Fig4.png` | Task 2 filename-grouped seed-42 confusion matrix and ROC curve |
| `Fig5.png` | Controlled Task 2 comparison across five paired seeds |
| `Fig6.png` | Task 1 saved occlusion examples |
| `Fig7.png` | Task 2 saved occlusion examples |

The PNG resolution metadata is 300 DPI (stored as approximately 299.9994 DPI by the PNG pixels-per-metre conversion). The full notebook exports TIFF and PDF versions and separate captions. Existing occlusion raster detail is not increased merely by exporting at 300 DPI. Selection records and source occlusion assets remain under `evidence/runs/`.

BACTERIAL/VIRAL are supplied dataset labels. “Incorrect” denotes a model decision that differs from the supplied label. Patient identities are not independently verified. Dataset attribution and image transformations are documented in [LICENSE_NOTES.md](../LICENSE_NOTES.md).

`provenance.json` records the executed notebook's checksum, output positions, image dimensions and image checksums. Absolute local paths have been replaced with repository-relative paths in this derived provenance record; the image bytes are unchanged.
