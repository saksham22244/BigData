# Dataset acquisition and snapshot verification

## Reference source

The upstream reference is Kermany, Zhang and Goldbaum, **Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification**, Mendeley Data, version 2, published 6 January 2018, DOI [10.17632/rscbjbr9sj.2](https://data.mendeley.com/datasets/rscbjbr9sj/2). The landing page lists a CC BY 4.0 license. A separate [version 3](https://data.mendeley.com/datasets/rscbjbr9sj/3) also exists; equivalence between versions requires a comparison of their contents.

Related source paper: Kermany DS et al. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell. 2018;172:1122–1131.e9. [DOI: 10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010).

The historical download date, upstream archive checksum and archive version used in the analysis are not recorded in the saved evidence. Version 2 serves as an acquisition reference; the version of the historical download remains unverified. The image-level manifest below defines the analyzed snapshot. For a new download, record the source URL, selected version, download date and downloaded archive SHA-256.

## Analyzed snapshot

| Item | Recorded value |
|---|---|
| Total images | 5,856 |
| Original TRAIN images | 5,232 |
| Original TEST images | 624 |
| VAL rows in the saved manifest | 0 |
| Image-level identity and labels | `evidence/audit/manifest.csv` |
| Snapshot identity | `evidence/dataset_fingerprint.json` |
| Content fingerprint | `7cedb1d089c850957118f56f91aa50e46b33474fc7d5e48dcb07b0f138dee290` |

The fingerprint is SHA-256 of a UTF-8, LF-terminated CSV serialization of the ordered columns `image_id`, `file_sha256`, `pixel_sha256`, `qualified_key`, `split_group`, `label_task1`, `label_task2`, with missing values represented by empty strings. It is not the checksum of the original download archive.

## Acquisition steps

1. Open the version-specific upstream page and download the chest X-ray component used in this study. Retain the original archive and its accompanying documentation. The OCT component is outside the study scope.
2. Extract it to a new directory. Point the verifier to the directory immediately containing `train/` and `test/`. The exact relative image paths must match `image_id` in the saved manifest, for example `test/NORMAL/IM-0001-0001.jpeg`.
3. Run the read-only verification command from the repository root:

   ```bash
   python scripts/verify_dataset.py --dataset /path/to/chest_xray
   ```

4. To also check the decoded RGB-image hashes used in the duplicate audit, run:

   ```bash
   python scripts/verify_dataset.py --dataset /path/to/chest_xray --decoded
   ```

   Pillow is required for `--decoded`. The original decoder version was Pillow 11.3.0. Matching file bytes with differing decoded hashes indicate a discrepancy in decoding that requires investigation; the saved manifest remains the verification reference.

5. Confirm that the snapshot matches before figure export or a new experiment. The verifier writes nothing to the dataset or evidence directory. A missing image, extra image, wrong relative path, changed byte hash or decoded hash mismatch causes failure.

Mirrors may redistribute some TRAIN images into a `val/` directory, duplicate nested folders or contain a different image count. Reproducing the study snapshot requires the recorded original-folder assignments, so merging `val/` into TRAIN alone does not establish a match. An archive that fails verification requires file-level comparison with the manifest and provenance review. Any reconstruction should have a separate record. A different dataset requires a new experiment with its own evidence.

The general audit code reports TRAIN/TEST overlap using those source folders only; validation-folder images are excluded from those denominators. The supplied study manifest contains no VAL rows. These filenames do not establish patient identity, and the subtype labels do not establish uniformly microbiologically confirmed diagnoses.
