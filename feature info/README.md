# Feature Information Files

This folder contains supporting analysis files for the pothole segmentation workflow. These files help explain the dataset composition and the relative importance of extracted image features.

## Files

| File | Description |
| --- | --- |
| `dataset_audit.csv` | Dataset audit table with image paths, mask paths, image names, mask names, dataset source, mask ratio, height, and width. |
| `feature_importance_lgbm.csv` | LightGBM feature-importance summary for the engineered computer vision features. |

## Dataset Audit

`dataset_audit.csv` records metadata for the training images used in the audit.

| Column | Description |
| --- | --- |
| `id` | Image identifier. |
| `image_path` | Original image path from the training environment. |
| `mask_path` | Ground truth mask path from the training environment. |
| `image_name` | Image filename. |
| `mask_name` | Mask filename. |
| `source` | Dataset source label. |
| `mask_ratio` | Ratio of pothole pixels to total image pixels. |
| `height` | Image height in pixels. |
| `width` | Image width in pixels. |

Current audit summary:

| Item | Value |
| --- | ---: |
| Audited images | 498 |
| Source label | `ara` |
| Average mask ratio | 0.135 |
| Minimum mask ratio | 0.000 |
| Maximum mask ratio | 0.674 |

## Feature Importance

`feature_importance_lgbm.csv` lists feature names and their LightGBM importance scores. Higher scores indicate features that contributed more often or more strongly to LightGBM split decisions during training.

Top features in the current file include:

| Feature | Importance |
| --- | ---: |
| `y_norm` | 9732 |
| `illum_norm` | 5097 |
| `blackhat61` | 4818 |
| `wet_like` | 4186 |
| `x_norm` | 4148 |
| `mean15` | 3864 |
| `clahe` | 3606 |

## Dataset Source Note

The main segmentation dataset is ARA 7.0. RDD2022 India hard-negative images were sourced from [sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector) to help reduce false positives on non-pothole road surfaces.

## Notes

- These files are analysis outputs and are not loaded directly by the Streamlit app.
- Regenerate them from the training notebook if the dataset, feature extractor, or model changes.
- The paths inside `dataset_audit.csv` reflect the original training environment and may not exist on every local machine.
