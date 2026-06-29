# Submission CSV Files

This folder contains CSV outputs from the pothole segmentation workflow, including the final submission file, validation metrics, threshold-search results, and model comparison summaries.

## Files

| File | Description |
| --- | --- |
| `submission.csv` | Final prediction file with image IDs and run-length encoded segmentation masks. |
| `validation_metrics.csv` | Validation metrics for the selected post-processing configuration. |
| `train_val_metrics.csv` | Train-subset and validation metrics in one comparison table. |
| `threshold_search.csv` | Results from testing different post-processing parameter combinations. |
| `sample_model_comparison.csv` | Model comparison summary for individual models and the soft ensemble. |

## Submission Format

`submission.csv` uses the following columns:

| Column | Description |
| --- | --- |
| `ImageId` | Test image filename. |
| `rle` | Run-length encoded pothole segmentation mask. |

The `rle` column stores the predicted binary mask in run-length encoding format, which is commonly used for image segmentation submissions.

## Validation Metrics

`validation_metrics.csv` contains the selected validation result:

| Metric | Value |
| --- | ---: |
| IoU Pothole | 0.320 |
| IoU Background | 0.863 |
| mIoU | 0.592 |
| Dice | 0.451 |
| Pixel Accuracy | 0.874 |
| Precision | 0.452 |
| Recall | 0.619 |
| Macro F1 | 0.687 |

## Threshold Search

`threshold_search.csv` records different post-processing settings and their resulting metrics. It can be used to compare how threshold, component area, morphological kernels, hole filling, maximum area ratio, and largest-component filtering affect segmentation quality.

The selected default configuration is:

| Parameter | Value |
| --- | ---: |
| Threshold | 0.80 |
| Minimum component area | 1200 px |
| Close kernel | 3 |
| Open kernel | 3 |
| Fill holes | true |
| Maximum area ratio | 0.18 |
| Keep largest component | false |

## Notes

- Keep these CSV files unchanged when reproducing the submitted result.
- Regenerate these files from the notebook if the model, validation split, or post-processing configuration changes.
- The final submission file can be large because each image mask is stored as text-based RLE data.
