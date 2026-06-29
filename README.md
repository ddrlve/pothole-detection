# Pothole Detection - Classical Computer Vision Segmentation

This project is a Streamlit application for pixel-wise pothole segmentation using classical machine learning. It combines hand-crafted computer vision features with LightGBM, XGBoost, and CatBoost models to detect pothole regions without using deep learning.

Live application: [pothole-detection-cv.streamlit.app](https://pothole-detection-cv.streamlit.app/)

## Overview

The application takes a road image as input and produces:

- A binary pothole segmentation mask
- An overlay of the prediction on the original image
- A probability heatmap
- Optional evaluation metrics when a ground truth mask is uploaded
- A downloadable predicted mask in PNG format

The app supports two inference modes:

| Mode | Description |
| --- | --- |
| Real-time | Uses the fast single model for lower latency. |
| Accuracy | Uses the full model artifact intended for the best segmentation quality. |

## Main Features

- Pixel-wise pothole segmentation
- Classical machine learning pipeline, not deep learning
- 36 image features per pixel, including color, texture, gradients, local statistics, spatial priors, and scene context
- Automatic road-region filtering through a road mask feature
- Configurable post-processing: thresholding, morphological filtering, hole filling, area filtering, and maximum predicted-area control
- Optional validation against a ground truth mask
- Streamlit web interface with image upload and result preview

## Technology Stack

| Component | Tools |
| --- | --- |
| Web application | Streamlit |
| Image processing | OpenCV, Pillow, NumPy |
| Feature extraction | OpenCV and scikit-image |
| Machine learning | LightGBM, XGBoost, CatBoost, scikit-learn |
| Model storage | Pickle artifacts and JSON configuration |

## Dataset Notes

The model artifacts were prepared for pothole segmentation using:

| Dataset | Role |
| --- | --- |
| ARA 7.0 | Main pothole segmentation dataset |
| RDD2022 India | Hard-negative road images without potholes |

The additional hard negatives help reduce false positives on normal road surfaces.

## Model Pipeline

The inference pipeline follows this flow:

```text
Input image
  -> resize to working resolution
  -> extract per-pixel feature map
  -> predict pothole probability
  -> apply road mask and post-processing
  -> output segmentation mask, overlay, and heatmap
```

The feature extractor uses the following feature groups:

| Feature group | Examples |
| --- | --- |
| RGB color | R, G, B |
| HSV color | Hue, saturation, value |
| LAB color | L, A, B channels |
| Intensity | Grayscale, CLAHE, illumination-normalized intensity |
| Edges and gradients | Sobel magnitude, Sobel angle, Laplacian |
| Local statistics | Local mean and standard deviation |
| Texture | LBP, blackhat transforms, Gabor filters |
| Spatial priors | Normalized x/y position, bottom prior, center distance |
| Scene context | Road mask, wet-like, shadow-like, dark-edge, specular-like indicators |

## Validation Metrics

The following validation metrics are stored in `pothole_output/pothole_config.json` and `submission csv/validation_metrics.csv`.

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

## Project Structure

```text
pothole_cv_new/
|-- app.py
|-- requirements.txt
|-- README.md
|-- feature info/
|   |-- dataset_audit.csv
|   `-- feature_importance_lgbm.csv
|-- notebook/
|   `-- pothole-cv-xgb-cat-lightgbm.ipynb
|-- pothole_output/
|   |-- README.md
|   |-- pothole_config.json
|   |-- pothole_model.pkl
|   |-- pothole_model_accuracy.pkl
|   |-- pothole_model_fast.pkl
|   |-- train_items_30.pkl
|   `-- val_items_accuracy.pkl
`-- submission csv/
    |-- README.md
    |-- sample_model_comparison.csv
    |-- submission.csv
    |-- threshold_search.csv
    |-- train_val_metrics.csv
    `-- validation_metrics.csv
```

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Running Locally

Start the Streamlit application from the project root:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

The app automatically looks for model artifacts in the `pothole_output` folder.

## How to Use the Application

1. Open the live app or run the project locally.
2. Upload a road image in JPG, PNG, BMP, WebP, or TIFF format.
3. Choose the inference mode from the sidebar.
4. Adjust post-processing parameters if needed.
5. Click `Run Detection`.
6. Review the original image, predicted mask, overlay, and probability heatmap.
7. Optionally upload a ground truth mask to calculate evaluation metrics.
8. Download the predicted mask if needed.

## Post-processing Parameters

| Parameter | Default | Purpose |
| --- | ---: | --- |
| Probability threshold | 0.80 | Minimum probability required for a pixel to be classified as pothole. |
| Minimum component area | 1200 px | Removes small noisy connected components. |
| Close kernel | 3 | Fills small gaps in predicted regions. |
| Open kernel | 3 | Removes small isolated noise. |
| Fill holes | true | Fills holes inside predicted pothole regions. |
| Maximum predicted area ratio | 0.18 | Prevents over-labeling large image regions as potholes. |
| Keep largest component | false | Optionally keeps only the largest connected component. |

If the predicted area exceeds the maximum allowed ratio, the application increases the threshold automatically until the prediction is within the configured limit.

## Repository Notes

- `pothole_output/` contains model artifacts and configuration used by the Streamlit app.
- `submission csv/` contains submission output, validation metrics, threshold-search results, and model comparison files.
- `notebook/` contains the experiment and training notebook.
- `feature info/` contains supporting feature and dataset analysis files.

## Limitations

This project uses classical computer vision and machine learning. It may be sensitive to lighting changes, unusual road textures, shadows, water reflections, and camera perspective. For best results, use clear road images where pothole boundaries are visible.
