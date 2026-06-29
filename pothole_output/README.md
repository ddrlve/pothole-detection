# Pothole Output Artifacts

This folder contains the trained model artifacts and configuration files used by the Streamlit pothole detection application.

## Files

| File | Description |
| --- | --- |
| `pothole_config.json` | Main configuration file. It stores feature names, default post-processing parameters, validation metrics, model filenames, and dataset notes. |
| `pothole_model_fast.pkl` | Fast inference model used by the Real-time mode in the application. |
| `pothole_model_accuracy.pkl` | Accuracy-focused model artifact used by the Accuracy mode in the application. |
| `pothole_model.pkl` | Main fallback model artifact. |
| `train_items_30.pkl` | Serialized training subset items used for analysis or reproducibility checks. |
| `val_items_accuracy.pkl` | Serialized validation items used for accuracy evaluation. |

## How the Application Uses This Folder

When `app.py` starts, it searches for this folder and loads:

1. `pothole_config.json`
2. `pothole_model_fast.pkl` when Real-time mode is selected
3. `pothole_model_accuracy.pkl` when Accuracy mode is selected
4. `pothole_model.pkl` as a fallback model when needed

The app expects this folder to remain named `pothole_output` unless the loading logic in `app.py` is updated.

## Configuration Summary

The current configuration uses:

| Setting | Value |
| --- | ---: |
| Training work size | 320 |
| Application work size | 256 |
| Default threshold | 0.80 |
| Minimum component area | 1200 px |
| Maximum predicted area ratio | 0.18 |

The configuration also stores validation metrics, including IoU, Dice, pixel accuracy, precision, recall, and macro F1.

## Important Notes

- Do not rename the model files unless `pothole_config.json` and `app.py` are updated accordingly.
- Keep this folder in the project root for local Streamlit runs.
- The `.pkl` files are binary artifacts and should not be edited manually.
- Use the training notebook to regenerate artifacts if model changes are required.
