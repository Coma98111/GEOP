# Dual-label sensitivity analysis for debris-flow outbreak potential

This repository provides the Python scripts used for the dual-label sensitivity analysis and figure generation in a debris-flow susceptibility study. The analysis compares two label systems: an observation-based erosion label and a GEOP-derived outbreak-potential label.

The scripts are intended to support reproducibility of the XGBoost-based model evaluation, permutation importance, SHAP interpretation and figure generation presented in the manuscript.

## Repository contents

- `Supplementary_Code_Fig3_DualLabelSensitivity_EnglishComments.py`  
  Generates the main dual-label sensitivity analysis figure, including permutation importance, SHAP dependence, ROC curves, calibration curves, ΔAUC–SHAP comparison and F1-threshold curves.

- `Supplementary_Code_Fig2c_Performance_EnglishComments.py`  
  Generates model-performance panels, including accuracy, F1 score, AUC, confusion matrices and positive-prediction fraction curves.

- `Supplementary_Code_XGBoost_SHAP_EnglishComments.py`  
  Provides a general XGBoost and SHAP workflow for model training, threshold analysis, permutation importance, SHAP interpretation, ROC analysis and result export.

## Input data

The scripts require prepared CSV files containing conditioning factors and binary labels.

Expected input files include:

- `erosion_label.csv`  
  Samples labelled using mapped erosion and stable reference areas.

- `geop_label.csv`  
  Samples labelled using high- and low-potential zones derived from the GEOP outbreak-potential index.

- `taget - qinshi.csv`  
  Input table used by the general XGBoost-SHAP workflow.

In these files, each row represents one sample, columns represent conditioning factors, and the label column is named `CID` unless otherwise specified in the script.

## Main analysis

The workflow includes:

1. Data loading and train-test splitting.
2. XGBoost binary classification.
3. Model-performance evaluation using AUC, accuracy, F1 score and confusion matrix.
4. Permutation importance based on AUC decrease.
5. SHAP-based feature-importance and dependence analysis.
6. ROC, calibration and threshold-response plotting.
7. Export of selected model results to CSV files.

The dual-label framework is used for sensitivity and structural-consistency analysis. The GEOP-derived label is not intended as an independent validation target. Instead, it is used to diagnose the internal sensitivity structure of the GEOP field, while the erosion label represents the mapped geomorphic response.

## Requirements

The scripts were written in Python and require the following packages:

```bash
numpy
pandas
matplotlib
scikit-learn
xgboost
shap
scipy
seaborn
