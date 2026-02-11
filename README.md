(The file `d:\Machine_Learning\ML_Assignment_2\README.md` exists, but is empty)
# ML Assignment 2 — Bank Marketing Classification

## Problem statement

Predict whether a client will subscribe to a term deposit (`y`) using the Bank Marketing dataset. The app implements six classification models and reports evaluation metrics for model comparison.

## Dataset

- Source: `data/bank-full.csv` (UCI / publicly available bank marketing dataset)
- Instances: 45211
- Note: target column is `y` with values `yes`/`no`.

## Models implemented

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbours (k-NN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

## How to run

1. Activate virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

2. Install dependencies (already provided in this repo):

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

Or run model scripts individually, e.g.:

```powershell
python models\logistic.py
```

## Results (evaluation metrics)

All metrics computed on a held-out test split (20%). Values rounded.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9016 | 0.9054 | 0.6474 | 0.3488 | 0.4533 | 0.4280 |
| Decision Tree | 0.8777 | 0.7135 | 0.4783 | 0.4991 | 0.4884 | 0.4191 |
| k-NN | 0.8961 | 0.8373 | 0.5931 | 0.3554 | 0.4444 | 0.4067 |
| Naive Bayes | 0.8639 | 0.8088 | 0.4282 | 0.4877 | 0.4560 | 0.3797 |
| Random Forest | 0.9045 | 0.9272 | 0.6554 | 0.3866 | 0.4863 | 0.4561 |
| XGBoost | 0.9080 | 0.9291 | 0.6348 | 0.5028 | 0.5612 | 0.5149 |

## Observations

- Ensemble methods (Random Forest, XGBoost) produced the highest AUC and competitive accuracy. XGBoost showed the best overall MCC and balanced precision/recall tradeoff.
- Logistic Regression and Random Forest achieved high accuracy, but XGBoost had the best MCC and F1.
- Decision Tree had lower AUC compared to ensembles, indicating overfitting / lower probability calibration.

## Repository contents

- `app.py` — Streamlit app and console fallback for dataset upload, model selection, training, and metrics display.
- `requirements.txt` — Python dependencies (includes `xgboost`).
- `models/` — Individual model scripts and JSON results for each model.
- `data/` — Contains `bank-full.csv` used in experiments.

## Deployment

Deploy to Streamlit Community Cloud: create a GitHub repo with this code, then follow Streamlit Cloud's "New app" flow and select `app.py` as the entrypoint.

## Submission checklist

- Include this README content in the final PDF.
- Provide GitHub repository link and Streamlit app link in the PDF.
- Include a screenshot from BITS Virtual Lab as proof of execution.

