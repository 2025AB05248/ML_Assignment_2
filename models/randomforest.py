import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "bank-full.csv"
RESULTS_PATH = Path(__file__).resolve().parents[0] / "randomforest_results.json"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=';', quotechar='"')
    return df


def preprocess(df):
    df = df.copy()
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != 'y']
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    X = df.drop(columns=['y'])
    y = df['y']
    return X, y, num_cols


def train_and_evaluate(X, y, random_state=42, n_estimators=100):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    numeric_like = [c for c in X_train.columns if X_train[c].nunique() > 2]
    if numeric_like:
        scaler.fit(X_train[numeric_like])
        X_train_scaled[numeric_like] = scaler.transform(X_train[numeric_like])
        X_test_scaled[numeric_like] = scaler.transform(X_test[numeric_like])

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    except Exception:
        pass

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'mcc': float(matthews_corrcoef(y_test, y_pred)),
    }

    if y_prob is not None:
        try:
            metrics['auc'] = float(roc_auc_score(y_test, y_prob))
        except Exception:
            metrics['auc'] = None
    else:
        metrics['auc'] = None

    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics['confusion_matrix'] = cm

    return model, metrics


def main():
    print(f"Loading data from: {DATA_PATH}")
    df = load_data()
    X, y, _ = preprocess(df)
    print(f"Data shape after preprocessing: X={X.shape}, y={y.shape}")
    model, metrics = train_and_evaluate(X, y)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
