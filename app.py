try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except Exception:
    _STREAMLIT_AVAILABLE = False

    class _ConsoleSt:
        def title(self, *args, **kwargs):
            print(*args)

        def write(self, *args, **kwargs):
            print(*args)

        def button(self, label):
            resp = input(f"{label} [y/N]: ").strip().lower()
            return resp in ("y", "yes")

        def checkbox(self, label):
            resp = input(f"{label} [y/N]: ").strip().lower()
            return resp in ("y", "yes")

    st = _ConsoleSt()


import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


DATA_DEFAULT = Path(__file__).resolve().parents[0] / "data" / "bank-full.csv"


def load_dataframe(uploaded_file):
    if uploaded_file is None:
        df = pd.read_csv(DATA_DEFAULT, sep=';', quotechar='"')
    else:
        # uploaded_file may be BytesIO
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
    return df


def preprocess(df, target_col='y'):
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    # map target if it's yes/no
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({'yes': 1, 'no': 0}).fillna(df[target_col])

    # one-hot encode categoricals
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def scale_numeric(X_train, X_test):
    scaler = StandardScaler()
    numeric_like = [c for c in X_train.columns if X_train[c].nunique() > 2]
    if numeric_like:
        scaler.fit(X_train[numeric_like])
        X_train.loc[:, numeric_like] = scaler.transform(X_train[numeric_like])
        X_test.loc[:, numeric_like] = scaler.transform(X_test[numeric_like])
    return X_train, X_test


def train_model(model_name, X_train, y_train, params):
    if model_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier(max_depth=params.get('max_depth', None), random_state=42)
    elif model_name == 'KNN':
        model = KNeighborsClassifier(n_neighbors=params.get('n_neighbors', 5))
    elif model_name == 'Naive Bayes':
        model = GaussianNB()
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=params.get('n_estimators', 100), random_state=42)
    elif model_name == 'XGBoost':
        if XGBClassifier is None:
            raise RuntimeError('xgboost is not available in the environment')
        model = XGBClassifier(n_estimators=params.get('n_estimators', 100), use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        raise ValueError('Unknown model')

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

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

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    return metrics, cm, report


def run_app():
    st.title('ML Assignment — Model Playground')
    st.write('Upload a CSV or use the default `bank-full.csv` dataset.')

    uploaded_file = st.file_uploader('Upload CSV', type=['csv'])
    use_default = st.checkbox('Use default dataset', value=(uploaded_file is None))

    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox('Select model', ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Random Forest', 'XGBoost'])
    with col2:
        test_size = st.slider('Test size (%)', 10, 50, 20)

    params = {}
    if model_name == 'KNN':
        params['n_neighbors'] = st.number_input('n_neighbors', value=5, min_value=1)
    if model_name in ('Random Forest', 'XGBoost'):
        params['n_estimators'] = st.number_input('n_estimators', value=100, min_value=10)
    if model_name == 'Decision Tree':
        params['max_depth'] = st.number_input('max_depth (0 for None)', value=0, min_value=0)
        if params['max_depth'] == 0:
            params['max_depth'] = None

    run = st.button('Run')

    if run:
        try:
            df = load_dataframe(uploaded_file if not use_default else None)
        except Exception as e:
            st.error(f'Could not load dataset: {e}')
            return

        if 'y' not in df.columns:
            st.error("Dataset must contain a 'y' column as target (values 'yes'/'no' or 1/0)")
            return

        X, y = preprocess(df, target_col='y')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42, stratify=y)
        X_train, X_test = scale_numeric(X_train, X_test)

        try:
            model = train_model(model_name, X_train, y_train, params)
        except Exception as e:
            st.error(f'Error training model: {e}')
            return

        metrics, cm, report = evaluate_model(model, X_test, y_test)

        st.subheader('Evaluation Metrics')
        st.table(pd.DataFrame([metrics]).T.rename(columns={0: 'value'}))

        st.subheader('Confusion Matrix')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha='center', va='center', color='black')
        st.pyplot(fig)

        st.subheader('Classification Report')
        st.json(report)


if __name__ == '__main__':
    if _STREAMLIT_AVAILABLE:
        run_app()
    else:
        # simple console fallback
        print('Streamlit not available — running console fallback')
        print('Loading default dataset...')
        df = pd.read_csv(DATA_DEFAULT, sep=';', quotechar='"')
        X, y = preprocess(df, target_col='y')
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_test = scale_numeric(X_train, X_test)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        metrics, cm, report = evaluate_model(model, X_test, y_test)
        print('Metrics:')
        print(json.dumps(metrics, indent=2))
        print('Confusion matrix:')
        print(cm.tolist())



