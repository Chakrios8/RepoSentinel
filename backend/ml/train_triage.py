"""Synthetic data generation and XGBoost model training for RepoSentinel commit triage."""

import json
from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

FEATURE_COLUMNS = [
    "lines_of_code",
    "num_functions",
    "max_nesting_depth",
    "num_branches",
    "num_imports",
    "num_external_calls",
    "has_eval_or_exec",
    "has_raw_sql_indicators",
]


def generate_synthetic_dataset(num_samples: int = 2000, random_seed: int = 42) -> pd.DataFrame:
    """Generate synthetic commit records with realistic structural and security features.
    
    Creates a realistic distribution where simple, clean code has very low risk probability,
    and constructs with dynamic code execution, raw SQL calls, or abnormal complexity have high risk.
    """
    np.random.seed(random_seed)

    # 1. Generate structural features
    # 70% benign / low-complexity commits, 30% complex or suspicious commits
    is_suspicious_cluster = np.random.binomial(1, 0.30, size=num_samples)

    # Lines of code: benign is smaller (3 to 60), suspicious can be larger (10 to 180)
    loc_benign = np.random.gamma(shape=2.5, scale=10.0, size=num_samples).astype(int) + 3
    loc_suspicious = np.random.gamma(shape=4.0, scale=18.0, size=num_samples).astype(int) + 10
    lines_of_code = np.where(is_suspicious_cluster == 1, loc_suspicious, loc_benign)

    # Number of functions: 0 to 6
    num_functions = np.random.poisson(lam=1.5, size=num_samples)
    num_functions = np.clip(num_functions, 0, 8)

    # Max nesting depth: benign (1-2), suspicious (2-6)
    nesting_benign = np.random.choice([1, 2, 3], size=num_samples, p=[0.55, 0.35, 0.10])
    nesting_suspicious = np.random.choice([2, 3, 4, 5, 6], size=num_samples, p=[0.25, 0.30, 0.25, 0.15, 0.05])
    max_nesting_depth = np.where(is_suspicious_cluster == 1, nesting_suspicious, nesting_benign)

    # Number of branches: If, While, For, Try
    branches_benign = np.random.poisson(lam=1.0, size=num_samples)
    branches_suspicious = np.random.poisson(lam=4.5, size=num_samples)
    num_branches = np.where(is_suspicious_cluster == 1, branches_suspicious, branches_benign)

    # Number of imports: 0 to 10
    num_imports = np.random.poisson(lam=2.5, size=num_samples)

    # External calls: calls to functions/methods
    calls_benign = np.random.poisson(lam=2.0, size=num_samples)
    calls_suspicious = np.random.poisson(lam=7.0, size=num_samples)
    num_external_calls = np.where(is_suspicious_cluster == 1, calls_suspicious, calls_benign)

    # Security indicators: has_eval_or_exec and has_raw_sql_indicators
    # Almost never in clean code, concentrated in suspicious cluster
    has_eval_or_exec = np.where(
        is_suspicious_cluster == 1,
        np.random.binomial(1, 0.40, size=num_samples),
        np.random.binomial(1, 0.005, size=num_samples)
    )

    has_raw_sql_indicators = np.where(
        is_suspicious_cluster == 1,
        np.random.binomial(1, 0.50, size=num_samples),
        np.random.binomial(1, 0.008, size=num_samples)
    )

    # Construct realistic vulnerability probability based on domain knowledge:
    # Strongest indicators: eval/exec and raw SQL indicators
    # Secondary indicators: excessive nesting, high branch counts, high external calls
    logits = (
        -3.8
        + 4.8 * has_eval_or_exec
        + 4.5 * has_raw_sql_indicators
        + 0.55 * (max_nesting_depth >= 4).astype(float)
        + 0.35 * (num_branches >= 5).astype(float)
        + 0.25 * (num_external_calls >= 8).astype(float)
        + 0.008 * lines_of_code
        + np.random.normal(0, 0.25, size=num_samples)  # realistic noise
    )

    probabilities = 1.0 / (1.0 + np.exp(-logits))
    labels = (probabilities >= 0.50).astype(int)

    df = pd.DataFrame({
        "lines_of_code": lines_of_code,
        "num_functions": num_functions,
        "max_nesting_depth": max_nesting_depth,
        "num_branches": num_branches,
        "num_imports": num_imports,
        "num_external_calls": num_external_calls,
        "has_eval_or_exec": has_eval_or_exec,
        "has_raw_sql_indicators": has_raw_sql_indicators,
        "is_vulnerable": labels,
        "risk_prob": probabilities,
    })

    return df


def train_and_save_model(
    model_output_path: str = None,
    columns_output_path: str = None,
    num_samples: int = 2000,
    random_seed: int = 42
) -> Tuple[XGBClassifier, dict]:
    """Train XGBoost triage classifier on synthetic AST metrics and serialize artifacts."""
    base_dir = Path(__file__).resolve().parent
    model_path = Path(model_output_path or (base_dir / "ast_triage_model.joblib"))
    columns_path = Path(columns_output_path or (base_dir / "feature_columns.json"))

    # Ensure parent directories exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    columns_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_dataset(num_samples=num_samples, random_seed=random_seed)

    X = df[FEATURE_COLUMNS]
    y = df["is_vulnerable"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_seed, stratify=y
    )

    classifier = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_seed,
        eval_metric="logloss"
    )

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save artifacts
    joblib.dump(classifier, model_path)
    with open(columns_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    print(f"XGBoost model trained successfully. Test ROC-AUC: {roc_auc:.4f}")
    print(f"Saved model to: {model_path}")
    print(f"Saved feature columns to: {columns_path}")

    return classifier, {"roc_auc": roc_auc, "classification_report": report}


if __name__ == "__main__":
    train_and_save_model()
