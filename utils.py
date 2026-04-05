"""Utility functions for solar power prediction pipeline."""
import numpy as np
import pandas as pd
from typing import Tuple

def load_and_preprocess(filepath: str, target_col: str = "power") -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(filepath).dropna()
    return df.drop(columns=[target_col]), df[target_col]

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["hour"] = df["date"].dt.hour
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear
    return df

def normalize_features(X: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        min_val, max_val = X[col].min(), X[col].max()
        if max_val > min_val:
            X[col] = (X[col] - min_val) / (max_val - min_val)
    return X
