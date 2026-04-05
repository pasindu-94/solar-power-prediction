"""ML Pipeline for solar power prediction."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    predictions: np.ndarray
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None


class SolarPredictionPipeline:
    """End-to-end ML pipeline for solar power prediction."""

    def __init__(self, target_col: str = "ac_power", test_size: float = 0.2):
        self.target_col = target_col
        self.test_size = test_size
        self._models: Dict = {}
        self._results: List[ModelResult] = []
        self._feature_stats: Dict = {}

    def load_data(self, filepath: str) -> pd.DataFrame:
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = df.copy()
        df = self._handle_missing(df)
        df = self._engineer_features(df)
        df = self._remove_outliers(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)

        self._feature_stats = {
            col: {"mean": df[col].mean(), "std": df[col].std(),
                  "min": df[col].min(), "max": df[col].max()}
            for col in numeric_cols
        }
        return df[numeric_cols], df[[self.target_col]]

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "date_time" in df.columns:
            df["date_time"] = pd.to_datetime(df["date_time"])
            df["hour"] = df["date_time"].dt.hour
            df["month"] = df["date_time"].dt.month
            df["day_of_year"] = df["date_time"].dt.dayofyear
            df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            df.drop(columns=["date_time"], inplace=True)

        if "ambient_temperature" in df.columns and "module_temperature" in df.columns:
            df["temp_diff"] = df["module_temperature"] - df["ambient_temperature"]

        if "irradiation" in df.columns:
            df["irradiation_squared"] = df["irradiation"] ** 2

        return df

    def _remove_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        mask = pd.Series([True] * len(df), index=df.index)
        for col in numeric_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            mask &= z_scores < threshold
        removed = (~mask).sum()
        if removed > 0:
            logger.info(f"Removed {removed} outlier rows")
        return df[mask].reset_index(drop=True)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
        return {
            "mse": round(mse, 4), "rmse": round(rmse, 4), "mae": round(mae, 4),
            "r2": round(r2, 4), "mape": round(mape, 2)
        }

    def compare_models(self) -> pd.DataFrame:
        if not self._results:
            return pd.DataFrame()
        rows = [{"model": r.name, **r.metrics} for r in self._results]
        return pd.DataFrame(rows).sort_values("r2", ascending=False)

    @property
    def feature_statistics(self) -> Dict:
        return self._feature_stats
