"""Visualization utilities for solar power prediction analysis."""
from typing import Dict, List, Optional, Tuple
import numpy as np


def create_prediction_summary(actuals: np.ndarray, predictions: np.ndarray) -> Dict:
    """Create a statistical summary of predictions vs actuals."""
    errors = actuals - predictions
    return {
        "actual_mean": float(np.mean(actuals)),
        "predicted_mean": float(np.mean(predictions)),
        "error_mean": float(np.mean(errors)),
        "error_std": float(np.std(errors)),
        "max_overestimate": float(np.min(errors)),
        "max_underestimate": float(np.max(errors)),
        "within_10pct": float(np.mean(np.abs(errors / np.where(actuals == 0, 1, actuals)) < 0.1) * 100),
        "within_20pct": float(np.mean(np.abs(errors / np.where(actuals == 0, 1, actuals)) < 0.2) * 100),
    }


def calculate_hourly_performance(hours: np.ndarray, actuals: np.ndarray,
                                  predictions: np.ndarray) -> Dict[int, Dict]:
    """Calculate prediction performance by hour of day."""
    results = {}
    for hour in range(24):
        mask = hours == hour
        if mask.sum() == 0:
            continue
        h_actual = actuals[mask]
        h_pred = predictions[mask]
        results[hour] = {
            "count": int(mask.sum()),
            "avg_actual": float(np.mean(h_actual)),
            "avg_predicted": float(np.mean(h_pred)),
            "mae": float(np.mean(np.abs(h_actual - h_pred))),
            "rmse": float(np.sqrt(np.mean((h_actual - h_pred) ** 2))),
        }
    return results


def generate_report(model_name: str, metrics: Dict, feature_importance: Optional[Dict] = None) -> str:
    """Generate a text report for model performance."""
    lines = [
        f"Model Performance Report: {model_name}",
        "=" * 50,
        f"R² Score:  {metrics.get('r2', 'N/A')}",
        f"RMSE:      {metrics.get('rmse', 'N/A')}",
        f"MAE:       {metrics.get('mae', 'N/A')}",
        f"MAPE:      {metrics.get('mape', 'N/A')}%",
    ]
    if feature_importance:
        lines.append("\nTop Features:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feat, imp in sorted_features:
            lines.append(f"  {feat}: {imp:.4f}")
    return "\n".join(lines)
