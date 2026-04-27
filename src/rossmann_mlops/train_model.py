from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import yaml

from rossmann_mlops.config import load_config, resolve_path


class TrainingError(ValueError):
    """Raised when training input/configuration is invalid."""


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Percentage Error."""
    safe_true = np.maximum(np.asarray(y_true, dtype=float), 1e-8)
    safe_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(np.square((safe_true - safe_pred) / safe_true))))


def _load_processed_data(
    paths: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = resolve_path(paths.get("train_final_data", "data/processed/train_final.csv"))
    val_path = resolve_path(paths.get("val_final_data", "data/processed/val_final.csv"))

    if not train_path.exists():
        raise FileNotFoundError(f"Preprocessed training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Preprocessed validation data not found: {val_path}")

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    # MỚI - không gây FutureWarning
    for col in train_df.columns:
        try:
            train_df[col] = pd.to_numeric(train_df[col])
        except (ValueError, TypeError):
            pass
    for col in val_df.columns:
        try:
            val_df[col] = pd.to_numeric(val_df[col])
        except (ValueError, TypeError):
            pass

    logger.info(f"Loaded training data: {len(train_df)} rows from {train_path}")
    logger.info(f"Loaded validation data: {len(val_df)} rows from {val_path}")

    return train_df, val_df


def _prepare_xgb_inputs(x_train: pd.DataFrame, x_val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure XGBoost-compatible numeric dtypes with stable encoding across splits."""
    prepared_train = x_train.copy()
    prepared_val = x_val.copy()

    categorical_cols = prepared_train.select_dtypes(include=["object", "category"]).columns.tolist()
    for column in categorical_cols:
        # Normalize to string to avoid mixed-type category issues (e.g., int + str) in XGBoost.
        train_values = prepared_train[column].astype(str)
        val_values = prepared_val[column].astype(str)

        categories = pd.Index(train_values.unique())
        mapping = {value: idx for idx, value in enumerate(categories)}

        prepared_train[column] = train_values.map(mapping).fillna(-1).astype(np.int32)
        prepared_val[column] = val_values.map(mapping).fillna(-1).astype(np.int32)

    return prepared_train, prepared_val


def _compact_model_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert model parameters to JSON-serializable format."""
    compact: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        if isinstance(value, (np.generic,)):
            compact[key] = value.item()
        else:
            compact[key] = value
    return compact


def _resolve_artifacts_dir(paths: dict[str, Any], model_path: Path) -> Path:
    """Resolve the artifacts directory path."""
    artifacts_dir = paths.get("artifacts_dir")
    if artifacts_dir:
        return resolve_path(artifacts_dir)
    return model_path.parent


def _save_feature_mappings(train_df: pd.DataFrame, artifacts_dir: Path) -> None:
    """
    Save feature engineering mappings from preprocessed training data.
    These will be used later for prediction on new data.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Global mean sales (for imputation of unseen values)
    global_mean = float(train_df["Sales_log"].mean())
    joblib.dump(global_mean, artifacts_dir / "global_mean_sales.pkl")
    logger.info(f"Saved global_mean_sales: {global_mean}")

    # Store + DayOfWeek + Promo average
    if {"Store", "DayOfWeek", "Promo", "Sales_log"}.issubset(train_df.columns):
        store_dw_promo_avg_map = (
            train_df.groupby(["Store", "DayOfWeek", "Promo"])["Sales_log"]
            .mean()
            .reset_index()
            .rename(columns={"Sales_log": "Store_DW_Promo_Avg"})
        )
        joblib.dump(store_dw_promo_avg_map, artifacts_dir / "store_dw_promo_mapping.pkl")
        logger.info("Saved store_dw_promo_mapping")

    # Month average
    if {"Month", "Sales_log"}.issubset(train_df.columns):
        month_avg_map = (
            train_df.groupby("Month")["Sales_log"]
            .mean()
            .reset_index()
            .rename(columns={"Sales_log": "Month_Avg_Sales"})
        )
        joblib.dump(month_avg_map, artifacts_dir / "month_mapping.pkl")
        logger.info("Saved month_mapping")

def _log_mlflow_payload(
    mlflow_cfg: dict[str, Any],
    model: xgb.XGBRegressor,
    metrics: dict[str, float],
    run_name: str,
) -> None:
    """Log model training results to MLflow."""
    if not bool(mlflow_cfg.get("enabled", False)):
        return

    tracking_uri = mlflow_cfg.get("tracking_uri") or "http://127.0.0.1:5000"
    experiment_name = mlflow_cfg.get("experiment_name") or "Rossmann_Final_Training"

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(model.get_params())
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, float(metric_value))
            mlflow.sklearn.log_model(model, "model")
        logger.info("MLflow logging completed successfully")
    except Exception as exc:  # pragma: no cover
        logger.warning("Skipping MLflow logging due to error: %s", exc)


def train_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """
    Train XGBoost model on preprocessed data.
    
    Args:
        config: Configuration dictionary with paths, training hyperparameters, and mlflow settings
    
    Returns:
        Dictionary with model path, metrics, and training statistics
    """
    paths = config.get("paths", {})
    training = config.get("training", {})
    mlflow_cfg = config.get("mlflow", {})

    # Load preprocessed data
    train_df, val_df = _load_processed_data(paths)

    # Prepare output paths
    model_path = resolve_path(paths["model_file"])
    metrics_path = resolve_path(paths["metrics_file"])
    artifacts_dir = _resolve_artifacts_dir(paths, model_path)
    default_model_config_path = resolve_path(paths.get("model_config_file", "configs/model_config.yaml"))

    # Verify data has required target column
    if "Sales_log" not in train_df.columns or "Sales_log" not in val_df.columns:
        raise TrainingError("Preprocessed data must contain 'Sales_log' column (target variable)")

    logger.info(f"Train set shape: {train_df.shape}")
    logger.info(f"Validation set shape: {val_df.shape}")

    # Prepare features and target
    # Drop non-feature columns (target and identifiers)
    drop_cols = ["Sales", "Sales_log", "Customers", "Date", "Id"]
    x_train = train_df.drop(columns=drop_cols, errors="ignore")
    x_val = val_df.drop(columns=drop_cols, errors="ignore")

    # Ensure consistent data types
    x_train, x_val = _prepare_xgb_inputs(x_train, x_val)

    # Get target variables
    y_train_log = train_df["Sales_log"].astype(float)
    y_val_log = val_df["Sales_log"].astype(float)
    
    # Convert from log space to original scale for RMSPE calculation
    y_train_true = np.exp(y_train_log)
    y_val_true = np.exp(y_val_log)

    logger.info(f"Features for training: {list(x_train.columns)}")

    # Initialize and train model with hyperparameters from config
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method=training.get("tree_method", "hist"),
        n_estimators=int(training.get("n_estimators", 1000)),
        max_depth=int(training.get("max_depth", 11)),
        learning_rate=float(training.get("learning_rate", 0.02484904299575523)),
        subsample=float(training.get("subsample", 0.8437880308889292)),
        colsample_bytree=float(training.get("colsample_bytree", 0.7688520150734801)),
        min_child_weight=float(training.get("min_child_weight", 14)),
        gamma=float(training.get("gamma", 0.00023429788475798095)),
        random_state=int(training.get("random_state", 42)),
        n_jobs=int(training.get("n_jobs", -1)),
    )

    logger.info("Training XGBoost model")
    model.fit(x_train, y_train_log)
    logger.info("Model training completed")

    # Generate predictions
    y_train_pred = np.exp(model.predict(x_train))
    y_val_pred = np.exp(model.predict(x_val))

    # Calculate metrics
    train_rmspe = rmspe(y_train_true, y_train_pred)
    val_rmspe = rmspe(y_val_true, y_val_pred)
    rmspe_gap = float(val_rmspe - train_rmspe)
    rmse = float(np.sqrt(mean_squared_error(y_val_true, y_val_pred)))
    mae = float(mean_absolute_error(y_val_true, y_val_pred))
    r2 = float(r2_score(y_val_true, y_val_pred))

    metrics = {
        "train_rmspe": float(train_rmspe),
        "val_rmspe": float(val_rmspe),
        "rmspe_gap": rmspe_gap,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

    logger.info(f"Training metrics: {metrics}")

    # Log to MLflow
    _log_mlflow_payload(mlflow_cfg, model, metrics, run_name=training.get("run_name", "Production_Train_Logic"))

    # Create output directories
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")

    # Save feature engineering mappings for later use in prediction
    _save_feature_mappings(train_df, artifacts_dir)

    # Save metrics
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(f"Metrics saved to: {metrics_path}")

    # Prepare and save model configuration
    model_params = _compact_model_params(model.get_params())
    model_config_payload = {
        "project": "Rossmann Store Sales",
        "best_model": {
            "name": "XGBoost",
            "val_rmspe": float(val_rmspe),
            "params": model_params,
        },
        "features": {
            "input_columns": list(x_train.columns),
            "target": "Sales_log",
        },
    }
# Lưu candidate config (luôn lưu sau mỗi lần train)
    candidate_config_key = paths.get("model_config_candidate_file")
    candidate_config_path = resolve_path(candidate_config_key) if candidate_config_key else None
    if candidate_config_path:
        candidate_config_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_config_path.write_text(
            yaml.safe_dump(model_config_payload, sort_keys=False), encoding="utf-8"
        )
        logger.info(f"Candidate model config saved to: {candidate_config_path}")

    # Official config chỉ ghi khi không có candidate path
    model_config_overwritten = False
    model_config_path = default_model_config_path
    if not candidate_config_path:
        model_config_path.parent.mkdir(parents=True, exist_ok=True)
        model_config_path.write_text(
            yaml.safe_dump(model_config_payload, sort_keys=False), encoding="utf-8"
        )
        logger.info(f"Model config saved to: {model_config_path}")
        model_config_overwritten = True

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "model_config_path": str(model_config_path),
        "model_config_candidate_path": str(candidate_config_path) if candidate_config_path else None,
        "model_config_overwritten": model_config_overwritten,
        "artifacts_dir": str(artifacts_dir),
        "metrics": metrics,
        "n_train": int(len(x_train)),
        "n_validation": int(len(x_val)),
    }


def main() -> None:
    """Main entry point for training pipeline."""
    parser = argparse.ArgumentParser(description="Train Rossmann model from preprocessed data")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    result = train_pipeline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
