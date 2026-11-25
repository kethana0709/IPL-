"""
train_model.py
==============
BioSight AI Model Training Pipeline
-----------------------------------
Trains lightweight, production-ready Scikit-Learn regression models on the
preprocessed BioSight datasets and saves serialized artifacts into `models/`
for deployment in web backends (e.g. FastAPI, Flask, Django).

Models Trained & Serialized:
1. IoT Biogas Digester Methane Predictor (models/biogas_digester_model.joblib):
   - Predicts real-time methane concentration (% Vol) from continuous IoT telemetry:
     pH levels (top/mid/bottom), multi-depth temperature probes, IR core & ambient
     temperatures, gas temperature/humidity/pressure/resistance, and gradient features.
   - Evaluated on test set with R², RMSE, MAE, and 5-fold cross-validation.

2. Biomass Feedstock Methane Yield Predictor (models/biochemical_yield_model.joblib):
   - Predicts cumulative methane yield (ml/g of VS) from biochemical composition:
     digestion duration, biomass storage duration, moisture, C/N ratio, cellulose,
     hemicellulose, lignin, and solid waste percentage.
   - Evaluated on test set with R², RMSE, MAE, and 5-fold cross-validation.

Metadata:
- JSON schemas containing feature order, metrics, and sample inference payloads
  are saved alongside the models for seamless web API integration.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("BioSightTraining")


# ==============================================================================
# 1. TRAIN IOT BIOGAS DIGESTER METHANE CONCENTRATION MODEL
# ==============================================================================

def train_biogas_digester_model(
    data_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains and serializes the IoT Digester Methane Concentration prediction model.
    """
    logger.info("==================================================================")
    logger.info("TRAINING MODEL 1: IoT Biogas Digester Methane Concentration Predictor")
    logger.info("==================================================================")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Telemetry data not found at: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset: {data_path} (Shape: {df.shape})")

    # Feature selection & Target definition
    target_col = "target_methane_conc_pct_vol"
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not in dataset.")

    # Drop non-feature columns
    exclude_cols = ["datetime", target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    logger.info(f"Feature count: {len(feature_cols)}")
    logger.info(f"Features: {feature_cols}")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Dataset split: Train = {len(X_train)} samples, Test = {len(X_test)} samples")

    # Build Pipeline: Standard Scaler + Gradient Boosting Regressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=random_state
        ))
    ])

    logger.info("Fitting Gradient Boosting Pipeline...")
    pipeline.fit(X_train, y_train)

    # Predictions & Test Evaluation
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-5))) * 100)

    # 5-Fold Cross Validation on full dataset
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    logger.info("------------------------------------------------------------------")
    logger.info("MODEL 1 EVALUATION RESULTS:")
    logger.info(f"  Test R² Score:              {r2:.4f} ({r2 * 100:.2f}% variance explained)")
    logger.info(f"  Root Mean Squared Error:    {rmse:.4f} % Vol")
    logger.info(f"  Mean Absolute Error (MAE):  {mae:.4f} % Vol")
    logger.info(f"  Mean Absolute % Error:      {mape:.2f} %")
    logger.info(f"  5-Fold CV Mean R²:          {cv_mean:.4f} (+/- {cv_std:.4f})")
    logger.info("------------------------------------------------------------------")

    # Extract Feature Importances
    regressor = pipeline.named_steps['regressor']
    importances = dict(zip(feature_cols, [float(x) for x in regressor.feature_importances_]))
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    logger.info("Top 5 Dominant Predictor Features:")
    for feat, imp in list(sorted_importances.items())[:5]:
        logger.info(f"  * {feat:28s}: {imp * 100:.2f}%")

    # Serialize Model & Metadata
    os.makedirs(output_dir, exist_ok=True)
    model_filename = "biogas_digester_model.joblib"
    model_path = os.path.join(output_dir, model_filename)
    joblib.dump(pipeline, model_path)
    logger.info(f"Model successfully saved to: {model_path} ({os.path.getsize(model_path):,} bytes)")

    # Create Sample Payload for Web Backends
    sample_payload = X_test.iloc[0].to_dict()

    metadata = {
        "model_name": "IoT Biogas Digester Real-Time Methane Concentration Predictor",
        "model_type": "GradientBoostingRegressor Pipeline with StandardScaler",
        "created_at": datetime.now().isoformat(),
        "target": {
            "name": target_col,
            "unit": "% Vol",
            "description": "Methane gas output concentration percentage by volume"
        },
        "features": feature_cols,
        "feature_importances": sorted_importances,
        "metrics": {
            "test_r2": round(r2, 4),
            "test_rmse": round(rmse, 4),
            "test_mae": round(mae, 4),
            "test_mape_percent": round(mape, 2),
            "cv_5fold_mean_r2": round(cv_mean, 4),
            "cv_5fold_std": round(cv_std, 4)
        },
        "sample_input": sample_payload,
        "sample_prediction": float(pipeline.predict(pd.DataFrame([sample_payload]))[0])
    }

    meta_path = os.path.join(output_dir, "biogas_digester_model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata schema saved to: {meta_path}")

    return metadata


# ==============================================================================
# 2. TRAIN BIOCHEMICAL FEEDSTOCK METHANE YIELD MODEL
# ==============================================================================

def train_biochemical_yield_model(
    data_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains and serializes the Biochemical Kinetics Methane Yield prediction model.
    """
    logger.info("\n==================================================================")
    logger.info("TRAINING MODEL 2: Biomass Feedstock Methane Yield Predictor")
    logger.info("==================================================================")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Kinetics data not found at: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset: {data_path} (Shape: {df.shape})")

    target_col = "target_ch4_yield_ml_g_vs"
    feature_cols = [
        "digestion_days",
        "storage_duration_months",
        "moisture_content_pct",
        "cn_ratio",
        "cellulose_pct",
        "hemicellulose_pct",
        "lignin_pct",
        "solid_waste_pct"
    ]

    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Build Pipeline: StandardScaler + GradientBoostingRegressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.09,
            max_depth=3,
            subsample=0.9,
            random_state=random_state
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    logger.info("------------------------------------------------------------------")
    logger.info("MODEL 2 EVALUATION RESULTS:")
    logger.info(f"  Test R² Score:              {r2:.4f} ({r2 * 100:.2f}% variance explained)")
    logger.info(f"  Root Mean Squared Error:    {rmse:.2f} ml/g of VS")
    logger.info(f"  Mean Absolute Error (MAE):  {mae:.2f} ml/g of VS")
    logger.info(f"  5-Fold CV Mean R²:          {cv_mean:.4f} (+/- {cv_std:.4f})")
    logger.info("------------------------------------------------------------------")

    # Feature Importances
    regressor = pipeline.named_steps['regressor']
    importances = dict(zip(feature_cols, [float(x) for x in regressor.feature_importances_]))
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    logger.info("Feature Importance Breakdown:")
    for feat, imp in sorted_importances.items():
        logger.info(f"  * {feat:28s}: {imp * 100:.2f}%")

    # Save Model Artifact
    os.makedirs(output_dir, exist_ok=True)
    model_filename = "biochemical_yield_model.joblib"
    model_path = os.path.join(output_dir, model_filename)
    joblib.dump(pipeline, model_path)
    logger.info(f"Model successfully saved to: {model_path} ({os.path.getsize(model_path):,} bytes)")

    sample_payload = X_test.iloc[0].to_dict()

    metadata = {
        "model_name": "Biomass Feedstock Biochemical Methane Yield Predictor",
        "model_type": "GradientBoostingRegressor Pipeline with StandardScaler",
        "created_at": datetime.now().isoformat(),
        "target": {
            "name": target_col,
            "unit": "ml/g of VS",
            "description": "Specific cumulative methane yield per gram of volatile solids"
        },
        "features": feature_cols,
        "feature_importances": sorted_importances,
        "metrics": {
            "test_r2": round(r2, 4),
            "test_rmse": round(rmse, 2),
            "test_mae": round(mae, 2),
            "cv_5fold_mean_r2": round(cv_mean, 4),
            "cv_5fold_std": round(cv_std, 4)
        },
        "sample_input": sample_payload,
        "sample_prediction": float(pipeline.predict(pd.DataFrame([sample_payload]))[0])
    }

    meta_path = os.path.join(output_dir, "biochemical_yield_model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata schema saved to: {meta_path}")

    return metadata


# ==============================================================================
# 3. VERIFY MODEL LOADING & WEB BACKEND COMPATIBILITY
# ==============================================================================

def verify_saved_models(models_dir: str):
    """
    Simulates a web backend loading the serialized models and serving predictions.
    """
    logger.info("\n==================================================================")
    logger.info("VERIFYING MODEL LOADING & WEB BACKEND INTEGRATION")
    logger.info("==================================================================")

    models_to_test = [
        ("biogas_digester_model.joblib", "biogas_digester_model_meta.json"),
        ("biochemical_yield_model.joblib", "biochemical_yield_model_meta.json")
    ]

    for model_file, meta_file in models_to_test:
        model_path = os.path.join(models_dir, model_file)
        meta_path = os.path.join(models_dir, meta_file)

        logger.info(f"Testing {model_file}...")
        # 1. Load metadata schema
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # 2. Load model
        loaded_model = joblib.load(model_path)
        logger.info(f"  Loaded model artifact: {type(loaded_model)}")

        # 3. Predict on sample input
        sample_input = meta["sample_input"]
        df_input = pd.DataFrame([sample_input])
        prediction = float(loaded_model.predict(df_input)[0])

        target_name = meta["target"]["name"]
        unit = meta["target"]["unit"]
        logger.info(f"  Web Backend Inference Test:")
        logger.info(f"    Target: {target_name}")
        logger.info(f"    Predicted Value: {prediction:.3f} {unit}")
        assert not np.isnan(prediction), "Prediction resulted in NaN!"
        logger.info(f"  [PASSED] {model_file} is verified ready for production deployment.")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="BioSight Model Training Script")
    parser.add_argument("--data_dir", type=str, default="data/preprocessed", help="Path to preprocessed data directory")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save trained models")
    args = parser.parse_args()

    base_path = Path(".")
    data_dir = base_path / args.data_dir
    models_dir = base_path / args.models_dir
    os.makedirs(models_dir, exist_ok=True)

    # 1. Train IoT Digester Methane Model
    feed_data_path = data_dir / "biogas_feed_monitoring_fused_full.csv"
    train_biogas_digester_model(
        data_path=str(feed_data_path),
        output_dir=str(models_dir)
    )

    # 2. Train Biochemical Methane Yield Model
    kinetics_data_path = data_dir / "biogas_kinetics_tidy_full.csv"
    train_biochemical_yield_model(
        data_path=str(kinetics_data_path),
        output_dir=str(models_dir)
    )

    # 3. Verify models can be loaded and served
    verify_saved_models(str(models_dir))

    logger.info("==================================================================")
    logger.info("ALL MODELS TRAINED, EVALUATED, AND SAVED SUCCESSFULLY")
    logger.info(f"Output directory: {models_dir.resolve()}")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
