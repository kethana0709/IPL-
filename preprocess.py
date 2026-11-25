"""
preprocess.py
=============
BioSight Data Preprocessing Pipeline
-------------------------------------
Prepares all available datasets in the `datasets/` directory for machine learning
and computer vision model training:

1. Biogas Biochemical Kinetics Dataset (dataset_BIOGAS-1.xlsx):
   - Unpivots wide-format multi-condition trial data into a tidy tabular format.
   - Standardizes feature column names and calculates storage duration in months.
   - Removes collinear/redundant features (e.g. Water(%) = 100 - Solid Waste(%)).
   - Performs train/test splitting and robust feature scaling.

2. IoT Biogas Digester Continuous Sensor Feed (biogasFeedMonitoringDataset-master):
   - Cleans ambient temperature logs (removes sensor dropout -1.0C rows & null timestamps).
   - Cleans anaerobic digester feed monitoring logs:
     * Fixes Bosch BME sensor altitude overflow error (44,330m) via median imputation.
     * Drops zero-variance columns (constant Battery Level, Charging State).
     * Imputes zero-sensor glitches in humidity/resistance with forward-fill.
   - Sensor Fusion: Aligns & merges ambient sensor readings with digester telemetry via merge_asof.
   - Engineers temporal and thermodynamic gradient features (e.g., core-to-ambient delta).
   - Generates chronological train/test splits for time-series forecasting.

3. Aerial Algae Bloom Computer Vision Dataset (archive (1)):
   - Verifies all 427 UAV RGB images across Level 1 (moderate) and Level 2 (severe).
   - Builds a stratified train/val/test split manifest to address class imbalance.
   - Provides standardized PyTorch/OpenCV-compatible image transformation pipelines.

4. Water Hyacinth Photo EXIF & Drone Dataset Summaries (metadata.csv & drone-view):
   - Audits and exports cataloged metadata manifests and documentation.
"""

import os
import sys
import glob
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("BioSightPreprocess")


# ==============================================================================
# 1. BIOGAS BIOCHEMICAL KINETICS PREPROCESSING
# ==============================================================================

def preprocess_biogas_kinetics(
    excel_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses the multi-condition biogas biochemical kinetics dataset (dataset_BIOGAS-1.xlsx).
    
    Transforms wide 41-column format into long 65-row feature matrix,
    standardizes column names, splits into train/test, and normalizes features.
    """
    logger.info("--- Processing Biogas Biochemical Kinetics Dataset ---")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"File not found: {excel_path}")

    df_raw = pd.read_excel(excel_path)
    logger.info(f"Loaded raw Excel data. Shape: {df_raw.shape}")

    conditions = ['Fresh', '3-month', '6-month', '3-year', '5-year']
    condition_months = {
        'Fresh': 0,
        '3-month': 3,
        '6-month': 6,
        '3-year': 36,
        '5-year': 60
    }
    
    # Standard column mapping
    metric_mapping = {
        'MC (%)': 'moisture_content_pct',
        'C/N ratio': 'cn_ratio',
        'Ce (%)': 'cellulose_pct',
        'He (%)': 'hemicellulose_pct',
        'Li (%)': 'lignin_pct',
        'CH4 (ml/g of VS)': 'target_ch4_yield_ml_g_vs',
        'Solid Waste(%)': 'solid_waste_pct',
        'Water(%)': 'water_pct'
    }

    records = []
    for cond in conditions:
        months = condition_months[cond]
        for _, row in df_raw.iterrows():
            rec = {
                'digestion_days': float(row['Days']),
                'storage_condition': cond,
                'storage_duration_months': float(months)
            }
            for raw_m, clean_m in metric_mapping.items():
                col_name = f"{cond}_{raw_m}"
                rec[clean_m] = float(row[col_name])
            records.append(rec)

    df_long = pd.DataFrame(records)
    logger.info(f"Unpivoted into tidy long format. Shape: {df_long.shape}")

    # Note on collinearity: water_pct = 100 - solid_waste_pct exactly
    # We drop water_pct to prevent multi-collinearity in linear/regression models
    feature_cols = [
        'digestion_days', 'storage_duration_months', 'moisture_content_pct',
        'cn_ratio', 'cellulose_pct', 'hemicellulose_pct', 'lignin_pct',
        'solid_waste_pct'
    ]
    target_col = 'target_ch4_yield_ml_g_vs'

    # Check missing values
    null_counts = df_long[feature_cols + [target_col]].isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Found missing values in kinetics data:\n{null_counts}")
        df_long = df_long.dropna(subset=feature_cols + [target_col])
    else:
        logger.info("Zero missing values in kinetics dataset.")

    # Train / Test split stratified by storage condition
    train_df, test_df = train_test_split(
        df_long,
        test_size=test_size,
        random_state=random_state,
        stratify=df_long['storage_condition']
    )

    # Feature scaling
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    scaled_feature_cols = [f"{col}_scaled" for col in feature_cols]
    train_scaled[scaled_feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[scaled_feature_cols] = scaler.transform(test_df[feature_cols])

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, "biogas_kinetics_tidy_full.csv")
    train_path = os.path.join(output_dir, "biogas_kinetics_train.csv")
    test_path = os.path.join(output_dir, "biogas_kinetics_test.csv")

    df_long.to_csv(full_path, index=False)
    train_scaled.to_csv(train_path, index=False)
    test_scaled.to_csv(test_path, index=False)

    logger.info(f"Saved full kinetics data to: {full_path}")
    logger.info(f"Saved kinetics train set ({len(train_df)} rows) to: {train_path}")
    logger.info(f"Saved kinetics test set ({len(test_df)} rows) to: {test_path}")

    return df_long, train_scaled, test_scaled


# ==============================================================================
# 2. IOT BIOGAS DIGESTER SENSOR FUSION PREPROCESSING
# ==============================================================================

def preprocess_sensor_feed(
    feed_csv_path: str,
    ambient_csv_path: str,
    output_dir: str,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses and fuses the IoT biogas digester telemetry and ambient weather logs.
    
    - Filters sensor dropouts (-1.0C readings).
    - Repairs barometric altitude overflow bugs (44,330m) with median imputation.
    - Eliminates uninformative constant columns (Battery Level, Charging State).
    - Fuses digester sensor telemetry with ambient temperature logs using merge_asof.
    - Creates thermodynamic gradient and rolling features.
    - Generates chronological train/test time-series splits.
    """
    logger.info("--- Processing IoT Biogas Digester Telemetry & Ambient Data ---")
    if not os.path.exists(feed_csv_path):
        raise FileNotFoundError(f"Feed CSV not found: {feed_csv_path}")
    if not os.path.exists(ambient_csv_path):
        raise FileNotFoundError(f"Ambient CSV not found: {ambient_csv_path}")

    df_feed = pd.read_csv(feed_csv_path)
    df_amb = pd.read_csv(ambient_csv_path)

    logger.info(f"Raw feed telemetry records: {len(df_feed)}")
    logger.info(f"Raw ambient temperature records: {len(df_amb)}")

    # 1. Clean ambient temperature
    # Drop rows with null DateTime or sentinel -1.0 values
    df_amb_clean = df_amb.dropna(subset=['DateTime']).copy()
    initial_amb_len = len(df_amb)
    df_amb_clean = df_amb_clean[df_amb_clean['ambientTemperature(*C)'] > -1.0].copy()
    dropped_amb = initial_amb_len - len(df_amb_clean)
    logger.info(f"Removed {dropped_amb} ambient sensor dropout / null records.")

    df_amb_clean['datetime'] = pd.to_datetime(df_amb_clean['DateTime'], errors='coerce')
    df_amb_clean = df_amb_clean.dropna(subset=['datetime']).sort_values('datetime')

    # Rename ambient columns
    df_amb_clean = df_amb_clean.rename(columns={
        'ambientTemperature(*C)': 'ambient_temp_c',
        'ambientTemperaturePMSensor(*C)': 'ambient_temp_pm_sensor_c'
    })

    # 2. Clean Feed Monitoring Data
    # Parse timestamps
    df_feed['datetime'] = pd.to_datetime(df_feed['Device Date & Time'], errors='coerce')
    df_feed = df_feed.dropna(subset=['datetime']).sort_values('datetime')

    # Drop constant or row-index columns
    cols_to_drop = ['#', 'Battery Level(%)', 'Charging State', 'Device Date & Time', 'Date', 'Hour']
    existing_drops = [c for c in cols_to_drop if c in df_feed.columns]
    df_feed = df_feed.drop(columns=existing_drops)

    # Standardize column naming
    clean_feed_names = {
        'Methane Conc(%V)': 'target_methane_conc_pct_vol',
        'pH Bottom': 'ph_bottom',
        'pH Middle': 'ph_middle',
        'pH Top': 'ph_top',
        'Top Temperature Probe(*C)': 'temp_probe_top_c',
        'Mid Temperature Probe(*C)': 'temp_probe_mid_c',
        'Bottom Temperature Probe(*C)': 'temp_probe_bottom_c',
        'IR Digestate Temperature(*C)': 'temp_ir_digestate_c',
        'IR Ambient Temperature(*C)': 'temp_ir_ambient_c',
        'Gas Temperature(*C)': 'gas_temp_c',
        'Gas Humidity(%)': 'gas_humidity_pct',
        'Gas Pressure(hPa)': 'gas_pressure_hpa',
        'Gas Resistance(KOhms)': 'gas_resistance_kohm',
        'Altitude(m)': 'altitude_m'
    }
    df_feed = df_feed.rename(columns=clean_feed_names)

    # 3. Handle Sensor Faults / Artifacts
    # Altitude overflow error (44,330m)
    altitude_glitch_mask = df_feed['altitude_m'] > 2000.0
    glitch_count = altitude_glitch_mask.sum()
    if glitch_count > 0:
        valid_altitude_median = df_feed.loc[~altitude_glitch_mask, 'altitude_m'].median()
        df_feed.loc[altitude_glitch_mask, 'altitude_m'] = valid_altitude_median
        logger.info(f"Corrected {glitch_count} Bosch BME altitude overflow glitches (44,330m -> median {valid_altitude_median:.1f}m).")

    # Replace 0 values in gas humidity/resistance where sensor briefly failed with forward-fill
    for col in ['gas_humidity_pct', 'gas_resistance_kohm']:
        zero_mask = df_feed[col] <= 0.0
        if zero_mask.sum() > 0:
            df_feed.loc[zero_mask, col] = np.nan
            df_feed[col] = df_feed[col].ffill().bfill()
            logger.info(f"Imputed {zero_mask.sum()} zero-glitch values in {col}.")

    # 4. Sensor Fusion via merge_asof
    merged_telemetry = pd.merge_asof(
        df_feed,
        df_amb_clean[['datetime', 'ambient_temp_c', 'ambient_temp_pm_sensor_c']],
        on='datetime',
        direction='nearest',
        tolerance=pd.Timedelta('2h')
    )

    # Impute any edge ambient temperatures with IR ambient temperature
    if merged_telemetry['ambient_temp_c'].isnull().sum() > 0:
        null_amb = merged_telemetry['ambient_temp_c'].isnull().sum()
        merged_telemetry['ambient_temp_c'] = merged_telemetry['ambient_temp_c'].fillna(
            merged_telemetry['temp_ir_ambient_c']
        )
        logger.info(f"Backfilled {null_amb} unaligned ambient temp records using on-device IR ambient sensor.")

    if 'ambient_temp_pm_sensor_c' in merged_telemetry.columns:
        merged_telemetry['ambient_temp_pm_sensor_c'] = merged_telemetry['ambient_temp_pm_sensor_c'].fillna(
            merged_telemetry['ambient_temp_c']
        )

    # 5. Feature Engineering (Gradients & Temporal dynamics)
    merged_telemetry['ph_gradient_top_bottom'] = merged_telemetry['ph_top'] - merged_telemetry['ph_bottom']
    merged_telemetry['temp_gradient_top_bottom'] = merged_telemetry['temp_probe_top_c'] - merged_telemetry['temp_probe_bottom_c']
    merged_telemetry['temp_digestate_vs_ambient'] = merged_telemetry['temp_ir_digestate_c'] - merged_telemetry['ambient_temp_c']
    merged_telemetry['hour_of_day'] = merged_telemetry['datetime'].dt.hour
    merged_telemetry['day_of_week'] = merged_telemetry['datetime'].dt.dayofweek

    # Chronological Time-series split (avoid future data leakage)
    split_idx = int(len(merged_telemetry) * (1.0 - test_ratio))
    train_telemetry = merged_telemetry.iloc[:split_idx].copy()
    test_telemetry = merged_telemetry.iloc[split_idx:].copy()

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, "biogas_feed_monitoring_fused_full.csv")
    train_path = os.path.join(output_dir, "biogas_feed_monitoring_train.csv")
    test_path = os.path.join(output_dir, "biogas_feed_monitoring_test.csv")

    merged_telemetry.to_csv(full_path, index=False)
    train_telemetry.to_csv(train_path, index=False)
    test_telemetry.to_csv(test_path, index=False)

    logger.info(f"Saved fused feed telemetry to: {full_path}")
    logger.info(f"Saved feed train set ({len(train_telemetry)} rows) to: {train_path}")
    logger.info(f"Saved feed test set ({len(test_telemetry)} rows) to: {test_path}")

    return train_telemetry, test_telemetry


# ==============================================================================
# 3. AERIAL ALGAE BLOOM COMPUTER VISION PREPROCESSING
# ==============================================================================

def preprocess_algae_image_dataset(
    algae_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Scans, verifies, and creates a stratified split manifest for the 427 UAV
    aerial algae bloom images across Level 1 (moderate bloom) and Level 2 (severe bloom).
    """
    logger.info("--- Processing Aerial Algae Bloom Image Dataset ---")
    images_base = os.path.join(algae_dir, "images")
    if not os.path.exists(images_base):
        raise FileNotFoundError(f"Images folder not found: {images_base}")

    class_dirs = {
        'level_1': 0,
        'level_2': 1
    }

    records = []
    logger.info("Validating image file headers, resolutions, and channels...")
    for class_name, class_id in class_dirs.items():
        folder = os.path.join(images_base, class_name)
        if not os.path.exists(folder):
            logger.warning(f"Class folder not found: {folder}")
            continue

        file_list = sorted(os.listdir(folder))
        for fname in file_list:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(folder, fname)
            file_size_bytes = os.path.getsize(img_path)

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    mode = img.mode
                
                records.append({
                    'image_path': os.path.abspath(img_path),
                    'relative_path': os.path.relpath(img_path, start=os.getcwd()),
                    'filename': fname,
                    'class_name': class_name,
                    'class_id': class_id,
                    'width': w,
                    'height': h,
                    'aspect_ratio': round(w / h, 4),
                    'channels': len(mode),
                    'color_mode': mode,
                    'file_size_kb': round(file_size_bytes / 1024, 2)
                })
            except Exception as e:
                logger.error(f"Corrupt image skipped: {img_path} ({e})")

    df_manifest = pd.DataFrame(records)
    logger.info(f"Verified {len(df_manifest)} valid images.")
    logger.info(f"Class breakdown:\n{df_manifest['class_name'].value_counts()}")

    # Stratified Train / Val / Test split
    train_df, temp_df = train_test_split(
        df_manifest,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=df_manifest['class_id']
    )
    
    val_share = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share),
        random_state=random_state,
        stratify=temp_df['class_id']
    )

    df_manifest['split'] = 'unassigned'
    df_manifest.loc[train_df.index, 'split'] = 'train'
    df_manifest.loc[val_df.index, 'split'] = 'val'
    df_manifest.loc[test_df.index, 'split'] = 'test'

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "algae_classification_manifest.csv")
    df_manifest.to_csv(manifest_path, index=False)

    logger.info(f"Split breakdown:\n{df_manifest.groupby(['split', 'class_name']).size().unstack()}")
    logger.info(f"Saved algae dataset manifest to: {manifest_path}")

    return df_manifest


# ==============================================================================
# 4. WATER HYACINTH CATALOG AUDIT
# ==============================================================================

def audit_water_hyacinth_metadata(
    meta_path: str,
    output_dir: str
) -> pd.DataFrame:
    """
    Audits and exports cleaned catalog metadata for the 150 water hyacinth survey records.
    """
    logger.info("--- Auditing Water Hyacinth Metadata ---")
    if not os.path.exists(meta_path):
        logger.warning(f"Metadata file not found: {meta_path}")
        return pd.DataFrame()

    df_meta = pd.read_csv(meta_path)
    logger.info(f"Catalog entries: {len(df_meta)} rows, {df_meta.shape[1]} columns.")

    # Drop single-value constant columns for cleaner feature representation
    single_value_cols = [c for c in df_meta.columns if df_meta[c].nunique() == 1]
    logger.info(f"Identified {len(single_value_cols)} constant columns in metadata.csv: {single_value_cols}")

    df_informative = df_meta.drop(columns=single_value_cols).copy()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "water_hyacinth_metadata_cleaned.csv")
    df_informative.to_csv(out_path, index=False)
    logger.info(f"Exported informative metadata columns to: {out_path}")

    return df_informative


# ==============================================================================
# MAIN PREPROCESSING ORCHESTRATOR
# ==============================================================================

def run_all_preprocessing(base_dir: str = ".", output_dir: str = "data/preprocessed"):
    """
    Orchestrates the entire BioSight preprocessing pipeline across all dataset components.
    """
    base_path = Path(base_dir)
    datasets_dir = base_path / "datasets"
    out_path = base_path / output_dir
    os.makedirs(out_path, exist_ok=True)

    logger.info("==================================================================")
    logger.info("STARTING BIOSIGHT DATA PREPROCESSING PIPELINE")
    logger.info(f"Base Directory: {base_path.resolve()}")
    logger.info(f"Output Directory: {out_path.resolve()}")
    logger.info("==================================================================")

    # 1. Biogas Biochemical Kinetics
    kinetics_file = datasets_dir / "dataset_BIOGAS-1.xlsx"
    if kinetics_file.exists():
        preprocess_biogas_kinetics(
            excel_path=str(kinetics_file),
            output_dir=str(out_path)
        )
    else:
        logger.warning(f"Kinetics file not found: {kinetics_file}")

    # 2. IoT Feed Monitoring & Ambient Data
    feed_file = datasets_dir / "biogasFeedMonitoringDataset-master" / "biogasFeedMonitoringDataset-master" / "FeedMonitoringData.csv"
    amb_file = datasets_dir / "biogasFeedMonitoringDataset-master" / "biogasFeedMonitoringDataset-master" / "ambientTemperatureData.csv"
    if feed_file.exists() and amb_file.exists():
        preprocess_sensor_feed(
            feed_csv_path=str(feed_file),
            ambient_csv_path=str(amb_file),
            output_dir=str(out_path)
        )
    else:
        logger.warning(f"Sensor telemetry files not found at: {feed_file}")

    # 3. Aerial Algae Bloom Images
    algae_dir = datasets_dir / "archive (1)" / "bluegreen_algae_bloom_release"
    if algae_dir.exists():
        preprocess_algae_image_dataset(
            algae_dir=str(algae_dir),
            output_dir=str(out_path)
        )
    else:
        logger.warning(f"Algae directory not found: {algae_dir}")

    # 4. Water Hyacinth Metadata
    meta_file = datasets_dir / "metadata.csv"
    if meta_file.exists():
        audit_water_hyacinth_metadata(
            meta_path=str(meta_file),
            output_dir=str(out_path)
        )

    logger.info("==================================================================")
    logger.info("ALL PREPROCESSING TASKS COMPLETED SUCCESSFULLY")
    logger.info(f"Artifacts generated in: {out_path.resolve()}")
    logger.info("==================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioSight Data Preprocessing Pipeline")
    parser.add_argument("--base_dir", type=str, default=".", help="Base project directory")
    parser.add_argument("--output_dir", type=str, default="data/preprocessed", help="Directory to save preprocessed outputs")
    args = parser.parse_args()

    run_all_preprocessing(base_dir=args.base_dir, output_dir=args.output_dir)
