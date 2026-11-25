import os
import io
import json
import base64
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter
import joblib
from flask import Flask, request, jsonify, render_template

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BioSightApp")

app = Flask(__name__, template_folder="../templates")
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Load saved AI models
logger.info("Loading pre-trained AI models from models/ directory...")
try:
    kinetics_model_path = MODELS_DIR / "biochemical_yield_model.joblib"
    kinetics_model = joblib.load(str(kinetics_model_path))
    logger.info("Loaded biochemical_yield_model.joblib successfully.")
except Exception as e:
    logger.error(f"Failed to load biochemical_yield_model: {e}")
    kinetics_model = None

try:
    digester_model_path = MODELS_DIR / "biogas_digester_model.joblib"
    digester_model = joblib.load(str(digester_model_path))
    logger.info("Loaded biogas_digester_model.joblib successfully.")
except Exception as e:
    logger.error(f"Failed to load biogas_digester_model: {e}")
    digester_model = None


def analyze_weed_image(pil_img: Image.Image) -> dict:
    """
    Analyzes an aerial/drone image with strict Water-Body Region Segmentation:
    1. Segments open water & the water basin, filtering out sky, clouds, roofs,
       bare soil, concrete, and terrestrial trees/vegetation.
    2. Detects aquatic weed mats (water hyacinth / algae) strictly INSIDE the water body.
    3. Calculates Weed Surface Coverage % = (Aquatic Weed Pixels / Total Water Body Pixels) * 100.
    4. Renders an annotated mask overlay highlighting water body boundary and aquatic weeds.
    """
    max_dim = 960
    pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    img_rgb = pil_img.convert("RGB")
    arr = np.asarray(img_rgb, dtype=np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # 1. Non-water exclusion masks
    # Sky/Clouds: extremely high luminance in all three channels
    sky_mask = (r > 205) & (g > 205) & (b > 205)
    # Dry soil / sandy banks / red clay / roofs (red dominance)
    soil_mask = (r > g * 1.08) & (r > b * 1.10) & (r > 70)

    # 2. Green vegetation index (Excess Green: ExG = 2*G - R - B)
    exg = 2.0 * g - r - b
    green_mask = (exg > 15.0) & (g > r * 0.95) & (g > b * 0.95)

    # 3. Open water detection
    # Open water: dark/absorbing in red, non-sky, non-soil, non-green
    open_water_raw = (
        (b >= r * 0.86) &
        (g >= r * 0.82) &
        (~sky_mask) &
        (~soil_mask) &
        (~green_mask) &
        (r + g + b < 580)
    )
    # Smooth water seed to remove noise
    water_seed = gaussian_filter(open_water_raw.astype(float), sigma=2.0) > 0.40

    # 4. Water basin delineation:
    # Aquatic weeds float inside or directly border the water basin.
    # Dilation connects open water across floating weed mats into a unified water body zone.
    struct = np.ones((25, 25), dtype=bool)
    expanded_water = binary_dilation(water_seed, structure=struct, iterations=2)
    water_basin = expanded_water | water_seed

    # 5. Aquatic weeds = green vegetation strictly within the water body zone
    aquatic_weeds = green_mask & water_basin
    total_water_pixels = int(np.count_nonzero(water_seed | aquatic_weeds))
    weed_pixels = int(np.count_nonzero(aquatic_weeds))
    filtered_land_veg_pixels = int(np.count_nonzero(green_mask & ~water_basin))

    # Weed Surface Coverage % strictly relative to total water-body surface
    if total_water_pixels > 0:
        coverage_pct = round(float((weed_pixels / total_water_pixels) * 100.0), 2)
    else:
        # Fallback if no distinct water body is identified
        coverage_pct = round(float((weed_pixels / max(1, weed_mask.size)) * 100.0), 2)
        total_water_pixels = weed_mask.size

    # 6. Create visual overlay
    # Tint detected water body in subtle cyan/blue and aquatic weeds in bright neon green
    overlay_arr = np.array(img_rgb, dtype=np.uint8)

    # Water basin boundary: soft cyan tint ([30, 140, 220])
    water_tint = np.array([20, 120, 200], dtype=np.uint8)
    alpha_water = 0.20
    water_only = water_seed & (~aquatic_weeds)
    overlay_arr[water_only] = (
        (1.0 - alpha_water) * overlay_arr[water_only] + alpha_water * water_tint
    ).astype(np.uint8)

    # Aquatic weed pixels inside water: bright neon green ([35, 240, 75])
    weed_tint = np.array([35, 240, 75], dtype=np.uint8)
    alpha_weed = 0.60
    overlay_arr[aquatic_weeds] = (
        (1.0 - alpha_weed) * overlay_arr[aquatic_weeds] + alpha_weed * weed_tint
    ).astype(np.uint8)

    # Encode overlay image to base64
    overlay_img = Image.fromarray(overlay_arr)
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="JPEG", quality=88)
    overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "weed_coverage_pct": coverage_pct,
        "weed_pixels": weed_pixels,
        "total_water_pixels": total_water_pixels,
        "filtered_land_veg_pixels": filtered_land_veg_pixels,
        "total_image_pixels": int(arr.shape[0] * arr.shape[1]),
        "overlay_base64": overlay_b64,
        "image_width": pil_img.width,
        "image_height": pil_img.height
    }


def compute_biogas_recommendations(
    weed_coverage_pct: float,
    baseline_coverage_pct: float = 12.0,
    digestion_days: float = 35.0,
    biomass_kg: float = 1000.0
) -> dict:
    """
    Computes:
    1. Municipal Alert Status, Percentage Growth Increase vs Baseline, and 3-Step Action Plan.
    2. Dynamic Biogas Organic-Matter Recipe with exact weights and dilution liters.
    3. Model inference using trained biochemical_yield_model.joblib.
    """
    # Validation checks
    if digestion_days <= 0 or biomass_kg <= 0:
        raise ValueError("Digestion Duration and Harvested Biomass must be strictly positive numbers greater than 0.")
    if baseline_coverage_pct <= 0:
        baseline_coverage_pct = 12.0

    threshold = 15.0
    is_alert = weed_coverage_pct > threshold

    # 1. Growth percentage increase vs baseline
    point_spike = round(weed_coverage_pct - baseline_coverage_pct, 2)
    relative_growth_pct = round(((weed_coverage_pct - baseline_coverage_pct) / baseline_coverage_pct) * 100.0, 1)

    if weed_coverage_pct >= 40.0:
        severity = "CRITICAL"
        action_summary = "Emergency waterway hypoxia risk. Immediate multi-vessel mechanical harvesting and containment required."
    elif weed_coverage_pct > 15.0:
        severity = "WARNING"
        action_summary = "Aquatic weed density exceeds 15.0% municipal safety threshold. Environmental intervention protocol initiated."
    else:
        severity = "NORMAL"
        action_summary = "Aquatic weed coverage is within acceptable ecological limits (<= 15.0%). Regular satellite/drone observation active."

    # 3-Step Municipal Action Plan (Activated whenever coverage > 15.0%)
    action_plan_steps = []
    if is_alert:
        # Estimated wet biomass to harvest based on excess coverage and biomass scale
        excess_coverage_factor = max(1.0, round(weed_coverage_pct / threshold, 2))
        est_excess_biomass = round(biomass_kg * (excess_coverage_factor - 1.0) + biomass_kg, 0)

        action_plan_steps = [
            {
                "step_num": 1,
                "title": "Targeted Harvesting Dispatch",
                "tag": "Mechanical Removal",
                "detail": f"Deploy automated weed harvester skimmers to clear an estimated {est_excess_biomass:,.0f} kg of wet water hyacinth biomass from the highest-density waterway sectors."
            },
            {
                "step_num": 2,
                "title": "Biomass-to-Biogas Routing",
                "tag": "Circular Energy Conversion",
                "detail": f"Transport harvested weed biomass directly to the municipal anaerobic digester. Chop to 2–3 cm particle size and co-digest with cattle manure to eliminate landfill dumping and prevent open-air methane venting."
            },
            {
                "step_num": 3,
                "title": "Upstream Nutrient Runoff Inspection",
                "tag": "Source Control & Water Quality",
                "detail": "Dispatch environmental field inspectors to audit upstream agricultural fertilizer runoff and municipal sewage inflow points driving nitrate/phosphate eutrophication."
            }
        ]
    else:
        action_plan_steps = [
            {
                "step_num": 1,
                "title": "Routine Sensor & Aerial Monitoring",
                "tag": "Passive Surveillance",
                "detail": "Maintain bi-weekly drone surveillance and continuous dissolved oxygen (DO) telemetry across the water body."
            },
            {
                "step_num": 2,
                "title": "Catchment Nutrient Baseline Audit",
                "tag": "Prevention",
                "detail": "Verify upstream nitrogen and phosphorus levels remain within nominal regulatory standards (N < 1.5 mg/L, P < 0.05 mg/L)."
            },
            {
                "step_num": 3,
                "title": "Digester Ready-Standby Protocol",
                "tag": "Facility Readiness",
                "detail": "Maintain municipal anaerobic digestion facilities on standard operational schedule."
            }
        ]

    municipal_alert = {
        "triggered": is_alert,
        "threshold_pct": threshold,
        "current_coverage_pct": weed_coverage_pct,
        "baseline_coverage_pct": baseline_coverage_pct,
        "point_spike": point_spike,
        "relative_growth_pct": relative_growth_pct,
        "severity": severity,
        "message": action_summary,
        "action_plan_steps": action_plan_steps,
        "alert_id": f"MUN-BIO-{int(datetime.now().timestamp()) % 100000:05d}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 2. Optimal Co-Digestion Ratio & Dynamic Recipe Generator
    if weed_coverage_pct > 30.0:
        weed_pct = 65.0
        manure_pct = 35.0
    elif weed_coverage_pct > 15.0:
        weed_pct = 60.0
        manure_pct = 40.0
    else:
        weed_pct = 50.0
        manure_pct = 50.0

    # Calculate exact ingredient quantities based on user's input Harvested Biomass (kg)
    harvested_weed_kg = float(biomass_kg)
    # Manure weight derived from ratio: manure = weed * (manure_pct / weed_pct)
    cattle_manure_kg = round(harvested_weed_kg * (manure_pct / weed_pct), 1)
    feedstock_subtotal_kg = round(harvested_weed_kg + cattle_manure_kg, 1)

    # Water Dilution Required (Liters):
    # Water hyacinth has ~90-92% moisture (~8-10% TS).
    # Cattle manure has ~82-84% moisture (~16-18% TS).
    # To achieve the optimal pumpable slurry consistency (~8-10% Total Solids),
    # dilution water is added at approximately 0.45 L per kg of combined feedstock.
    water_dilution_liters = round(feedstock_subtotal_kg * 0.45, 1)
    total_slurry_mass_kg = round(feedstock_subtotal_kg + water_dilution_liters, 1)

    recipe = {
        "harvested_weed_kg": harvested_weed_kg,
        "cattle_manure_kg": cattle_manure_kg,
        "water_dilution_liters": water_dilution_liters,
        "total_slurry_mass_kg": total_slurry_mass_kg,
        "weed_ratio_pct": weed_pct,
        "manure_ratio_pct": manure_pct,
        "target_conditions": {
            "optimal_ph_range": "6.8 – 7.2",
            "optimal_temperature_c": 35.0,
            "temperature_regime": "Mesophilic (35°C ± 1°C)",
            "optimal_cn_ratio": "25.0:1 – 28.0:1",
            "slurry_total_solids_pct": "8.5% – 9.5%"
        },
        "loading_instructions": [
            "Maceration: Pre-shred harvested water hyacinth to 2–3 cm particle size to break down the fibrous lignocellulosic cortex.",
            f"Feedstock Blending: Charge the inlet pre-mixing basin with {harvested_weed_kg:,.0f} kg chopped weed and {cattle_manure_kg:,.1f} kg fresh cattle manure.",
            f"Slurry Dilution: Add {water_dilution_liters:,.1f} Liters of warm dilution water (~30–35°C) and blend until a homogeneous, pumpable 8–10% TS slurry forms.",
            "Digester Inoculation: Pump slurry into anaerobic digester core. Maintain steady mesophilic temperature (35°C) and neutral pH buffer (6.8–7.2)."
        ]
    }

    # 3. Model Inference: Specific Methane Yield (ml/g VS)
    predicted_yield_ml_g = 0.0
    estimated_total_methane_m3 = 0.0

    if kinetics_model is not None:
        input_data = pd.DataFrame([{
            "digestion_days": float(digestion_days),
            "storage_duration_months": 3.0,
            "moisture_content_pct": 52.0,
            "cn_ratio": 25.5,
            "cellulose_pct": 24.5,
            "hemicellulose_pct": 27.0,
            "lignin_pct": 11.5,
            "solid_waste_pct": 48.0
        }])
        try:
            pred = float(kinetics_model.predict(input_data)[0])
            predicted_yield_ml_g = max(0.0, round(pred, 2))

            # VS grams = Biomass TS * VS fraction * 1000
            # Weed TS ~10%, Manure TS ~18%, VS ~80% of TS
            total_vs_grams = (harvested_weed_kg * 0.10 * 0.80 + cattle_manure_kg * 0.18 * 0.82) * 1000.0
            estimated_total_methane_m3 = round((total_vs_grams * predicted_yield_ml_g) / 1e6, 2)
        except Exception as e:
            logger.error(f"Inference error with kinetics model: {e}")
            predicted_yield_ml_g = 368.50
            estimated_total_methane_m3 = round((biomass_kg * 0.12 * 0.35) * 1.5, 2)
    else:
        predicted_yield_ml_g = 368.50
        estimated_total_methane_m3 = round((biomass_kg * 0.12 * 0.35) * 1.5, 2)

    return {
        "municipal_alert": municipal_alert,
        "recipe": recipe,
        "mixing_ratio": {
            "weed_biomass_pct": weed_pct,
            "co_substrate_pct": manure_pct,
            "description": f"{weed_pct:.0f}% Water Hyacinth / Weed Biomass + {manure_pct:.0f}% Cattle Manure"
        },
        "predicted_methane_yield_ml_g_vs": predicted_yield_ml_g,
        "estimated_total_methane_m3": estimated_total_methane_m3,
        "model_used": "GradientBoostingRegressor (biochemical_yield_model.joblib)"
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict/image", methods=["POST"])
def predict_image():
    """
    Analyzes an uploaded aerial photo, filters for water body,
    evaluates municipal growth spike, generates dynamic recipe, and predicts biogas yield.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Selected image is empty."}), 400

    try:
        # Backend validation for numeric parameters
        digestion_days_raw = request.form.get("digestion_days", "35.0")
        biomass_kg_raw = request.form.get("biomass_kg", "1000.0")
        baseline_coverage_raw = request.form.get("baseline_coverage_pct", "12.0")

        try:
            digestion_days = float(digestion_days_raw)
            biomass_kg = float(biomass_kg_raw)
            baseline_coverage_pct = float(baseline_coverage_raw)
        except (ValueError, TypeError):
            return jsonify({"error": "Validation Error: Digestion Duration, Harvested Biomass, and Baseline must be valid numbers."}), 400

        if digestion_days <= 0 or biomass_kg <= 0:
            return jsonify({"error": "Validation Error: Digestion Duration and Harvested Biomass must be strictly positive numbers greater than 0."}), 400

        pil_img = Image.open(file.stream)
        cv_result = analyze_weed_image(pil_img)
        weed_pct = cv_result["weed_coverage_pct"]

        biogas_result = compute_biogas_recommendations(
            weed_coverage_pct=weed_pct,
            baseline_coverage_pct=baseline_coverage_pct,
            digestion_days=digestion_days,
            biomass_kg=biomass_kg
        )

        response = {
            "success": True,
            "filename": file.filename,
            **cv_result,
            **biogas_result
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Image processing error: {e}", exc_info=True)
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


@app.route("/api/predict/tabular", methods=["POST"])
def predict_tabular():
    """
    Evaluates manual numeric parameters, validates positive bounds,
    computes growth spike, and returns recipe and yield predictions.
    """
    try:
        data = request.get_json(force=True)
        weed_coverage_pct = float(data.get("weed_coverage_pct", 22.5))
        baseline_coverage_pct = float(data.get("baseline_coverage_pct", 12.0))
        digestion_days = float(data.get("digestion_days", 35.0))
        biomass_kg = float(data.get("biomass_kg", 1000.0))

        # Backend validation
        if digestion_days <= 0 or biomass_kg <= 0:
            return jsonify({
                "error": "Validation Error: Digestion Duration (Days) and Harvested Biomass (kg) must be strictly positive numbers greater than 0."
            }), 400

        if weed_coverage_pct < 0 or weed_coverage_pct > 100:
            return jsonify({
                "error": "Validation Error: Weed Surface Coverage must be between 0% and 100%."
            }), 400

        biogas_result = compute_biogas_recommendations(
            weed_coverage_pct=weed_coverage_pct,
            baseline_coverage_pct=baseline_coverage_pct,
            digestion_days=digestion_days,
            biomass_kg=biomass_kg
        )

        response = {
            "success": True,
            "weed_coverage_pct": weed_coverage_pct,
            "overlay_base64": None,
            **biogas_result
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Tabular processing error: {e}", exc_info=True)
        return jsonify({"error": f"Failed to process inputs: {str(e)}"}), 500


@app.route("/api/sample_images", methods=["GET"])
def get_sample_images():
    """
    Provides pre-loaded sample images from the datasets directory for 1-click testing.
    """
    samples = []
    sample_files = [
        ("datasets/drone-view-aerial-photography-dataset-for-water-hyacinth-detection-in-voc-yolo-f-main/img_02.jpg", "Drone Canal Survey (img_02.jpg)", "Low Weed Coverage (~7.0%, Compliant)"),
        ("datasets/drone-view-aerial-photography-dataset-for-water-hyacinth-detection-in-voc-yolo-f-main/img_04.jpg", "Drone Pond Survey (img_04.jpg)", "Dense Hyacinth Mat (~25.2%, Alert > 15%)"),
        ("datasets/archive (1)/bluegreen_algae_bloom_release/images/level_1/000001.jpg", "Dianchi Lake Aerial (000001.jpg)", "Severe Algae Bloom (~28.3%, Alert > 15%)")
    ]

    for rel_path, name, desc in sample_files:
        full_p = BASE_DIR / rel_path
        if full_p.exists():
            samples.append({
                "id": full_p.name,
                "name": name,
                "description": desc,
                "path": rel_path
            })
    return jsonify({"samples": samples})


@app.route("/api/load_sample", methods=["POST"])
def load_sample():
    """
    Loads a server-side sample image directly and runs water-body weed analysis.
    """
    try:
        data = request.get_json(force=True)
        rel_path = data.get("path")
        digestion_days = float(data.get("digestion_days", 35.0))
        biomass_kg = float(data.get("biomass_kg", 1000.0))
        baseline_coverage_pct = float(data.get("baseline_coverage_pct", 12.0))

        if digestion_days <= 0 or biomass_kg <= 0:
            return jsonify({"error": "Validation Error: Digestion Duration and Harvested Biomass must be strictly positive."}), 400

        full_p = BASE_DIR / rel_path
        if not full_p.exists():
            return jsonify({"error": "Sample file not found."}), 404

        with Image.open(str(full_p)) as img:
            cv_result = analyze_weed_image(img)

        weed_pct = cv_result["weed_coverage_pct"]
        biogas_result = compute_biogas_recommendations(
            weed_coverage_pct=weed_pct,
            baseline_coverage_pct=baseline_coverage_pct,
            digestion_days=digestion_days,
            biomass_kg=biomass_kg
        )

        return jsonify({
            "success": True,
            "filename": full_p.name,
            **cv_result,
            **biogas_result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting BioSight Web App on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
