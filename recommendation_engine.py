"""
SolarClean AI — Real-Time Cleaning Recommendation Engine
=========================================================
Synthesizes live meteorological evidence, recent dry streak history,
empirical/ML soiling accumulation models, and configurable economic thresholds
to produce transparent, defensible cleaning guidance.

Labels:
- CLEAN NOW
- MONITOR
- WAIT FOR RAIN
- INSUFFICIENT DATA
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

import solarclean_config as cfg
from live_weather_service import fetch_live_weather

MODELS_DIR = cfg.MODELS_DIR
MODEL_FILE = os.path.join(MODELS_DIR, "soiling_model.joblib")

# Load model if available
_gbr_model = None
if os.path.exists(MODEL_FILE):
    try:
        _gbr_model = joblib.load(MODEL_FILE)
    except Exception as e:
        print(f"[RecommendationEngine] Could not load GBR model: {e}")

def estimate_current_soiling_loss(dry_days, rain_last_3d, rain_last_7d, current_temp):
    """
    Computes defensible soiling loss estimate using empirical baseline
    and trained model features.
    """
    # 1. Empirical baseline from NREL rdtools SRR dry intervals (~0.16% loss / dry day)
    # Caps at 14% based on typical Colorado equilibrium between wind scour and deposition
    empirical_loss = 0.4 + (dry_days * cfg.OPERATIONAL_CONFIG.get("daily_soiling_rate_pct", 0.16))
    empirical_loss = max(0.4, min(14.0, empirical_loss))

    # If GBR model is loaded, also compute ML prediction
    ml_loss = None
    if _gbr_model is not None:
        try:
            cum_insolation = dry_days * 4.5
            feat_df = pd.DataFrame([{
                'consecutive_dry_days': dry_days,
                'cum_insolation_since_rain': cum_insolation,
                'rain_last_3d': rain_last_3d,
                'rain_last_7d': rain_last_7d,
                'temp_ambient_avg': current_temp if current_temp is not None else 15.0,
                'temp_module_avg': (current_temp + 12.0) if current_temp is not None else 27.0
            }])
            ml_pred = float(_gbr_model.predict(feat_df)[0])
            ml_loss = round(max(0.4, min(15.0, ml_pred)), 2)
        except Exception as e:
            print(f"[RecommendationEngine] GBR inference note: {e}")

    # Use empirical rate as primary (grounded in rdtools interval slope), with ML as supporting
    primary_loss = round(empirical_loss, 2)
    return {
        "primary_loss_pct": primary_loss,
        "empirical_loss_pct": round(empirical_loss, 2),
        "ml_model_loss_pct": ml_loss,
        "soiling_ratio": round(1.0 - (primary_loss / 100.0), 4)
    }

def evaluate_cleaning_decision(weather_data=None, custom_overrides=None):
    """
    Core Recommendation Rule Engine.
    Combines live weather evidence and configured thresholds.
    """
    op_cfg = cfg.OPERATIONAL_CONFIG.copy()
    sys_specs = cfg.SYSTEM_SPECS.copy()

    # Apply any custom operational overrides (e.g. from user slider or what-if)
    if custom_overrides:
        op_cfg.update(custom_overrides.get("operational_config", {}))
        sys_specs.update(custom_overrides.get("system_specs", {}))

    # Fetch live weather if not provided
    if weather_data is None:
        weather_data = fetch_live_weather()

    # 1. Check for data insufficiency
    if not weather_data or weather_data.get("status") == "UNAVAILABLE" or not weather_data.get("current"):
        return {
            "recommendation": "INSUFFICIENT DATA",
            "status_color": "dim",
            "action_title": "Weather Data Unavailable",
            "reason": "Live meteorological telemetry and forecasts could not be retrieved from Open-Meteo, and no valid cache exists. Decision support is suspended until connectivity is restored.",
            "data_confidence": "none",
            "uncertainty_note": "A valid weather observation or forecast is required to formulate an actionable recommendation.",
            "weather_status": "OFFLINE",
            "metrics": None,
            "financials": None,
            "config_used": op_cfg
        }

    # Extract weather observations
    current = weather_data["current"]
    dry_info = weather_data.get("dry_streak", {})
    horizons = weather_data.get("forecast_horizons", {})

    dry_days = dry_info.get("consecutive_dry_days", 0)
    rain_last_3d = dry_info.get("rain_last_3d_mm", 0.0)
    rain_last_7d = dry_info.get("rain_last_7d_mm", 0.0)
    current_temp = current.get("temperature_c", 15.0)

    # Forecast values
    rain_72h = horizons.get("rain_next_72h_mm", 0.0)
    prob_72h = horizons.get("max_rain_probability_72h_pct", 0)
    rain_24h = horizons.get("rain_next_24h_mm", 0.0)

    # Allow overriding dry days or rain forecast if testing
    if custom_overrides and "dry_days" in custom_overrides:
        dry_days = int(custom_overrides["dry_days"])
    if custom_overrides and "rain_forecast_72h" in custom_overrides:
        rain_72h = float(custom_overrides["rain_forecast_72h"])
        prob_72h = int(custom_overrides.get("rain_prob_72h", 60))

    # 2. Estimate Soiling Loss
    soiling_est = estimate_current_soiling_loss(dry_days, rain_last_3d, rain_last_7d, current_temp)
    loss_pct = soiling_est["primary_loss_pct"]
    soiling_ratio = soiling_est["soiling_ratio"]

    # 3. Calculate Financials using Documented Formulas
    cap_kw = sys_specs["capacity_kw"]
    tariff = op_cfg["electricity_tariff_usd_kwh"]
    wash_cost = op_cfg["manual_wash_cost_usd"]
    daily_yield_factor = op_cfg["daily_yield_kwh_per_kw"]

    daily_clean_gen_kwh = cap_kw * daily_yield_factor
    daily_lost_kwh = daily_clean_gen_kwh * (loss_pct / 100.0)
    daily_lost_usd = daily_lost_kwh * tariff
    monthly_lost_usd = daily_lost_usd * 30.0

    # 4. Decision Rules
    clean_threshold = op_cfg["urgent_clean_loss_pct"]
    monitor_threshold = op_cfg["monitor_loss_pct"]
    rain_wash_thresh = op_cfg["rain_wash_threshold_mm"]
    min_prob = op_cfg["min_forecast_prob_pct"]

    # Rule A: Meaningful rain forecast soon and loss is not extreme -> WAIT FOR RAIN
    if (rain_72h >= rain_wash_thresh) and (prob_72h >= min_prob) and (loss_pct < 12.0):
        recommendation = "WAIT FOR RAIN"
        status_color = "cyan"
        action_title = "Natural Cleaning Forecast Within 72h"
        reason = (
            f"Precipitation of {rain_72h:.1f} mm is forecast within the next 72 hours ({prob_72h}% probability). "
            f"Because estimated soiling loss is currently {loss_pct:.1f}% (below critical threshold), "
            f"natural rain wash-off is expected to rinse panels without manual expenditure (${wash_cost:.0f})."
        )
        data_confidence = "high"

    # Rule B: High soiling loss and no rain forecast -> CLEAN NOW
    elif (loss_pct >= clean_threshold) and (rain_72h < rain_wash_thresh or prob_72h < 30):
        recommendation = "CLEAN NOW"
        status_color = "rose"
        action_title = "Action Required: Clean Array"
        reason = (
            f"Estimated soiling loss has reached {loss_pct:.1f}% after {dry_days} consecutive dry days. "
            f"No meaningful rain is forecast over the next 72 hours ({rain_72h:.1f} mm expected). "
            f"Manual cleaning is recommended to recover approximately ${daily_lost_usd:.2f}/day in solar generation."
        )
        data_confidence = "medium"

    # Rule C: Moderate soiling or dry streak continuing -> MONITOR
    elif loss_pct >= monitor_threshold:
        recommendation = "MONITOR"
        status_color = "gold"
        action_title = "Monitor: Accumulating Soiling"
        reason = (
            f"Estimated soiling loss is {loss_pct:.1f}% after {dry_days} dry days. "
            f"Generation loss is approximately ${daily_lost_usd:.2f}/day (${monthly_lost_usd:.2f}/month). "
            f"Upcoming rain forecast is {rain_72h:.1f} mm. Immediate washing is not yet cost-effective; continue monitoring."
        )
        data_confidence = "medium"

    # Rule D: Minimal soiling -> MONITOR (nominal)
    else:
        recommendation = "MONITOR"
        status_color = "emerald"
        action_title = "Optimal Performance"
        reason = (
            f"Estimated soiling loss is low ({loss_pct:.1f}%), and panels are operating near {soiling_ratio*100:.1f}% efficiency. "
            f"Revenue impact is negligible (${daily_lost_usd:.2f}/day). No cleaning action required."
        )
        data_confidence = "high"

    uncertainty_note = (
        "Weather-aware screening reduces the influence of selected confounding conditions but "
        "cannot completely distinguish soiling from clouds, snow, shading, equipment faults, or "
        "model error. Direct soiling measurements are not available."
    )

    # 5. Integrated What-If Strategy Simulation
    try:
        from cleaning_simulator import simulate_cleaning_strategies
        simulation_res = simulate_cleaning_strategies(
            current_loss_pct=loss_pct,
            forecast_rain_mm=rain_72h,
            forecast_rain_day=2,
            capacity_kw=cap_kw,
            tariff_usd=tariff,
            wash_cost_usd=wash_cost
        )
    except Exception as e:
        print(f"[RecommendationEngine] Simulation note: {e}")
        simulation_res = None

    return {
        "recommendation": recommendation,
        "status_color": status_color,
        "action_title": action_title,
        "reason": reason,
        "data_confidence": data_confidence,
        "uncertainty_note": uncertainty_note,
        "weather_status": weather_data.get("status", "ONLINE"),
        "is_stale_weather": weather_data.get("is_stale", False),
        "strategy_simulation": simulation_res,
        "metrics": {
            "estimated_soiling_ratio": soiling_ratio,
            "estimated_soiling_loss_pct": loss_pct,
            "consecutive_dry_days": dry_days,
            "rain_next_72h_mm": rain_72h,
            "rain_prob_72h_pct": prob_72h,
            "current_temp_c": current_temp,
            "weather_condition": current.get("weather_desc", "Unknown"),
            "last_rain_date": dry_info.get("last_rain_date", "N/A"),
            "soiling_est_breakdown": soiling_est
        },
        "financials": {
            "daily_loss_kwh": round(daily_lost_kwh, 2),
            "daily_loss_usd": round(daily_lost_usd, 2),
            "monthly_loss_usd": round(monthly_lost_usd, 2),
            "assumed_capacity_kw": cap_kw,
            "assumed_tariff_usd": tariff,
            "assumed_wash_cost_usd": wash_cost
        },
        "config_used": {
            "clean_loss_threshold": clean_threshold,
            "monitor_threshold": monitor_threshold,
            "rain_wash_threshold_mm": rain_wash_thresh
        }
    }

if __name__ == "__main__":
    print("Evaluating Live Cleaning Decision...")
    rec = evaluate_cleaning_decision()
    print(f"RECOMMENDATION: {rec['recommendation']} ({rec['action_title']})")
    print(f"Reason: {rec['reason']}")
    print(f"Metrics: {rec['metrics']}")
    print(f"Financials: {rec['financials']}")
