"""
SolarClean AI — Central Configuration & Operational Parameters
==============================================================
Single source of truth for geographical, physical, and operational parameters.
Every assumption is explicitly labeled and documented.
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

# Geographical Location: NREL PVDAQ System 10 (Golden, Colorado, USA)
LOCATION = {
    "name": "NREL PVDAQ — System 10",
    "city": "Golden",
    "state": "Colorado",
    "country": "USA",
    "full_location": "Golden, Colorado, USA",
    "latitude": 39.7425,
    "longitude": -105.1774,
    "timezone": "America/Denver"
}

# Physical & System Characteristics (PVDAQ System 10)
SYSTEM_SPECS = {
    "capacity_kw": 3.0,  # Configured nameplate DC capacity assumption
    "tilt_deg": 40.0,    # Fixed tilt angle
    "azimuth_deg": 180.0 # South-facing
}

# Configurable Economic & Operational Thresholds
OPERATIONAL_CONFIG = {
    # Thresholds for Cleaning Recommendations
    "urgent_clean_loss_pct": 7.5,   # % soiling loss above which cleaning is considered
    "monitor_loss_pct": 3.0,        # % soiling loss above which close monitoring begins
    "rain_wash_threshold_mm": 3.0,  # mm of rainfall considered effective for natural dust wash
    "rain_forecast_window_days": 3, # Horizon (days) to check for upcoming natural cleaning
    "min_forecast_prob_pct": 50,    # Minimum rain probability to factor into cleaning decisions

    # Financial Assumptions (clearly identified as configurable estimates, not verified ground truth)
    "electricity_tariff_usd_kwh": 0.15, # Assumed electricity revenue / savings rate ($/kWh)
    "manual_wash_cost_usd": 150.00,     # Commercial manual cleaning service fee ($/wash)
    "daily_yield_kwh_per_kw": 4.2       # Historical average daily yield factor in Colorado (kWh/kW/day)
}

# Open-Meteo Live Weather API Configuration
WEATHER_API_CONFIG = {
    "base_url": "https://api.open-meteo.com/v1/forecast",
    "timeout_sec": 12,
    "cache_expiry_sec": 1800, # 30-minute cache validity
    "past_days": 30,          # Days of recent history to fetch for dry streak tracking
    "forecast_days": 7        # Days of upcoming forecast
}

def get_config_dict():
    """Return all configuration parameters in a serializable dictionary."""
    return {
        "location": LOCATION,
        "system_specs": SYSTEM_SPECS,
        "operational_config": OPERATIONAL_CONFIG,
        "weather_api": WEATHER_API_CONFIG
    }

def export_config_json(filepath=None):
    """Save config to JSON for dashboard consumption."""
    if filepath is None:
        filepath = os.path.join(DASHBOARD_DIR, "config.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(get_config_dict(), f, indent=2)
    return filepath

if __name__ == "__main__":
    p = export_config_json()
    print(f"Centralized configuration exported to {p}")
