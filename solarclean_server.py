"""
SolarClean AI — Real-Time Operational Decision Support Server
=============================================================
Provides real-time REST API endpoints and serves the clean, minimal dashboard.
Integrates live Open-Meteo weather, the trained soiling models, and central configuration.
"""

import os
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import pandas as pd

import solarclean_config as cfg
from live_weather_service import fetch_live_weather
from recommendation_engine import evaluate_cleaning_decision

PORT = 8085
DASHBOARD_DIR = cfg.DASHBOARD_DIR

class SolarCleanHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/live-status":
            self.send_json_response(self.handle_live_status())
        elif path == "/api/historical":
            self.send_json_response(self.handle_historical())
        elif path == "/api/config":
            self.send_json_response(cfg.get_config_dict())
        else:
            # Fall back to static dashboard files
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/evaluate":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                overrides = json.loads(post_data) if post_data else {}
            except Exception:
                overrides = {}
            
            rec = evaluate_cleaning_decision(custom_overrides=overrides)
            self.send_json_response(rec)
        else:
            self.send_error(404, "Endpoint not found")

def get_live_status_data():
    """Fetch live weather and run recommendation engine."""
    weather = fetch_live_weather()
    recommendation = evaluate_cleaning_decision(weather_data=weather)
    
    meta_path = os.path.join(cfg.MODELS_DIR, "model_metadata.json")
    model_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            model_meta = json.load(f)

    return {
        "timestamp": weather.get("fetched_at_utc"),
        "status": "ONLINE",
        "weather": weather,
        "recommendation": recommendation,
        "model_metadata": model_meta,
        "config": cfg.get_config_dict()
    }

def get_historical_data():
    """Return validated 2022 historical time series for the secondary chart."""
    soiling_file = os.path.join(cfg.DATA_DIR, "soiling_estimation_2022.csv")
    daily_file = os.path.join(cfg.DATA_DIR, "daily_clean_2022.csv")
    
    if not os.path.exists(soiling_file) or not os.path.exists(daily_file):
        return {"error": "Historical data files not found"}

    df_soiling = pd.read_csv(soiling_file)
    df_daily = pd.read_csv(daily_file)

    df = df_daily.merge(
        df_soiling[['date', 'estimated_soiling_ratio', 'estimated_soiling_loss_pct', 'cleaning_event_candidate']],
        on='date',
        how='inner'
    )
    df = df.sort_values('date').reset_index(drop=True)

    records = []
    for _, row in df.iterrows():
        ratio = float(row['estimated_soiling_ratio']) if pd.notnull(row['estimated_soiling_ratio']) else None
        records.append({
            "date": str(row['date']),
            "performance_index": round(float(row['normalized_pi']), 3) if pd.notnull(row['normalized_pi']) else None,
            "soiling_ratio": round(ratio, 4) if ratio is not None else None,
            "soiling_loss_pct": round(float(row['estimated_soiling_loss_pct']), 2) if pd.notnull(row['estimated_soiling_loss_pct']) else None,
            "precipitation_mm": round(float(row['precipitation_sum']), 1),
            "is_recovery_event": bool(row['cleaning_event_candidate']),
            "ac_energy_kwh": round(float(row['daily_ac_energy_Wh']) / 1000.0, 2) if pd.notnull(row['daily_ac_energy_Wh']) else 0.0
        })

    return {
        "total_days": len(records),
        "date_range": ["2022-01-01", "2022-12-31"],
        "records": records
    }

def get_rainfall_analysis_data():
    """Return historical rainfall effectiveness analysis."""
    try:
        from rainfall_analyzer import analyze_historical_rainfall_events
        return analyze_historical_rainfall_events()
    except Exception as e:
        return {"error": str(e), "events": [], "summary": {}}

def get_simulation_data(overrides=None):
    """Return What-If strategy simulation comparison."""
    try:
        from cleaning_simulator import simulate_cleaning_strategies
        args = {}
        if overrides and isinstance(overrides, dict):
            op = overrides.get("operational_config", {})
            sys = overrides.get("system_specs", {})

            if "capacity_kw" in sys:
                args["capacity_kw"] = float(sys["capacity_kw"])
            elif "capacity_kw" in overrides:
                args["capacity_kw"] = float(overrides["capacity_kw"])

            if "electricity_tariff_usd_kwh" in op:
                args["tariff_usd"] = float(op["electricity_tariff_usd_kwh"])
            elif "tariff_usd" in overrides:
                args["tariff_usd"] = float(overrides["tariff_usd"])

            if "manual_wash_cost_usd" in op:
                args["wash_cost_usd"] = float(op["manual_wash_cost_usd"])
            elif "wash_cost_usd" in overrides:
                args["wash_cost_usd"] = float(overrides["wash_cost_usd"])

            if "fixed_interval_days" in overrides:
                args["fixed_interval_days"] = int(overrides["fixed_interval_days"])
            if "eval_period_days" in overrides:
                args["eval_period_days"] = int(overrides["eval_period_days"])
            if "current_loss_pct" in overrides:
                args["current_loss_pct"] = float(overrides["current_loss_pct"])
            if "forecast_rain_mm" in overrides:
                args["forecast_rain_mm"] = float(overrides["forecast_rain_mm"])
        return simulate_cleaning_strategies(**args)
    except Exception as e:
        return {"error": str(e)}

def get_screening_data():
    """Return weather-aware screening metrics and exclusion summary."""
    try:
        from weather_screening import load_or_compute_screening_summary
        return load_or_compute_screening_summary()
    except Exception as e:
        return {"error": str(e)}

class SolarCleanHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/live-status":
            self.send_json_response(get_live_status_data())
        elif path == "/api/historical":
            self.send_json_response(get_historical_data())
        elif path == "/api/rainfall-analysis":
            self.send_json_response(get_rainfall_analysis_data())
        elif path == "/api/simulate-strategies":
            self.send_json_response(get_simulation_data())
        elif path == "/api/screening-status":
            self.send_json_response(get_screening_data())
        elif path == "/api/config":
            self.send_json_response(cfg.get_config_dict())
        else:
            # Fall back to static dashboard files
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else "{}"
        try:
            overrides = json.loads(post_data) if post_data else {}
        except Exception:
            overrides = {}

        if parsed.path == "/api/evaluate":
            rec = evaluate_cleaning_decision(custom_overrides=overrides)
            self.send_json_response(rec)
        elif parsed.path == "/api/simulate-strategies":
            sim = get_simulation_data(overrides=overrides)
            self.send_json_response(sim)
        else:
            self.send_error(404, "Endpoint not found")

    def send_json_response(self, data):
        payload = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def export_static_dashboard_bundle():
    """Exports initial static bundle to data.js for offline/direct file access."""
    live_status = get_live_status_data()
    historical = get_historical_data()
    rainfall_analysis = get_rainfall_analysis_data()
    simulation = get_simulation_data()
    screening = get_screening_data()

    bundle = {
        "live_status": live_status,
        "historical": historical,
        "rainfall_analysis": rainfall_analysis,
        "simulation": simulation,
        "screening": screening
    }

    out_file = os.path.join(DASHBOARD_DIR, "data.js")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("window.SOLARCLEAN_INITIAL_DATA = ")
        json.dump(bundle, f, indent=2)
        f.write(";\n")
    print(f"Static fallback bundle written to {out_file}")

if __name__ == "__main__":
    export_static_dashboard_bundle()
    server = HTTPServer(('127.0.0.1', PORT), SolarCleanHandler)
    print(f"SolarClean AI Server listening on http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()
