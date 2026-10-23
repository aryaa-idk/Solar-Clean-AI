"""
SolarClean AI — Open-Meteo Live Weather & Forecast Service
==========================================================
Retrieves real-time meteorological observations and 7-day forecasts for NREL PVDAQ System 10
(Latitude: 39.7425, Longitude: -105.1774, Denver/Golden, Colorado).
Calculates exact consecutive dry days from actual recent precipitation history.
Implements robust timeout handling, error trapping, disk caching, and stale indicators.
"""

import os
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

import solarclean_config as cfg

CACHE_FILE = os.path.join(cfg.OUTPUT_DIR, "live_weather_cache.json")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# WMO Weather Interpretation Codes
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

def get_weather_description(code):
    return WMO_CODES.get(int(code), f"Code {code}")

def build_api_url():
    loc = cfg.LOCATION
    w_cfg = cfg.WEATHER_API_CONFIG
    params = [
        f"latitude={loc['latitude']}",
        f"longitude={loc['longitude']}",
        "current=temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
        "hourly=temperature_2m,precipitation_probability,precipitation",
        "daily=temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code",
        f"past_days={w_cfg['past_days']}",
        f"forecast_days={w_cfg['forecast_days']}",
        f"timezone={loc['timezone'].replace('/', '%2F')}"
    ]
    return f"{w_cfg['base_url']}?{'&'.join(params)}"

def fetch_live_weather(force_refresh=False):
    """
    Fetch live weather data from Open-Meteo with caching and error handling.
    Returns a validated, structured meteorological summary.
    """
    now_ts = time.time()
    w_cfg = cfg.WEATHER_API_CONFIG

    # Check cache first if not forced
    if not force_refresh and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_age = now_ts - cached.get("cached_timestamp", 0)
            if cached_age < w_cfg["cache_expiry_sec"]:
                cached["data_source"] = "LIVE_CACHE"
                cached["is_stale"] = False
                return cached
        except Exception as e:
            print(f"[WeatherService] Cache read error: {e}")

    # Fetch from Open-Meteo
    url = build_api_url()
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SolarCleanAI/2.0 (NREL PVDAQ Decision Engine)"}
        )
        with urllib.request.urlopen(req, timeout=w_cfg["timeout_sec"]) as resp:
            if resp.status != 200:
                raise ValueError(f"HTTP status {resp.status}")
            raw_data = json.loads(resp.read().decode("utf-8"))

        processed = process_weather_payload(raw_data)
        processed["cached_timestamp"] = now_ts
        processed["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
        processed["data_source"] = "LIVE_API"
        processed["is_stale"] = False
        processed["status"] = "ONLINE"

        # Save to disk cache
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2)

        return processed

    except Exception as e:
        err_msg = str(e)
        print(f"[WeatherService] Live API fetch failed ({err_msg}). Attempting fallback to disk cache...")
        
        # Fallback to stale cache if available
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                cached["data_source"] = "STALE_CACHE"
                cached["is_stale"] = True
                cached["status"] = "STALE_OFFLINE"
                cached["error_note"] = f"Live fetch failed: {err_msg}"
                return cached
            except Exception:
                pass

        # If completely unavailable
        return {
            "status": "UNAVAILABLE",
            "is_stale": True,
            "error_note": f"Weather API unreachable and no cache exists: {err_msg}",
            "current": None,
            "forecast": None,
            "dry_days_history": None
        }

def process_weather_payload(raw):
    """Parse and normalize Open-Meteo response into domain-specific structures."""
    current_raw = raw.get("current", {})
    daily_raw = raw.get("daily", {})
    hourly_raw = raw.get("hourly", {})

    # 1. Current conditions
    curr_weather_code = current_raw.get("weather_code", 0)
    current_summary = {
        "time": current_raw.get("time"),
        "temperature_c": current_raw.get("temperature_2m"),
        "relative_humidity_pct": current_raw.get("relative_humidity_2m"),
        "precipitation_mm": current_raw.get("precipitation", 0.0),
        "wind_speed_kmh": current_raw.get("wind_speed_10m"),
        "weather_code": curr_weather_code,
        "weather_desc": get_weather_description(curr_weather_code)
    }

    # 2. Daily Timeline & Dry Day Calculation
    daily_times = daily_raw.get("time", [])
    daily_precip = [float(p) if p is not None else 0.0 for p in daily_raw.get("precipitation_sum", [])]
    daily_prob = [int(p) if p is not None else 0 for p in daily_raw.get("precipitation_probability_max", [])]
    daily_tmax = [float(t) if t is not None else 0.0 for t in daily_raw.get("temperature_2m_max", [])]
    daily_tmin = [float(t) if t is not None else 0.0 for t in daily_raw.get("temperature_2m_min", [])]
    daily_codes = [int(c) if c is not None else 0 for c in daily_raw.get("weather_code", [])]

    # Find index of current date in daily series
    today_str = current_summary["time"].split("T")[0] if current_summary["time"] else datetime.now().strftime("%Y-%m-%d")
    
    if today_str in daily_times:
        today_idx = daily_times.index(today_str)
    else:
        today_idx = max(0, len(daily_times) - 8) # Approx index

    # Historical precipitation up to yesterday
    past_days_precip = daily_precip[:today_idx]
    
    # Calculate exact consecutive dry days (< 1.5 mm precipitation threshold for rain cleaning)
    rain_threshold = 1.5
    consecutive_dry_days = 0
    last_rain_date = "None in past 30 days"
    last_rain_amount_mm = 0.0

    for idx in range(len(past_days_precip) - 1, -1, -1):
        p_amount = past_days_precip[idx]
        if p_amount < rain_threshold:
            consecutive_dry_days += 1
        else:
            last_rain_date = daily_times[idx]
            last_rain_amount_mm = p_amount
            break

    # Precipitation in last 3 and 7 days
    rain_last_3d = sum(daily_precip[max(0, today_idx-3):today_idx])
    rain_last_7d = sum(daily_precip[max(0, today_idx-7):today_idx])

    # 3. Forecast Horizons (Next 24h, 48h, 72h)
    forecast_days = []
    future_precip_72h = 0.0
    max_prob_72h = 0

    for i in range(today_idx, min(len(daily_times), today_idx + 8)):
        p_val = daily_precip[i]
        prob_val = daily_prob[i]
        d_item = {
            "date": daily_times[i],
            "is_today": (i == today_idx),
            "temp_max_c": daily_tmax[i],
            "temp_min_c": daily_tmin[i],
            "precipitation_mm": p_val,
            "rain_probability_pct": prob_val,
            "weather_code": daily_codes[i],
            "weather_desc": get_weather_description(daily_codes[i])
        }
        forecast_days.append(d_item)

        # Accumulate next 3 days (today + next 2 days)
        if 0 <= (i - today_idx) <= 3:
            future_precip_72h += p_val
            max_prob_72h = max(max_prob_72h, prob_val)

    # Hourly forecast for the next 24 hours
    hourly_times = hourly_raw.get("time", [])
    hourly_precip = hourly_raw.get("precipitation", [])
    hourly_prob = hourly_raw.get("precipitation_probability", [])
    hourly_temp = hourly_raw.get("temperature_2m", [])

    next_24h_hourly = []
    now_hour_str = current_summary["time"][:13] if current_summary["time"] else ""
    start_h = 0
    for idx, ht in enumerate(hourly_times):
        if now_hour_str and ht.startswith(now_hour_str):
            start_h = idx
            break

    for h in range(start_h, min(len(hourly_times), start_h + 24)):
        next_24h_hourly.append({
            "time": hourly_times[h],
            "temp_c": hourly_temp[h] if h < len(hourly_temp) else None,
            "precipitation_mm": hourly_precip[h] if h < len(hourly_precip) else 0.0,
            "probability_pct": hourly_prob[h] if h < len(hourly_prob) else 0
        })

    next_24h_rain_sum = sum(item["precipitation_mm"] for item in next_24h_hourly)

    return {
        "location": cfg.LOCATION,
        "current": current_summary,
        "dry_streak": {
            "consecutive_dry_days": consecutive_dry_days,
            "last_rain_date": last_rain_date,
            "last_rain_amount_mm": round(last_rain_amount_mm, 1),
            "rain_last_3d_mm": round(rain_last_3d, 1),
            "rain_last_7d_mm": round(rain_last_7d, 1),
            "definition": f"Days since precipitation >= {rain_threshold} mm"
        },
        "forecast_horizons": {
            "rain_next_24h_mm": round(next_24h_rain_sum, 1),
            "rain_next_72h_mm": round(future_precip_72h, 1),
            "max_rain_probability_72h_pct": max_prob_72h
        },
        "daily_forecast": forecast_days,
        "hourly_next_24h": next_24h_hourly
    }

if __name__ == "__main__":
    print("Testing Live Weather Service...")
    res = fetch_live_weather(force_refresh=True)
    print(f"Status: {res['status']} (Source: {res['data_source']})")
    print(f"Current Temp: {res['current']['temperature_c']}°C, Condition: {res['current']['weather_desc']}")
    print(f"Dry Days Streak: {res['dry_streak']['consecutive_dry_days']} days (Last rain: {res['dry_streak']['last_rain_date']})")
    print(f"Next 72h Rain: {res['forecast_horizons']['rain_next_72h_mm']} mm (Max prob: {res['forecast_horizons']['max_rain_probability_72h_pct']}%)")
