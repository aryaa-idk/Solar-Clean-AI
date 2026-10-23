"""
Prepare consolidated data for SolarClean AI Interactive Web Dashboard.
Outputs a JavaScript file defining window.SOLARCLEAN_DATA so that the dashboard
can run both as a standalone file (file://) and over HTTP without CORS issues.
"""

import json
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")
os.makedirs(DASHBOARD_DIR, exist_ok=True)

# 1. Load pipeline report
report_path = os.path.join(OUTPUT_DIR, "pipeline_report.json")
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

# 2. Load CSV datasets
daily_clean_path = os.path.join(DATA_DIR, "daily_clean_2022.csv")
soiling_path = os.path.join(DATA_DIR, "soiling_estimation_2022.csv")
recs_path = os.path.join(DATA_DIR, "cleaning_recommendations_2022.csv")

df_daily = pd.read_csv(daily_clean_path)
df_soiling = pd.read_csv(soiling_path)
df_recs = pd.read_csv(recs_path)

# Merge datasets on 'date'
df_merged = df_daily.merge(df_soiling[['date', 'cleaning_event_candidate', 'estimated_soiling_ratio', 'estimated_soiling_loss_pct', 'estimation_quality_flag']], on="date", how="left")
df_merged = df_merged.merge(df_recs[['date', 'recommendation', 'reason', 'confidence']], on="date", how="left")

# Ensure dates are sorted
df_merged['date'] = pd.to_datetime(df_merged['date'])
df_merged = df_merged.sort_values('date').reset_index(drop=True)

# Helper calculations
df_merged['energy_kwh'] = df_merged['daily_ac_energy_Wh'].clip(lower=0) / 1000.0
df_merged['insolation_kwh'] = df_merged['daily_insolation_Wh'].clip(lower=0) / 1000.0

# Extract time-series arrays
records = []
for _, row in df_merged.iterrows():
    date_str = row['date'].strftime('%Y-%m-%d')
    records.append({
        "date": date_str,
        "energy_kwh": round(float(row.get('energy_kwh', 0.0) or 0.0), 2),
        "insolation_kwh": round(float(row.get('insolation_kwh', 0.0) or 0.0), 2),
        "temp_ambient": round(float(row.get('ambient_temp_mean', 0.0) or 0.0), 1),
        "temp_module": round(float(row.get('module_temp_mean', 0.0) or 0.0), 1),
        "precipitation_mm": round(float(row.get('precipitation_sum', 0.0) or 0.0), 1),
        "perf_index": round(float(row.get('normalized_pi', 0.0) or 0.0), 3) if pd.notnull(row.get('normalized_pi')) else None,
        "soiling_ratio": round(float(row.get('estimated_soiling_ratio', 0.0) or 0.0), 4) if pd.notnull(row.get('estimated_soiling_ratio')) else None,
        "soiling_loss_pct": round(float(row.get('estimated_soiling_loss_pct', 0.0) or 0.0), 2) if pd.notnull(row.get('estimated_soiling_loss_pct')) else None,
        "is_cleaning_event": bool(row.get('cleaning_event_candidate', False)),
        "recommendation": str(row.get('recommendation', 'no_action')),
        "rec_reason": str(row.get('reason', '')) if pd.notnull(row.get('reason')) else "",
        "confidence": str(row.get('confidence', 'low')),
        "quality_flag": str(row.get('data_quality_flag', 'good'))
    })

# Monthly aggregates
df_merged['month'] = df_merged['date'].dt.strftime('%b')
monthly_summary = []
for month_name, group in df_merged.groupby(df_merged['date'].dt.to_period('M')):
    m_name = group['date'].iloc[0].strftime('%b %Y')
    tot_energy = round(float(group['energy_kwh'].sum()), 1)
    tot_rain = round(float(group['precipitation_sum'].sum()), 1)
    valid_pi = group['normalized_pi'].dropna()
    avg_pi = round(float(valid_pi.mean()), 3) if len(valid_pi) > 0 else 0
    valid_soil = group['estimated_soiling_ratio'].dropna()
    avg_soil = round(float(valid_soil.mean()), 4) if len(valid_soil) > 0 else None
    monthly_summary.append({
        "month": m_name,
        "total_energy_kwh": tot_energy,
        "total_rain_mm": tot_rain,
        "avg_pi": avg_pi,
        "avg_soiling_ratio": avg_soil
    })

# High-level KPIs
kpis = {
    "system_name": "PVDAQ System 10 (Denver, Colorado)",
    "year": 2022,
    "capacity_kw": 3.0,
    "total_energy_kwh": round(float(df_merged['energy_kwh'].sum()), 1),
    "annual_soiling_loss_pct": 2.71,
    "soiling_ratio_p50": 0.9729,
    "ci_soiling_loss": [1.42, 4.91],
    "ci_soiling_ratio": [0.9509, 0.9858],
    "cleaning_events_detected": 24,
    "days_analyzed": len(df_merged),
    "rain_days": int((df_merged['precipitation_sum'] > 0).sum()),
    "total_precipitation_mm": round(float(df_merged['precipitation_sum'].sum()), 1),
    "estimated_annual_loss_usd": 39.55,
    "cleaning_cost_usd": 150.00,
    "recommendations_breakdown": {
        "no_action": int((df_recs['recommendation'] == 'no_action').sum()),
        "monitor": int((df_recs['recommendation'] == 'monitor').sum()),
        "clean": int((df_recs['recommendation'] == 'clean').sum())
    }
}

dashboard_data = {
    "kpis": kpis,
    "monthly_summary": monthly_summary,
    "daily_records": records
}

output_js = os.path.join(DASHBOARD_DIR, "data.js")
with open(output_js, "w", encoding="utf-8") as f:
    f.write("window.SOLARCLEAN_DATA = ")
    json.dump(dashboard_data, f, indent=2)
    f.write(";\n")

print(f"Dashboard data successfully prepared at {output_js}")
print(f"Total daily records: {len(records)}, Monthly bins: {len(monthly_summary)}")
