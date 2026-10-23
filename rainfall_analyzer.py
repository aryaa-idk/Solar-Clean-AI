"""
SolarClean AI — Historical Rainfall Effectiveness Analyzer
===========================================================
Investigates how recorded PV performance changed around historical rainfall events
in the validated 2022 NREL PVDAQ System 10 dataset (Golden, Colorado).
Accounts for temperature and irradiance normalization using temperature-corrected PI.
Avoids unfounded causal claims: states observable performance changes.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import timedelta

import solarclean_config as cfg

def analyze_historical_rainfall_events(min_rain_mm=2.0, window_days=3):
    """
    Identifies historical rain events and compares pre- and post-event PV performance.
    
    Parameters:
    - min_rain_mm: Minimum daily precipitation to qualify as an analyzed rain event (default: 2.0 mm).
    - window_days: Number of days before and after the event to evaluate (default: 3 days).
    """
    daily_file = os.path.join(cfg.DATA_DIR, "daily_clean_2022.csv")
    if not os.path.exists(daily_file):
        raise FileNotFoundError(f"Missing {daily_file}")

    df = pd.read_csv(daily_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Use temperature-corrected normalized performance index (pi_temp_corrected or normalized_pi)
    # Filter out extreme outliers / invalid flags
    df['valid_pi'] = df['normalized_pi'].where(
        (df['normalized_pi'] > 0.1) & (df['normalized_pi'] < 1.4) & (df['data_quality_flag'] != 'pi_outlier')
    )

    # 1. Identify discrete rain events (cluster contiguous rain days into single events)
    rain_mask = df['precipitation_sum'] >= min_rain_mm
    rain_indices = df[rain_mask].index.tolist()

    if not rain_indices:
        return {"events": [], "summary": {}}

    # Group contiguous rain days into single rain events
    events = []
    current_event = [rain_indices[0]]

    for idx in rain_indices[1:]:
        if idx == current_event[-1] + 1:
            current_event.append(idx)
        else:
            events.append(current_event)
            current_event = [idx]
    events.append(current_event)

    analyzed_events = []
    pi_changes = []

    for ev in events:
        start_idx = ev[0]
        end_idx = ev[-1]
        
        event_start_date = df.loc[start_idx, 'date']
        event_end_date = df.loc[end_idx, 'date']
        total_precip_mm = round(float(df.loc[ev, 'precipitation_sum'].sum()), 1)
        max_daily_precip = round(float(df.loc[ev, 'precipitation_sum'].max()), 1)
        duration_days = len(ev)

        # Pre-event window (1 to window_days before start_idx)
        pre_indices = [i for i in range(start_idx - window_days, start_idx) if i >= 0]
        # Post-event window (1 to window_days after end_idx)
        post_indices = [i for i in range(end_idx + 1, end_idx + 1 + window_days) if i < len(df)]

        pre_pi_vals = df.loc[pre_indices, 'valid_pi'].dropna()
        post_pi_vals = df.loc[post_indices, 'valid_pi'].dropna()

        # Check comparability and data quality
        if len(pre_pi_vals) == 0 or len(post_pi_vals) == 0:
            status = "Inconclusive (Insufficient Valid Days)"
            outcome = "inconclusive"
            pre_pi_mean = None
            post_pi_mean = None
            delta_pi = None
            pct_change = None
        else:
            pre_pi_mean = round(float(pre_pi_vals.mean()), 3)
            post_pi_mean = round(float(post_pi_vals.mean()), 3)
            delta_pi = round(post_pi_mean - pre_pi_mean, 3)
            pct_change = round((delta_pi / pre_pi_mean) * 100.0, 1)

            # Check temperature and irradiance comparability
            pre_insolation = df.loc[pre_indices, 'daily_insolation_Wh'].dropna().mean()
            post_insolation = df.loc[post_indices, 'daily_insolation_Wh'].dropna().mean()
            insolation_ratio = post_insolation / pre_insolation if pre_insolation > 0 else 1.0

            if delta_pi > 0.02:
                outcome = "increased"
                status = "Performance Increased"
            elif delta_pi < -0.02:
                outcome = "decreased"
                status = "Performance Decreased"
            else:
                outcome = "neutral"
                status = "No Significant Change"

            # If extreme insolation difference, mark comparability limitation
            if insolation_ratio < 0.5 or insolation_ratio > 2.0:
                status += " (High Irradiance Variance)"

            pi_changes.append(delta_pi)

        event_dict = {
            "event_id": len(analyzed_events) + 1,
            "start_date": event_start_date.strftime('%Y-%m-%d'),
            "end_date": event_end_date.strftime('%Y-%m-%d'),
            "date_display": event_start_date.strftime('%b %d') if duration_days == 1 else f"{event_start_date.strftime('%b %d')}–{event_end_date.strftime('%b %d')}",
            "precipitation_total_mm": total_precip_mm,
            "duration_days": duration_days,
            "pre_event_pi": pre_pi_mean,
            "post_event_pi": post_pi_mean,
            "delta_pi": delta_pi,
            "pct_change": pct_change,
            "outcome": outcome,
            "status": status,
            "pre_valid_days": len(pre_pi_vals),
            "post_valid_days": len(post_pi_vals)
        }
        analyzed_events.append(event_dict)

    # 2. Compute Summary Statistics across valid comparative events
    valid_events = [e for e in analyzed_events if e["delta_pi"] is not None]
    increased_count = sum(1 for e in valid_events if e["outcome"] == "increased")
    decreased_count = sum(1 for e in valid_events if e["outcome"] == "decreased")
    neutral_count = sum(1 for e in valid_events if e["outcome"] == "neutral")
    inconclusive_count = len(analyzed_events) - len(valid_events)

    mean_delta = round(float(np.mean(pi_changes)), 4) if pi_changes else 0.0
    median_delta = round(float(np.median(pi_changes)), 4) if pi_changes else 0.0

    summary = {
        "total_rain_events_identified": len(analyzed_events),
        "comparable_events_count": len(valid_events),
        "inconclusive_events_count": inconclusive_count,
        "increased_performance_events": increased_count,
        "decreased_performance_events": decreased_count,
        "neutral_performance_events": neutral_count,
        "increased_pct": round((increased_count / len(valid_events) * 100), 1) if valid_events else 0,
        "mean_pi_change": mean_delta,
        "median_pi_change": median_delta,
        "interpretation": (
            f"Across {len(valid_events)} comparable historical rain events in 2022, "
            f"PV performance increased following {increased_count} events ({round(increased_count/len(valid_events)*100,1)}%), "
            f"decreased following {decreased_count} events, and showed no significant change following {neutral_count} events. "
            f"The mean observed change was {mean_delta:+.3f} in normalized performance index. "
            f"Note: Historical performance changed following rainfall events. Cloud variability and seasonal factors confound day-to-day changes; "
            f"correlation with rain events does not prove complete dust removal."
        )
    }

    return {
        "location": "NREL PVDAQ System 10 (Golden, CO)",
        "period": "2022 Full Year",
        "threshold_used_mm": min_rain_mm,
        "window_days": window_days,
        "summary": summary,
        "events": analyzed_events
    }

if __name__ == "__main__":
    res = analyze_historical_rainfall_events()
    print("Rainfall Effectiveness Analysis Summary:")
    print(json.dumps(res["summary"], indent=2))
    print(f"\nTotal events: {len(res['events'])}. First 3 events:")
    for ev in res["events"][:3]:
        print(f"  {ev['date_display']}: {ev['precipitation_total_mm']}mm -> Delta PI: {ev['delta_pi']} ({ev['status']})")
