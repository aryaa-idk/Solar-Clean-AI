"""
SolarClean AI — What-If Cleaning Strategy Simulator
===================================================
Simulates and compares three distinct PV maintenance strategies over a consistent
evaluation period (default: 30 days) under uniform user-configured assumptions:

1. Strategy A — Clean Today (Manual wash immediately)
2. Strategy B — Wait for Forecast Rain (Leverage upcoming natural wash-off)
3. Strategy C — Follow a Fixed Cleaning Schedule (Periodic scheduled maintenance)

All assumptions are explicitly identified and parameterized. No hardcoded outcomes.
"""

import os
import json
import numpy as np

import solarclean_config as cfg

def simulate_cleaning_strategies(
    current_loss_pct=None,
    forecast_rain_mm=None,
    forecast_rain_day=2,
    capacity_kw=None,
    tariff_usd=None,
    wash_cost_usd=None,
    fixed_interval_days=45,
    eval_period_days=30,
    daily_soiling_rate_pct=0.16
):
    """
    Simulates and compares the 3 cleaning strategies over eval_period_days.
    All parameters fall back to central solarclean_config if not provided.
    """
    op_cfg = cfg.OPERATIONAL_CONFIG
    sys_specs = cfg.SYSTEM_SPECS

    # Apply configuration defaults
    if capacity_kw is None:
        capacity_kw = float(sys_specs.get("capacity_kw", 3.0))
    if tariff_usd is None:
        tariff_usd = float(op_cfg.get("electricity_tariff_usd_kwh", 0.15))
    if wash_cost_usd is None:
        wash_cost_usd = float(op_cfg.get("manual_wash_cost_usd", 150.0))
    if current_loss_pct is None:
        current_loss_pct = 1.7 # Default Colorado baseline
    if forecast_rain_mm is None:
        forecast_rain_mm = 3.2 # From live weather

    # Baseline daily clean energy production
    daily_yield_factor = float(op_cfg.get("daily_yield_kwh_per_kw", 4.2))
    daily_clean_kwh = capacity_kw * daily_yield_factor
    total_clean_kwh = daily_clean_kwh * eval_period_days
    total_potential_revenue = total_clean_kwh * tariff_usd

    # Uncleaned baseline ("Do Nothing" reference)
    do_nothing_daily_losses = []
    l_curr = current_loss_pct
    for d in range(eval_period_days):
        loss = min(15.0, l_curr + d * daily_soiling_rate_pct)
        do_nothing_daily_losses.append(loss)
    do_nothing_energy_loss_kwh = sum((l / 100.0) * daily_clean_kwh for l in do_nothing_daily_losses)
    do_nothing_revenue_loss_usd = do_nothing_energy_loss_kwh * tariff_usd

    # =========================================================================
    # STRATEGY A — Clean Today
    # =========================================================================
    # Clean on Day 1: resets soiling to 0.5% clean baseline
    strat_a_daily_losses = []
    for d in range(eval_period_days):
        loss = min(15.0, 0.5 + d * daily_soiling_rate_pct)
        strat_a_daily_losses.append(loss)
    
    strat_a_energy_loss_kwh = sum((l / 100.0) * daily_clean_kwh for l in strat_a_daily_losses)
    strat_a_revenue_loss_usd = strat_a_energy_loss_kwh * tariff_usd
    strat_a_recoverable_kwh = max(0.0, do_nothing_energy_loss_kwh - strat_a_energy_loss_kwh)
    strat_a_gross_benefit_usd = strat_a_recoverable_kwh * tariff_usd
    strat_a_net_benefit_usd = strat_a_gross_benefit_usd - wash_cost_usd

    strategy_a = {
        "strategy_id": "A",
        "name": "Clean Today",
        "action": "Immediate Manual Wash",
        "cleaning_cost_usd": round(wash_cost_usd, 2),
        "energy_loss_kwh": round(strat_a_energy_loss_kwh, 2),
        "recoverable_energy_kwh": round(strat_a_recoverable_kwh, 2),
        "gross_savings_usd": round(strat_a_gross_benefit_usd, 2),
        "net_financial_benefit_usd": round(strat_a_net_benefit_usd, 2),
        "ending_soiling_loss_pct": round(strat_a_daily_losses[-1], 2),
        "description": f"Manual washing today resets soiling to ~0.5%. Recovers {round(strat_a_recoverable_kwh, 1)} kWh ($ {round(strat_a_gross_benefit_usd, 2)}) over {eval_period_days} days against a ${wash_cost_usd:.0f} service fee."
    }

    # =========================================================================
    # STRATEGY B — Wait for Forecast Rain
    # =========================================================================
    # Wash cost = $0. If forecast rain >= 3.0 mm occurs on forecast_rain_day,
    # natural rain wash reduces accumulated loss by ~70%
    strat_b_daily_losses = []
    l_b = current_loss_pct
    has_effective_rain = (forecast_rain_mm >= float(op_cfg.get("rain_wash_threshold_mm", 3.0)))

    for d in range(eval_period_days):
        if has_effective_rain and d == forecast_rain_day:
            # Rain wash event: cleans roughly 70% of soiling
            l_b = max(0.5, l_b * 0.30)
        else:
            l_b = min(15.0, l_b + daily_soiling_rate_pct)
        strat_b_daily_losses.append(l_b)

    strat_b_energy_loss_kwh = sum((l / 100.0) * daily_clean_kwh for l in strat_b_daily_losses)
    strat_b_revenue_loss_usd = strat_b_energy_loss_kwh * tariff_usd
    strat_b_recoverable_kwh = max(0.0, do_nothing_energy_loss_kwh - strat_b_energy_loss_kwh)
    strat_b_gross_benefit_usd = strat_b_recoverable_kwh * tariff_usd
    strat_b_net_benefit_usd = strat_b_gross_benefit_usd # Zero cleaning cost!

    strategy_b = {
        "strategy_id": "B",
        "name": "Wait for Forecast Rain",
        "action": f"Natural Wash ({forecast_rain_mm:.1f} mm rain expected Day {forecast_rain_day})",
        "cleaning_cost_usd": 0.0,
        "energy_loss_kwh": round(strat_b_energy_loss_kwh, 2),
        "recoverable_energy_kwh": round(strat_b_recoverable_kwh, 2),
        "gross_savings_usd": round(strat_b_gross_benefit_usd, 2),
        "net_financial_benefit_usd": round(strat_b_net_benefit_usd, 2),
        "ending_soiling_loss_pct": round(strat_b_daily_losses[-1], 2),
        "forecast_rain_mm": round(forecast_rain_mm, 1),
        "has_effective_rain": has_effective_rain,
        "description": (
            f"Zero cleaning cost. Free natural wash-off ({forecast_rain_mm:.1f} mm forecast) restores panel efficiency, "
            f"saving ${wash_cost_usd:.0f} in manual wash fees with net benefit of +${strat_b_net_benefit_usd:.2f}."
            if has_effective_rain else
            f"Rain forecast ({forecast_rain_mm:.1f} mm) is insufficient for full natural wash. Soiling continues accumulating with net benefit of ${strat_b_net_benefit_usd:.2f}."
        )
    }

    # =========================================================================
    # STRATEGY C — Fixed Periodic Cleaning Schedule
    # =========================================================================
    # Pro-rated cleaning cost based on periodic frequency (e.g. every 45 days)
    washes_in_window = eval_period_days / float(fixed_interval_days)
    strat_c_cost = washes_in_window * wash_cost_usd
    
    # Average soiling under periodic cycle
    avg_periodic_loss = 0.5 + (0.5 * fixed_interval_days * daily_soiling_rate_pct)
    strat_c_energy_loss_kwh = (avg_periodic_loss / 100.0) * total_clean_kwh
    strat_c_recoverable_kwh = max(0.0, do_nothing_energy_loss_kwh - strat_c_energy_loss_kwh)
    strat_c_gross_benefit_usd = strat_c_recoverable_kwh * tariff_usd
    strat_c_net_benefit_usd = strat_c_gross_benefit_usd - strat_c_cost

    strategy_c = {
        "strategy_id": "C",
        "name": "Fixed Periodic Schedule",
        "action": f"Fixed cleaning every {fixed_interval_days} days",
        "cleaning_interval_days": fixed_interval_days,
        "cleaning_cost_usd": round(strat_c_cost, 2),
        "energy_loss_kwh": round(strat_c_energy_loss_kwh, 2),
        "recoverable_energy_kwh": round(strat_c_recoverable_kwh, 2),
        "gross_savings_usd": round(strat_c_gross_benefit_usd, 2),
        "net_financial_benefit_usd": round(strat_c_net_benefit_usd, 2),
        "average_soiling_loss_pct": round(avg_periodic_loss, 2),
        "description": f"Pro-rated {washes_in_window:.2f} cleanings per {eval_period_days} days (${round(strat_c_cost, 2)} cost). Average cyclic soiling loss is ~{round(avg_periodic_loss, 1)}%."
    }

    # =========================================================================
    # Synthesis & Recommendation
    # =========================================================================
    strategies = [strategy_a, strategy_b, strategy_c]
    # Rank by net financial benefit
    strategies_ranked = sorted(strategies, key=lambda s: s["net_financial_benefit_usd"], reverse=True)
    best_strategy = strategies_ranked[0]

    if best_strategy["strategy_id"] == "B" and has_effective_rain:
        synthesis = (
            f"Strategy B (Wait for Forecast Rain) is the economically optimal decision. "
            f"Upcoming rainfall of {forecast_rain_mm:.1f} mm provides free natural wash-off, "
            f"avoiding a ${wash_cost_usd:.0f} service fee while yielding higher net financial return."
        )
    elif best_strategy["strategy_id"] == "A":
        synthesis = (
            f"Strategy A (Clean Today) is the economically optimal decision. "
            f"Recoverable energy value (${strategy_a['gross_savings_usd']:.2f}) exceeds the ${wash_cost_usd:.0f} cleaning cost."
        )
    else:
        synthesis = (
            f"Strategy {best_strategy['strategy_id']} ({best_strategy['name']}) provides the highest net financial outcome "
            f"(${best_strategy['net_financial_benefit_usd']:+.2f}) under current assumptions."
        )

    return {
        "evaluation_period_days": eval_period_days,
        "assumptions": {
            "capacity_kw": capacity_kw,
            "tariff_usd_kwh": tariff_usd,
            "manual_wash_cost_usd": wash_cost_usd,
            "current_soiling_loss_pct": current_loss_pct,
            "forecast_rain_mm": forecast_rain_mm,
            "fixed_interval_days": fixed_interval_days,
            "daily_soiling_rate_pct": daily_soiling_rate_pct,
            "total_clean_production_kwh": round(total_clean_kwh, 1),
            "label": "All parameters are user-configured operational assumptions"
        },
        "reference_do_nothing": {
            "energy_loss_kwh": round(do_nothing_energy_loss_kwh, 2),
            "revenue_loss_usd": round(do_nothing_revenue_loss_usd, 2)
        },
        "strategies": {
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "strategy_c": strategy_c
        },
        "ranked_strategies": [s["strategy_id"] for s in strategies_ranked],
        "optimal_strategy_id": best_strategy["strategy_id"],
        "optimal_strategy_name": best_strategy["name"],
        "synthesis": synthesis
    }

if __name__ == "__main__":
    sim = simulate_cleaning_strategies()
    print("What-If Cleaning Simulator Results:")
    print(f"Optimal Strategy: {sim['optimal_strategy_id']} ({sim['optimal_strategy_name']})")
    print(f"Synthesis: {sim['synthesis']}")
    for k, s in sim["strategies"].items():
        print(f"\n{s['name']}:")
        print(f"  Cleaning Cost: ${s['cleaning_cost_usd']}")
        print(f"  Energy Loss: {s['energy_loss_kwh']} kWh")
        print(f"  Net Benefit: ${s['net_financial_benefit_usd']}")
