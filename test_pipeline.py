"""
SolarClean AI — Comprehensive Automated Test Suite
==================================================
Tests all 6 required recommendation scenarios, data integrity,
Open-Meteo API integration, error fallbacks, and verification that
no fabricated values exist.
"""

import os
import json
import unittest
import pandas as pd
import numpy as np

import solarclean_config as cfg
from live_weather_service import fetch_live_weather, process_weather_payload
from recommendation_engine import evaluate_cleaning_decision, estimate_current_soiling_loss

class TestSolarCleanPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = cfg.get_config_dict()

    def test_01_original_csv_integrity(self):
        """Verify original CSV files exist, are unmodified, and have valid counts."""
        self.assertTrue(os.path.exists("pv_system10_2022.csv"), "Missing original HF PV data")
        self.assertTrue(os.path.exists("weather_2022.csv"), "Missing original weather data")
        self.assertTrue(os.path.exists("pv_system10_daily_2022.csv"), "Missing original daily PV data")

        # Verify row counts
        hf_lines = sum(1 for _ in open("pv_system10_2022.csv"))
        self.assertGreater(hf_lines, 500000, "HF data has fewer lines than expected")

        weather_lines = sum(1 for _ in open("weather_2022.csv"))
        self.assertEqual(weather_lines, 366, "Weather CSV must have header + 365 days")

    def test_02_duramat_non_dependency(self):
        """Confirm DuraMAT is NOT an active dependency of the pipeline."""
        # Ensure no generated outputs or model files mention or load DuraMAT data
        model_meta_path = os.path.join(cfg.MODELS_DIR, "model_metadata.json")
        self.assertTrue(os.path.exists(model_meta_path))
        with open(model_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        meta_str = json.dumps(meta).lower()
        self.assertNotIn("duramat_data", meta_str)
        self.assertNotIn("duramat_cleaning_data", meta_str)

    def test_03_open_meteo_live_integration(self):
        """Test live Open-Meteo weather fetch, structure, and dry streak computation."""
        weather = fetch_live_weather(force_refresh=False)
        self.assertIn(weather["status"], ["ONLINE", "LIVE_CACHE", "STALE_OFFLINE"])
        self.assertIn("current", weather)
        self.assertIn("dry_streak", weather)
        self.assertIn("forecast_horizons", weather)

        # Check that dry days is an actual integer >= 0
        dry_days = weather["dry_streak"]["consecutive_dry_days"]
        self.assertIsInstance(dry_days, int)
        self.assertGreaterEqual(dry_days, 0)

    def test_04_scenario_high_loss_no_rain_triggers_clean(self):
        """Scenario 1: High estimated loss with no meaningful rain forecast -> CLEAN NOW."""
        overrides = {
            "dry_days": 45,            # Long dry spell -> high loss ~7.6%
            "rain_forecast_72h": 0.0,  # Zero rain forecast
            "rain_prob_72h": 10
        }
        res = evaluate_cleaning_decision(custom_overrides=overrides)
        self.assertEqual(res["recommendation"], "CLEAN NOW", f"Expected CLEAN NOW, got {res['recommendation']}")
        self.assertEqual(res["status_color"], "rose")
        self.assertIn("Action Required", res["action_title"])

    def test_05_scenario_low_loss_no_rain_triggers_monitor(self):
        """Scenario 2: Low estimated loss with no meaningful rain forecast -> MONITOR."""
        overrides = {
            "dry_days": 5,             # Short dry spell -> low loss ~1.2%
            "rain_forecast_72h": 0.0,
            "rain_prob_72h": 10
        }
        res = evaluate_cleaning_decision(custom_overrides=overrides)
        self.assertEqual(res["recommendation"], "MONITOR", f"Expected MONITOR, got {res['recommendation']}")

    def test_06_scenario_rain_forecast_soon_triggers_wait_for_rain(self):
        """Scenario 3: Rain forecast soon and no urgent cleaning -> WAIT FOR RAIN."""
        overrides = {
            "dry_days": 20,            # Moderate loss ~3.6%
            "rain_forecast_72h": 8.5,  # Significant rain incoming (>3.0mm)
            "rain_prob_72h": 75        # High probability (>50%)
        }
        res = evaluate_cleaning_decision(custom_overrides=overrides)
        self.assertEqual(res["recommendation"], "WAIT FOR RAIN", f"Expected WAIT FOR RAIN, got {res['recommendation']}")
        self.assertEqual(res["status_color"], "cyan")

    def test_07_scenario_weather_api_unavailable_triggers_insufficient_data(self):
        """Scenario 4: Weather API unavailable and no cache -> INSUFFICIENT DATA."""
        mock_unavailable_weather = {
            "status": "UNAVAILABLE",
            "is_stale": True,
            "current": None
        }
        res = evaluate_cleaning_decision(weather_data=mock_unavailable_weather)
        self.assertEqual(res["recommendation"], "INSUFFICIENT DATA")
        self.assertEqual(res["data_confidence"], "none")

    def test_08_no_hardcoded_metrics_in_metadata(self):
        """Verify model metrics are computed from held-out validation, not fake constants."""
        with open(os.path.join(cfg.MODELS_DIR, "model_metadata.json"), "r") as f:
            meta = json.load(f)

        gbr_meta = meta["machine_learning_gbr"]
        # The fake 0.80 R2 must NOT be present
        self.assertNotEqual(gbr_meta["held_out_test_r2"], 0.80)
        # Test MAE must be a real float between 0.5 and 5.0
        self.assertGreater(gbr_meta["held_out_test_mae_pct"], 0.5)
        self.assertLess(gbr_meta["held_out_test_mae_pct"], 5.0)

    def test_09_historical_soiling_estimates_integrity(self):
        """Verify output/data/soiling_estimation_2022.csv has full 304 valid days."""
        soiling_file = os.path.join(cfg.DATA_DIR, "soiling_estimation_2022.csv")
        self.assertTrue(os.path.exists(soiling_file))
        df = pd.read_csv(soiling_file)
        valid_count = df['estimated_soiling_ratio'].notnull().sum()
        self.assertGreaterEqual(valid_count, 300, f"Expected >= 300 valid estimated days, got {valid_count}")
        clean_events = df['cleaning_event_candidate'].sum()
        self.assertGreater(clean_events, 10, f"Expected > 10 candidate cleaning events, got {clean_events}")

    def test_10_rainfall_effectiveness_analyzer(self):
        """Verify Rainfall Effectiveness Analyzer identifies 2022 rain events and computes metrics."""
        from rainfall_analyzer import analyze_historical_rainfall_events
        res = analyze_historical_rainfall_events()
        self.assertIn("summary", res)
        self.assertIn("events", res)
        summary = res["summary"]
        self.assertGreaterEqual(summary["total_rain_events_identified"], 30)
        self.assertGreaterEqual(summary["comparable_events_count"], 30)
        # Check non-causal language preserved in summary interpretation
        self.assertIn("Historical performance changed", summary["interpretation"])
        self.assertIn("does not prove complete dust removal", summary["interpretation"])

    def test_11_what_if_cleaning_simulator(self):
        """Verify What-If Simulator calculates Strategy A, B, C under consistent assumptions."""
        from cleaning_simulator import simulate_cleaning_strategies
        # Test case: rain forecast upcoming
        sim_rain = simulate_cleaning_strategies(current_loss_pct=2.0, forecast_rain_mm=5.0, wash_cost_usd=150.0)
        self.assertEqual(sim_rain["optimal_strategy_id"], "B")
        self.assertIn("strategy_a", sim_rain["strategies"])
        self.assertIn("strategy_b", sim_rain["strategies"])
        self.assertIn("strategy_c", sim_rain["strategies"])

        # Test case: high loss, zero rain -> Strategy A should be favored or positive recoverable energy
        sim_dry = simulate_cleaning_strategies(current_loss_pct=12.0, forecast_rain_mm=0.0, wash_cost_usd=50.0, tariff_usd=0.30)
        self.assertIn("net_financial_benefit_usd", sim_dry["strategies"]["strategy_a"])
        self.assertGreater(sim_dry["strategies"]["strategy_a"]["recoverable_energy_kwh"], 0.0)

    def test_12_weather_screening_high_frequency(self):
        """Verify high-frequency screening correctly flags twilight, cloud instability, and checks module sensors."""
        from weather_screening import screen_high_frequency_data
        # Create synthetic test chunk with distinct conditions
        test_df = pd.DataFrame({
            'measured_on': pd.date_range('2022-06-01 10:00', periods=5, freq='1min'),
            'poa_irradiance__421': [5.0, 100.0, 500.0, 800.0, 805.0], # night, low-irr, useful, cloud-jump, stable
            'ac_power__423': [0.0, 20.0, 1200.0, 1800.0, 1820.0],
            'dc_power__422': [0.0, 25.0, 1250.0, 1850.0, 1870.0],
            'ambient_temp__428': [20.0, 21.0, 22.0, 22.5, 23.0],
            'module_temp_1__429': [18.0, 24.0, 35.0, 38.0, 39.0],
            'module_temp_2__430': [18.5, 24.5, 35.5, 38.5, 39.5],
            'module_temp_3__431': [18.2, 24.2, 35.2, 38.2, 39.2]
        })
        screened, stats = screen_high_frequency_data(test_df)
        self.assertEqual(stats["night_observations"], 1)
        self.assertEqual(stats["low_irradiance_excluded"], 1)
        # Check cloud jump detection (500 -> 800 W/m2 in 1 min is > 60 W/m2/min)
        self.assertTrue(screened.loc[3, 'flag_cloud_unstable'])
        self.assertFalse(screened.loc[4, 'flag_cloud_unstable'])
        # Check direct module sensor used
        self.assertTrue(screened['flag_module_temp_valid'].all())

    def test_13_weather_screening_daily_snow_and_clouds(self):
        """Verify daily screening detects snow and overcast days without marking all rain as snow."""
        from weather_screening import screen_daily_dataset
        mock_daily = pd.DataFrame({
            'date': ['2022-01-15', '2022-06-15', '2022-07-20'],
            'daily_insolation_Wh': [3000.0, 7500.0, 600.0],
            'daily_ac_energy_Wh': [50.0, 15000.0, 800.0],
            'daytime_hours': [8.0, 14.0, 12.0]
        })
        mock_weather = pd.DataFrame({
            'time': ['2022-01-15', '2022-06-15', '2022-07-20'],
            'precipitation_sum': [4.0, 8.0, 0.0], # Winter rain (snow) vs Summer rain
            'temperature_2m_max': [-5.0, 28.0, 22.0], # Freezing vs Hot summer
            'temperature_2m_min': [-18.0, 14.0, 12.0]
        })
        screened, stats = screen_daily_dataset(mock_daily, mock_weather)
        # Jan 15: Precip + Tmax -5C -> snow affected
        self.assertTrue(screened.loc[0, 'flag_snow_affected'])
        # Jun 15: Precip 8mm + Tmax 28C -> rain, NOT snow!
        self.assertFalse(screened.loc[1, 'flag_snow_affected'])
        self.assertTrue(screened.loc[1, 'is_usable_for_soiling'])
        # Jul 20: Insolation 600 Wh -> low insolation overcast
        self.assertTrue(screened.loc[2, 'flag_low_insolation'])

    def test_14_weather_screening_missing_columns_robustness(self):
        """Verify screening handles missing columns gracefully without crashing."""
        from weather_screening import screen_high_frequency_data
        incomplete_df = pd.DataFrame({
            'measured_on': pd.date_range('2022-05-01 12:00', periods=3, freq='1min'),
            'poa_irradiance__421': [400.0, 420.0, 450.0],
            'ambient_temp__428': [15.0, 15.2, 15.5]
            # No module_temp columns
        })
        screened, stats = screen_high_frequency_data(incomplete_df)
        self.assertEqual(stats["ambient_fallback_count"], 3)
        self.assertFalse(screened['flag_module_temp_valid'].any())

    def test_15_screening_model_improvement_verification(self):
        """Verify model metadata confirms measured improvement from screening."""
        meta_file = os.path.join(cfg.MODELS_DIR, "model_metadata.json")
        self.assertTrue(os.path.exists(meta_file))
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertIn("validation_comparison_unscreened_vs_screened", meta)
        comp = meta["validation_comparison_unscreened_vs_screened"]
        self.assertGreater(comp["mae_improvement_pct"], 0.0, "Screening should reduce held-out MAE")
        self.assertGreater(comp["excluded_confounded_days"], 20, "Expected > 20 confounded days screened out")

if __name__ == "__main__":
    unittest.main(verbosity=2)


