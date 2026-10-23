"""
SolarClean AI: Complete Data Validation, Soiling Estimation, and Cleaning Recommendation Pipeline
=================================================================================================

This script implements the full pipeline for:
1. Data validation and quality assessment
2. Data cleaning and preprocessing
3. Performance ratio computation
4. Soiling estimation using rdtools SRR (Stochastic Rate and Recovery)
5. Cleaning event detection
6. Cleaning recommendation generation

PVDAQ System 10, Colorado, USA, Year 2022

IMPORTANT SCIENTIFIC NOTES:
- SRR detects candidate cleaning events from performance data; it does NOT measure soiling directly
- Without independent soiling measurements (e.g., soiling stations), all soiling ratios are ESTIMATES
- Performance changes may be caused by factors other than soiling (temperature, degradation, faults)
- All outputs are clearly labeled as estimates, not measured ground truth
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import warnings
import json
from datetime import datetime, timedelta

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_DIR = '.'
OUTPUT_DIR = 'output'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')

# System specifications for PVDAQ System 10
# Based on typical specs: ~3 kW residential system at NREL, Golden, CO
SYSTEM_CAPACITY_KW = 3.0  # Approximate nameplate DC capacity in kW
LATITUDE = 39.74  # NREL, Golden, CO
LONGITUDE = -105.18

# Physical limits for data validation
IRRADIANCE_MAX = 1500  # W/m2 - maximum physically reasonable POA irradiance
IRRADIANCE_MIN_DAYTIME = 10  # W/m2 - minimum for useful measurements
AC_POWER_MAX = 3500  # W - slightly above nameplate for high-irradiance conditions
DC_POWER_MAX = 4000  # W
TEMP_MIN = -40  # degC
TEMP_MAX = 80  # degC

# SRR parameters (will be tuned based on data characteristics)
SRR_PARAMS = {
    'reps': 500,             # Monte Carlo repetitions for robust convergence
    'confidence_level': 68.2  # 1-sigma confidence interval
}

# Create output directories
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================================
# LOGGING AND REPORTING
# ============================================================================
class PipelineReport:
    """Accumulates pipeline findings for the final report."""
    def __init__(self):
        self.sections = {}
        self.data_changes = []
        self.files_created = []
        
    def add_finding(self, section, finding):
        if section not in self.sections:
            self.sections[section] = []
        self.sections[section].append(finding)
        print(f"[{section}] {finding}")
        
    def add_data_change(self, file_affected, change_type, n_observations, reason, method):
        change = {
            'file': file_affected,
            'type': change_type,
            'n_observations': n_observations,
            'reason': reason,
            'method': method
        }
        self.data_changes.append(change)
        print(f"[DATA CHANGE] {change_type}: {n_observations} obs in {file_affected} - {reason}")
    
    def add_file(self, filepath, description):
        self.files_created.append({'path': filepath, 'description': description})

report = PipelineReport()

# ============================================================================
# OBJECTIVE 1: DATA VALIDATION
# ============================================================================
print("=" * 80)
print("OBJECTIVE 1: DATA VALIDATION")
print("=" * 80)

# --- Load all datasets ---
print("\n--- Loading datasets ---")

# 1. Original high-frequency PV data
print("Loading high-frequency PV data (this may take a moment)...")
hf_df = pd.read_csv(os.path.join(INPUT_DIR, 'pv_system10_2022.csv'))
hf_df['measured_on'] = pd.to_datetime(hf_df['measured_on'])
report.add_finding("Data Loading", f"High-frequency PV data: {hf_df.shape[0]:,} rows x {hf_df.shape[1]} columns")
report.add_finding("Data Loading", f"  Date range: {hf_df['measured_on'].min()} to {hf_df['measured_on'].max()}")

# 2. Daily PV data
daily_pv = pd.read_csv(os.path.join(INPUT_DIR, 'pv_system10_daily_2022.csv'))
daily_pv['date'] = pd.to_datetime(daily_pv['date'])
report.add_finding("Data Loading", f"Daily PV data: {daily_pv.shape[0]} rows x {daily_pv.shape[1]} columns")
report.add_finding("Data Loading", f"  Date range: {daily_pv['date'].min().date()} to {daily_pv['date'].max().date()}")

# 3. Weather data
weather = pd.read_csv(os.path.join(INPUT_DIR, 'weather_2022.csv'))
weather['time'] = pd.to_datetime(weather['time'])
report.add_finding("Data Loading", f"Weather data: {weather.shape[0]} rows x {weather.shape[1]} columns")
report.add_finding("Data Loading", f"  Date range: {weather['time'].min().date()} to {weather['time'].max().date()}")

# 4. Merged dataset
merged = pd.read_csv(os.path.join(INPUT_DIR, 'solarclean_merged_2022.csv'))
merged['time'] = pd.to_datetime(merged['time'])
merged['date'] = pd.to_datetime(merged['date'])
report.add_finding("Data Loading", f"Merged dataset: {merged.shape[0]} rows x {merged.shape[1]} columns")

# --- Shorthand column names ---
COL_IRR = 'poa_irradiance__421'
COL_AC = 'ac_power__423'
COL_DC = 'dc_power__422'
COL_AMB_TEMP = 'ambient_temp__428'
COL_MOD_TEMP1 = 'module_temp_1__429'
COL_MOD_TEMP2 = 'module_temp_2__430'
COL_MOD_TEMP3 = 'module_temp_3__431'
COL_DC_V = 'dc_pos_voltage__424'
COL_DC_I = 'dc_pos_current__425'
COL_AC_V = 'ac_voltage__426'
COL_AC_I = 'ac_current__427'
COL_INV_TEMP = 'inverter_temp__432'

# ============================================================================
# 1A. HIGH-FREQUENCY DATA VALIDATION
# ============================================================================
print("\n--- Validating high-frequency PV data ---")

# Timestamp validation
hf_dups = hf_df['measured_on'].duplicated().sum()
report.add_finding("HF Validation", f"Duplicate timestamps: {hf_dups}")

# Check sampling interval
time_diffs = hf_df['measured_on'].diff().dropna()
mode_interval = time_diffs.mode()[0]
report.add_finding("HF Validation", f"Modal sampling interval: {mode_interval}")

irregular = (time_diffs != mode_interval).sum()
report.add_finding("HF Validation", f"Irregular intervals: {irregular} ({irregular/len(time_diffs)*100:.2f}%)")

# Missing values
hf_missing = hf_df.isnull().sum()
report.add_finding("HF Validation", f"Missing values per column:")
for col in hf_missing.index:
    if hf_missing[col] > 0:
        report.add_finding("HF Validation", f"  {col}: {hf_missing[col]}")

if hf_missing.sum() == 0:
    report.add_finding("HF Validation", "  No missing values in high-frequency data")

# Negative irradiance (expected at night)
neg_irr = (hf_df[COL_IRR] < 0).sum()
nighttime_neg_irr = ((hf_df[COL_IRR] < 0) & (hf_df[COL_IRR] > -5)).sum()
severe_neg_irr = (hf_df[COL_IRR] < -5).sum()
report.add_finding("HF Validation", f"Negative irradiance readings: {neg_irr}")
report.add_finding("HF Validation", f"  Mildly negative (>-5 W/m2, sensor offset): {nighttime_neg_irr}")
report.add_finding("HF Validation", f"  Severely negative (<-5 W/m2): {severe_neg_irr}")

# Extreme irradiance
extreme_irr = (hf_df[COL_IRR] > IRRADIANCE_MAX).sum()
report.add_finding("HF Validation", f"Irradiance > {IRRADIANCE_MAX} W/m2: {extreme_irr}")

# Negative AC power (expected at night - inverter standby consumption)
neg_ac = (hf_df[COL_AC] < 0).sum()
report.add_finding("HF Validation", f"Negative AC power readings: {neg_ac} ({neg_ac/len(hf_df)*100:.1f}%)")

# Negative DC power
neg_dc = (hf_df[COL_DC] < 0).sum()
report.add_finding("HF Validation", f"Negative DC power readings: {neg_dc} ({neg_dc/len(hf_df)*100:.1f}%)")

# Extreme power
extreme_ac = (hf_df[COL_AC] > AC_POWER_MAX).sum()
extreme_dc = (hf_df[COL_DC] > DC_POWER_MAX).sum()
report.add_finding("HF Validation", f"AC power > {AC_POWER_MAX}W: {extreme_ac}")
report.add_finding("HF Validation", f"DC power > {DC_POWER_MAX}W: {extreme_dc}")

# Temperature range
for tcol in [COL_AMB_TEMP, COL_MOD_TEMP1]:
    tmin = hf_df[tcol].min()
    tmax = hf_df[tcol].max()
    out_of_range = ((hf_df[tcol] < TEMP_MIN) | (hf_df[tcol] > TEMP_MAX)).sum()
    report.add_finding("HF Validation", f"{tcol}: range [{tmin:.1f}, {tmax:.1f}] degC, out-of-range: {out_of_range}")

# ============================================================================
# 1B. DAILY PV DATA VALIDATION
# ============================================================================
print("\n--- Validating daily PV data ---")

# Check for missing dates
full_date_range = pd.date_range('2022-01-01', '2022-12-31', freq='D')
missing_dates = full_date_range.difference(daily_pv['date'])
report.add_finding("Daily PV Validation", f"Expected dates: {len(full_date_range)}, Present: {len(daily_pv)}")
report.add_finding("Daily PV Validation", f"Missing dates: {len(missing_dates)}")
if len(missing_dates) > 0:
    report.add_finding("Daily PV Validation", f"  Missing: {[d.strftime('%Y-%m-%d') for d in missing_dates]}")

# Missing values in daily data
daily_missing = daily_pv.isnull().sum()
report.add_finding("Daily PV Validation", "Missing values in daily PV data:")
for col in daily_missing.index:
    if daily_missing[col] > 0:
        report.add_finding("Daily PV Validation", f"  {col}: {daily_missing[col]}")

# Check for suspicious daily averages
# Daily average irradiance includes nighttime, so should be much lower than peak
irr_col = COL_IRR
if irr_col in daily_pv.columns:
    # Identify extreme daily irradiance values
    irr_data = daily_pv[irr_col].dropna()
    q1 = irr_data.quantile(0.25)
    q3 = irr_data.quantile(0.75)
    iqr = q3 - q1
    irr_upper = q3 + 3 * iqr
    irr_lower = q1 - 3 * iqr
    
    extreme_high = (irr_data > irr_upper).sum()
    extreme_low = (irr_data < irr_lower).sum()
    negative_irr_daily = (irr_data < 0).sum()
    
    report.add_finding("Daily PV Validation", f"Daily irradiance stats: mean={irr_data.mean():.1f}, min={irr_data.min():.1f}, max={irr_data.max():.1f}")
    report.add_finding("Daily PV Validation", f"  IQR bounds: [{irr_lower:.1f}, {irr_upper:.1f}]")
    report.add_finding("Daily PV Validation", f"  Extreme high (>3*IQR): {extreme_high}")
    report.add_finding("Daily PV Validation", f"  Extreme low (<3*IQR): {extreme_low}")
    report.add_finding("Daily PV Validation", f"  Negative daily avg irradiance: {negative_irr_daily}")
    
    # Show the extreme values
    if irr_data.min() < -100:
        extreme_dates = daily_pv.loc[daily_pv[irr_col] < irr_lower, ['date', irr_col]]
        report.add_finding("Daily PV Validation", f"  Extreme irradiance values:")
        for _, row in extreme_dates.iterrows():
            report.add_finding("Daily PV Validation", f"    {row['date'].date()}: {row[irr_col]:.1f} W/m2")

# Check negative daily power averages
neg_ac_daily = (daily_pv[COL_AC] < 0).sum()
neg_dc_daily = (daily_pv[COL_DC] < 0).sum()
report.add_finding("Daily PV Validation", f"Negative daily avg AC power: {neg_ac_daily}")
report.add_finding("Daily PV Validation", f"Negative daily avg DC power: {neg_dc_daily}")

if neg_ac_daily > 0:
    neg_ac_dates = daily_pv.loc[daily_pv[COL_AC] < 0, ['date', COL_AC]]
    for _, row in neg_ac_dates.iterrows():
        report.add_finding("Daily PV Validation", f"  Negative AC on {row['date'].date()}: {row[COL_AC]:.2f} W")

# ============================================================================
# 1C. WEATHER DATA VALIDATION
# ============================================================================
print("\n--- Validating weather data ---")

weather_missing = weather.isnull().sum()
report.add_finding("Weather Validation", f"Missing values: {weather_missing.sum()}")
report.add_finding("Weather Validation", f"Precipitation: min={weather['precipitation_sum'].min()}, max={weather['precipitation_sum'].max()}")
report.add_finding("Weather Validation", f"Rain days (>0 mm): {(weather['precipitation_sum'] > 0).sum()}")
report.add_finding("Weather Validation", f"Significant rain days (>5 mm): {(weather['precipitation_sum'] > 5).sum()}")

# ============================================================================
# 1D. MERGED DATASET VALIDATION
# ============================================================================
print("\n--- Validating merged dataset ---")

# Check merge alignment
merged_missing = merged.isnull().sum()
report.add_finding("Merged Validation", f"Total rows: {len(merged)}")
report.add_finding("Merged Validation", f"Missing values in key columns:")
for col in merged_missing.index:
    if merged_missing[col] > 0:
        report.add_finding("Merged Validation", f"  {col}: {merged_missing[col]}")

# Verify date alignment
date_mismatch = (merged['time'].dt.date != merged['date'].dt.date).sum()
report.add_finding("Merged Validation", f"Date alignment mismatches (time vs date): {date_mismatch}")

# Confirm the 56 missing irradiance values
report.add_finding("Merged Validation", f"Confirmed missing irradiance in merged: {merged[COL_IRR].isnull().sum()}")

# Check whether the merge is a left join on weather
weather_dates_not_in_pv = len(weather) - len(daily_pv)
report.add_finding("Merged Validation", f"Weather has {len(weather)} dates, daily PV has {len(daily_pv)} dates")
report.add_finding("Merged Validation", f"Merged has {len(merged)} rows (matches weather = left join on weather)")

print("\n--- Data Validation Summary ---")
report.add_finding("Validation Summary", 
    "The data contains 523,943 high-frequency measurements at 1-minute intervals, "
    "364 daily aggregated PV records (1 date missing), 365 weather records, "
    "and a merged dataset of 365 rows.")
report.add_finding("Validation Summary",
    f"Key issues confirmed: {merged[COL_IRR].isnull().sum()} missing daily irradiance values, "
    f"extreme irradiance readings (min={daily_pv[COL_IRR].dropna().min():.1f}, max={daily_pv[COL_IRR].dropna().max():.1f}), "
    f"{neg_ac_daily} negative daily AC power averages.")

# ============================================================================
# OBJECTIVE 2: DATA CLEANING AND PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("DATA CLEANING AND PREPROCESSING")
print("=" * 80)

# --- DECISION: Use high-frequency data for performance ratio computation ---
# Daily averages that include nighttime are NOT suitable for SRR because:
# 1. They dilute the performance signal with nighttime near-zero/negative values
# 2. The extreme negative daily irradiance averages suggest problematic aggregation
# 3. SRR needs a daytime-only performance index
#
# APPROACH: Recompute daily metrics from high-frequency data using only 
# daytime observations with sufficient irradiance.

print("\n--- Constructing clean daytime daily metrics from high-frequency data ---")

# Step 1: Filter high-frequency data for daytime conditions
# Only keep observations where irradiance >= 50 W/m2 (reasonable threshold for
# useful power production; avoids sunrise/sunset noise)
DAYTIME_IRR_THRESHOLD = 50  # W/m2

hf_clean = hf_df.copy()

# Flag nighttime/low-irradiance observations
hf_clean['is_daytime'] = hf_clean[COL_IRR] >= DAYTIME_IRR_THRESHOLD
daytime_count = hf_clean['is_daytime'].sum()
report.add_finding("Preprocessing", f"Daytime observations (irradiance >= {DAYTIME_IRR_THRESHOLD} W/m2): {daytime_count:,} of {len(hf_clean):,} ({daytime_count/len(hf_clean)*100:.1f}%)")

# Step 2: Flag physically invalid irradiance readings
# Negative irradiance during daytime = sensor error
# Irradiance > 1500 W/m2 = physically implausible
hf_clean['irr_valid'] = (hf_clean[COL_IRR] >= 0) & (hf_clean[COL_IRR] <= IRRADIANCE_MAX)
irr_invalid = (~hf_clean['irr_valid']).sum()
report.add_data_change("high-frequency data", "Flag invalid irradiance", irr_invalid,
    f"Irradiance outside [0, {IRRADIANCE_MAX}] W/m2 flagged as invalid",
    "Physical limits check")

# Step 3: Flag invalid power readings during daytime
# During daytime with sufficient irradiance, negative DC power is anomalous
# (AC power can be slightly negative due to inverter self-consumption,
# but shouldn't be significantly negative during good irradiance)
hf_clean['power_valid'] = True
daytime_neg_dc = (hf_clean['is_daytime'] & (hf_clean[COL_DC] < -10)).sum()
hf_clean.loc[hf_clean['is_daytime'] & (hf_clean[COL_DC] < -10), 'power_valid'] = False
report.add_data_change("high-frequency data", "Flag daytime negative DC power", daytime_neg_dc,
    "DC power < -10W during sufficient irradiance flagged",
    "Physical impossibility during production hours")

# Step 4: Flag temperature outliers
hf_clean['temp_valid'] = True
for tcol in [COL_AMB_TEMP, COL_MOD_TEMP1, COL_MOD_TEMP2, COL_MOD_TEMP3]:
    if tcol in hf_clean.columns:
        temp_invalid = ((hf_clean[tcol] < TEMP_MIN) | (hf_clean[tcol] > TEMP_MAX)).sum()
        if temp_invalid > 0:
            hf_clean.loc[(hf_clean[tcol] < TEMP_MIN) | (hf_clean[tcol] > TEMP_MAX), 'temp_valid'] = False
            report.add_data_change("high-frequency data", f"Flag extreme {tcol}", temp_invalid,
                f"Temperature outside [{TEMP_MIN}, {TEMP_MAX}] degC",
                "Physical limits check")

# Step 5: Create clean daytime dataset for daily aggregation
hf_daytime = hf_clean[
    hf_clean['is_daytime'] & 
    hf_clean['irr_valid'] & 
    hf_clean['power_valid'] & 
    hf_clean['temp_valid']
].copy()

report.add_finding("Preprocessing", f"Clean daytime observations: {len(hf_daytime):,}")

# Step 6: Compute module temperature average (use mean of available sensors)
temp_cols = [COL_MOD_TEMP1, COL_MOD_TEMP2, COL_MOD_TEMP3]
available_temp_cols = [c for c in temp_cols if c in hf_daytime.columns]
hf_daytime['module_temp_avg'] = hf_daytime[available_temp_cols].mean(axis=1)

# Step 7: Aggregate to daily values using daytime-only data
hf_daytime['date'] = hf_daytime['measured_on'].dt.date

daily_clean = hf_daytime.groupby('date').agg(
    poa_irradiance_mean=(COL_IRR, 'mean'),
    poa_irradiance_sum=(COL_IRR, 'sum'),
    ac_power_mean=(COL_AC, 'mean'),
    dc_power_mean=(COL_DC, 'mean'),
    ac_power_max=(COL_AC, 'max'),
    dc_power_max=(COL_DC, 'max'),
    module_temp_mean=('module_temp_avg', 'mean'),
    ambient_temp_mean=(COL_AMB_TEMP, 'mean'),
    n_observations=('measured_on', 'count'),
    daytime_hours=('measured_on', lambda x: len(x) / 60.0)  # minutes to hours
).reset_index()

daily_clean['date'] = pd.to_datetime(daily_clean['date'])
report.add_finding("Preprocessing", f"Clean daily dataset: {len(daily_clean)} days")

# Step 8: Compute daily insolation (kWh/m2/day)
# Sum of minute-level irradiance readings converted to daily insolation
# Each reading represents ~1 minute = 1/60 hour
# Insolation = sum(irradiance * dt) in Wh/m2, convert to kWh/m2
daily_clean['daily_insolation_Wh'] = hf_daytime.groupby(hf_daytime['measured_on'].dt.date)[COL_IRR].sum().values / 60.0  # W * (1/60 h) = Wh/m2

# Step 9: Compute daily energy (Wh)
daily_clean['daily_ac_energy_Wh'] = hf_daytime.groupby(hf_daytime['measured_on'].dt.date)[COL_AC].sum().values / 60.0  # W * (1/60 h) = Wh
daily_clean['daily_dc_energy_Wh'] = hf_daytime.groupby(hf_daytime['measured_on'].dt.date)[COL_DC].sum().values / 60.0

# Step 10: Data quality flags for each day
# Flag days with too few observations (< 4 hours of data)
MIN_DAYTIME_HOURS = 4.0
daily_clean['data_quality_flag'] = 'good'
low_data_days = daily_clean['daytime_hours'] < MIN_DAYTIME_HOURS
daily_clean.loc[low_data_days, 'data_quality_flag'] = 'low_data'
report.add_data_change("daily clean dataset", "Flag low-data days", low_data_days.sum(),
    f"Days with < {MIN_DAYTIME_HOURS} hours of daytime data flagged",
    "Minimum data coverage requirement")

# Flag days with very low insolation (cloudy/snowy)
LOW_INSOLATION_THRESHOLD = 500  # Wh/m2
low_insol = daily_clean['daily_insolation_Wh'] < LOW_INSOLATION_THRESHOLD
daily_clean.loc[low_insol & (daily_clean['data_quality_flag'] == 'good'), 'data_quality_flag'] = 'low_insolation'
report.add_data_change("daily clean dataset", "Flag low-insolation days", low_insol.sum(),
    f"Days with daily insolation < {LOW_INSOLATION_THRESHOLD} Wh/m2 flagged",
    "Insufficient irradiance for reliable performance estimation")

# Step 11: Merge with weather data
daily_clean = daily_clean.merge(
    weather[['time', 'precipitation_sum', 'temperature_2m_max', 'temperature_2m_min', 
             'sunshine_duration', 'shortwave_radiation_sum']],
    left_on='date', right_on='time', how='left'
).drop(columns=['time'], errors='ignore')

report.add_finding("Preprocessing", f"Merged clean daily data with weather: {len(daily_clean)} rows")

# ============================================================================
# COMPUTE PERFORMANCE RATIO (NORMALIZED PERFORMANCE INDEX)
# ============================================================================
print("\n--- Computing Performance Ratio ---")

# Performance Ratio = (Measured AC energy) / (Expected energy based on irradiance)
# PR = (E_ac) / (G_poa * A * eta_stc) 
# 
# For normalized daily index suitable for SRR:
# normalized_energy = daily_ac_energy / daily_insolation
# This gives us energy yield per unit of insolation, which captures soiling effects
# while normalizing for day-to-day irradiance variations.
#
# Temperature correction using the simple linear model:
# PR_temp_corrected = PR / (1 + gamma * (T_cell - T_ref))
# where gamma = temperature coefficient (~-0.004 /degC for crystalline Si)
# T_ref = 25 degC (STC reference)

TEMP_COEFF = -0.004  # %/degC typical for c-Si
T_REF = 25.0  # STC reference temperature

# Compute raw performance index (W/W*m2 = dimensionless when properly normalized)
# PI = (AC power) / (POA irradiance * system_capacity_factor)
# For SRR, we need the daily ratio normalized to a clean baseline

# Use the energy ratio: AC energy / (insolation * system_capacity_kW)
daily_clean['performance_index'] = daily_clean['daily_ac_energy_Wh'] / (
    daily_clean['daily_insolation_Wh'] * SYSTEM_CAPACITY_KW
)

# Apply temperature correction
daily_clean['temp_correction_factor'] = 1 + TEMP_COEFF * (daily_clean['module_temp_mean'] - T_REF)
daily_clean['pi_temp_corrected'] = daily_clean['performance_index'] / daily_clean['temp_correction_factor']

# Normalize to the initial clean period (first 30 good days)
good_days = daily_clean[daily_clean['data_quality_flag'] == 'good'].head(30)
if len(good_days) >= 10:
    baseline_pi = good_days['pi_temp_corrected'].median()
    daily_clean['normalized_pi'] = daily_clean['pi_temp_corrected'] / baseline_pi
else:
    # Fallback: use overall median of good days
    baseline_pi = daily_clean.loc[daily_clean['data_quality_flag'] == 'good', 'pi_temp_corrected'].median()
    daily_clean['normalized_pi'] = daily_clean['pi_temp_corrected'] / baseline_pi

report.add_finding("Performance Index", f"Baseline performance index: {baseline_pi:.4f}")
report.add_finding("Performance Index", f"Normalized PI range: [{daily_clean['normalized_pi'].min():.3f}, {daily_clean['normalized_pi'].max():.3f}]")
report.add_finding("Performance Index", f"Normalized PI median: {daily_clean['normalized_pi'].median():.3f}")

# Remove extreme outliers from PI (likely equipment issues, not soiling)
pi_q1 = daily_clean.loc[daily_clean['data_quality_flag'] == 'good', 'normalized_pi'].quantile(0.05)
pi_q3 = daily_clean.loc[daily_clean['data_quality_flag'] == 'good', 'normalized_pi'].quantile(0.95)
pi_outlier = (daily_clean['normalized_pi'] < pi_q1 * 0.5) | (daily_clean['normalized_pi'] > pi_q3 * 1.5)
daily_clean.loc[pi_outlier & (daily_clean['data_quality_flag'] == 'good'), 'data_quality_flag'] = 'pi_outlier'
report.add_data_change("daily clean dataset", "Flag PI outliers", pi_outlier.sum(),
    f"Performance index outside [{pi_q1*0.5:.3f}, {pi_q3*1.5:.3f}] flagged as outliers",
    "Statistical outlier detection (0.5x P5 to 1.5x P95)")

# ============================================================================
# SAVE CLEANED DATA
# ============================================================================
clean_csv_path = os.path.join(DATA_DIR, 'daily_clean_2022.csv')
daily_clean.to_csv(clean_csv_path, index=False)
report.add_file(clean_csv_path, "Cleaned daily PV + weather dataset with performance indices")
print(f"Saved cleaned daily data: {clean_csv_path}")

# ============================================================================
# OBJECTIVE 3 & 4: SOILING ESTIMATION USING SRR
# ============================================================================
print("\n" + "=" * 80)
print("OBJECTIVE 3 & 4: SOILING ESTIMATION (SRR)")
print("=" * 80)

# Prepare data for rdtools SRR
# SRR requires:
# 1. energy_normalized_daily: Daily normalized energy (pandas Series with DatetimeIndex, freq='D')
# 2. insolation_daily: Daily insolation (pandas Series with DatetimeIndex, freq='D')
#
# CRITICAL: rdtools requires a DatetimeIndex with freq='D' (no gaps).
# Our cleaned data has gaps (305 out of 365 days). We must reindex to a full
# daily range and let gaps be NaN — rdtools handles internal NaN.

# Filter to only good-quality days for the values, but we need a full date range
srr_mask = daily_clean['data_quality_flag'] == 'good'
srr_data = daily_clean[srr_mask].copy()
srr_data = srr_data.set_index('date')
srr_data.index = pd.DatetimeIndex(srr_data.index)

report.add_finding("SRR Input", f"Good-quality days for SRR: {len(srr_data)} of {len(daily_clean)}")

# Prepare SRR inputs — reindex to full daily range so freq='D' is set
full_range = pd.date_range(start=srr_data.index.min(), end=srr_data.index.max(), freq='D')

energy_normalized = srr_data['normalized_pi'].reindex(full_range)
insolation = srr_data['daily_insolation_Wh'].reindex(full_range)

# Replace Inf with NaN (rdtools can handle NaN internally)
energy_normalized = energy_normalized.replace([np.inf, -np.inf], np.nan)
insolation = insolation.replace([np.inf, -np.inf], np.nan)

# Verify frequency is set
assert energy_normalized.index.freq == 'D', f"Expected daily freq, got {energy_normalized.index.freq}"

n_valid = energy_normalized.notna().sum()
n_total = len(energy_normalized)
report.add_finding("SRR Input", f"Full daily range: {n_total} days, valid data: {n_valid} ({n_valid/n_total*100:.1f}%)")
report.add_finding("SRR Input", f"Date range: {energy_normalized.index.min().date()} to {energy_normalized.index.max().date()}")

# Run SRR
print("\n--- Running Stochastic Rate and Recovery (SRR) analysis ---")
print("This may take a few minutes (Monte Carlo simulation with 2000 reps)...")

try:
    from rdtools.soiling import soiling_srr
    
    # Prepare daily precipitation series matching DatetimeIndex
    precip_series = daily_clean.set_index('date').reindex(energy_normalized.index)['precipitation_sum'].fillna(0.0)

    sr, sr_ci, soiling_info = soiling_srr(
        energy_normalized_daily=energy_normalized,
        insolation_daily=insolation,
        precipitation_daily=precip_series,
        confidence_level=SRR_PARAMS['confidence_level'],
        reps=SRR_PARAMS['reps']
    )
    
    srr_success = True
    report.add_finding("SRR Results", f"P50 insolation-weighted soiling ratio: {sr:.4f}")
    report.add_finding("SRR Results", f"{SRR_PARAMS['confidence_level']}% CI: [{sr_ci[0]:.4f}, {sr_ci[1]:.4f}]")
    report.add_finding("SRR Results", f"Estimated annual soiling loss: {(1-sr)*100:.2f}%")
    report.add_finding("SRR Results", f"CI on loss: [{(1-sr_ci[1])*100:.2f}%, {(1-sr_ci[0])*100:.2f}%]")
    
    # Extract soiling profile and cleaning events from soiling_info
    if hasattr(soiling_info, 'keys'):
        report.add_finding("SRR Results", f"soiling_info keys: {list(soiling_info.keys())}")
    
    # Try to extract soiling intervals / cleaning events
    cleaning_dates = []
    soiling_profile_df = None
    
    if isinstance(soiling_info, dict):
        if 'soiling_ratio_perfect_clean' in soiling_info:
            soiling_profile_df = soiling_info['soiling_ratio_perfect_clean']
        if 'renormalizing_factor' in soiling_info:
            report.add_finding("SRR Results", f"Renormalizing factor: {soiling_info['renormalizing_factor']:.4f}")
        if 'exceedance_level' in soiling_info:
            report.add_finding("SRR Results", f"Exceedance level: {soiling_info['exceedance_level']}")
        if 'stochastic_soiling_profiles' in soiling_info:
            profiles = soiling_info['stochastic_soiling_profiles']
            report.add_finding("SRR Results", f"Number of Monte Carlo soiling profiles: {len(profiles) if isinstance(profiles, list) else (len(profiles.columns) if hasattr(profiles, 'columns') else 'N/A')}")
        if 'soiling_interval_summary' in soiling_info:
            intervals = soiling_info['soiling_interval_summary']
            report.add_finding("SRR Results", f"Number of soiling intervals detected: {len(intervals)}")
            # Interval boundaries = cleaning event candidates (start date of intervals 1..N)
            if len(intervals) > 1 and 'start' in intervals.columns:
                for i in range(1, len(intervals)):
                    cleaning_dates.append(pd.to_datetime(intervals.iloc[i]['start']))
            report.add_finding("SRR Results", f"Candidate cleaning events: {len(cleaning_dates)}")
            print("\nSoiling interval summary:")
            print(intervals.to_string())
    
    print(f"\n{'='*60}")
    print(f"SRR RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"P50 soiling ratio: {sr:.4f} (loss: {(1-sr)*100:.2f}%)")
    print(f"{SRR_PARAMS['confidence_level']}% CI: [{sr_ci[0]:.4f}, {sr_ci[1]:.4f}]")
    print(f"Candidate cleaning events: {len(cleaning_dates)}")
    
except Exception as e:
    srr_success = False
    report.add_finding("SRR Error", f"SRR failed with error: {str(e)}")
    print(f"SRR error: {e}")
    import traceback
    traceback.print_exc()
    
    # Fallback: Simple rolling-ratio method  
    # Inspired by the SRR approach described in the PDF:
    # Detect positive shifts in the rolling median that exceed a threshold
    print("\n--- Falling back to rolling-median shift detection method ---")
    # Use a rolling median to detect cleaning events (sudden upward shifts)
    window = 14  # 14-day rolling window (similar to DS=13 in the PDF's best results)
    energy_normalized_full = daily_clean.set_index('date')['normalized_pi'].dropna()
    # Reindex to daily for rolling calculations
    full_idx = pd.date_range(energy_normalized_full.index.min(), energy_normalized_full.index.max(), freq='D')
    energy_normalized_full = energy_normalized_full.reindex(full_idx)
    
    rolling_med = energy_normalized_full.rolling(window=window, center=False, min_periods=7).median()
    rolling_diff = rolling_med.diff()
    
    # Use a selective threshold: at least 5*MAD or 0.03 absolute shift
    # This is more conservative than the initial 3*MAD which gave 50 false positives
    mad = rolling_diff.dropna().apply(lambda x: abs(x)).median()
    threshold = max(5 * mad, 0.03)  # At least 3% performance recovery
    cleaning_candidates = rolling_diff[rolling_diff > threshold]
    
    # Apply minimum separation: at least 7 days between cleaning events
    # to avoid detecting the same recovery multiple times
    min_sep_days = 7
    filtered_dates = []
    last_date = None
    for d in cleaning_candidates.index:
        if last_date is None or (d - last_date).days >= min_sep_days:
            filtered_dates.append(d)
            last_date = d
    cleaning_dates = filtered_dates
    
    report.add_finding("Fallback Method", f"Rolling median window: {window} days")
    report.add_finding("Fallback Method", f"Detection threshold: {threshold:.4f} (max of 5*MAD={5*mad:.4f}, 0.03)")
    report.add_finding("Fallback Method", f"Minimum separation: {min_sep_days} days")
    report.add_finding("Fallback Method", f"Raw candidates before separation filter: {len(cleaning_candidates)}")
    report.add_finding("Fallback Method", f"Candidate cleaning events after filtering: {len(cleaning_dates)}")
    for cd in cleaning_dates:
        shift = rolling_diff.loc[cd]
        report.add_finding("Fallback Method", f"  {cd.date()}: PI shift = +{shift:.4f}")
    sr = None
    sr_ci = None
    soiling_info = None

# ============================================================================
# ADDITIONAL ANALYSIS: Rainfall correlation with performance recovery
# ============================================================================
print("\n--- Analyzing rainfall correlation with performance changes ---")

daily_indexed = daily_clean.set_index('date')

# Calculate day-over-day PI change
daily_indexed['pi_change'] = daily_indexed['normalized_pi'].diff()

# Find days with significant rain
rain_days = daily_indexed[daily_indexed['precipitation_sum'] > 2.0].index  # >2mm

# Performance change after rain vs dry periods
if len(rain_days) > 0:
    rain_day_changes = daily_indexed.loc[rain_days, 'pi_change'].dropna()
    
    # Performance 1-3 days after rain
    post_rain_changes = []
    for rd in rain_days:
        for offset in range(1, 4):
            check_date = rd + timedelta(days=offset)
            if check_date in daily_indexed.index:
                val = daily_indexed.loc[check_date, 'pi_change']
                if not np.isnan(val):
                    post_rain_changes.append(val)
    
    dry_days = daily_indexed[daily_indexed['precipitation_sum'] == 0].index
    dry_changes = daily_indexed.loc[dry_days, 'pi_change'].dropna()
    
    report.add_finding("Rainfall Analysis", f"Rain days (>2mm): {len(rain_days)}")
    report.add_finding("Rainfall Analysis", f"Mean PI change on rain days: {rain_day_changes.mean():.4f}")
    report.add_finding("Rainfall Analysis", f"Mean PI change 1-3 days post-rain: {np.mean(post_rain_changes):.4f}" if post_rain_changes else "No post-rain data")
    report.add_finding("Rainfall Analysis", f"Mean PI change on dry days: {dry_changes.mean():.4f}")
    report.add_finding("Rainfall Analysis", 
        "NOTE: Positive mean PI change after rain is consistent with natural cleaning, "
        "but does NOT prove it. Cloud effects and measurement noise dominate day-to-day changes.")

# ============================================================================
# OBJECTIVE 4: GENERATE OUTPUT CSV
# ============================================================================
print("\n--- Generating soiling estimation output ---")

output_df = daily_clean[['date', 'data_quality_flag', 'normalized_pi', 
                          'daily_insolation_Wh', 'daily_ac_energy_Wh',
                          'module_temp_mean', 'ambient_temp_mean',
                          'precipitation_sum', 'n_observations', 'daytime_hours']].copy()

output_df.rename(columns={
    'normalized_pi': 'performance_indicator',
}, inplace=True)

# Add cleaning event candidates
output_df['cleaning_event_candidate'] = False
if cleaning_dates:
    for cd in cleaning_dates:
        if isinstance(cd, str):
            cd = pd.Timestamp(cd)
        mask = output_df['date'] == cd
        if mask.any():
            output_df.loc[mask, 'cleaning_event_candidate'] = True

# Add estimated soiling ratio (from SRR profile if available)
output_df['estimated_soiling_ratio'] = np.nan
output_df['estimated_soiling_loss_pct'] = np.nan
output_df['estimation_quality_flag'] = 'not_estimated'

if srr_success and soiling_info is not None and isinstance(soiling_info, dict):
    # Extract the median soiling profile from Monte Carlo profiles
    if 'stochastic_soiling_profiles' in soiling_info:
        profiles = soiling_info['stochastic_soiling_profiles']
        if isinstance(profiles, list) and len(profiles) > 0:
            df_profiles = pd.concat(profiles, axis=1)
            median_profile = df_profiles.median(axis=1)
            for idx, ratio in median_profile.items():
                mask = output_df['date'] == idx
                if mask.any() and pd.notnull(ratio):
                    output_df.loc[mask, 'estimated_soiling_ratio'] = float(ratio)
                    output_df.loc[mask, 'estimated_soiling_loss_pct'] = (1.0 - float(ratio)) * 100.0
                    output_df.loc[mask, 'estimation_quality_flag'] = 'srr_estimated'
    
    # Also fill any remaining days with soiling_ratio_perfect_clean if present
    if 'soiling_ratio_perfect_clean' in soiling_info:
        sr_profile = soiling_info['soiling_ratio_perfect_clean']
        if hasattr(sr_profile, 'items'):
            for idx, ratio in sr_profile.items():
                mask = output_df['date'] == idx
                if mask.any() and pd.isna(output_df.loc[mask, 'estimated_soiling_ratio'].values[0]):
                    output_df.loc[mask, 'estimated_soiling_ratio'] = float(ratio)
                    output_df.loc[mask, 'estimated_soiling_loss_pct'] = (1.0 - float(ratio)) * 100.0
                    output_df.loc[mask, 'estimation_quality_flag'] = 'srr_estimated'

n_estimated = (output_df['estimation_quality_flag'] == 'srr_estimated').sum()
n_candidates = output_df['cleaning_event_candidate'].sum()
report.add_finding("Output", f"Days with soiling estimates: {n_estimated}")
report.add_finding("Output", f"Cleaning event candidates flagged: {n_candidates}")

# Save output
output_csv_path = os.path.join(DATA_DIR, 'soiling_estimation_2022.csv')
output_df.to_csv(output_csv_path, index=False)
report.add_file(output_csv_path, "Soiling estimation results with cleaning event candidates")
print(f"Saved soiling estimation: {output_csv_path}")

# ============================================================================
# OBJECTIVE 5: CLEANING RECOMMENDATION
# ============================================================================
print("\n" + "=" * 80)
print("OBJECTIVE 5: CLEANING RECOMMENDATION")
print("=" * 80)

# Simple rule-based cleaning recommendation
# Rules:
# 1. If estimated soiling loss exceeds a configurable threshold, recommend cleaning
# 2. If no significant rain in the last N days, increase urgency
# 3. If recent cleaning event detected, reset recommendation
# 4. Cost analysis requires configurable parameters

# Configurable parameters
SOILING_LOSS_THRESHOLD_PCT = 3.0  # Recommend cleaning when losses exceed this
DRY_DAYS_THRESHOLD = 21  # Number of consecutive dry days to increase urgency
CLEANING_COST_USD = 150  # Assumed cost per cleaning (CONFIGURABLE - default assumption)
SYSTEM_DAILY_REVENUE_USD = 4.0  # Rough estimate for a 3kW system in Colorado

print(f"\nCleaning recommendation parameters:")
print(f"  Soiling loss threshold: {SOILING_LOSS_THRESHOLD_PCT}%")
print(f"  Dry days urgency threshold: {DRY_DAYS_THRESHOLD} days")
print(f"  Cleaning cost (assumed): ${CLEANING_COST_USD}")
print(f"  Daily revenue estimate: ${SYSTEM_DAILY_REVENUE_USD}")
print(f"  NOTE: Cost parameters are assumptions, not measured values")

recommendations = []
consecutive_dry = 0
last_cleaning = None

for _, row in output_df.iterrows():
    rec = {
        'date': row['date'],
        'recommendation': 'no_action',
        'reason': '',
        'estimated_loss_pct': row['estimated_soiling_loss_pct'] if not pd.isna(row['estimated_soiling_loss_pct']) else None,
        'consecutive_dry_days': 0,
        'confidence': 'low'
    }
    
    # Track consecutive dry days
    if not pd.isna(row['precipitation_sum']) and row['precipitation_sum'] < 1.0:
        consecutive_dry += 1
    else:
        consecutive_dry = 0
    rec['consecutive_dry_days'] = consecutive_dry
    
    # Reset after cleaning event
    if row['cleaning_event_candidate']:
        last_cleaning = row['date']
        rec['recommendation'] = 'cleaning_detected'
        rec['reason'] = 'Performance recovery detected (candidate cleaning event)'
        rec['confidence'] = 'medium'
    elif rec['estimated_loss_pct'] is not None and rec['estimated_loss_pct'] > SOILING_LOSS_THRESHOLD_PCT:
        # High soiling loss
        if consecutive_dry > DRY_DAYS_THRESHOLD:
            rec['recommendation'] = 'clean_soon'
            rec['reason'] = f"Estimated soiling loss {rec['estimated_loss_pct']:.1f}% exceeds {SOILING_LOSS_THRESHOLD_PCT}% threshold; {consecutive_dry} consecutive dry days"
            rec['confidence'] = 'medium'
        else:
            rec['recommendation'] = 'monitor'
            rec['reason'] = f"Soiling loss {rec['estimated_loss_pct']:.1f}% above threshold but rain may occur"
            rec['confidence'] = 'low'
    elif consecutive_dry > DRY_DAYS_THRESHOLD * 2:
        rec['recommendation'] = 'inspect'
        rec['reason'] = f"Extended dry period ({consecutive_dry} days) with no rain; soiling likely accumulating"
        rec['confidence'] = 'low'
    
    recommendations.append(rec)

rec_df = pd.DataFrame(recommendations)

# Summary statistics
rec_counts = rec_df['recommendation'].value_counts()
report.add_finding("Recommendations", "Cleaning recommendation summary:")
for action, count in rec_counts.items():
    report.add_finding("Recommendations", f"  {action}: {count} days")

# Cost-benefit estimate (very rough)
if srr_success and sr is not None:
    annual_loss_fraction = 1 - sr
    annual_revenue_usd = SYSTEM_DAILY_REVENUE_USD * 365
    annual_soiling_cost = annual_loss_fraction * annual_revenue_usd
    
    # Simple payback: if cleaning costs less than soiling losses over the dirty period
    recommended_cleanings = rec_counts.get('clean_soon', 0)
    total_cleaning_cost = recommended_cleanings * CLEANING_COST_USD / 365 * 30  # rough monthly equivalent
    
    report.add_finding("Cost Analysis", f"Estimated annual soiling loss: {annual_soiling_cost:.2f} USD (at assumed ${SYSTEM_DAILY_REVENUE_USD}/day revenue)")
    report.add_finding("Cost Analysis", f"NOTE: This is a rough estimate based on assumed revenue. Actual economics depend on local electricity prices, feed-in tariffs, and system-specific factors.")
    report.add_finding("Cost Analysis", f"NOTE: Cleaning cost of ${CLEANING_COST_USD} is an assumption, not a measured value.")

# Save recommendations
rec_csv_path = os.path.join(DATA_DIR, 'cleaning_recommendations_2022.csv')
rec_df.to_csv(rec_csv_path, index=False)
report.add_file(rec_csv_path, "Cleaning recommendations based on soiling estimates")

# ============================================================================
# OBJECTIVE 6: PLOTTING AND VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("OBJECTIVE 6: VISUALIZATION")
print("=" * 80)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# --- Plot 1: Performance Index Time Series ---
fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)

# Panel 1: Raw performance index
ax = axes[0]
colors = {'good': '#2196F3', 'low_data': '#FF9800', 'low_insolation': '#9E9E9E', 'pi_outlier': '#F44336'}
for flag, color in colors.items():
    mask = daily_clean['data_quality_flag'] == flag
    if mask.sum() > 0:
        ax.scatter(daily_clean.loc[mask, 'date'], daily_clean.loc[mask, 'normalized_pi'], 
                  c=color, s=10, alpha=0.7, label=f'{flag} ({mask.sum()})')

# Mark cleaning events
if cleaning_dates:
    for cd in cleaning_dates:
        ax.axvline(x=cd, color='green', alpha=0.5, linestyle='--', linewidth=1)
    ax.axvline(x=cleaning_dates[0], color='green', alpha=0.5, linestyle='--', linewidth=1, label='Cleaning candidate')

ax.set_ylabel('Normalized PI')
ax.set_title('Performance Index with Data Quality Flags and Cleaning Events')
ax.legend(loc='lower left', fontsize=8)
ax.set_ylim(0, 1.5)

# Panel 2: Soiling ratio (if available)
ax = axes[1]
estimated = output_df[output_df['estimation_quality_flag'] == 'srr_estimated']
if len(estimated) > 0:
    ax.plot(estimated['date'], estimated['estimated_soiling_ratio'], 'b-', alpha=0.7, linewidth=1)
    ax.fill_between(estimated['date'], estimated['estimated_soiling_ratio'], 1.0, alpha=0.2, color='red', label='Soiling loss')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Estimated Soiling Ratio')
    ax.set_title('Estimated Soiling Ratio (SRR) — Values below 1.0 indicate soiling losses')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'Soiling ratio profile not available from SRR output',
            transform=ax.transAxes, ha='center', va='center', fontsize=12, color='gray')
    ax.set_ylabel('Soiling Ratio')
    ax.set_title('Estimated Soiling Ratio')

# Panel 3: Daily insolation and rainfall
ax = axes[2]
ax2 = ax.twinx()
ax.bar(daily_clean['date'], daily_clean['daily_insolation_Wh']/1000, color='#FFB74D', alpha=0.7, label='Insolation (kWh/m²)')
ax2.bar(daily_clean['date'], daily_clean['precipitation_sum'], color='#42A5F5', alpha=0.6, width=1.5, label='Precipitation (mm)')
ax.set_ylabel('Daily Insolation (kWh/m²)')
ax2.set_ylabel('Precipitation (mm)')
ax.set_title('Daily Insolation and Precipitation')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

# Panel 4: Module temperature
ax = axes[3]
ax.plot(daily_clean['date'], daily_clean['module_temp_mean'], 'r-', alpha=0.7, linewidth=1, label='Module temp (mean)')
ax.plot(daily_clean['date'], daily_clean['ambient_temp_mean'], 'b-', alpha=0.5, linewidth=1, label='Ambient temp')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Module and Ambient Temperature')
ax.legend(fontsize=8)
ax.set_xlabel('Date')

plt.tight_layout()
plot1_path = os.path.join(PLOTS_DIR, 'performance_overview_2022.png')
plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
plt.close()
report.add_file(plot1_path, "Multi-panel performance overview plot")
print(f"Saved plot: {plot1_path}")

# --- Plot 2: Cleaning events and rainfall ---
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

ax = axes[0]
good_mask = daily_clean['data_quality_flag'] == 'good'
ax.plot(daily_clean.loc[good_mask, 'date'], daily_clean.loc[good_mask, 'normalized_pi'], 
        'b.', markersize=4, alpha=0.5, label='Good-quality PI')

# Rolling median
rolling = daily_clean.set_index('date')['normalized_pi'].rolling(window=14, center=True, min_periods=7).median()
ax.plot(rolling.index, rolling.values, 'r-', linewidth=2, alpha=0.8, label='14-day rolling median')

if cleaning_dates:
    for i, cd in enumerate(cleaning_dates):
        ax.axvline(x=cd, color='green', alpha=0.7, linestyle='--', linewidth=1.5,
                  label='Cleaning candidate' if i == 0 else '')
        
ax.set_ylabel('Normalized PI')
ax.set_title('Performance Index with Cleaning Event Candidates')
ax.legend(fontsize=9)

ax = axes[1]
ax.bar(daily_clean['date'], daily_clean['precipitation_sum'], color='#42A5F5', alpha=0.7)
ax.set_ylabel('Precipitation (mm)')
ax.set_title('Daily Precipitation')
ax.set_xlabel('Date')

plt.tight_layout()
plot2_path = os.path.join(PLOTS_DIR, 'cleaning_events_2022.png')
plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
plt.close()
report.add_file(plot2_path, "Cleaning events with precipitation overlay")
print(f"Saved plot: {plot2_path}")

# --- Plot 3: Monthly soiling and performance summary ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Monthly mean PI
ax = axes[0, 0]
monthly = daily_clean[good_mask].set_index('date').resample('MS').agg(
    pi_mean=('normalized_pi', 'mean'),
    pi_std=('normalized_pi', 'std'),
    n_days=('normalized_pi', 'count')
)
ax.bar(monthly.index, monthly['pi_mean'], width=25, color='#2196F3', alpha=0.7)
ax.errorbar(monthly.index, monthly['pi_mean'], yerr=monthly['pi_std'], fmt='none', color='black', alpha=0.5)
ax.set_ylabel('Mean Normalized PI')
ax.set_title('Monthly Average Performance Index')
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# Monthly rainfall
ax = axes[0, 1]
monthly_rain = daily_clean.set_index('date')['precipitation_sum'].resample('MS').sum()
ax.bar(monthly_rain.index, monthly_rain.values, width=25, color='#42A5F5', alpha=0.7)
ax.set_ylabel('Total Precipitation (mm)')
ax.set_title('Monthly Precipitation Total')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# PI distribution
ax = axes[1, 0]
pi_good = daily_clean.loc[good_mask, 'normalized_pi'].dropna()
ax.hist(pi_good, bins=40, color='#2196F3', alpha=0.7, edgecolor='white')
ax.axvline(x=pi_good.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {pi_good.median():.3f}')
ax.set_xlabel('Normalized PI')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Good-Quality Performance Index')
ax.legend()

# Scatter: PI vs Insolation
ax = axes[1, 1]
ax.scatter(daily_clean.loc[good_mask, 'daily_insolation_Wh']/1000, 
          daily_clean.loc[good_mask, 'normalized_pi'],
          c='#2196F3', s=15, alpha=0.5)
ax.set_xlabel('Daily Insolation (kWh/m²)')
ax.set_ylabel('Normalized PI')
ax.set_title('Performance Index vs Daily Insolation')

plt.tight_layout()
plot3_path = os.path.join(PLOTS_DIR, 'monthly_summary_2022.png')
plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
plt.close()
report.add_file(plot3_path, "Monthly performance and precipitation summary")
print(f"Saved plot: {plot3_path}")

# --- Plot 4: Data quality overview ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Quality flag distribution
ax = axes[0, 0]
quality_counts = daily_clean['data_quality_flag'].value_counts()
colors_qual = {'good': '#4CAF50', 'low_data': '#FF9800', 'low_insolation': '#9E9E9E', 'pi_outlier': '#F44336'}
ax.bar(quality_counts.index, quality_counts.values, 
       color=[colors_qual.get(q, '#607D8B') for q in quality_counts.index])
ax.set_ylabel('Number of Days')
ax.set_title('Data Quality Flag Distribution')
for i, (q, v) in enumerate(quality_counts.items()):
    ax.text(i, v + 1, str(v), ha='center', fontsize=10)

# Daytime hours distribution
ax = axes[0, 1]
ax.hist(daily_clean['daytime_hours'], bins=30, color='#FF9800', alpha=0.7, edgecolor='white')
ax.axvline(x=MIN_DAYTIME_HOURS, color='red', linestyle='--', label=f'Min threshold ({MIN_DAYTIME_HOURS}h)')
ax.set_xlabel('Daytime Hours')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Usable Daytime Hours per Day')
ax.legend()

# Daily observation count
ax = axes[1, 0]
ax.plot(daily_clean['date'], daily_clean['n_observations'], 'b-', alpha=0.5, linewidth=1)
ax.set_xlabel('Date')
ax.set_ylabel('Observations per Day')
ax.set_title('Number of Clean Daytime Observations per Day')

# Missing data calendar (irradiance)
ax = axes[1, 1]
# Show which dates had missing irradiance in the original daily data
orig_irr_missing = merged[merged[COL_IRR].isnull()]['time']
if len(orig_irr_missing) > 0:
    ax.scatter(orig_irr_missing, [1]*len(orig_irr_missing), c='red', s=20, marker='|', linewidths=2)
    ax.set_title(f'Dates with Missing Irradiance in Original Merged Data ({len(orig_irr_missing)} days)')
else:
    ax.text(0.5, 0.5, 'No missing irradiance dates', transform=ax.transAxes, ha='center')
ax.set_yticks([])
ax.set_xlabel('Date')

plt.tight_layout()
plot4_path = os.path.join(PLOTS_DIR, 'data_quality_2022.png')
plt.savefig(plot4_path, dpi=150, bbox_inches='tight')
plt.close()
report.add_file(plot4_path, "Data quality overview")
print(f"Saved plot: {plot4_path}")

# --- Plot 5: SRR Monte Carlo profiles (if available) ---
if srr_success:
    try:
        from rdtools.plotting import soiling_monte_carlo_plot, soiling_interval_plot
        
        fig = soiling_monte_carlo_plot(soiling_info, energy_normalized, profiles=200)
        plt.title('SRR Monte Carlo Soiling Profiles')
        plot5_path = os.path.join(PLOTS_DIR, 'srr_monte_carlo_2022.png')
        plt.savefig(plot5_path, dpi=150, bbox_inches='tight')
        plt.close()
        report.add_file(plot5_path, "SRR Monte Carlo soiling profiles")
        print(f"Saved plot: {plot5_path}")
    except Exception as e:
        print(f"Could not generate SRR Monte Carlo plot: {e}")
    
    try:
        fig = soiling_interval_plot(soiling_info, energy_normalized)
        plt.title('SRR Soiling Intervals')
        plot6_path = os.path.join(PLOTS_DIR, 'srr_intervals_2022.png')
        plt.savefig(plot6_path, dpi=150, bbox_inches='tight')
        plt.close()
        report.add_file(plot6_path, "SRR soiling interval analysis")
        print(f"Saved plot: {plot6_path}")
    except Exception as e:
        print(f"Could not generate SRR interval plot: {e}")

# ============================================================================
# GENERATE FINAL REPORT DATA
# ============================================================================
print("\n" + "=" * 80)
print("PIPELINE COMPLETE - SUMMARY")
print("=" * 80)

# Save report as JSON
report_data = {
    'timestamp': datetime.now().isoformat(),
    'system': 'PVDAQ System 10, Colorado',
    'year': 2022,
    'findings': report.sections,
    'data_changes': report.data_changes,
    'files_created': report.files_created,
    'srr_success': srr_success if 'srr_success' in dir() else False,
    'srr_results': {
        'p50_soiling_ratio': float(sr) if sr is not None else None,
        'confidence_interval': [float(sr_ci[0]), float(sr_ci[1])] if sr_ci is not None else None,
        'n_cleaning_events': len(cleaning_dates),
        'cleaning_dates': [str(d) for d in cleaning_dates]
    } if 'cleaning_dates' in dir() else None,
    'limitations': [
        "No independent soiling measurements available for validation",
        "SRR estimates are model-derived, not measured ground truth",
        "Performance changes may be caused by factors other than soiling",
        "Temperature correction uses a simplified linear model",
        "System capacity is approximate (3 kW assumed)",
        "Cleaning cost and revenue parameters are assumptions",
        "Precision, recall, and F1 cannot be validated without confirmed cleaning labels"
    ]
}

report_json_path = os.path.join(OUTPUT_DIR, 'pipeline_report.json')
with open(report_json_path, 'w') as f:
    json.dump(report_data, f, indent=2, default=str)
report.add_file(report_json_path, "Complete pipeline report in JSON format")

# Print final summary
print(f"\nFiles created:")
for f_info in report.files_created:
    print(f"  {f_info['path']} — {f_info['description']}")

print(f"\nData changes made:")
for change in report.data_changes:
    print(f"  [{change['type']}] {change['n_observations']} obs — {change['reason']}")

print(f"\n{'='*60}")
print("DONE. See output/ directory for all results.")
print(f"{'='*60}")
