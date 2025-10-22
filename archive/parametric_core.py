import pandas as pd
import numpy as np
import os
import imdlib as imd # Assumes IMD library is installed and configured
from datetime import datetime
import matplotlib.pyplot as plt # Kept for AEP plotting, though display is often suppressed in scripts

# ==============================================================================
# 0. Configuration Constants & Setup (Placeholders)
# ==============================================================================

# NOTE: These constants must be defined before running the analysis functions
# LPA_START_YR and LPA_END_YR define the period used to calculate the Long Period Average (LPA)
LPA_START_YR = 1981
LPA_END_YR = 2010
DATA_DIR = './imd_data'
LPA_DIR = './lpa_cache'
BASE_TRIGGER_MM = 204.5 # IMD's "Extremely Heavy Rainfall" threshold
MAX_PAYOUT = 100000.0

# ==============================================================================
# 1. Core Utilities and Data Handling
# ==============================================================================

def setup_data_directories(data_dir, lpa_dir):
    """Ensures necessary data directories exist."""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(lpa_dir, exist_ok=True)

def ensure_historical_data_exists(data_dir, start_yr, end_yr):
    """Placeholder function to confirm data availability."""
    print("✅ Historical data files assumed available or previously downloaded.")

def load_localized_data(lat, lon, data_dir, start_yr, end_yr, cumulative_days=1):
    """
    Loads and localizes the daily rainfall data for a single grid point (lat, lon).
    Returns a Pandas Series with a DateTime index.
    
    :param cumulative_days: If > 1, applies a rolling sum for cumulative index calculation.
    """
    try:
        # Simulate loading the IMD data object
        ds = imd.open_data('rain', start_yr, end_yr, file_dir=data_dir, fn_format='yearwise')
        # Select the nearest grid point for the specified coordinates
        regional_ds = ds.get_xarray().sel(lat=lat, lon=lon, method='nearest')['rain']
        df = regional_ds.to_dataframe().dropna()
    except Exception as e:
        print(f"Error loading IMD data: {e}")
        return pd.Series([], dtype=float)

    if cumulative_days > 1:
        # Rolling sum over the defined number of days for cumulative index
        df['rain'] = df['rain'].rolling(window=cumulative_days, min_periods=cumulative_days).sum()
        df = df.dropna()
        
    return df['rain'].rename('Localized_Rainfall')

def get_climatological_data(lat, lon, data_dir, start_yr, end_yr, cumulative_days=1):
    """
    Calculates localized LPA and Std Dev over the reference period.
    Returns LPA, Std Dev, and the full rain series DATA.
    """
    rain_series = load_localized_data(lat, lon, data_dir, start_yr, end_yr, cumulative_days)
    if rain_series.empty:
        return 0.0, 0.0, rain_series
        
    lpa = rain_series.mean()
    std_dev = rain_series.std()
    return lpa, std_dev, rain_series 

# ==============================================================================
# 2. Contract Type 1: Statistical Deviation Payout (LPA Sigma)
# ==============================================================================

def calculate_payout_statistical(index_value, lpa, std_dev, trigger_std_dev, max_payout):
    """
    Calculates payout based on the index value's distance from the LPA in standard deviations.
    This simulates a graduated payout for moderate to severe anomalies (e.g., Z-score > 2.0).
    """
    if std_dev == 0 or index_value < lpa:
        return 0.0

    # Calculate Z-score
    z_score = (index_value - lpa) / std_dev
    
    # Check if the Z-score is above the trigger threshold
    if z_score < trigger_std_dev:
        return 0.0
    
    # Simple linear graduation from the trigger Z-score up to 1.0 sigma above the trigger
    payout_range_sigma = 1.0 # Payout scales over 1.0 standard deviation band
    max_z = trigger_std_dev + payout_range_sigma
    
    # Normalize the excess Z-score over the defined range (clamped at 1.0)
    normalized_payout_factor = min(1.0, (z_score - trigger_std_dev) / payout_range_sigma)
    
    return normalized_payout_factor * max_payout

# ==============================================================================
# 3. Contract Type 2: Graduated Catastrophic Payout (Extreme Tail)
# ==============================================================================

def calibrate_extreme_tail(rain_series, base_trigger_mm):
    """
    CALIBRATION STEP: Calculates the quartiles (25th, 50th, 75th percentiles) 
    of historical rainfall events that were >= the base_trigger_mm.
    """
    # Filter the entire historical data to only include extreme events
    extreme_tail = rain_series.values[rain_series.values >= base_trigger_mm]
    
    if len(extreme_tail) < 4:
        # Fallback if insufficient historical data exists above the trigger
        max_val = np.max(extreme_tail) if len(extreme_tail) > 0 else base_trigger_mm
        print("Warning: Insufficient data for precise quartile calibration. Using fallback steps.")
        return {'count': len(extreme_tail), 'step_1': base_trigger_mm + 10, 'step_2': max_val * 0.75, 'step_3': max_val}

    # Calculate the quartiles (percentiles) of the extreme tail
    steps = np.percentile(extreme_tail, [25, 50, 75])
    
    return {
        'count': len(extreme_tail),
        'step_1': steps[0],  # 25% Payout Start
        'step_2': steps[1],  # 50% Payout Start
        'step_3': steps[2]   # 75% Payout Start
    }

def calculate_payout_graduated(index_value, base_trigger, steps, max_payout):
    """
    Calculates a graduated payout using the calibrated steps from the extreme tail.
    Packs 0-100% payout into 4 zones defined by the base trigger and 3 historical steps.
    """
    if index_value < base_trigger:
        return 0.0

    step_1_val = steps['step_1']
    step_2_val = steps['step_2']
    step_3_val = steps['step_3']
    
    # Zone 1: Below Step 1 (0% to 25% Payout) - Uses a linear scale between trigger and step 1
    if index_value < step_1_val:
        if step_1_val <= base_trigger: return max_payout * 0.25
        return (index_value - base_trigger) / (step_1_val - base_trigger) * (max_payout * 0.25)
    
    # Zone 2: Between Step 1 and Step 2 (25% to 50% Payout)
    elif index_value < step_2_val:
        if step_2_val <= step_1_val: return max_payout * 0.50
        payout = max_payout * 0.25 
        payout += (index_value - step_1_val) / (step_2_val - step_1_val) * (max_payout * 0.25)
        return payout
        
    # Zone 3: Between Step 2 and Step 3 (50% to 75% Payout)
    elif index_value < step_3_val:
        if step_3_val <= step_2_val: return max_payout * 0.75
        payout = max_payout * 0.50
        payout += (index_value - step_2_val) / (step_3_val - step_2_val) * (max_payout * 0.25)
        return payout

    # Zone 4: Above Step 3 (75% to 100% Payout) - Maximum payout achieved
    else: 
        return max_payout 

# ==============================================================================
# 4. Contract Type 3: Binary Catastrophic Payout (Hard Floor)
# ==============================================================================

def calculate_payout_binary(index_value, base_trigger, max_payout):
    """
    Calculates the binary payout: Max Payout if index >= trigger, else 0.
    This is the simplest, pure catastrophe risk transfer model.
    """
    if index_value >= base_trigger:
        return max_payout
    return 0.0

# ==============================================================================
# 5. Actuarial and Distribution Analysis
# ==============================================================================

def calculate_historical_loss_expectancy(lat, lon, data_dir, lpa_start_yr, lpa_end_yr, base_trigger_mm, max_payout, study_start_yr, study_end_yr, payout_function, **kwargs):
    """
    Runs the historical stress test for any defined payout function (Type 1, 2, or 3).
    Calculates Expected Annual Loss (EAL) and Maximum Annual Loss (MAL).
    """
    print(f"\n--- Historical Stress Test using {payout_function.__name__} ---")
    
    # 1. Load data
    lpa, std_dev, rain_series = get_climatological_data(lat, lon, data_dir, lpa_start_yr, lpa_end_yr)
    df = rain_series.to_frame(name='Rainfall') 
    
    # --- Filter to the specific Study Period ---
    df['Year'] = df.index.year
    study_df = df[(df['Year'] >= study_start_yr) & (df['Year'] <= study_end_yr)].copy()

    # 2. Apply payout logic
    
    if payout_function.__name__ == 'calculate_payout_binary':
        study_df['Payout'] = study_df['Rainfall'].apply(
            lambda x: payout_function(x, base_trigger_mm, max_payout)
        )
    elif payout_function.__name__ == 'calculate_payout_statistical':
        # Requires 'trigger_std_dev' to be passed in kwargs
        trigger_std_dev = kwargs.get('trigger_std_dev', 2.5) 
        study_df['Payout'] = study_df['Rainfall'].apply(
            lambda x: payout_function(x, lpa, std_dev, trigger_std_dev, max_payout)
        )
    elif payout_function.__name__ == 'calculate_payout_graduated':
        # Calibrate steps based on historical data for the LPA period
        steps = calibrate_extreme_tail(rain_series, base_trigger_mm)
        study_df['Payout'] = study_df['Rainfall'].apply(
            lambda x: payout_function(x, base_trigger_mm, steps, max_payout)
        )
    else:
        print("⚠️ Payout function not supported or requires specific arguments.")
        return

    # Calculate financial metrics
    max_years_for_avg = len(study_df['Year'].unique())
    total_loss_expectancy = study_df['Payout'].sum()
    expected_annual_loss = total_loss_expectancy / max_years_for_avg if max_years_for_avg > 0 else 0
    max_annual_loss = study_df.groupby('Year')['Payout'].sum().max()
    
    print(f"💰 Expected Annual Loss (EAL): {expected_annual_loss:,.2f}")
    print(f"💸 Maximum Annual Loss (MAL): {max_annual_loss:,.2f}")
    return expected_annual_loss, max_annual_loss


def analyze_aep_and_distribution(lat, lon, data_dir, start_yr, end_yr, base_trigger_mm):
    """
    Analyzes the Annual Exceedance Probability (AEP) and Return Period (RP)
    for the single BASE_TRIGGER_MM using the full climatology range.
    """
    print("\n--- Trigger Validation (AEP and Return Period) ---")
    
    rain_series = load_localized_data(lat, lon, data_dir, start_yr, end_yr)
    df = rain_series.to_frame()
    
    # Calculate Annual Maxima (AMAX) Series
    df['Year'] = df.index.year
    amax_series = df.groupby('Year')['Localized_Rainfall'].max()
    
    # AEP/RP Calculation for the BASE_TRIGGER_MM
    years_exceeded = amax_series[amax_series >= base_trigger_mm].count()
    total_years = amax_series.shape[0]
    
    aep = (years_exceeded / total_years) * 100
    return_period = total_years / years_exceeded if years_exceeded > 0 else float('inf')

    print(f"BASE TRIGGER ({base_trigger_mm:.1f} mm) AEP/RP Analysis:")
    print(f" - AEP (Annual Exceedance Probability): {aep:.2f}%")
    print(f" - Return Period (RP): {return_period:.1f} Years")
    
    # --- PLOTTING CODE REMOVED (or suppressed in a script environment) ---
    # The plotting part is commented out as requested for a pure script file.
    # plt.figure(figsize=(10, 5))
    # amax_series.plot(kind='bar', color='skyblue')
    # plt.axhline(base_trigger_mm, color='red', linestyle='--', linewidth=2, label=f'Base Trigger ({base_trigger_mm:.1f} mm)')
    # plt.title(f'Annual Maximum Rainfall Series', fontsize=14)
    # plt.ylabel('Annual Maximum Rainfall (mm)', fontsize=12)
    # plt.xlabel('Year', fontsize=12)
    # plt.legend()
    # plt.grid(axis='y', linestyle='dotted')
    # plt.tight_layout()
    # plt.show()
    # ---------------------------------------------------------------------
    
    return aep, return_period

def run_live_parametric_engine(base_trigger_mm, max_payout, policy_start_date, policy_end_date):
    """
    A placeholder to simulate checking for a live payout on the current day 
    using the Type 3 (Binary) Contract logic.
    """
    print("\n--- Final Live Check (Using Binary Contract Logic) ---")
    
    current_date = datetime.now()
    # Simulate current index reading for a check
    simulated_rain_mm = 250.0 # Hypothetically, the latest reading
    
    # Check if we are in the policy window
    start_dt = datetime.strptime(policy_start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(policy_end_date, '%Y-%m-%d')

    if not (start_dt <= current_date <= end_dt):
        print(f"Policy is OUTSIDE the policy window ({policy_start_date} to {policy_end_date}). No payout possible.")
        return 0.0

    print(f"Current Date: {current_date.strftime('%Y-%m-%d')}")
    print(f"Simulated Rainfall Index: {simulated_rain_mm:.2f} mm")
    
    payout = calculate_payout_binary(simulated_rain_mm, base_trigger_mm, max_payout)
    
    if payout > 0:
        print(f"✅ EVENT TRIGGERED! Full Payout: {payout:,.2f}")
    else:
        print("❌ Index not high enough. No Payout.")
        
    return payout

# ==============================================================================
# 6. Example Execution Block (For demonstration purposes)
# ==============================================================================

if __name__ == '__main__':
    # --- Example Configuration ---
    # NOTE: You must have IMD data downloaded to DATA_DIR for this to run fully.
    
    LAT = 18.5  # Pune, Maharashtra
    LON = 73.8
    STUDY_START = 2011
    STUDY_END = 2020
    POLICY_START = '2025-06-01'
    POLICY_END = '2025-09-30'
    
    # 1. Setup
    setup_data_directories(DATA_DIR, LPA_DIR)
    ensure_historical_data_exists(DATA_DIR, LPA_START_YR, STUDY_END)
    
    # 2. Run Analysis for Type 3 (Binary Catastrophe)
    eal_binary, mal_binary = calculate_historical_loss_expectancy(
        lat=LAT, lon=LON, data_dir=DATA_DIR, 
        lpa_start_yr=LPA_START_YR, lpa_end_yr=LPA_END_YR, 
        base_trigger_mm=BASE_TRIGGER_MM, max_payout=MAX_PAYOUT,
        study_start_yr=STUDY_START, study_end_yr=STUDY_END, 
        payout_function=calculate_payout_binary
    )

    # 3. Run Analysis for Type 1 (Statistical Deviation - 2.5 Sigma Trigger)
    eal_stat, mal_stat = calculate_historical_loss_expectancy(
        lat=LAT, lon=LON, data_dir=DATA_DIR, 
        lpa_start_yr=LPA_START_YR, lpa_end_yr=LPA_END_YR, 
        base_trigger_mm=BASE_TRIGGER_MM, max_payout=MAX_PAYOUT,
        study_start_yr=STUDY_START, study_end_yr=STUDY_END, 
        payout_function=calculate_payout_statistical,
        trigger_std_dev=2.5 # Specific parameter for statistical model
    )
    
    # 4. AEP/RP Analysis for the base trigger
    aep, rp = analyze_aep_and_distribution(LAT, LON, DATA_DIR, LPA_START_YR, STUDY_END, BASE_TRIGGER_MM)

    # 5. Live Check
    # run_live_parametric_engine(BASE_TRIGGER_MM, MAX_PAYOUT, POLICY_START, POLICY_END)
    pass
