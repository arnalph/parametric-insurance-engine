# =============================================================================
# PARAMETRIC CORE MODULE (parametric_core.py) - V8 (PERCENTILE TRIGGER)
#
# CRITICAL CHANGE: Partial Trigger is now set via Percentile Rank (e.g., 99.9) 
# to accurately model skewed rainfall data, replacing the unreliable LPA + X*sigma.
# =============================================================================

# --- Dependencies ---
import imdlib as imd
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import xarray 
# Import the data check utility from the new data_ingestion module
from data_ingestion import ensure_historical_data_exists 


# =============================================================================
# 1. CORE DATA UTILITIES
# =============================================================================

def get_climatological_data(lat, lon, data_dir, lpa_file, lpa_start_yr, lpa_end_yr, percentile_rank):
    """
    Calculates the Localized LPA, Std Dev, AND the Percentile Trigger Value, 
    using cache if available.
    """
    print(f"\n--- Calculating Climatological LPA and Percentile Trigger for Target Point ({lat:.4f}°N, {lon:.4f}°E) ---")

    # --- CHECK FOR CACHED DATA ---
    cache_key = f"{lat:.4f}_{lon:.4f}_{percentile_rank:.1f}"
    
    if os.path.exists(lpa_file):
        try:
            cache_df = pd.read_csv(lpa_file, index_col='location')
            # Check if the specific location's LPA and PERCENTILE are already cached
            if cache_key in cache_df.index:
                lpa_daily_mm = cache_df.loc[cache_key, 'LPA_Daily_MM']
                std_daily_mm = cache_df.loc[cache_key, 'StdDev_Daily_MM']
                percentile_val = cache_df.loc[cache_key, 'Percentile_Trigger_MM']
                print(f"✅ Loaded LPA, Std Dev, and {percentile_rank:.1f}th Percentile from cache.")
                return round(lpa_daily_mm, 2), round(std_daily_mm, 2), round(percentile_val, 2)
        except Exception as e:
            print(f"Warning: Could not read or parse cache file. Recalculating. Error: {e}")

    # --- CALCULATION (if cache miss or read failure) ---
    print("Cache miss or invalid. Calculating metrics from historical IMD data...")
    try:
        data = imd.open_data('rain', lpa_start_yr, lpa_end_yr, 'yearwise', file_dir=data_dir)
        ds = data.get_xarray()
        
        # Selects only the time series for the grid cell nearest the target lat/lon.
        regional_ds = ds.sel(lat=lat, lon=lon, method='nearest')['rain']
        
        clean_rainfall = regional_ds.where(regional_ds != -999.0).values.flatten()
        clean_rainfall = clean_rainfall[~np.isnan(clean_rainfall)]

        if clean_rainfall.size == 0:
            print("Climatology calculation failed: No valid data points found.")
            return 0.0, 0.0, 0.0
            
        # 1. Standard Climatology
        lpa_daily_mm = np.mean(clean_rainfall)
        std_daily_mm = np.std(clean_rainfall) 
        
        # 2. *** NEW: PERCENTILE TRIGGER CALCULATION ***
        percentile_val = np.percentile(clean_rainfall, percentile_rank)
        
        # --- SAVE TO CACHE ---
        new_data = pd.DataFrame({
            'location': [cache_key], 
            'LPA_Daily_MM': [lpa_daily_mm], 
            'StdDev_Daily_MM': [std_daily_mm],
            'Percentile_Trigger_MM': [percentile_val]
        }).set_index('location')
        
        if os.path.exists(lpa_file):
            cache_df = pd.read_csv(lpa_file, index_col='location')
            cache_df = pd.concat([cache_df, new_data]).drop_duplicates(keep='last')
        else:
            cache_df = new_data

        cache_df.to_csv(lpa_file)
        print(f"✅ Successfully calculated and saved localized climatology to cache.")
        
        return round(lpa_daily_mm, 2), round(std_daily_mm, 2), round(percentile_val, 2)
        
    except Exception as e:
        print(f"Error reading local IMD data for Climatology: {e}")
        return None, None, None


def get_cumulative_rainfall_index(lat, lon, index_days, data_dir):
    """
    Fetches the necessary recent data and calculates the rolling index.
    """
    index_description = "Single-Day" if index_days == 1 else f"{index_days}-Day Cumulative"
    print(f"\n--- Fetching Real-Time IMD Data for {index_description} Index ---")
    latest_available_date = datetime.now() - timedelta(days=1)
    end_dy = latest_available_date.strftime('%Y-%m-%d')
    start_dy = (latest_available_date - timedelta(days=index_days + 2)).strftime('%Y-%m-%d')
    
    print(f"Requesting data range: {start_dy} to {end_dy} (Latest available)")
    
    try:
        data = imd.get_real_data('rain', start_dy, end_dy, file_dir=data_dir) 
        ds = data.get_xarray()
        regional_ds = ds.sel(lat=lat, lon=lon, method='nearest')['rain']
        rainfall_series = regional_ds.to_pandas().replace(-999.0, np.nan)
        
        # Use a rolling sum of 1 for single-day, or N for cumulative
        cumulative_series = rainfall_series.rolling(window=index_days, min_periods=index_days).sum()
        latest_index = cumulative_series.dropna().iloc[-1]
        
        print(f"Latest {index_description} index calculation successful.")
        return round(latest_index, 1)

    except Exception as e:
        print(f"Error fetching real-time IMD data: {e}")
        return None

# =============================================================================
# 2. CORE ENGINE & STRESS TESTING
# =============================================================================

def calculate_payout_triggers(partial_trigger_val, full_payout_threshold):
    """Calculates the two payout triggers based on the new Percentile value."""
    if partial_trigger_val is None:
        return None, None
        
    partial_trigger = partial_trigger_val
    full_trigger = full_payout_threshold 
    
    # Safety check: ensure partial trigger is always lower than full trigger
    if partial_trigger >= full_trigger:
         partial_trigger = full_trigger * 0.75 
         
    return partial_trigger, full_trigger


def run_live_parametric_engine(lat, lon, data_dir, lpa_file, lpa_start_yr, lpa_end_yr, 
                               full_payout_threshold, percentile_rank, index_days, 
                               payout_amount_max, policy_start_date, policy_end_date):
    """Executes the parametric index check and determines the financial payout for the latest date."""
    
    index_description = "Single-Day Rainfall" if index_days == 1 else f"{index_days}-Day Cumulative Rainfall"
    
    print(f"\n====================================================================")
    print(f"⛈️  LIVE PARAMETRIC PAYOUT DECISION ({index_description} Index) ⛈️")
    
    current_date = datetime.now().date()
    policy_start = datetime.strptime(policy_start_date, '%Y-%m-%d').date()
    policy_end = datetime.strptime(policy_end_date, '%Y-%m-%d').date()

    # 1. Setup & Climatology
    lpa_daily, std_dev_daily, partial_trigger_val = get_climatological_data(
        lat, lon, data_dir, lpa_file, lpa_start_yr, lpa_end_yr, percentile_rank)
    
    partial_trigger, full_trigger = calculate_payout_triggers(partial_trigger_val, full_payout_threshold)
    if partial_trigger is None: 
        print("\nPAYOUT DECISION: INDETERMINATE (Climatology data missing/invalid)."); return

    print("\n--- Model Payout Parameters ---")
    print(f"Partial Payout Start Trigger ({percentile_rank:.1f}% Rank): {partial_trigger:.2f} mm")
    print(f"Full Payout Trigger (Fixed Severity): {full_trigger:.2f} mm")
    
    # 2. Get Live Index Value
    current_index = get_cumulative_rainfall_index(lat, lon, index_days, data_dir)
    if current_index is None:
        print("\nPAYOUT DECISION: INDETERMINATE\nReason: Index data is unavailable or invalid."); return

    print(f"\nINDEX VALUE ({index_description}): {current_index:.1f} mm")

    # 3. Payout Decision Logic
    payout_level = 0.0
    payout_reason = "Index below all defined triggers (NO PAYOUT)."
    
    if not (policy_start <= current_date <= policy_end):
        payout_reason = f"POLICY INACTIVE: Current date ({current_date}) is outside the policy period."
    elif current_index >= full_trigger:
        payout_level = 1.0 
        payout_reason = f"FULL PAYOUT: Observed Index >= fixed trigger ({full_trigger:.2f} mm)."
    elif current_index > partial_trigger:
        payout_range = full_trigger - partial_trigger
        deviation = current_index - partial_trigger
        payout_level = min(1.0, max(0.0, deviation / payout_range)) 
        payout_reason = f"PARTIAL PAYOUT: Index in the tiered range ({payout_level*100:.2f}% payout)."

    final_payout_amount = round(payout_level * payout_amount_max, 2)
    
    print("\n====================================================================")
    print(f"PAYOUT AMOUNT: {final_payout_amount:,.2f} (Out of {payout_amount_max:,.2f})")
    print(f"REASON: {payout_reason}")
    print("====================================================================")
    
    return final_payout_amount


def calculate_historical_loss_expectancy(lat, lon, data_dir, lpa_file, lpa_start_yr, lpa_end_yr, 
                                         full_payout_threshold, percentile_rank, index_days, 
                                         payout_amount_max):
    """
    Runs the full parametric logic against 30 years of historical data to 
    calculate Expected Annual Loss (EAL) and Maximum Annual Loss (MAL).
    """
    
    index_description = "Single-Day Rainfall" if index_days == 1 else f"{index_days}-Day Cumulative Rainfall"
    
    print(f"\n====================================================================")
    print(f"📈 STRESS TEST: HISTORICAL LOSS EXPECTANCY (EAL/MAL) 📈")
    print(f"Index Type: {index_description} | Period: {lpa_start_yr}-{lpa_end_yr}")
    print(f"====================================================================")
    
    # 1. Load Data and Climatology
    lpa_daily, std_dev_daily, partial_trigger_val = get_climatological_data(
        lat, lon, data_dir, lpa_file, lpa_start_yr, lpa_end_yr, percentile_rank)
    
    partial_trigger, full_trigger = calculate_payout_triggers(partial_trigger_val, full_payout_threshold)
    if partial_trigger is None: return

    # 2. Get Full Time Series Data for the point
    try:
        data = imd.open_data('rain', lpa_start_yr, lpa_end_yr, 'yearwise', file_dir=data_dir)
        ds = data.get_xarray()
        regional_ds = ds.sel(lat=lat, lon=lon, method='nearest')['rain']
        rainfall_series = regional_ds.to_pandas().replace(-999.0, np.nan)
        cumulative_series = rainfall_series.rolling(window=index_days, min_periods=index_days).sum().dropna()
        
    except Exception as e:
        print(f"Error loading historical time series: {e}")
        return

    # 3. Simulate Payouts for every day in the history
    historical_losses = {}
    
    for date, index_value in cumulative_series.items():
        year = date.year
        payout_level = 0.0
        
        # Payout Logic
        if index_value >= full_trigger:
            payout_level = 1.0 
        elif index_value > partial_trigger:
            payout_range = full_trigger - partial_trigger
            deviation = index_value - partial_trigger
            payout_level = min(1.0, max(0.0, deviation / payout_range))
            
        payout_amount = payout_level * payout_amount_max
        
        # Filter: Only events within the monsoon policy window (June to Sept) for realistic EAL
        if date.month >= 6 and date.month <= 9 and payout_amount > 0:
            historical_losses.setdefault(year, 0)
            historical_losses[year] += payout_amount

    # 4. Calculate Final Metrics
    num_years = lpa_end_yr - lpa_start_yr + 1
    
    if not historical_losses:
        total_loss_expectancy = 0.0
        expected_annual_loss = 0.0
        maximum_annual_loss = 0.0
    else:
        total_loss_expectancy = sum(historical_losses.values())
        expected_annual_loss = total_loss_expectancy / num_years
        maximum_annual_loss = max(historical_losses.values())
        worst_year = max(historical_losses, key=historical_losses.get)


    # 5. Output Financial Metrics
    print("\n--- Historical Loss Analysis ---")
    print(f"Index Type: {index_description}")
    print(f"Total Years Analyzed: {num_years}")
    print(f"Partial Trigger: {partial_trigger:.2f} mm | Full Trigger: {full_trigger:.2f} mm")
    print("--------------------------------")
    print(f"Total Loss Expectancy ({num_years}yr HLE): {total_loss_expectancy:,.2f}")
    print(f"💸 Expected Annual Loss (EAL): {expected_annual_loss:,.2f} (Base Premium)")
    
    if historical_losses:
        print(f"🔥 Maximum Annual Loss (MAL): {maximum_annual_loss:,.2f} (Worst Year: {worst_year})")
        print("--------------------------------")
        print("Historical losses by year (Payouts Triggered):")
        for year, loss in sorted(historical_losses.items()):
            print(f"  {year}: {loss:,.2f}")

    print("Stress Testing complete.")


# =============================================================================
# 3. ADVANCED ANALYSIS & VISUALIZATION
# =============================================================================

def create_payoff_frequency_heatmap(lat, lon, data_dir, lpa_start_yr, lpa_end_yr,
                                    full_payout_threshold, percentile_rank, index_days):
    """
    Calculates the historical frequency of the partial payout trigger being hit
    across the entire IMD grid (India) to generate a risk heatmap.
    """
    index_description = "Single-Day" if index_days == 1 else f"{index_days}-Day Cumulative"
    
    print(f"\n====================================================================")
    print(f"🗺️  GENERATING PARAMETRIC RISK HEATMAP 🗺️  ")
    print(f"Trigger: {percentile_rank:.1f}% Rank | Index: {index_description}")
    print(f"Data Period: {lpa_start_yr} to {lpa_end_yr}")
    print(f"====================================================================")
    

    try:
        data = imd.open_data('rain', lpa_start_yr, lpa_end_yr, 'yearwise', file_dir=data_dir)
        ds = data.get_xarray()
        
        # Calculate Climatology and Percentile Triggers for ALL Grid Points
        # Note: We need to calculate the Nth percentile for *every* grid cell separately
        # This is computationally intensive but necessary for localized triggers.
        
        # 1. Get the 99.9th percentile value for every grid cell's historical daily rainfall
        partial_payout_trigger = ds['rain'].reduce(np.percentile, dim='time', q=percentile_rank)

        # 2. Calculate Index Time Series for ALL Grid Points
        cumulative_index = ds['rain'].rolling(time=index_days, min_periods=index_days).sum()

        # Count Trigger Events (where index > localized trigger)
        trigger_hit_count = (cumulative_index > partial_payout_trigger).sum(dim='time')
        total_days = ds['time'].size - index_days + 1
        payoff_frequency = (trigger_hit_count / total_days) * 100
        
        # Generate Heatmap
        plt.figure(figsize=(10, 8))
        # Using RdYlGn_r to show GREEN (low frequency) to RED (high frequency)
        im = plt.imshow(payoff_frequency.values, cmap='RdYlGn_r', origin='lower',
                        extent=[ds['lon'].min(), ds['lon'].max(), ds['lat'].min(), ds['lat'].max()])
        
        plt.colorbar(im, label='Historical Trigger Frequency (%)')
        
        # Highlight the target location
        plt.plot(lon, lat, 'o', color='green', markersize=8, markeredgecolor='white', label=f'Target ({lat:.1f}N, {lon:.1f}E)')
        plt.legend()
        
        plt.title(f'Parametric Rain Risk Map: {index_description} Trigger Frequency ({percentile_rank:.1f}% Rank)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
        print("Heatmap generation complete.")
        
    except Exception as e:
        print(f"Error during Heatmap Calculation: {e}")


def analyze_aep_and_distribution(lat, lon, data_dir, lpa_start_yr, lpa_end_yr,
                                 lpa_daily_mm, std_dev_daily_mm, 
                                 partial_payout_trigger, full_payout_threshold, index_days):
    """
    Calculates the Annual Exceedance Probability (AEP) for key rainfall thresholds 
    and plots the histogram of the historical index values.
    """
    index_description = "Single-Day" if index_days == 1 else f"{index_days}-Day Cumulative"
    
    print(f"\n====================================================================")
    print(f"📊 AEP & DISTRIBUTION ANALYSIS ({index_description} Index) 📊")
    print(f"Period: {lpa_start_yr}-{lpa_end_yr}")
    print(f"====================================================================")

    try:
        # Calculate LPA index for plotting context
        lpa_index_for_plot = lpa_daily_mm * index_days
        
        # Load full historical data for the point
        data = imd.open_data('rain', lpa_start_yr, lpa_end_yr, 'yearwise', file_dir=data_dir)
        ds = data.get_xarray()
        regional_ds = ds.sel(lat=lat, lon=lon, method='nearest')['rain']
        rainfall_series = regional_ds.to_pandas().replace(-999.0, np.nan)
        
        # Calculate the N-day rolling index
        cumulative_series = rainfall_series.rolling(window=index_days, min_periods=index_days).sum().dropna()
        
        # Filter only for the active policy season (June to September) for realistic frequency
        seasonal_series = cumulative_series[(cumulative_series.index.month >= 6) & (cumulative_series.index.month <= 9)]
        
        # Find the max event in each year
        max_annual_events = seasonal_series.groupby(seasonal_series.index.year).max().dropna()
        
        if max_annual_events.empty:
            print("No valid data found for seasonal AEP calculation.")
            return

        # ----------------------------------------------------------------------
        # AEP CALCULATION
        # ----------------------------------------------------------------------
        num_years = max_annual_events.size
        print(f"Total years with max event recorded: {num_years}")
        
        # Define thresholds to analyze
        thresholds_mm = [
            partial_payout_trigger,
            full_payout_threshold,
            lpa_index_for_plot, # Include LPA in the AEP table for reference
            full_payout_threshold * 1.25 # A severe event threshold
        ]
        
        results = {}
        for t_mm in sorted(list(set(thresholds_mm))):
            exceedance_count = (max_annual_events >= t_mm).sum()
            aep_percent = (exceedance_count / num_years) * 100
            return_period = num_years / exceedance_count if exceedance_count > 0 else np.inf
            
            # Label for printout
            label = "LPA" if abs(t_mm - lpa_index_for_plot) < 1e-6 else (
                    "Partial Trigger" if abs(t_mm - partial_payout_trigger) < 1e-6 else (
                        "Full Payout" if abs(t_mm - full_payout_threshold) < 1e-6 else "Extreme Event"))
            
            results[label] = {
                'Threshold (mm)': round(t_mm, 2),
                'AEP (%)': round(aep_percent, 2),
                'Return Period (Years)': round(return_period, 1)
            }
            
        print("\n--- Annual Exceedance Probability (AEP) ---")
        for label, res in results.items():
            print(f"{label:<15} ({res['Threshold (mm)']:.2f} mm): AEP={res['AEP (%)']:>6.2f}% | RP={res['Return Period (Years)'] if res['Return Period (Years)'] != np.inf else '>29.0':>6} yr")

        '''# ----------------------------------------------------------------------
        # PLOT HISTOGRAM (IMPROVED CLARITY)
        # ----------------------------------------------------------------------
        
        plt.figure(figsize=(12, 7))
        # Use cleaner, distinct colors
        plt.hist(max_annual_events, bins=20, edgecolor='#1f2937', color='#93c5fd', alpha=0.9)
        
        # Add LPA baseline for context (Blue dotted line)
        if lpa_index_for_plot > 0:
            plt.axvline(lpa_index_for_plot, color='#3b82f6', linestyle=':', linewidth=2, label=f'LPA Index Baseline ({lpa_index_for_plot:.1f} mm)')
        
        # Add key thresholds to the plot (Orange and Red lines for Payouts)
        plt.axvline(partial_payout_trigger, color='#f97316', linestyle='--', linewidth=2, label=f'Partial Payout Start ({partial_payout_trigger:.1f} mm)')
        plt.axvline(full_payout_threshold, color='#ef4444', linestyle='-', linewidth=3, label=f'Full Payout Limit ({full_payout_threshold:.1f} mm)')

        plt.title(f'Historical Annual Max {index_description} Rainfall vs. Parametric Triggers\nTarget Location: {lat}°N, {lon}°E ({lpa_start_yr}-{lpa_end_yr} Data)', fontsize=14)
        plt.xlabel(f'Annual Maximum {index_description} Rainfall (mm)', fontsize=12)
        plt.ylabel('Frequency (Number of Years Observed)', fontsize=12)
        plt.grid(axis='y', alpha=0.4, linestyle='--')
        plt.legend(fontsize=10)
        plt.tight_layout() # Ensures labels don't get cut off
        plt.show()'''

    except Exception as e:
        print(f"Error during AEP and Distribution Analysis: {e}")
