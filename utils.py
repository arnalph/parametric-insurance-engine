import pandas as pd
import numpy as np
import os
import imdlib as imd
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Union, Tuple, Callable, Any
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ==================== CORE UTILITIES ====================

def setup_data_directories(data_dir, lpa_dir):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(lpa_dir, exist_ok=True)

def ensure_historical_data_exists(data_dir, start_yr, end_yr):
    print("✅ Historical data files assumed available or previously downloaded.")

def load_localized_data(lat, lon, data_dir, start_yr, end_yr, cumulative_days=1):
    try:
        ds = imd.open_data('rain', start_yr, end_yr, file_dir=data_dir, fn_format='yearwise')
        regional_ds = ds.get_xarray().sel(lat=lat, lon=lon, method='nearest')['rain']
        df = regional_ds.to_dataframe().dropna()
    except Exception as e:
        print(f"Error loading IMD data: {e}")
        return pd.Series([], dtype=float)

    if cumulative_days > 1:
        df['rain'] = df['rain'].rolling(window=cumulative_days, min_periods=cumulative_days).sum()
        df = df.dropna()

    return df['rain'].rename('Localized_Rainfall')

def get_climatological_data(lat, lon, data_dir, start_yr, end_yr, cumulative_days=1):
    rain_series = load_localized_data(lat, lon, data_dir, start_yr, end_yr, cumulative_days)
    if rain_series.empty:
        return 0.0, 0.0, rain_series
    return rain_series.mean(), rain_series.std(), rain_series

# ==================== COLOR NORMALIZATION ====================

def normalize_colors(values, cmap_name="viridis"):
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    cmap = cm.get_cmap(cmap_name)
    return [cmap(norm(val)) for val in values]

# ==================== GENERIC PLOTTER ====================

def generate_single_payout_plot(x, y, title, max_payout, trigger_info=None, lpa=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=title, linewidth=2)
    if lpa:
        plt.axvline(lpa, color='red', linestyle=':', alpha=0.6, label=f'LPA {lpa:.1f}mm')
    if trigger_info:
        for label, val in trigger_info.items():
            plt.axvline(val, linestyle='--', alpha=0.7, label=f'{label} {val:.1f}mm')
    plt.axhline(max_payout, color='gray', linestyle='-.', alpha=0.5)
    plt.xlabel("Rainfall Index (mm)")
    plt.ylabel(f"Payout (₹, Max {max_payout:,.0f})")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="dotted", alpha=0.6)
    plt.tight_layout()
    plt.show()

# ==================== PAYOUT FUNCTIONS ====================

def calculate_payout_statistical(index_value, lpa, std_dev, trigger_std_dev, max_payout, plot_curve=False, **kwargs):
    if std_dev == 0:
        return 0.0
    lower_trigger = lpa + trigger_std_dev * std_dev
    upper_trigger = lpa + (trigger_std_dev + 1.0) * std_dev
    if index_value < lower_trigger:
        payout = 0.0
    elif index_value >= upper_trigger:
        payout = max_payout
    else:
        fraction = (index_value - lower_trigger) / (upper_trigger - lower_trigger)
        payout = fraction * max_payout
    if plot_curve:
        x = np.linspace(max(0, lpa - 0.5 * std_dev), lpa + 8 * std_dev, 400)
        y = [calculate_payout_statistical(val, lpa, std_dev, trigger_std_dev, max_payout) for val in x]
        generate_single_payout_plot(x, y, f"Statistical Payout", max_payout,
                                    {f"{trigger_std_dev}σ": lower_trigger, f"{trigger_std_dev+1}σ": upper_trigger}, lpa)
    return payout

def calculate_payout_graduated(index_value, base_trigger_mm, max_payout, steps, plot_curve=False, **kwargs):
    s1, s2, s3 = steps['step_1'], steps['step_2'], steps['step_3']
    payout = 0.0
    if index_value >= base_trigger_mm:
        if index_value < s1:
            payout = 0.25 * max_payout
        elif index_value < s2:
            payout = 0.5 * max_payout
        elif index_value < s3:
            payout = 0.75 * max_payout
        else:
            payout = max_payout
    if plot_curve:
        x = np.linspace(0, s3 * 1.5, 400)
        y = [calculate_payout_graduated(val, base_trigger_mm, max_payout, steps) for val in x]
        generate_single_payout_plot(x, y, "Graduated", max_payout,
                                    {"Trigger": base_trigger_mm, "Step1": s1, "Step2": s2, "Step3": s3})
    return min(max(payout, 0), max_payout)

def calculate_payout_binary(index_value, base_trigger_mm, max_payout, plot_curve=False, **kwargs):
    payout = max_payout if index_value >= base_trigger_mm else 0.0
    if plot_curve:
        x = np.linspace(base_trigger_mm * 0.8, base_trigger_mm * 1.2, 200)
        y = [calculate_payout_binary(val, base_trigger_mm, max_payout) for val in x]
        generate_single_payout_plot(x, y, "Binary", max_payout, {"Trigger": base_trigger_mm})
    return payout

# ==================== LOSS EXPECTANCY ====================

def calculate_historical_loss_expectancy(lat, lon, data_dir, start_yr, end_yr, payout_function, contract_start_mmdd="06-01", contract_end_mmdd="09-30", plot_curve=False, **kwargs):
    rain_series = load_localized_data(lat, lon, data_dir, start_yr, end_yr)
    if rain_series.empty:
        return 0.0, 0.0
    df = rain_series.to_frame()
    df['Year'] = df.index.year
    df['MonthDay'] = df.index.strftime("%m-%d")
    df = df[(df['MonthDay'] >= contract_start_mmdd) & (df['MonthDay'] <= contract_end_mmdd)]
    annual_index_values = df.groupby('Year')['Localized_Rainfall'].max()
    annual_payouts = annual_index_values.apply(lambda x: payout_function(index_value=x, **kwargs))
    eal, mal = annual_payouts.mean(), annual_payouts.max()
    if plot_curve:
        plt.figure(figsize=(10, 6))
        plt.bar(annual_payouts.index, annual_payouts.values, color="skyblue")
        plt.axhline(eal, color="red", linestyle="--", label=f"EAL ₹{eal:,.0f}")
        plt.axhline(mal, color="green", linestyle=":", label=f"MAL ₹{mal:,.0f}")
        plt.title(f"Historical Annual Payouts ({contract_start_mmdd} to {contract_end_mmdd})")
        plt.xlabel("Year")
        plt.ylabel("Payout (₹)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return eal, mal

def run_pointwise_loss_analysis(locations, data_dir, start_yr, end_yr, payout_function, contract_start_mmdd="06-01", contract_end_mmdd="09-30", cumulative_days=1, **kwargs):
    results = []
    for name, (lat, lon) in locations.items():
        eal, mal = calculate_historical_loss_expectancy(lat, lon, data_dir, start_yr, end_yr, payout_function, contract_start_mmdd, contract_end_mmdd, False, **kwargs)
        results.append({"Location": name, "Latitude": lat, "Longitude": lon, "EAL": eal, "MAL": mal})
    return pd.DataFrame(results)

# ==================== PLOTTING ====================

def plot_loss_map(df, value_col="EAL", title="Loss Map"):
    import plotly.express as px
    fig = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color=value_col,
        size=value_col,
        text="Location",
        color_continuous_scale="Viridis",
        size_max=15,
        zoom=7,
        height=600,
        title=title
    )
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    fig.show()
