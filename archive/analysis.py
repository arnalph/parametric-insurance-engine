import os
import imdlib as imd # Import the IMD library
import pandas as pd
import numpy as np
import xarray as xr # imdlib works internally with xarray

# 1. Define the directory structure
# file_dir should point to the top-level directory (IMD_Rain_Data) 
# which contains the 'rain' subdirectory.
TOP_DIR = 'IMD_Rain_Data'
VARIABLE = 'rain' # The variable we are analyzing

# List to store the results for all years
yearly_results = []

print(f"--- Starting Advanced Rainfall Analysis using IMDLIB in: {TOP_DIR}/{VARIABLE} ---")

# 2. Get the list of all yearly .grd files
rain_dir = os.path.join(TOP_DIR, VARIABLE)
if not os.path.exists(rain_dir):
    print(f"Error: Directory '{rain_dir}' not found. Please check the path.")
else:
    # Extract unique years from the filenames (e.g., 2020.grd -> 2020)
    years_to_process = sorted([
        int(f.split('.')[0]) 
        for f in os.listdir(rain_dir) 
        if f.endswith('.grd')
    ])

    if not years_to_process:
        print("No .grd files found in the 'rain' subdirectory.")
    
    # 3. Process each year individually using imdlib
    for year in years_to_process:
        try:
            print(f"-> Processing year {year}...")
            
            # Use imdlib.open_data for a single year in 'yearwise' format
            # This handles the complex binary reading and coordinates setup.
            imd_data_obj = imd.open_data(VARIABLE, year, year, 'yearwise', TOP_DIR)
            
            # Convert the IMD object to the xarray DataArray for analysis
            ds = imd_data_obj.get_xarray()
            
            # The DataArray 'ds' contains daily rainfall data for the entire year
            # ds is typically (lat: 129, lon: 135, time: 365/366)
            
            # --- ANALYSIS 1: Total Annual Rainfall Sum (for each grid cell) ---
            # Sum the daily values across the time dimension to get the total annual rainfall grid
            annual_total_grid = ds.sum(dim='time', skipna=False)
            
            # --- ANALYSIS 2: Domain-wide Statistics ---
            # Total accumulated rainfall for the entire domain (sum of all grid cells)
            total_sum = annual_total_grid.sum(skipna=True).item()
            
            # Average annual rainfall intensity across all grid cells
            mean_annual_rainfall = annual_total_grid.mean(skipna=True).item()
            
            # --- ANALYSIS 3: Extreme Value and Location (on the total annual grid) ---
            max_annual_rainfall = annual_total_grid.max(skipna=True).item()
            
            # Find the coordinates (Lat, Lon) of the max value
            max_loc = annual_total_grid.where(annual_total_grid == max_annual_rainfall, drop=True)
            
            max_lat = max_loc.lat.item() if max_loc.lat.size > 0 else np.nan
            max_lon = max_loc.lon.item() if max_loc.lon.size > 0 else np.nan

            # Store results
            yearly_results.append({
                'Year': year,
                'Total Grid Sum (mm-units)': total_sum,
                'Avg Annual Rainfall (mm/yr)': mean_annual_rainfall,
                'Max Annual Rainfall (mm/yr)': max_annual_rainfall,
                'Max Location (Lat, Lon)': f'({max_lat:.2f}, {max_lon:.2f})'
            })
            
        except Exception as e:
            print(f"❌ Error using imdlib to process year {year}: {e}")
            
# --- Final Results Presentation ---
if yearly_results:
    # Convert the list of dicts to a pandas DataFrame and sort by year
    results_df = pd.DataFrame(yearly_results).sort_values(by='Year').set_index('Year')
    
    print("\n" + "="*85)
    print("      SUMMARY: YEARLY RAINFALL ANALYSIS (IMDLIB & Xarray)")
    print("="*85)
    
    # Format the numeric columns for better readability
    format_dict = {
        'Total Grid Sum (mm-units)': '{:,.0f}'.format,
        'Avg Annual Rainfall (mm/yr)': '{:.2f}'.format,
        'Max Annual Rainfall (mm/yr)': '{:,.2f}'.format
    }
    
    print(results_df.to_string(formatters=format_dict))
    print("="*85)
    
    # Highlight the extreme years based on the Total Grid Sum
    total_col = 'Total Grid Sum (mm-units)'
    
    # Need to convert formatted back to numeric to find max/min
    numeric_totals = results_df[total_col].apply(lambda x: float(x.replace(',', '')))
    
    wettest_year = numeric_totals.idxmax()
    driest_year = numeric_totals.idxmin()

    print(f"\nKey Findings (based on Total Grid Sum):")
    print(f"  - Wettest Year: {wettest_year}")
    print(f"  - Driest Year: {driest_year}")

else:
    print("\nAnalysis complete, but no valid yearly data was processed.")

# This YouTube video discusses the use of Python for downloading and processing IMD data, which is directly relevant to the libraries and data handling discussed here.
#[Download IMD gridded data into CSV without any software | IMDLIB | Python](https://www.youtube.com/watch?v=fG1jF5f05j8)