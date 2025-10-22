# =============================================================================
# DATA INGESTION UTILITIES (data_ingestion.py)
#
# This module handles the setup of the local data environment, specifically 
# checking for and downloading the necessary historical IMD data files.
# =============================================================================

import os
import imdlib as imd
import time

def setup_data_directories(data_dir, lpa_dir):
    """Ensures the necessary local directories for data and cache exist."""
    print(f"--- Ensuring Data Directories Exist ---")
    # Setup main data directory and the 'rain' subdirectory
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'rain'), exist_ok=True)
    # Setup cache directory for LPA
    os.makedirs(lpa_dir, exist_ok=True)
    print(f"Data directory: '{data_dir}' is ready.")

def ensure_historical_data_exists(data_dir, start_yr, end_yr):
    """
    Iterates through each year in the range and downloads the data ONLY for 
    the missing years using the imdlib utility.
    """
    print(f"\n--- Checking and Downloading Historical IMD Data ({start_yr}-{end_yr}) ---")
    
    all_data_present = True
    rain_data_dir = os.path.join(data_dir, 'rain')
    
    for year in range(start_yr, end_yr + 1):
        file_path = os.path.join(rain_data_dir, f'{year}.grd')
        
        if not os.path.exists(file_path):
            print(f"File for year {year} is MISSING. Initiating download...")
            all_data_present = False
            try:
                # Download only the missing year
                imd.get_data('rain', year, year, fn_format='yearwise', file_dir=data_dir)
                print(f"✅ Downloaded data for year {year} successfully.")
                time.sleep(1) # Introduce a small delay to be polite to the server
            except Exception as e:
                print(f"❌ ERROR: Failed to download data for year {year}. Error: {e}")
                
        else:
            # print(f"File for year {year} is PRESENT.") # Commented out for cleaner output
            pass
            
    if all_data_present:
        print(f"All historical data files for range {start_yr}-{end_yr} are locally present.")
    else:
        print(f"Data check and necessary downloads complete for range {start_yr}-{end_yr}.")
        
    return True
