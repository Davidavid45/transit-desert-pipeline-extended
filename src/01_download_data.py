"""
01_download_data.py
Download GTFS, Census ACS, and Census tract shapefiles for transit desert analysis.

Usage:
    python src/01_download_data.py

Outputs:
    - data/raw/gtfs/          : GTFS feed files
    - data/raw/census/        : Census tract shapefile
    - data/raw/acs/           : ACS demographic data
"""

import os
import sys
import yaml
import zipfile
import requests
from pathlib import Path
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import geopandas as gpd
from census import Census
from us import states

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_config():
    """Load configuration from YAML file."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_gtfs(config, output_dir):
    """
    Download GTFS feed from URL.
    
    Handles various GTFS packaging formats:
    - Single zip with txt files
    - Zip containing multiple mode-specific zips
    - Multiple separate URLs for different modes
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with GTFS URL(s)
    output_dir : Path
        Directory to save GTFS files
    """
    gtfs_dir = output_dir / "gtfs"
    gtfs_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if URL is a single string or list of URLs
    gtfs_config = config['gtfs']
    
    if isinstance(gtfs_config.get('url'), list):
        # Multiple URLs (multi-modal)
        urls = gtfs_config['url']
        print(f"Downloading {len(urls)} GTFS feeds (multi-modal)...")
    else:
        # Single URL
        urls = [gtfs_config['url']]
        print(f"Downloading GTFS from: {urls[0]}")
    
    downloaded_files = []
    
    for i, url in enumerate(urls):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            # Determine filename
            if len(urls) == 1:
                filename = "gtfs.zip"
            else:
                # Try to get name from URL or use index
                url_parts = url.rstrip('/').split('/')
                filename = url_parts[-1] if url_parts[-1].endswith('.zip') else f"gtfs_{i+1}.zip"
            
            zip_path = gtfs_dir / filename
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            downloaded_files.append(zip_path)
            print(f"  ✓ Downloaded: {filename}")
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error downloading {url}: {e}")
    
    if not downloaded_files:
        print("  ✗ No GTFS files downloaded")
        print("  → Please download manually and place in data/raw/gtfs/")
        return None
    
    # Analyze what was downloaded
    print(f"\n  Analyzing downloaded GTFS data...")
    
    # Only extract if single feed — multiple feeds stay as zips for gtfs_utils to merge
    if len(downloaded_files) == 1:
        zip_path = downloaded_files[0]
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                names = zf.namelist()
                txt_files = [n for n in names if n.endswith('.txt') and '/' not in n]
                nested_zips = [n for n in names if n.endswith('.zip')]
                
                if txt_files:
                    print(f"    {zip_path.name}: Standard GTFS ({len(txt_files)} txt files)")
                    zf.extractall(gtfs_dir)
                elif nested_zips:
                    print(f"    {zip_path.name}: Multi-modal package ({len(nested_zips)} zip files)")
                    zf.extractall(gtfs_dir)
                else:
                    nested_txt = [n for n in names if n.endswith('.txt')]
                    if nested_txt:
                        print(f"    {zip_path.name}: Nested folder structure")
                        zf.extractall(gtfs_dir)
                    else:
                        print(f"    {zip_path.name}: Unknown format")
        except zipfile.BadZipFile:
            print(f"    ⚠ {zip_path.name}: Not a valid zip file")
    else:
        print(f"    Multiple feeds detected ({len(downloaded_files)} zips)")
        print(f"    Keeping as separate zips for merging in step 02")
        for zip_path in downloaded_files:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    names = zf.namelist()
                    txt_files = [n for n in names if n.endswith('.txt') and '/' not in n]
                    print(f"    {zip_path.name}: {len(txt_files)} txt files (not extracted)")
            except zipfile.BadZipFile:
                print(f"    ⚠ {zip_path.name}: Not a valid zip file")
    
    # List what's in the directory now
    all_files = list(gtfs_dir.glob("*"))
    txt_files = [f for f in all_files if f.suffix == '.txt']
    zip_files = [f for f in all_files if f.suffix == '.zip']
    
    print(f"\n  GTFS directory contents:")
    print(f"    txt files: {len(txt_files)}")
    print(f"    zip files: {len(zip_files)}")
    
    if txt_files:
        for f in txt_files[:5]:
            print(f"      - {f.name}")
        if len(txt_files) > 5:
            print(f"      ... and {len(txt_files) - 5} more")
    
    if zip_files:
        for f in zip_files[:5]:
            print(f"      - {f.name}")
        if len(zip_files) > 5:
            print(f"      ... and {len(zip_files) - 5} more")
    
    return gtfs_dir

def download_census_tracts(config, output_dir):
    """
    Download Census tract boundaries from TIGER/Line.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with FIPS codes
    output_dir : Path
        Directory to save shapefile
    """
    census_dir = output_dir / "census"
    census_dir.mkdir(parents=True, exist_ok=True)
    
    state_fips = config['study_area']['state_fips']
    county_fips = config['study_area']['county_fips']
    
    # TIGER/Line URL for tracts
    year = config['census']['acs_year']
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
    
    print(f"Downloading Census tracts from: {url}")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Extract shapefile
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(census_dir / "tracts_state")
        
        # Read and filter to county
        shp_files = list((census_dir / "tracts_state").glob("*.shp"))
        if shp_files:
            gdf = gpd.read_file(shp_files[0])
            
            # Filter to study area county
            gdf_county = gdf[gdf['COUNTYFP'] == county_fips].copy()
            
            # Save filtered shapefile
            output_path = census_dir / "study_area_tracts.shp"
            gdf_county.to_file(output_path)
            
            print(f"  ✓ Downloaded {len(gdf_county)} tracts for county {county_fips}")
            print(f"  ✓ Saved to {output_path}")
            
            return output_path
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error downloading tracts: {e}")
        return None

def download_acs_data(config, output_dir):
    """
    Download ACS 5-year estimates for TDI calculation.
    
    Variables needed:
    - B08201: Household Size by Vehicles Available
    - B17001: Poverty Status
    - B03002: Hispanic or Latino Origin by Race
    - B01001: Sex by Age (for 65+ elderly and youth ages 10-17)
    - B08122: Means of Transportation to Work by Poverty Status (forced car ownership proxy)
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_dir : Path
        Directory to save CSV
    """
    acs_dir = output_dir / "acs"
    acs_dir.mkdir(parents=True, exist_ok=True)
    
    state_fips = config['study_area']['state_fips']
    county_fips = config['study_area']['county_fips']
    year = config['census']['acs_year']
    
    print(f"Downloading ACS {year} 5-year estimates...")
    print("  Note: Requires Census API key. Set CENSUS_API_KEY environment variable.")
    
    api_key = os.environ.get('CENSUS_API_KEY')
    if not api_key:
        print("  ⚠ No API key found. Creating placeholder file with variable list.")
        print("  → Get a free key at: https://api.census.gov/data/key_signup.html")
        
        # Create placeholder with variable definitions
        variables_info = {
            'Variable': [
                'B08201_001E', 'B08201_002E',  # Vehicles available
                'B17001_001E', 'B17001_002E',  # Poverty status
                'B03002_001E', 'B03002_003E',  # Race/ethnicity
                'B01001_001E',  # Total population
                # Age 65+ variables (males + females)
                'B01001_020E', 'B01001_021E', 'B01001_022E', 'B01001_023E', 'B01001_024E', 'B01001_025E',
                'B01001_044E', 'B01001_045E', 'B01001_046E', 'B01001_047E', 'B01001_048E', 'B01001_049E',
                # Youth ages 10-17 (males + females)
                'B01001_005E', 'B01001_006E',
                'B01001_029E', 'B01001_030E',
                # Forced car ownership proxy (B08122)
                'B08122_001E', 'B08122_002E', 'B08122_003E',
                'B08122_005E', 'B08122_006E',
                'B08122_008E', 'B08122_009E',
                'B08122_011E', 'B08122_012E'
            ],
            'Description': [
                'Total households', 'Households with 0 vehicles',
                'Total population for poverty', 'Below poverty level',
                'Total population', 'White alone, not Hispanic',
                'Total population for age',
                'Male 65-66', 'Male 67-69', 'Male 70-74', 'Male 75-79', 'Male 80-84', 'Male 85+',
                'Female 65-66', 'Female 67-69', 'Female 70-74', 'Female 75-79', 'Female 80-84', 'Female 85+',
                'Male 10-14', 'Male 15-17',
                'Female 10-14', 'Female 15-17',
                'Total workers (poverty determined)', 'Workers below poverty', 'Workers 100-149% poverty',
                'Drove alone, below poverty', 'Drove alone, 100-149% poverty',
                'Carpooled, below poverty', 'Carpooled, 100-149% poverty',
                'Public transit, below poverty', 'Public transit, 100-149% poverty'
            ],
            'Used_For': [
                'Zero-vehicle rate', 'Zero-vehicle rate',
                'Poverty rate', 'Poverty rate',
                'Minority %', 'Minority %',
                'Elderly %',
                'Elderly %', 'Elderly %', 'Elderly %', 'Elderly %', 'Elderly %', 'Elderly %',
                'Elderly %', 'Elderly %', 'Elderly %', 'Elderly %', 'Elderly %', 'Elderly %',
                'Youth % (10-17)', 'Youth % (10-17)',
                'Youth % (10-17)', 'Youth % (10-17)',
                'Forced car ownership', 'Forced car ownership', 'Forced car ownership',
                'Forced car ownership', 'Forced car ownership',
                'Forced car ownership', 'Forced car ownership',
                'Forced car ownership', 'Forced car ownership'
            ]
        }
        pd.DataFrame(variables_info).to_csv(acs_dir / "acs_variables_needed.csv", index=False)
        print(f"  ✓ Variable list saved to {acs_dir / 'acs_variables_needed.csv'}")
        return None
    
    try:
        c = Census(api_key)
        
        # Define variables to download
        variables = [
            'NAME',
            # Vehicles available (B08201)
            'B08201_001E',  # Total households
            'B08201_002E',  # No vehicle available
            # Poverty (B17001)
            'B17001_001E',  # Total population for poverty determination
            'B17001_002E',  # Below poverty level
            # Race/Ethnicity (B03002)
            'B03002_001E',  # Total population
            'B03002_003E',  # White alone, not Hispanic
            # Age (B01001) - need 65+ for elderly
            'B01001_001E',  # Total population
            # Males 65+
            'B01001_020E', 'B01001_021E', 'B01001_022E', 
            'B01001_023E', 'B01001_024E', 'B01001_025E',
            # Females 65+
            'B01001_044E', 'B01001_045E', 'B01001_046E',
            'B01001_047E', 'B01001_048E', 'B01001_049E',
            # Males 10-17 (youth — independent transit users)
            'B01001_005E', 'B01001_006E',
            # Females 10-17
            'B01001_029E', 'B01001_030E',
            # ── Forced car ownership proxy (B08122) ──
            # Means of Transportation to Work by Poverty Status
            # Universe: Workers 16+ for whom poverty status is determined
            'B08122_001E',  # Total workers
            'B08122_002E',  # Workers below 100% poverty
            'B08122_003E',  # Workers 100-149% poverty
            'B08122_005E',  # Drove alone, below poverty
            'B08122_006E',  # Drove alone, 100-149% poverty
            'B08122_008E',  # Carpooled, below poverty
            'B08122_009E',  # Carpooled, 100-149% poverty
            'B08122_011E',  # Public transit, below poverty
            'B08122_012E',  # Public transit, 100-149% poverty
        ]
        
        # Download tract-level data
        data = c.acs5.state_county_tract(
            fields=variables,
            state_fips=state_fips,
            county_fips=county_fips,
            tract='*',
            year=year
        )
        
        df = pd.DataFrame(data)
        
        # Create GEOID
        df['GEOID'] = df['state'] + df['county'] + df['tract']
        
        # Calculate derived variables
        df['pct_zero_vehicle'] = df['B08201_002E'] / df['B08201_001E']
        df['pct_poverty'] = df['B17001_002E'] / df['B17001_001E']
        df['pct_minority'] = 1 - (df['B03002_003E'] / df['B03002_001E'])
        
        # Elderly (sum all 65+ age groups)
        elderly_male = df[['B01001_020E', 'B01001_021E', 'B01001_022E', 
                           'B01001_023E', 'B01001_024E', 'B01001_025E']].sum(axis=1)
        elderly_female = df[['B01001_044E', 'B01001_045E', 'B01001_046E',
                             'B01001_047E', 'B01001_048E', 'B01001_049E']].sum(axis=1)
        df['pct_elderly'] = (elderly_male + elderly_female) / df['B01001_001E']
        
        # Youth (ages 10-17 — independent transit users)
        # Male 10-14 (005E) + Male 15-17 (006E)
        youth_male = df[['B01001_005E', 'B01001_006E']].sum(axis=1)
        # Female 10-14 (029E) + Female 15-17 (030E)
        youth_female = df[['B01001_029E', 'B01001_030E']].sum(axis=1)
        df['pct_youth'] = (youth_male + youth_female) / df['B01001_001E']
        
        # ── Forced car ownership proxy (Allen & Farber 2021) ──
        # Share of below-poverty workers commuting by private vehicle
        # High values indicate forced car dependence: poor workers driving
        # because transit is unavailable, not because they prefer it.
        # Uses B08122: Means of Transportation to Work by Poverty Status
        #
        # We include both <100% and 100-149% poverty to capture near-poor
        # households who face similar forced-car dynamics (Mattioli 2017).
        low_income_workers = (
            df['B08122_002E'].fillna(0) + df['B08122_003E'].fillna(0)
        )
        low_income_car_commuters = (
            df['B08122_005E'].fillna(0) + df['B08122_006E'].fillna(0) +  # drove alone
            df['B08122_008E'].fillna(0) + df['B08122_009E'].fillna(0)    # carpooled
        )
        low_income_transit_commuters = (
            df['B08122_011E'].fillna(0) + df['B08122_012E'].fillna(0)
        )
        
        # Primary metric: share of low-income workers driving to work
        df['pct_low_income_car_commute'] = low_income_car_commuters / low_income_workers
        
        # Secondary metric: count of low-income car commuters (absolute)
        df['n_low_income_car_commuters'] = low_income_car_commuters
        df['n_low_income_workers'] = low_income_workers
        df['n_low_income_transit_commuters'] = low_income_transit_commuters
        
        # Handle division by zero and infinity
        for col in ['pct_zero_vehicle', 'pct_poverty', 'pct_minority', 
                     'pct_elderly', 'pct_youth', 'pct_low_income_car_commute']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            # Clip percentages to 0-1 range (guard against data anomalies)
            df[col] = df[col].clip(0, 1)
        
        # Save
        output_path = acs_dir / "acs_tvi_variables.csv"
        df.to_csv(output_path, index=False)
        
        print(f"  ✓ Downloaded ACS data for {len(df)} tracts")
        print(f"  ✓ Saved to {output_path}")
        
        # Summary stats
        print("\n  Summary of TVI variables:")
        for col in ['pct_zero_vehicle', 'pct_poverty', 'pct_minority', 
                     'pct_elderly', 'pct_youth', 'pct_low_income_car_commute']:
            print(f"    {col}: mean={df[col].mean():.2%}, max={df[col].max():.2%}")
        
        # Forced car ownership diagnostics
        tracts_with_low_inc_workers = (df['n_low_income_workers'] > 0).sum()
        print(f"\n  Forced car ownership proxy (B08122):")
        print(f"    Tracts with low-income workers: {tracts_with_low_inc_workers}/{len(df)}")
        if tracts_with_low_inc_workers > 0:
            valid = df[df['n_low_income_workers'] > 0]
            print(f"    Mean car commute rate (low-income): {valid['pct_low_income_car_commute'].mean():.1%}")
            print(f"    Mean transit commute rate (low-income): "
                  f"{(valid['n_low_income_transit_commuters'] / valid['n_low_income_workers']).mean():.1%}")
        
        return output_path
        
    except Exception as e:
        print(f"  ✗ Error downloading ACS data: {e}")
        return None

def download_sld_data(config, output_dir):
    """
    Download EPA Smart Location Database for land use diversity metrics.
    
    SLD v3.0 provides block-group-level built environment variables including:
    - D2a_EpHHm: Employment + household entropy (activity diversity)
    - D2b_E8MixA: 8-tier employment entropy
    - NatWalkInd: National Walkability Index
    - D3B: Street intersection density
    
    Strategy:
    1. Try EPA's official ArcGIS MapServer query (geodata.epa.gov)
    2. If that fails, try the data.gov national CSV (large ~200MB file)
    3. If both fail, check for a manually downloaded CSV in data/raw/sld/
    
    We aggregate block groups to tracts using population-weighted mean.
    Source: EPA Smart Location Database v3.0
    https://www.epa.gov/smartgrowth/smart-location-mapping
    """
    sld_dir = output_dir / "sld"
    sld_dir.mkdir(parents=True, exist_ok=True)
    
    full_fips = config['study_area']['full_fips']
    output_path = sld_dir / "sld_tract_metrics.csv"
    
    # Check if already processed
    if output_path.exists():
        df = pd.read_csv(output_path, dtype={'GEOID': str})
        print(f"  ✓ SLD data already exists: {len(df)} tracts")
        return output_path
    
    print("Downloading EPA Smart Location Database...")
    print("  Source: EPA SLD v3.0 (block group level)")
    
    bg_df = None
    
    # ── Strategy 1: EPA official MapServer query ──
    # This is the correct endpoint from EPA's web services page
    sld_url = (
        "https://geodata.epa.gov/arcgis/rest/services/OA/"
        "SmartLocationDatabase/MapServer/0/query"
    )
    
    sld_vars = ('GEOID20,D2a_EpHHm,D2b_E8MixA,D2B_Ranked,D2A_Ranked,'
                'NatWalkInd,D3B,Ac_Total,Ac_Land,TotPop,HH,TotEmp')
    
    params = {
        'where': f"GEOID20 LIKE '{full_fips}%'",
        'outFields': sld_vars,
        'returnGeometry': 'false',
        'f': 'json',
        'resultRecordCount': 5000,
    }
    
    try:
        print(f"  Trying EPA MapServer (geodata.epa.gov)...")
        response = requests.get(sld_url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        if 'features' in data and len(data['features']) > 0:
            records = [f['attributes'] for f in data['features']]
            bg_df = pd.DataFrame(records)
            print(f"  ✓ Downloaded {len(bg_df)} block groups from EPA MapServer")
        else:
            err_msg = data.get('error', {}).get('message', 'No features returned')
            print(f"  ⚠ MapServer returned no data: {err_msg}")
    except Exception as e:
        print(f"  ⚠ MapServer query failed: {e}")
    
    # ── Strategy 2: Check for manually placed CSV/GDB in sld_dir ──
    if bg_df is None:
        # Look for any CSV file the user may have downloaded manually
        manual_csvs = list(sld_dir.glob("*.csv"))
        # Also check for the national CSV specifically
        national_csv = sld_dir / "EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv"
        
        if national_csv.exists():
            manual_csvs = [national_csv] + [f for f in manual_csvs if f != national_csv]
        
        for csv_file in manual_csvs:
            if csv_file.name == output_path.name:
                continue  # Skip our own output file
            try:
                print(f"  Trying local file: {csv_file.name}...")
                # Read with GEOID20 as string to preserve leading zeros
                chunk = pd.read_csv(csv_file, dtype={'GEOID20': str, 'GEOID10': str},
                                     low_memory=False, nrows=5)
                
                # Detect which GEOID column exists
                geoid_col = None
                for col in ['GEOID20', 'GEOID10', 'GEOID']:
                    if col in chunk.columns:
                        geoid_col = col
                        break
                
                if geoid_col is None:
                    print(f"    No GEOID column found, skipping")
                    continue
                
                # Read only rows matching our FIPS (filter while reading for speed)
                all_rows = pd.read_csv(csv_file, dtype={geoid_col: str}, low_memory=False)
                bg_df = all_rows[all_rows[geoid_col].str.startswith(full_fips, na=False)].copy()
                
                if len(bg_df) > 0:
                    # Standardize column name
                    if geoid_col != 'GEOID20':
                        bg_df = bg_df.rename(columns={geoid_col: 'GEOID20'})
                    print(f"  ✓ Loaded {len(bg_df)} block groups from {csv_file.name}")
                    break
                else:
                    print(f"    No block groups matching FIPS {full_fips}")
                    bg_df = None
            except Exception as e:
                print(f"    Error reading {csv_file.name}: {e}")
                bg_df = None
    
    # ── No data obtained ──
    if bg_df is None or len(bg_df) == 0:
        print(f"\n  ⚠ Could not obtain SLD data for FIPS {full_fips}")
        print(f"  To download manually:")
        print(f"    1. Go to https://www.epa.gov/smartgrowth/smart-location-mapping")
        print(f"    2. Under 'Smart Location Database', click 'Download data for all areas'")
        print(f"    3. Extract the ZIP — it contains a geodatabase (.gdb)")
        print(f"    4. OR go to https://catalog.data.gov/dataset/smart-location-database7")
        print(f"       and download the CSV: EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv")
        print(f"    5. Place the CSV in: {sld_dir}/")
        print(f"    6. Re-run this step")
        print(f"\n  Pipeline will continue using OSM land use mix as fallback.")
        return None
    
    # ── Aggregate block groups → tracts ──
    bg_df['GEOID20'] = bg_df['GEOID20'].astype(str).str.zfill(12)
    bg_df['GEOID'] = bg_df['GEOID20'].str[:11]  # Tract = first 11 chars
    
    sld_metrics = ['D2a_EpHHm', 'D2b_E8MixA', 'D2B_Ranked', 'D2A_Ranked',
                   'NatWalkInd', 'D3B']
    available_vars = [v for v in sld_metrics if v in bg_df.columns]
    
    if 'TotPop' in bg_df.columns and bg_df['TotPop'].sum() > 0:
        print(f"  Aggregating {len(bg_df)} block groups → tracts (population-weighted)")
        
        tract_records = []
        for geoid, group in bg_df.groupby('GEOID'):
            pop = group['TotPop'].fillna(0)
            total_pop = pop.sum()
            
            record = {'GEOID': geoid}
            for var in available_vars:
                vals = group[var].fillna(0)
                if total_pop > 0:
                    record[var] = (vals * pop).sum() / total_pop
                else:
                    record[var] = vals.mean()
            
            record['sld_population'] = total_pop
            if 'TotEmp' in group.columns:
                record['sld_employment'] = group['TotEmp'].fillna(0).sum()
            
            tract_records.append(record)
        
        tract_df = pd.DataFrame(tract_records)
    else:
        print(f"  Aggregating {len(bg_df)} block groups → tracts (simple mean)")
        agg_dict = {var: 'mean' for var in available_vars}
        tract_df = bg_df.groupby('GEOID').agg(agg_dict).reset_index()
    
    # Save
    tract_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(tract_df)} tracts to {output_path}")
    
    # Summary
    if 'D2a_EpHHm' in tract_df.columns:
        print(f"\n  SLD Activity Diversity (D2a_EpHHm):")
        print(f"    Mean: {tract_df['D2a_EpHHm'].mean():.3f}")
        print(f"    Min:  {tract_df['D2a_EpHHm'].min():.3f}")
        print(f"    Max:  {tract_df['D2a_EpHHm'].max():.3f}")
    
    if 'NatWalkInd' in tract_df.columns:
        print(f"  National Walkability Index:")
        print(f"    Mean: {tract_df['NatWalkInd'].mean():.1f}")
    
    return output_path


def main():
    """Main function to download all required data."""
    print("=" * 60)
    print("Transit Desert Pipeline: Data Download")
    print("=" * 60)
    
    # Load config
    config = load_config()
    study_area = config['study_area']['name']
    print(f"\nStudy Area: {study_area}")
    print(f"FIPS: {config['study_area']['full_fips']}")
    
    # Set up output directory
    output_dir = PROJECT_ROOT / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "-" * 40)
    print("Step 1: Download GTFS Feed")
    print("-" * 40)
    download_gtfs(config, output_dir)
    
    print("\n" + "-" * 40)
    print("Step 2: Download Census Tract Boundaries")
    print("-" * 40)
    download_census_tracts(config, output_dir)
    
    print("\n" + "-" * 40)
    print("Step 3: Download ACS Demographic Data")
    print("-" * 40)
    download_acs_data(config, output_dir)
    
    print("\n" + "-" * 40)
    print("Step 4: Download EPA Smart Location Database")
    print("-" * 40)
    download_sld_data(config, output_dir)
    
    print("\n" + "=" * 60)
    print("Data download complete!")
    print("Next step: python src/02_compute_supply.py")
    print("=" * 60)

if __name__ == "__main__":
    main()