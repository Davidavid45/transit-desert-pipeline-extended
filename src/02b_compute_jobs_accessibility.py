"""
02b_compute_jobs_accessibility.py
Compute job accessibility metric using r5py travel time matrix.

This script calculates gravity-model job accessibility for each Census tract:

    A_i = Σ O_j × exp(-β × t_ij)

Where O_j = jobs at destination j, t_ij = travel time from tract i to j,
and β = decay parameter calibrated from commute time distributions.

Unlike a hard cutoff (e.g. "jobs within 45 min"), the decay model gives
full weight to nearby jobs and diminishing weight to distant ones, with
no abrupt boundary.

Prerequisites:
- r5py installed: pip install r5py
- Java 11+ installed
- GTFS data downloaded (run 01_download_data.py first)
- OpenStreetMap PBF file for the region

Usage:
    python src/02b_compute_jobs_accessibility.py

Outputs:
    - data/processed/jobs_accessibility.csv
    - Updates supply_metrics.csv with jobs_accessibility column
"""

import sys
import yaml
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, date, time, timedelta
import requests
import zipfile
from io import BytesIO
from state_osm import get_osm_pbf_path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gtfs_utils import is_valid_gtfs_zip

# Check for r5py
try:
    import r5py
    R5PY_AVAILABLE = True
except ImportError:
    R5PY_AVAILABLE = False
    print("⚠ r5py not installed. Run: pip install r5py")
    print("  Also requires Java 11+")


def load_config():
    """Load configuration from YAML file."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_osm_pbf(config, output_dir):
    """
    Download OpenStreetMap PBF file for Maryland.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_dir : Path
        Directory to save PBF file
    
    Returns
    -------
    Path
        Path to PBF file
    """
    osm_dir = output_dir / "osm"
    osm_dir.mkdir(parents=True, exist_ok=True)
    
    pbf_path = get_osm_pbf_path(config, osm_dir)
    
    return pbf_path


def download_lodes_data(config, output_dir):
    """
    Download LEHD LODES Workplace Area Characteristics (WAC) data.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_dir : Path
        Directory to save data
    
    Returns
    -------
    DataFrame
        Jobs by Census block
    """
    lodes_dir = output_dir / "lodes"
    lodes_dir.mkdir(parents=True, exist_ok=True)
    
    state_fips = config['study_area']['state_fips']
    
    # LODES year — check multiple config locations
    lodes_year = (config.get('accessibility', {}).get('job_accessibility', {}).get('lodes_year')
                  or config.get('accessibility', {}).get('lodes_year')
                  or 2021)  # Default to 2021 (most recent stable LODES release)
    
    # State FIPS to abbreviation mapping (common states)
    fips_to_abbrev = {
        '24': 'md', '42': 'pa', '47': 'tn', '48': 'tx',
        '06': 'ca', '36': 'ny', '17': 'il', '12': 'fl',
        '39': 'oh', '51': 'va', '11': 'dc', '34': 'nj',
        '13': 'ga', '37': 'nc', '25': 'ma', '53': 'wa',
    }
    state_abbrev = fips_to_abbrev.get(state_fips)
    if state_abbrev is None:
        # Fallback: try to derive from us library or just warn
        try:
            from us import states as us_states
            state_obj = us_states.lookup(state_fips)
            state_abbrev = state_obj.abbr.lower() if state_obj else None
        except ImportError:
            pass
    
    if state_abbrev is None:
        print(f"  ✗ Cannot determine state abbreviation for FIPS {state_fips}")
        print("  → Add it to the fips_to_abbrev mapping in 02b")
        return None
    
    # WAC file URL
    # Format: state_wac_S000_JT00_YYYY.csv.gz
    filename = f"{state_abbrev}_wac_S000_JT00_{lodes_year}.csv.gz"
    url = f"https://lehd.ces.census.gov/data/lodes/LODES8/{state_abbrev}/wac/{filename}"
    
    csv_path = lodes_dir / filename.replace('.gz', '')
    
    if csv_path.exists():
        print(f"  ✓ LODES data already exists: {csv_path}")
        return pd.read_csv(csv_path)
    
    print(f"  Downloading LODES WAC data from: {url}")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Decompress and save
        import gzip
        with gzip.open(BytesIO(response.content), 'rt') as gz:
            df = pd.read_csv(gz)
        
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Downloaded LODES data: {len(df)} blocks")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error downloading LODES: {e}")
        return None


def aggregate_jobs_to_tracts(lodes_df, config):
    """
    Aggregate block-level jobs to tract level.
    
    Parameters
    ----------
    lodes_df : DataFrame
        LODES WAC data with w_geocode (block FIPS)
    config : dict
        Configuration dictionary
    
    Returns
    -------
    DataFrame
        Jobs by tract
    """
    # Extract tract GEOID from block GEOID (first 11 digits)
    lodes_df['tract_geoid'] = lodes_df['w_geocode'].astype(str).str[:11]
    
    # Filter to study area
    county_fips = config['study_area']['full_fips']  # e.g., "24510"
    lodes_df = lodes_df[lodes_df['tract_geoid'].str.startswith(county_fips)]
    
    # Aggregate total jobs (C000) to tract
    tract_jobs = lodes_df.groupby('tract_geoid').agg({
        'C000': 'sum'  # Total jobs
    }).reset_index()
    
    tract_jobs.columns = ['GEOID', 'total_jobs']
    
    print(f"  ✓ Aggregated jobs for {len(tract_jobs)} tracts")
    print(f"  ✓ Total jobs in study area: {tract_jobs['total_jobs'].sum():,}")
    
    return tract_jobs


def load_tract_centroids(config):
    """
    Load tract centroids for travel time calculation.
    
    Returns
    -------
    GeoDataFrame
        Tract centroids
    """
    tracts_path = PROJECT_ROOT / "data" / "raw" / "census" / "study_area_tracts.shp"
    
    if not tracts_path.exists():
        raise FileNotFoundError(f"Tract shapefile not found: {tracts_path}")
    
    gdf = gpd.read_file(tracts_path)
    
    # Calculate centroids
    gdf_wgs84 = gdf.to_crs('EPSG:4326')
    centroids = gdf_wgs84.copy()
    centroids['geometry'] = gdf_wgs84.geometry.centroid
    
    # Add lat/lon columns
    centroids['lat'] = centroids.geometry.y
    centroids['lon'] = centroids.geometry.x
    centroids['id'] = centroids['GEOID']  # For r5py matching
    
    print(f"  ✓ Loaded {len(centroids)} tract centroids")
    
    return centroids[['GEOID', 'id', 'geometry', 'lat', 'lon']]


def compute_travel_time_matrix(origins, destinations, gtfs_path, osm_path, config):
    """
    Compute travel time matrix using r5py.
    
    Parameters
    ----------
    origins : GeoDataFrame
        Origin points (tract centroids)
    destinations : GeoDataFrame
        Destination points (tract centroids with jobs)
    gtfs_path : Path
        Path to GTFS zip file
    osm_path : Path
        Path to OSM PBF file
    config : dict
        Configuration dictionary
    
    Returns
    -------
    DataFrame
        Travel time matrix
    """
    if not R5PY_AVAILABLE:
        raise ImportError("r5py is required for travel time calculation")
    
    print("  Building transport network (this may take several minutes)...")
    
    # Create transport network
    # Handle both single zip and directory of zips
    # Filter out wrapper zips (e.g. zips containing only other zips, no GTFS txt files)
    if gtfs_path.is_dir():
        gtfs_files = [str(f) for f in sorted(gtfs_path.glob("*.zip"))
                      if is_valid_gtfs_zip(f)]
    else:
        gtfs_files = [str(gtfs_path)]
    
    transport_network = r5py.TransportNetwork(
        osm_pbf=str(osm_path),
        gtfs=gtfs_files
    )
    
    # Departure time from config
    job_config = config.get('accessibility', {}).get('job_accessibility', {})
    dep_time_str = job_config.get('departure_time', '08:00:00')
    dep_h, dep_m, dep_s = map(int, dep_time_str.split(':'))
    
    # Use the analysis date from config, or a sensible default
    analysis_date_str = config.get('gtfs', {}).get('analysis_date', '20241015')
    try:
        dep_date = datetime.strptime(analysis_date_str, '%Y%m%d').date()
    except ValueError:
        dep_date = date(2024, 10, 15)
    
    departure_time = datetime.combine(dep_date, time(dep_h, dep_m, dep_s))
    
    # Max travel time from config (upper bound for routing efficiency)
    max_minutes = job_config.get('max_travel_time_minutes', 90)
    max_time = timedelta(minutes=max_minutes)
    
    print(f"  Departure: {departure_time}")
    print(f"  Max travel time: {max_minutes} min (routing upper bound)")
    print(f"  Computing travel times...")
    
    # Compute travel time matrix
    travel_time_matrix = r5py.TravelTimeMatrix(
        transport_network,
        origins=origins,
        destinations=destinations,
        departure=departure_time,
        departure_time_window=timedelta(hours=1),
        transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
        max_time=max_time
    )

    travel_time_matrix = pd.DataFrame(travel_time_matrix)
    
    print(f"  ✓ Computed {len(travel_time_matrix)} origin-destination pairs")
    
    return travel_time_matrix


def calculate_jobs_accessible(travel_times, tract_jobs, config):
    """
    Calculate gravity-model job accessibility from each tract.
    
    A_i = Σ O_j × exp(-β × t_ij)
    
    Where:
      O_j = total jobs at destination tract j
      t_ij = travel time in minutes from tract i to tract j
      β = decay parameter (higher = steeper decay = only nearby jobs matter)
    
    A job 10 minutes away counts almost fully.
    A job 40 minutes away counts partially.
    A job 60+ minutes away barely counts.
    No hard cutoff — the decay handles it smoothly.
    
    Parameters
    ----------
    travel_times : DataFrame
        Travel time matrix with from_id, to_id, travel_time (minutes)
    tract_jobs : DataFrame
        Jobs per tract (GEOID, total_jobs)
    config : dict
        Configuration dictionary with decay_beta
    
    Returns
    -------
    DataFrame
        Gravity-weighted job accessibility from each tract
    """
    job_config = config.get('accessibility', {}).get('job_accessibility', {})
    decay_beta = job_config.get('decay_beta', 0.08)
    
    print(f"  Decay model: A_i = Σ O_j × exp(-{decay_beta} × t_ij)")
    
    # Drop rows with NaN travel time (unreachable OD pairs)
    valid_times = travel_times.dropna(subset=['travel_time']).copy()
    
    # Merge with jobs at destination
    valid_times = valid_times.merge(
        tract_jobs,
        left_on='to_id',
        right_on='GEOID',
        how='left'
    )
    valid_times['total_jobs'] = valid_times['total_jobs'].fillna(0)
    
    # Apply exponential decay weighting
    valid_times['decay_weight'] = np.exp(-decay_beta * valid_times['travel_time'])
    valid_times['weighted_jobs'] = valid_times['total_jobs'] * valid_times['decay_weight']
    
    # Sum weighted jobs reachable from each origin
    jobs_accessible = valid_times.groupby('from_id').agg({
        'weighted_jobs': 'sum',
        'total_jobs': 'sum',       # Also keep raw count for reference
        'to_id': 'count'           # Number of reachable destinations
    }).reset_index()
    
    jobs_accessible.columns = ['GEOID', 'jobs_accessibility', 'jobs_raw_sum', 'destinations_reached']
    
    # Show how decay compares to raw count
    print(f"\n  Jobs accessibility statistics (gravity-weighted):")
    print(f"    Mean:   {jobs_accessible['jobs_accessibility'].mean():,.0f}")
    print(f"    Median: {jobs_accessible['jobs_accessibility'].median():,.0f}")
    print(f"    Max:    {jobs_accessible['jobs_accessibility'].max():,.0f}")
    print(f"\n  For comparison — raw jobs sum (no decay):")
    print(f"    Mean:   {jobs_accessible['jobs_raw_sum'].mean():,.0f}")
    print(f"    Ratio (decay/raw): {jobs_accessible['jobs_accessibility'].mean() / max(jobs_accessible['jobs_raw_sum'].mean(), 1):.2%}")
    
    # Show effective decay at key travel times
    print(f"\n  Decay weights at key travel times (β={decay_beta}):")
    for t in [5, 10, 15, 20, 30, 45, 60, 90]:
        w = np.exp(-decay_beta * t)
        print(f"    {t:3d} min: {w:.3f} ({w:.0%} of full weight)")
    
    return jobs_accessible[['GEOID', 'jobs_accessibility']]


def update_supply_metrics(jobs_accessible):
    """
    Update supply_metrics.csv with jobs accessibility column.
    
    CPTA recalculation is handled separately by 02d_compute_cpta.py.
    
    Parameters
    ----------
    jobs_accessible : DataFrame
        Gravity-weighted job accessibility from each tract
    """
    supply_path = PROJECT_ROOT / "data" / "processed" / "supply_metrics.csv"
    
    if not supply_path.exists():
        print("  ⚠ supply_metrics.csv not found. Run 02_compute_supply.py first.")
        return
    
    supply_df = pd.read_csv(supply_path, dtype={'GEOID': str})
    
    # Drop existing column if present (re-run scenario)
    if 'jobs_accessibility' in supply_df.columns:
        supply_df = supply_df.drop(columns=['jobs_accessibility'])
    
    # Merge jobs accessibility
    supply_df = supply_df.merge(
        jobs_accessible,
        on='GEOID',
        how='left'
    )
    supply_df['jobs_accessibility'] = supply_df['jobs_accessibility'].fillna(0)
    
    supply_df.to_csv(supply_path, index=False)
    print(f"  ✓ Updated supply_metrics.csv with jobs_accessibility column")
    print(f"  → Run 02d_compute_cpta.py to incorporate into CPTA score")


def main():
    """Main function to compute job accessibility."""
    print("=" * 60)
    print("Transit Desert Pipeline: Job Accessibility (r5py)")
    print("=" * 60)
    
    if not R5PY_AVAILABLE:
        print("\n⚠ r5py is not installed.")
        print("  To install: pip install r5py")
        print("  Also requires Java 11+")
        print("\n  Alternatively, job accessibility can be computed using:")
        print("  - OpenTripPlanner")
        print("  - Conveyal Analysis")
        sys.exit(1)
    
    config = load_config()
    
    # Paths
    raw_dir = PROJECT_ROOT / "data" / "raw"
    external_dir = PROJECT_ROOT / "data" / "external"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    external_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download OSM data
    print("\n" + "-" * 40)
    print("Step 1: OpenStreetMap Data")
    print("-" * 40)
    
    osm_path = download_osm_pbf(config, external_dir)
    if osm_path is None:
        print("  ✗ Cannot proceed without OSM data")
        sys.exit(1)
    
    # Step 2: Download LODES data
    print("\n" + "-" * 40)
    print("Step 2: LEHD LODES Job Data")
    print("-" * 40)
    
    lodes_df = download_lodes_data(config, external_dir)
    if lodes_df is None:
        print("  ✗ Cannot proceed without LODES data")
        sys.exit(1)
    
    tract_jobs = aggregate_jobs_to_tracts(lodes_df, config)
    
    # Step 3: Load tract centroids
    print("\n" + "-" * 40)
    print("Step 3: Load Tract Centroids")
    print("-" * 40)
    
    centroids = load_tract_centroids(config)
    
    # Add jobs to centroids for destination weighting
    centroids_with_jobs = centroids.merge(tract_jobs, on='GEOID', how='left').fillna(0)
    centroids_with_jobs['id'] = centroids_with_jobs['GEOID']  # Ensure 'id' column exists for r5py matching
    
    # Step 4: Compute travel time matrix
    print("\n" + "-" * 40)
    print("Step 4: Compute Travel Time Matrix")
    print("-" * 40)
    
    gtfs_path = raw_dir / "gtfs"
    gtfs_zips = list(gtfs_path.glob("*.zip")) if gtfs_path.is_dir() else []
    if not gtfs_zips:
        print(f"  No GTFS zip files found in: {gtfs_path}")
        print("    Run 01_download_data.py first")
        sys.exit(1)
    print(f"  Found {len(gtfs_zips)} GTFS file(s): {[f.name for f in gtfs_zips]}")
    
    try:
        travel_times = compute_travel_time_matrix(
            origins=centroids,
            destinations=centroids_with_jobs,
            gtfs_path=gtfs_path,
            osm_path=osm_path,
            config=config
        )
    except Exception as e:
        print(f"  ✗ Error computing travel times: {e}")
        print("\n  Common issues:")
        print("  - Java not installed or wrong version (needs Java 11+)")
        print("  - Insufficient memory (try increasing Java heap)")
        print("  - GTFS or OSM data issues")
        sys.exit(1)
    
    # Step 5: Calculate jobs accessible
    print("\n" + "-" * 40)
    print("Step 5: Calculate Jobs Accessible")
    print("-" * 40)
    
    jobs_accessible = calculate_jobs_accessible(travel_times, tract_jobs, config)
    
    # Save results
    jobs_accessible.to_csv(processed_dir / "jobs_accessibility.csv", index=False)
    print(f"  ✓ Saved to {processed_dir / 'jobs_accessibility.csv'}")
    
    # Update supply metrics
    print("\n" + "-" * 40)
    print("Step 6: Update Supply Metrics")
    print("-" * 40)
    
    update_supply_metrics(jobs_accessible)
    
    print("\n" + "=" * 60)
    print("Job accessibility calculation complete!")
    print("\nNext steps:")
    print("  python src/02c_compute_poi_accessibility.py  (optional)")
    print("  python src/02d_compute_cpta.py               (computes CPTA score)")
    print("=" * 60)


if __name__ == "__main__":
    main()