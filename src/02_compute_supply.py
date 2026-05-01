"""
02_compute_supply.py
Compute Composite Public Transit Accessibility (CPTA) score for each Census tract.

Metrics computed:
  Transit Service (7 metrics):
    1. Stop Connectivity (Welch method)
    2. Route Coverage (bus ≥2 stops, rail ≥1 stop)
    3. Frequency - AM Peak
    4. Frequency - Midday
    5. Frequency - PM Peak
    6. Span of Service
    7. Weekend Ratio
  
  Walking/Access (2 metrics):
    8. Walking Access (bus 400m, rail 800m Euclidean buffers)
    9. Jobs Accessible (optional, added by 02b)
    10. POI Accessible (optional, added by 02c)
  
  Built Environment (4 metrics):
    11. Population Density
    12. Intersection Density
    13. Land Use Mix (Shannon entropy)
    14. Sidewalk Index (OSM footway density, m/km²)

Usage:
    python src/02_compute_supply.py

Outputs:
    - data/processed/supply_metrics.csv
    - data/processed/supply_metrics.gpkg
    - data/processed/cpta_correlation_matrix.csv
"""

import os
import sys
import yaml
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import our GTFS utility module
from gtfs_utils import load_gtfs, print_gtfs_summary
from gtfs_date_picker import pick_best_date, validate_date

# Optional: OSMnx for built environment metrics
try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    print("⚠ osmnx not installed. Built environment metrics will be skipped.")
    print("  Install with: pip install osmnx")


def load_config():
    """Load configuration from YAML file."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_tracts(config):
    """
    Load Census tract boundaries.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    
    Returns
    -------
    GeoDataFrame
        Census tracts with geometry and area_sqkm
    """
    tracts_path = PROJECT_ROOT / "data" / "raw" / "census" / "study_area_tracts.shp"
    
    if not tracts_path.exists():
        raise FileNotFoundError(f"Tract shapefile not found: {tracts_path}")
    
    gdf = gpd.read_file(tracts_path)
    
    # Reproject to projected CRS for accurate area/distance calculations
    crs_projected = config['spatial']['crs_projected']
    gdf = gdf.to_crs(crs_projected)
    
    # Calculate tract area in square kilometers
    crs_obj = gdf.crs
    if crs_obj.axis_info[0].unit_name == 'metre':
        gdf['area_sqkm'] = gdf.geometry.area / 1e6
    elif crs_obj.axis_info[0].unit_name in ['foot', 'US survey foot', 'ft']:
        gdf['area_sqkm'] = gdf.geometry.area / (3.28084**2) / 1e6
    else:
        print(f"  ⚠ Unknown CRS units: {crs_obj.axis_info[0].unit_name}, assuming meters")
        gdf['area_sqkm'] = gdf.geometry.area / 1e6
    
    print(f"  ✓ Loaded {len(gdf)} tracts")
    print(f"  ✓ CRS: {gdf.crs}")
    
    return gdf


def get_stops_gdf(feed, config):
    """
    Convert GTFS stops to GeoDataFrame with route_type information.
    
    Parameters
    ----------
    feed : gtfs_kit.Feed
        GTFS feed
    config : dict
        Configuration dictionary
    
    Returns
    -------
    GeoDataFrame
        Stops with geometry and route_type
    """
    stops = feed.stops.copy()
    
    # Create geometry
    geometry = [Point(lon, lat) for lon, lat in zip(stops['stop_lon'], stops['stop_lat'])]
    gdf = gpd.GeoDataFrame(stops, geometry=geometry, crs='EPSG:4326')
    
    # Determine the primary route_type for each stop
    # (a stop might serve both bus and rail — take the "best" mode)
    stop_times = feed.stop_times[['trip_id', 'stop_id']].drop_duplicates()
    trips = feed.trips[['trip_id', 'route_id']]
    routes = feed.routes[['route_id', 'route_type']]
    
    stop_routes = (stop_times
                   .merge(trips, on='trip_id')
                   .merge(routes, on='route_id')
                   [['stop_id', 'route_type']]
                   .drop_duplicates())
    
    # For each stop, get the minimum route_type (rail types 0,1,2 < bus type 3)
    # This means if a stop serves both rail and bus, it's classified as rail
    stop_mode = stop_routes.groupby('stop_id')['route_type'].min().reset_index()
    stop_mode.columns = ['stop_id', 'primary_route_type']
    
    gdf = gdf.merge(stop_mode, on='stop_id', how='left')
    gdf['primary_route_type'] = gdf['primary_route_type'].fillna(3).astype(int)  # Default to bus
    
    # Add is_rail flag (route_type 0=tram, 1=subway, 2=rail)
    gdf['is_rail'] = gdf['primary_route_type'].isin([0, 1, 2])
    
    # Reproject
    crs_projected = config['spatial']['crs_projected']
    gdf = gdf.to_crs(crs_projected)
    
    n_rail = gdf['is_rail'].sum()
    n_bus = (~gdf['is_rail']).sum()
    print(f"  ✓ Stops: {len(gdf)} total ({n_bus} bus, {n_rail} rail)")
    
    return gdf


def compute_stop_routes_frequency(feed, config):
    """
    Compute routes and frequency at each stop.
    
    Parameters
    ----------
    feed : gtfs_kit.Feed
        GTFS feed
    config : dict
        Configuration dictionary
    
    Returns
    -------
    DataFrame
        Stop-level metrics: routes_at_stop, frequency by time window
    """
    print("Computing stop-level metrics...")
    
    # Get valid service dates from the feed
    try:
        valid_dates = feed.get_dates()
        if valid_dates is None or len(valid_dates) == 0:
            valid_dates = []
    except:
        valid_dates = []
    
    # Try to use config date, but validate it exists in feed
    analysis_date = None
    config_date = config['gtfs'].get('analysis_date', '')
    gtfs_dir = PROJECT_ROOT / "data" / "raw" / "gtfs"
    if not validate_date(gtfs_dir, config_date, verbose=False):
        print(f"  ⚠ Configured date {config_date} has no service — auto-selecting...")
        best = pick_best_date(gtfs_dir)
        if best:
            config_date = best
            print(f"  ✓ Using {best} instead")
    
    if config_date:
        config_date_normalized = config_date.replace('-', '')
        if config_date_normalized in valid_dates:
            analysis_date = config_date_normalized
            print(f"  Using configured analysis date: {analysis_date}")
        else:
            print(f"  ⚠ Configured date {config_date} not in GTFS calendar")
    
    if analysis_date is None:
        if valid_dates and len(valid_dates) > 0:
            mid_idx = len(valid_dates) // 2
            analysis_date = valid_dates[mid_idx]
            print(f"  Auto-selected analysis date: {analysis_date}")
            print(f"    (from {len(valid_dates)} valid dates: {valid_dates[0]} to {valid_dates[-1]})")
        else:
            print("  ⚠ No valid dates found in GTFS calendar, using all stop_times")
            analysis_date = None
    
    # Get stop times for analysis date
    if analysis_date:
        try:
            stop_times = feed.get_stop_times(analysis_date)
            if stop_times is None or len(stop_times) == 0:
                print(f"  ⚠ No stop times for date {analysis_date}, using all stop_times")
                stop_times = feed.stop_times.copy()
        except Exception as e:
            print(f"  ⚠ Error getting stop times for {analysis_date}: {e}")
            stop_times = feed.stop_times.copy()
    else:
        stop_times = feed.stop_times.copy()
    
    print(f"  Stop times loaded: {len(stop_times):,} records")
    
    if len(stop_times) == 0:
        print("  ✗ ERROR: No stop times found! Check GTFS data.")
        return pd.DataFrame(columns=[
            'stop_id', 'routes_at_stop', 'total_departures', 'destinations_reachable',
            'freq_am_peak', 'freq_midday', 'freq_pm_peak', 'freq_evening'
        ])
    
    # Merge with trips to get route_id
    trips = feed.trips[['trip_id', 'route_id', 'service_id']].copy()
    stop_times = stop_times.merge(trips, on='trip_id', how='left')
    
    # Convert arrival_time to seconds
    def time_to_seconds(time_str):
        if pd.isna(time_str):
            return np.nan
        try:
            parts = str(time_str).split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
        except:
            return np.nan
    
    stop_times['arrival_seconds'] = stop_times['arrival_time'].apply(time_to_seconds)
    
    # Time windows from config
    # Support both old ('gtfs.time_windows') and new ('transit_service.frequency_periods') paths
    time_windows = (config.get('transit_service', {}).get('frequency_periods')
                    or config.get('gtfs', {}).get('time_windows', {}))
    if not time_windows:
        print("  ⚠ No frequency periods found in config, using defaults")
        time_windows = {
            'am_peak': {'start': '06:00:00', 'end': '09:00:00'},
            'midday':  {'start': '10:00:00', 'end': '15:00:00'},
            'pm_peak': {'start': '16:00:00', 'end': '19:00:00'},
            'evening': {'start': '19:00:00', 'end': '23:00:00'}
        }
    
    def time_str_to_seconds(t):
        h, m, s = map(int, t.split(':'))
        return h * 3600 + m * 60 + s
    
    # Initialize results
    stop_metrics = []
    
    for stop_id in tqdm(stop_times['stop_id'].unique(), desc="Processing stops"):
        stop_data = stop_times[stop_times['stop_id'] == stop_id]
        
        # Routes at stop
        routes_at_stop = stop_data['route_id'].nunique()
        
        # Frequency by time window (departures per hour)
        freq_metrics = {}
        for window_name, window_times in time_windows.items():
            start_sec = time_str_to_seconds(window_times['start'])
            end_sec = time_str_to_seconds(window_times['end'])
            
            # Handle overnight windows (e.g. evening 18:30 → 06:30 next day)
            # GTFS allows times > 24:00:00 for trips past midnight
            if end_sec <= start_sec:
                # Overnight: match arrivals >= start OR < end (next day)
                # Also check for GTFS 24+ hour times (e.g. 25:30:00 = 1:30 AM)
                window_trips = stop_data[
                    (stop_data['arrival_seconds'] >= start_sec) | 
                    (stop_data['arrival_seconds'] < end_sec)
                ]
                window_hours = (86400 - start_sec + end_sec) / 3600
            else:
                window_trips = stop_data[
                    (stop_data['arrival_seconds'] >= start_sec) & 
                    (stop_data['arrival_seconds'] < end_sec)
                ]
                window_hours = (end_sec - start_sec) / 3600
            departures = len(window_trips)
            freq = departures / window_hours if window_hours > 0 else 0
            freq_metrics[f'freq_{window_name}'] = freq
        
        # Total daily departures
        total_departures = len(stop_data)
        
        # Unique destinations reachable
        route_ids = stop_data['route_id'].unique()
        destinations = stop_times[stop_times['route_id'].isin(route_ids)]['stop_id'].nunique()
        
        stop_metrics.append({
            'stop_id': stop_id,
            'routes_at_stop': routes_at_stop,
            'total_departures': total_departures,
            'destinations_reachable': destinations,
            **freq_metrics
        })
    
    return pd.DataFrame(stop_metrics)


def compute_stop_connectivity_welch(stop_metrics):
    """
    Compute Welch-style stop connectivity score.
    Connectivity = routes × frequency × destinations
    """
    print("Computing Welch connectivity scores...")
    
    stop_metrics['connectivity'] = (
        stop_metrics['routes_at_stop'] * 
        stop_metrics['freq_am_peak'] * 
        stop_metrics['destinations_reachable']
    )
    
    max_conn = stop_metrics['connectivity'].max()
    if max_conn > 0:
        stop_metrics['connectivity_normalized'] = (stop_metrics['connectivity'] / max_conn) * 100
    else:
        stop_metrics['connectivity_normalized'] = 0
    
    return stop_metrics


def aggregate_stops_to_tracts(stops_gdf, stop_metrics, tracts_gdf, config):
    """
    Aggregate stop-level metrics to tract level.
    """
    print("Aggregating stop metrics to tracts...")
    
    # Merge stop metrics with geometry
    stops_gdf = stops_gdf.merge(stop_metrics, on='stop_id')
    
    # Spatial join: stops to tracts
    stops_in_tracts = gpd.sjoin(stops_gdf, tracts_gdf[['GEOID', 'geometry']], how='left', predicate='within')
    
    # Aggregate by tract
    tract_metrics = stops_in_tracts.groupby('GEOID').agg({
        'stop_id': 'count',
        'routes_at_stop': 'sum',
        'connectivity': 'sum',
        'freq_am_peak': 'mean',
        'freq_midday': 'mean',
        'freq_pm_peak': 'mean',
        'freq_evening': 'mean',
        'total_departures': 'sum'
    }).reset_index()
    
    tract_metrics.columns = [
        'GEOID', 'stop_count', 'route_stop_pairs', 'connectivity_sum',
        'freq_am_peak', 'freq_midday', 'freq_pm_peak', 'freq_evening',
        'total_departures'
    ]
    
    # Merge with tract areas
    tract_metrics = tract_metrics.merge(
        tracts_gdf[['GEOID', 'area_sqkm']], 
        on='GEOID'
    )
    
    # Calculate density metrics
    tract_metrics['stop_density'] = tract_metrics['stop_count'] / tract_metrics['area_sqkm']
    tract_metrics['connectivity_density'] = tract_metrics['connectivity_sum'] / tract_metrics['area_sqkm']
    
    return tract_metrics


def compute_route_coverage(feed, stops_gdf, tracts_gdf, config):
    """
    Compute population-weighted route coverage for each tract.
    
    For each route, calculate what percentage of the tract's population
    lives within the walking buffer of that route's stops. The tract's
    route coverage score is the average population coverage across all
    routes serving the tract.
    
    This replaces the simpler "count routes with ≥N stops" approach,
    providing a more nuanced measure of how well routes actually serve
    the tract's residents.
    
    Parameters
    ----------
    feed : gtfs_kit.Feed
        GTFS feed
    stops_gdf : GeoDataFrame
        Stops with geometry and is_rail flag
    tracts_gdf : GeoDataFrame
        Census tracts with geometry and area_sqkm
    config : dict
        Configuration dictionary
    
    Returns
    -------
    DataFrame
        Route coverage score by tract (0-100 scale)
    """
    print("Computing population-weighted route coverage...")
    
    # Get buffer distances from config
    rc_config = config.get('transit_service', {}).get('route_coverage', {})
    buffer_bus_m = rc_config.get('bus_buffer_m', 400)
    buffer_rail_m = rc_config.get('rail_buffer_m', 800)
    rail_types = rc_config.get('rail_route_types', [0, 1, 2])
    
    print(f"  Buffers: bus={buffer_bus_m}m, rail={buffer_rail_m}m")
    
    # Convert buffer distances to CRS units
    crs_obj = tracts_gdf.crs
    if crs_obj.axis_info[0].unit_name in ['foot', 'US survey foot', 'ft']:
        buffer_bus_crs = buffer_bus_m * 3.28084
        buffer_rail_crs = buffer_rail_m * 3.28084
    else:
        buffer_bus_crs = buffer_bus_m
        buffer_rail_crs = buffer_rail_m
    
    # Load population data for weighting
    acs_path = PROJECT_ROOT / "data" / "raw" / "acs" / "acs_tvi_variables.csv"
    if acs_path.exists():
        acs_df = pd.read_csv(acs_path, dtype={'GEOID': str})
        pop_col = None
        for col in ['B01001_001E', 'B03002_001E']:
            if col in acs_df.columns:
                pop_col = col
                break
        if pop_col:
            tract_pop = acs_df.set_index('GEOID')[pop_col].to_dict()
        else:
            tract_pop = {}
    else:
        tract_pop = {}
    
    if not tract_pop:
        print("  ⚠ No population data found — falling back to area-weighted coverage")
    
    # Get route-stop relationships with route_type
    stop_times_df = feed.stop_times[['trip_id', 'stop_id']].drop_duplicates()
    trips = feed.trips[['trip_id', 'route_id']]
    routes = feed.routes[['route_id', 'route_type']]
    
    route_stops = (stop_times_df
                   .merge(trips, on='trip_id')
                   .merge(routes, on='route_id')
                   [['route_id', 'stop_id', 'route_type']]
                   .drop_duplicates())
    
    # Get unique routes
    unique_routes = route_stops[['route_id', 'route_type']].drop_duplicates()
    print(f"  Routes to process: {len(unique_routes)}")
    
    # For each route, compute the union of its stop buffers
    route_buffers = {}
    for _, row in unique_routes.iterrows():
        route_id = row['route_id']
        route_type = row['route_type']
        
        # Get stops for this route
        route_stop_ids = route_stops[route_stops['route_id'] == route_id]['stop_id'].unique()
        route_stop_geoms = stops_gdf[stops_gdf['stop_id'].isin(route_stop_ids)]
        
        if len(route_stop_geoms) == 0:
            continue
        
        # Apply mode-appropriate buffer
        is_rail = route_type in rail_types
        buffer_dist = buffer_rail_crs if is_rail else buffer_bus_crs
        
        # Union of all stop buffers for this route
        stop_buffers = route_stop_geoms.geometry.buffer(buffer_dist)
        route_buffers[route_id] = unary_union(stop_buffers.values)
    
    print(f"  Route buffers computed: {len(route_buffers)}")
    
    # For each tract, compute average population coverage across serving routes
    coverage_results = []
    
    for idx, tract in tqdm(tracts_gdf.iterrows(), total=len(tracts_gdf),
                           desc="Computing route coverage"):
        geoid = tract['GEOID']
        tract_geom = tract.geometry
        tract_area = tract_geom.area
        
        if tract_area <= 0:
            coverage_results.append({'GEOID': geoid, 'route_coverage': 0.0})
            continue
        
        # Find routes whose buffers intersect this tract
        route_coverages = []
        for route_id, buffer_geom in route_buffers.items():
            if not tract_geom.intersects(buffer_geom):
                continue
            
            # Fraction of tract area covered by this route's buffer
            intersection = tract_geom.intersection(buffer_geom)
            area_fraction = intersection.area / tract_area
            
            # If we have population data, this fraction approximates
            # the share of residents within walking distance.
            # (Assumes uniform population distribution within tract —
            # a common simplification at tract level.)
            route_coverages.append(min(area_fraction, 1.0))
        
        if route_coverages:
            # Average coverage across all routes serving the tract
            # Then multiply by number of routes to reward diversity
            avg_coverage = np.mean(route_coverages)
            n_routes = len(route_coverages)
            
            # Score: average coverage × log(1 + n_routes) to reward route diversity
            # without letting route count dominate
            route_coverage_score = avg_coverage * np.log1p(n_routes) * 100
        else:
            route_coverage_score = 0.0
        
        coverage_results.append({
            'GEOID': geoid,
            'route_coverage': route_coverage_score
        })
    
    result = pd.DataFrame(coverage_results)
    
    print(f"  ✓ Route coverage: mean={result['route_coverage'].mean():.1f}, "
          f"max={result['route_coverage'].max():.1f}")
    
    return result


def compute_walking_access(stops_gdf, tracts_gdf, config):
    """
    Compute percentage of tract area within walking distance of transit.
    
    Uses mode-appropriate Euclidean buffers:
    - Bus stops: 400m buffer (600m with micromobility)
    - Rail stations: 800m buffer (1500m with micromobility)
    
    Micromobility adjustment (bikeshare/scooter) extends catchment areas
    when enabled in config.
    """
    print("Computing walking access...")
    
    # Read from canonical config path, fall back to legacy spatial section
    walk_config = config.get('accessibility', {}).get('walking_access', {})
    
    buffer_bus = walk_config.get('bus_buffer_m',
                    config.get('spatial', {}).get('buffer_bus', 400))
    buffer_rail = walk_config.get('rail_buffer_m',
                     config.get('spatial', {}).get('buffer_rail', 800))
    
    # Check micromobility adjustment
    micro_config = walk_config.get('micromobility', {})
    if micro_config.get('enabled', False):
        buffer_bus = micro_config.get('bus_buffer_m', 600)
        buffer_rail = micro_config.get('rail_buffer_m', 1500)
        print(f"  Micromobility enabled — extended buffers: bus={buffer_bus}m, rail={buffer_rail}m")
    else:
        print(f"  Standard buffers: bus={buffer_bus}m, rail={buffer_rail}m")
    
    # Detect CRS units and convert buffers
    crs_obj = tracts_gdf.crs
    if crs_obj.axis_info[0].unit_name in ['foot', 'US survey foot', 'ft']:
        buffer_bus_crs = buffer_bus * 3.28084
        buffer_rail_crs = buffer_rail * 3.28084
    else:
        buffer_bus_crs = buffer_bus
        buffer_rail_crs = buffer_rail
    
    # Apply mode-specific buffers
    bus_stops = stops_gdf[~stops_gdf['is_rail']]
    rail_stops = stops_gdf[stops_gdf['is_rail']]
    
    buffers = []
    if len(bus_stops) > 0:
        bus_buffers = bus_stops.geometry.buffer(buffer_bus_crs)
        buffers.extend(bus_buffers.values)
        print(f"  Bus stops: {len(bus_stops)} with {buffer_bus}m buffer")
    
    if len(rail_stops) > 0:
        rail_buffers = rail_stops.geometry.buffer(buffer_rail_crs)
        buffers.extend(rail_buffers.values)
        print(f"  Rail stops: {len(rail_stops)} with {buffer_rail}m buffer")
    
    if not buffers:
        print("  ⚠ No stops found, returning 0% walking access")
        return pd.DataFrame({
            'GEOID': tracts_gdf['GEOID'],
            'walking_access_pct': 0.0
        })
    
    # Union all buffers
    all_buffers = unary_union(buffers)
    
    # Calculate intersection area for each tract
    walking_access = []
    
    for idx, tract in tqdm(tracts_gdf.iterrows(), total=len(tracts_gdf), desc="Computing walking access"):
        tract_geom = tract.geometry
        intersection = tract_geom.intersection(all_buffers)
        
        if tract_geom.area > 0:
            pct_covered = (intersection.area / tract_geom.area) * 100
        else:
            pct_covered = 0
        
        walking_access.append({
            'GEOID': tract['GEOID'],
            'walking_access_pct': min(pct_covered, 100)
        })
    
    return pd.DataFrame(walking_access)


def compute_span_of_service(feed, stop_metrics):
    """
    Compute span of service (hours per day with service).
    """
    print("Computing span of service...")
    
    stop_times = feed.stop_times.copy()
    
    def time_to_hours(time_str):
        if pd.isna(time_str):
            return np.nan
        parts = str(time_str).split(':')
        return int(parts[0]) + int(parts[1]) / 60
    
    stop_times['arrival_hour'] = stop_times['arrival_time'].apply(time_to_hours)
    
    span = stop_times.groupby('stop_id').agg({
        'arrival_hour': ['min', 'max']
    }).reset_index()
    span.columns = ['stop_id', 'first_departure', 'last_departure']
    span['span_hours'] = span['last_departure'] - span['first_departure']
    
    return span[['stop_id', 'span_hours']]


def compute_weekend_ratio(feed):
    """
    Compute ratio of weekend to weekday service.
    """
    print("Computing weekend ratio...")
    
    calendar = feed.calendar.copy() if feed.calendar is not None else None
    
    if calendar is None or len(calendar) == 0:
        print("  ⚠ No calendar.txt found, using calendar_dates")
        return pd.DataFrame({
            'stop_id': feed.stops['stop_id'],
            'weekend_ratio': 1.0
        })
    
    weekday_services = calendar[calendar['monday'] == 1]['service_id'].tolist()
    weekend_services = calendar[
        (calendar['saturday'] == 1) | (calendar['sunday'] == 1)
    ]['service_id'].tolist()
    
    trips = feed.trips.copy()
    weekday_trips = trips[trips['service_id'].isin(weekday_services)]
    weekend_trips = trips[trips['service_id'].isin(weekend_services)]
    
    stop_times = feed.stop_times[['trip_id', 'stop_id']].drop_duplicates()
    
    weekday_stops = stop_times[stop_times['trip_id'].isin(weekday_trips['trip_id'])]
    weekday_count = weekday_stops.groupby('stop_id').size().reset_index(name='weekday_trips')
    
    weekend_stops = stop_times[stop_times['trip_id'].isin(weekend_trips['trip_id'])]
    weekend_count = weekend_stops.groupby('stop_id').size().reset_index(name='weekend_trips')
    
    ratio = weekday_count.merge(weekend_count, on='stop_id', how='outer').fillna(0)
    ratio['weekend_ratio'] = np.where(
        ratio['weekday_trips'] > 0,
        ratio['weekend_trips'] / ratio['weekday_trips'],
        0
    )
    
    return ratio[['stop_id', 'weekend_ratio']]



def compute_population_density(tracts_gdf, config):
    """
    Compute population density for each tract.
    
    Uses ACS total population already downloaded in step 01.
    
    Parameters
    ----------
    tracts_gdf : GeoDataFrame
        Census tracts with area_sqkm
    config : dict
        Configuration dictionary
    
    Returns
    -------
    DataFrame
        Population density by tract
    """
    print("Computing population density...")
    
    acs_path = PROJECT_ROOT / "data" / "raw" / "acs" / "acs_tvi_variables.csv"
    
    if not acs_path.exists():
        print("  ⚠ ACS data not found, skipping population density")
        return pd.DataFrame({'GEOID': tracts_gdf['GEOID'], 'population_density': 0.0})
    
    acs_df = pd.read_csv(acs_path, dtype={'GEOID': str})
    
    # Get total population — B01001_001E or B03002_001E
    pop_col = None
    for col in ['B01001_001E', 'B03002_001E']:
        if col in acs_df.columns:
            pop_col = col
            break
    
    if pop_col is None:
        # Fallback: check if 01_download_data computed pct columns but also
        # stored a total_population column, or if the tract shapefile has pop
        if 'total_population' in acs_df.columns:
            pop_col = 'total_population'
        elif 'TOTPOP' in acs_df.columns:
            pop_col = 'TOTPOP'
        else:
            # Last resort: try to get population from tract shapefile attributes
            # (Census TIGER/Line shapefiles sometimes include ALAND/POP fields)
            print("  ⚠ No population column found in ACS data.")
            print("    Checked: B01001_001E, B03002_001E, total_population, TOTPOP")
            print("    → Make sure 01_download_data.py saves raw ACS totals alongside pct columns")
            return pd.DataFrame({'GEOID': tracts_gdf['GEOID'], 'population_density': 0.0})
    
    pop_df = acs_df[['GEOID', pop_col]].copy()
    pop_df.columns = ['GEOID', 'total_population']
    
    # Merge with tract areas
    density = tracts_gdf[['GEOID', 'area_sqkm']].merge(pop_df, on='GEOID', how='left')
    density['total_population'] = density['total_population'].fillna(0)
    density['population_density'] = np.where(
        density['area_sqkm'] > 0,
        density['total_population'] / density['area_sqkm'],
        0
    )
    
    print(f"  ✓ Population density: mean={density['population_density'].mean():.0f}/km², "
          f"max={density['population_density'].max():.0f}/km²")
    
    return density[['GEOID', 'population_density']]


def compute_intersection_density(tracts_gdf, config):
    """
    Compute intersection density (3+ way intersections per km²).
    
    Uses OSMnx to extract street network and count true intersections.
    Follows walkability research conventions (Ewing & Cervero, 2010).
    
    Parameters
    ----------
    tracts_gdf : GeoDataFrame
        Census tracts
    config : dict
        Configuration dictionary
    
    Returns
    -------
    DataFrame
        Intersection density by tract
    """
    print("Computing intersection density...")
    
    if not OSMNX_AVAILABLE:
        print("  ⚠ osmnx not available, skipping intersection density")
        return pd.DataFrame({'GEOID': tracts_gdf['GEOID'], 'intersection_density': 0.0})
    
    min_degree = config.get('built_environment', {}).get('intersection_density', {}).get('min_degree', 3)
    network_type = config.get('built_environment', {}).get('intersection_density', {}).get('network_type', 'walk')
    
    # Get study area boundary in WGS84 for OSMnx query
    study_area = tracts_gdf.dissolve().to_crs('EPSG:4326')
    boundary = study_area.geometry.iloc[0]
    
    print(f"  Downloading street network from OSM (this may take a minute)...")
    
    try:
        # Download street network for entire study area at once (faster than per-tract)
        G = ox.graph_from_polygon(boundary, network_type=network_type)
        
        # Convert to GeoDataFrame of nodes
        nodes_gdf = ox.graph_to_gdfs(G, edges=False)
        
        # Calculate degree for each node
        node_degrees = dict(G.degree())
        nodes_gdf['degree'] = nodes_gdf.index.map(node_degrees)
        
        # Filter to true intersections (degree >= min_degree)
        intersections = nodes_gdf[nodes_gdf['degree'] >= min_degree].copy()
        
        print(f"  ✓ Total nodes: {len(nodes_gdf)}, intersections (degree≥{min_degree}): {len(intersections)}")
        
        # Reproject intersections to match tracts CRS
        intersections = intersections.to_crs(tracts_gdf.crs)
        
        # Spatial join: intersections to tracts
        intersections_in_tracts = gpd.sjoin(
            intersections, 
            tracts_gdf[['GEOID', 'geometry']], 
            how='left', 
            predicate='within'
        )
        
        # Count intersections per tract
        int_counts = intersections_in_tracts.groupby('GEOID').size().reset_index(name='intersection_count')
        
        # Merge with areas and compute density
        int_density = tracts_gdf[['GEOID', 'area_sqkm']].merge(int_counts, on='GEOID', how='left')
        int_density['intersection_count'] = int_density['intersection_count'].fillna(0)
        int_density['intersection_density'] = np.where(
            int_density['area_sqkm'] > 0,
            int_density['intersection_count'] / int_density['area_sqkm'],
            0
        )
        
        print(f"  ✓ Intersection density: mean={int_density['intersection_density'].mean():.0f}/km², "
              f"max={int_density['intersection_density'].max():.0f}/km²")
        
        return int_density[['GEOID', 'intersection_density']]
    
    except Exception as e:
        print(f"  ✗ Error computing intersection density: {e}")
        return pd.DataFrame({'GEOID': tracts_gdf['GEOID'], 'intersection_density': 0.0})


def compute_land_use_mix(tracts_gdf, config):
    """
    Compute land use mix using Shannon entropy from OSM building/amenity tags.
    
    Entropy is normalized to 0-1 range:
    - 0 = single-use (all one category)
    - 1 = perfectly mixed (equal distribution across categories)
    
    Based on Frank & Pivo (1994) and Cervero & Kockelman (1997).
    
    Parameters
    ----------
    tracts_gdf : GeoDataFrame
        Census tracts
    config : dict
        Configuration dictionary
    
    Returns
    -------
    DataFrame
        Land use mix by tract
    """
    print("Computing land use mix...")
    
    if not OSMNX_AVAILABLE:
        print("  ⚠ osmnx not available, skipping land use mix")
        return pd.DataFrame({'GEOID': tracts_gdf['GEOID'], 'land_use_mix': 0.0})
    
    min_features = config.get('built_environment', {}).get('land_use_mix', {}).get('min_features', 15)
    
    # Get study area boundary in WGS84
    study_area = tracts_gdf.dissolve().to_crs('EPSG:4326')
    boundary = study_area.geometry.iloc[0]
    
    print(f"  Downloading building/amenity data from OSM...")

    def _fetch_osm_chunked(poly, tags, grid_n=4, depth=0, max_depth=2):
        """Fetch OSM features by splitting `poly` into grid_n x grid_n sub-polygons.
        Recurses on any chunk that still times out (up to max_depth levels).
        Large cities (Philadelphia) blow past single-query Overpass timeouts;
        spatial subdivision keeps each query small enough to complete.
        """
        from shapely.geometry import box as _box
        minx, miny, maxx, maxy = poly.bounds
        dx = (maxx - minx) / grid_n
        dy = (maxy - miny) / grid_n
        parts = []
        total_cells = grid_n * grid_n
        cell_idx = 0
        for i in range(grid_n):
            for j in range(grid_n):
                cell_idx += 1
                cell = _box(minx + i * dx, miny + j * dy,
                            minx + (i + 1) * dx, miny + (j + 1) * dy)
                sub = cell.intersection(poly)
                if sub.is_empty:
                    continue
                try:
                    part = ox.features_from_polygon(sub, tags=tags)
                    if len(part) > 0:
                        parts.append(part)
                    if depth == 0:
                        tag_key = list(tags.keys())[0]
                        print(f"    ✓ chunk {cell_idx}/{total_cells} ({tag_key}): {len(part)} features")
                except Exception as ce:
                    msg = str(ce).lower()
                    if 'timed out' in msg or 'timeout' in msg or '504' in msg or '429' in msg:
                        if depth < max_depth:
                            if depth == 0:
                                print(f"    ⚠ chunk {cell_idx}/{total_cells} timed out — recursing")
                            sub_parts = _fetch_osm_chunked(sub, tags, grid_n=grid_n,
                                                           depth=depth + 1, max_depth=max_depth)
                            if sub_parts is not None and len(sub_parts) > 0:
                                parts.append(sub_parts)
                        else:
                            if depth == 0:
                                print(f"    ✗ chunk {cell_idx}/{total_cells} gave up after {max_depth} sub-splits")
                    else:
                        # Non-timeout error (e.g. empty result for tags) — skip this chunk
                        pass
        if not parts:
            return None
        combined = pd.concat(parts, ignore_index=True)
        if 'geometry' in combined.columns:
            combined = combined.drop_duplicates(subset=['geometry'])
        return combined

    try:
        # Increase Overpass timeout for large cities (Philadelphia, Dallas)
        ox.settings.timeout = 900
        ox.settings.max_query_area_size = 50 * 1000 * 50 * 1000  # 50km x 50km

        # Download buildings — try whole polygon first, fall back to spatial chunks
        buildings = None
        try:
            buildings = ox.features_from_polygon(boundary, tags={'building': True})
        except Exception as e:
            msg = str(e).lower()
            if 'timed out' in msg or 'timeout' in msg or '504' in msg or '429' in msg:
                print(f"  ⚠ Overpass timeout on whole-city buildings; switching to 4x4 spatial chunks...")
                buildings = _fetch_osm_chunked(boundary, {'building': True}, grid_n=4)
                if buildings is None or len(buildings) == 0:
                    raise RuntimeError("spatial chunking returned no buildings")
            else:
                raise

        # Download amenities — try whole polygon first, fall back to spatial chunks
        try:
            amenities = ox.features_from_polygon(boundary, tags={'amenity': True})
        except Exception as e:
            msg = str(e).lower()
            if 'timed out' in msg or 'timeout' in msg or '504' in msg or '429' in msg:
                print(f"  ⚠ Overpass timeout on whole-city amenities; switching to 4x4 spatial chunks...")
                amenities = _fetch_osm_chunked(boundary, {'amenity': True}, grid_n=4)
                if amenities is None:
                    amenities = gpd.GeoDataFrame(columns=['geometry', 'amenity'], geometry='geometry', crs='EPSG:4326')
            else:
                raise

        print(f"  ✓ Downloaded {len(buildings)} buildings, {len(amenities)} amenities")
        
        # Classify into land use categories
        def classify_building(row):
            building_type = str(row.get('building', '')).lower()
            amenity_type = str(row.get('amenity', '')).lower()
            
            residential = ['residential', 'house', 'apartments', 'detached', 
                          'semidetached_house', 'terrace', 'dormitory']
            commercial = ['commercial', 'retail', 'office', 'supermarket', 
                         'shop', 'mall', 'restaurant', 'cafe', 'fast_food',
                         'bar', 'pub', 'bank', 'pharmacy']
            industrial = ['industrial', 'warehouse', 'factory', 'manufacture']
            institutional = ['school', 'university', 'college', 'hospital', 
                           'clinic', 'church', 'place_of_worship', 'library',
                           'public', 'government', 'civic', 'fire_station',
                           'police', 'community_centre', 'social_facility']
            
            if building_type in residential or amenity_type in residential:
                return 'residential'
            elif building_type in commercial or amenity_type in commercial:
                return 'commercial'
            elif building_type in industrial or amenity_type in industrial:
                return 'industrial'
            elif building_type in institutional or amenity_type in institutional:
                return 'institutional'
            elif building_type == 'yes':
                return 'residential'  # Default untagged buildings to residential
            else:
                return 'other'
        
        # Combine buildings and amenities
        all_features = pd.concat([
            buildings[['geometry']].assign(building=buildings.get('building', 'unknown')),
            amenities[['geometry']].assign(amenity=amenities.get('amenity', 'unknown'))
        ], ignore_index=True)
        
        all_features['land_use'] = all_features.apply(classify_building, axis=1)
        
        # Convert to points (centroids) for spatial join
        all_features_gdf = gpd.GeoDataFrame(all_features, geometry='geometry', crs='EPSG:4326')
        all_features_gdf = all_features_gdf.to_crs(tracts_gdf.crs)
        
        # Use representative point instead of centroid for polygons
        all_features_gdf['point_geom'] = all_features_gdf.geometry.representative_point()
        points_gdf = all_features_gdf.set_geometry('point_geom')
        
        # Spatial join to tracts
        features_in_tracts = gpd.sjoin(
            points_gdf[['point_geom', 'land_use']].rename(columns={'point_geom': 'geometry'}).set_geometry('geometry'),
            tracts_gdf[['GEOID', 'geometry']],
            how='left',
            predicate='within'
        )
        
        # Compute Shannon entropy per tract
        categories = ['residential', 'commercial', 'industrial', 'institutional', 'other']
        n_categories = len(categories)
        
        land_use_results = []
        
        for geoid in tracts_gdf['GEOID'].values:
            tract_features = features_in_tracts[features_in_tracts['GEOID'] == geoid]
            n_features = len(tract_features)
            
            if n_features < min_features:
                # Too few features for reliable entropy — use NaN (will be filled later)
                land_use_results.append({
                    'GEOID': geoid,
                    'land_use_mix': np.nan,
                    'land_use_feature_count': n_features
                })
                continue
            
            # Count features per category
            counts = tract_features['land_use'].value_counts()
            proportions = counts / counts.sum()
            
            # Shannon entropy: H = -Σ(p_i × ln(p_i))
            entropy = 0
            for cat in categories:
                p = proportions.get(cat, 0)
                if p > 0:
                    entropy -= p * np.log(p)
            
            # Normalize by ln(n) to get 0-1 range
            max_entropy = np.log(n_categories)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            land_use_results.append({
                'GEOID': geoid,
                'land_use_mix': normalized_entropy,
                'land_use_feature_count': n_features
            })
        
        results_df = pd.DataFrame(land_use_results)
        
        # Flag tracts with insufficient data rather than filling with median
        n_insufficient = results_df['land_use_mix'].isna().sum()
        
        if n_insufficient > 0:
            print(f"  ⚠ {n_insufficient} tracts had <{min_features} features — flagged as insufficient data")
            # Set to 0 (low development / data-sparse area) rather than median
            # This avoids artificially inflating built environment scores
            results_df['land_use_insufficient'] = results_df['land_use_mix'].isna().astype(int)
            results_df['land_use_mix'] = results_df['land_use_mix'].fillna(0.0)
        
        print(f"  ✓ Land use mix: mean={results_df['land_use_mix'].mean():.2f}, "
              f"max={results_df['land_use_mix'].max():.2f}")
        
        return results_df[['GEOID', 'land_use_mix']]
    
    except Exception as e:
        print(f"  ✗ Error computing land use mix: {e}")
        return pd.DataFrame({'GEOID': tracts_gdf['GEOID'], 'land_use_mix': 0.0})


def compute_sidewalk_index(tracts_gdf, config):
    """
    Compute sidewalk infrastructure density per tract from OSM.

    Queries three common OSM sidewalk representation styles:
      1. highway=footway/pedestrian/path/steps  (dedicated footway geometries)
      2. footway=sidewalk/crossing               (sidewalk sub-tagged footways)
      3. sidewalk=yes/both/left/right             (sidewalk tags on road segments)

    Returns total sidewalk/footway length (meters) / tract area (km²).

    Parameters
    ----------
    tracts_gdf : GeoDataFrame
        Census tracts (projected CRS with area_sqkm column)
    config : dict
        Configuration dictionary

    Returns
    -------
    DataFrame
        Sidewalk index by tract (GEOID, sidewalk_index)
    """
    print("Computing sidewalk index...")

    if not OSMNX_AVAILABLE:
        print("  ⚠ osmnx not available, skipping sidewalk index")
        return pd.DataFrame({'GEOID': tracts_gdf['GEOID'], 'sidewalk_index': 0.0})

    sw_config = config.get('built_environment', {}).get('sidewalk_index', {})
    osm_tags = sw_config.get('osm_tags', {
        'highway': ['footway', 'pedestrian', 'path', 'steps'],
        'footway': ['sidewalk', 'crossing'],
        'sidewalk': ['yes', 'both', 'left', 'right'],
    })

    # Get study area boundary in WGS84
    study_area = tracts_gdf.dissolve().to_crs('EPSG:4326')
    boundary = study_area.geometry.iloc[0]

    print(f"  Downloading sidewalk/footway features from OSM...")

    all_features = []
    for tag_key, tag_values in osm_tags.items():
        for tag_value in tag_values:
            try:
                features = ox.features_from_polygon(boundary, tags={tag_key: tag_value})
                if len(features) > 0:
                    # Keep only linear features (LineString, MultiLineString)
                    lines = features[features.geometry.type.isin(
                        ['LineString', 'MultiLineString']
                    )][['geometry']].copy()
                    if len(lines) > 0:
                        lines['source_tag'] = f"{tag_key}={tag_value}"
                        all_features.append(lines)
                        print(f"    ✓ {tag_key}={tag_value}: {len(lines)} segments")
            except Exception as e:
                print(f"    ⚠ {tag_key}={tag_value}: {e}")

    if not all_features:
        print("  ⚠ No sidewalk features found — sidewalk_index will be 0")
        return pd.DataFrame({'GEOID': tracts_gdf['GEOID'], 'sidewalk_index': 0.0})

    sidewalks = pd.concat(all_features, ignore_index=True)
    sidewalks = gpd.GeoDataFrame(sidewalks, geometry='geometry', crs='EPSG:4326')

    # Deduplicate — same segment can match multiple tag queries
    before = len(sidewalks)
    sidewalks = sidewalks.drop_duplicates(subset=['geometry'])
    after = len(sidewalks)
    if before > after:
        print(f"  Deduplicated: {before} → {after} segments")

    print(f"  Total sidewalk/footway segments: {len(sidewalks)}")

    # Reproject to match tracts CRS for accurate length measurement
    sidewalks_proj = sidewalks.to_crs(tracts_gdf.crs)

    # Determine if CRS uses feet (for unit conversion)
    crs_units = tracts_gdf.crs.axis_info[0].unit_name if tracts_gdf.crs.axis_info else 'metre'
    is_feet = any(u in crs_units.lower() for u in ['foot', 'feet', 'ft'])

    # Spatial join: assign sidewalk segments to tracts
    print("  Computing sidewalk length per tract...")
    try:
        joined = gpd.sjoin(sidewalks_proj, tracts_gdf[['GEOID', 'geometry']],
                           how='inner', predicate='intersects')
    except Exception as e:
        print(f"  ✗ Spatial join failed: {e}")
        return pd.DataFrame({'GEOID': tracts_gdf['GEOID'], 'sidewalk_index': 0.0})

    # Compute segment lengths in meters
    if is_feet:
        joined['length_m'] = joined.geometry.length * 0.3048
    else:
        joined['length_m'] = joined.geometry.length

    # Sum length per tract, normalize by area
    tract_lengths = joined.groupby('GEOID')['length_m'].sum().reset_index()
    tract_lengths.columns = ['GEOID', 'sidewalk_length_m']

    result = tracts_gdf[['GEOID', 'area_sqkm']].merge(tract_lengths, on='GEOID', how='left')
    result['sidewalk_length_m'] = result['sidewalk_length_m'].fillna(0)

    # --- Road-class proxy for tracts with zero/missing sidewalk data ---
    # OSM sidewalk tagging is inconsistent: some mappers draw separate footway
    # geometries, others tag roads with sidewalk=yes, and many don't tag at all.
    # For tracts with no sidewalk features, we estimate sidewalk presence from
    # road segments by class using FHWA empirical sidewalk probability weights.
    #
    # Source: FHWA Pedestrian Safety Guide (Publication No. FHWA-SA-07-016)
    # and National Household Travel Survey sidewalk presence by road class.
    zero_tracts = result[result['sidewalk_length_m'] == 0]['GEOID'].tolist()

    if zero_tracts and len(zero_tracts) > 0:
        print(f"\n  Road-class proxy for {len(zero_tracts)} tracts with no OSM sidewalk data...")

        # FHWA-based sidewalk probability by OSM highway class
        # Higher for residential/urban streets, lower for highways
        sidewalk_probability = {
            'residential': 0.80,
            'tertiary': 0.70,
            'secondary': 0.55,
            'primary': 0.30,
            'trunk': 0.10,
            'motorway': 0.00,
            'unclassified': 0.50,
            'living_street': 0.95,
            'service': 0.40,
            'tertiary_link': 0.65,
            'secondary_link': 0.50,
            'primary_link': 0.25,
        }

        # Query road network for the study area (reuse boundary from above)
        try:
            road_tags = list(sidewalk_probability.keys())
            road_features = []
            for road_class in road_tags:
                try:
                    feats = ox.features_from_polygon(
                        boundary, tags={'highway': road_class}
                    )
                    if len(feats) > 0:
                        lines = feats[feats.geometry.type.isin(
                            ['LineString', 'MultiLineString']
                        )][['geometry']].copy()
                        if len(lines) > 0:
                            lines['highway_class'] = road_class
                            road_features.append(lines)
                except Exception:
                    pass

            if road_features:
                roads_gdf = pd.concat(road_features, ignore_index=True)
                roads_gdf = gpd.GeoDataFrame(roads_gdf, geometry='geometry', crs='EPSG:4326')
                roads_proj = roads_gdf.to_crs(tracts_gdf.crs)

                # Spatial join roads to zero-coverage tracts only
                zero_tracts_gdf = tracts_gdf[tracts_gdf['GEOID'].isin(zero_tracts)][['GEOID', 'geometry']]
                roads_in_tracts = gpd.sjoin(roads_proj, zero_tracts_gdf,
                                            how='inner', predicate='intersects')

                # Compute proxy: road_length × sidewalk_probability
                if is_feet:
                    roads_in_tracts['length_m'] = roads_in_tracts.geometry.length * 0.3048
                else:
                    roads_in_tracts['length_m'] = roads_in_tracts.geometry.length

                roads_in_tracts['proxy_length'] = (
                    roads_in_tracts['length_m'] *
                    roads_in_tracts['highway_class'].map(sidewalk_probability).fillna(0.3)
                )

                proxy_lengths = roads_in_tracts.groupby('GEOID')['proxy_length'].sum().reset_index()
                proxy_lengths.columns = ['GEOID', 'proxy_sidewalk_m']

                # Apply proxy values to zero tracts
                n_filled = 0
                for _, row in proxy_lengths.iterrows():
                    mask = result['GEOID'] == row['GEOID']
                    if mask.any():
                        result.loc[mask, 'sidewalk_length_m'] = row['proxy_sidewalk_m']
                        n_filled += 1

                print(f"    ✓ Estimated sidewalk coverage for {n_filled} tracts using road-class proxy")

                # Show proxy vs direct comparison for transparency
                if n_filled > 0:
                    proxy_vals = result[result['GEOID'].isin(proxy_lengths['GEOID'])]['sidewalk_length_m']
                    direct_vals = result[~result['GEOID'].isin(zero_tracts)]['sidewalk_length_m']
                    print(f"    Proxy tracts mean length: {proxy_vals.mean():.0f}m "
                          f"(vs direct-mapped mean: {direct_vals.mean():.0f}m)")
            else:
                print("    ⚠ No road features found for proxy — keeping zeros")

        except Exception as e:
            print(f"    ⚠ Road-class proxy failed: {e} — keeping zeros")

    # Mark which tracts used the proxy (for data quality transparency)
    result['sidewalk_data_source'] = np.where(
        result['GEOID'].isin(zero_tracts), 'road_class_proxy', 'osm_direct'
    )

    result['sidewalk_index'] = np.where(
        result['area_sqkm'] > 0,
        result['sidewalk_length_m'] / result['area_sqkm'],
        0
    )

    print(f"  ✓ Sidewalk index (m/km²): mean={result['sidewalk_index'].mean():.0f}, "
          f"median={result['sidewalk_index'].median():.0f}, "
          f"max={result['sidewalk_index'].max():.0f}")
    n_zero = (result['sidewalk_index'] == 0).sum()
    if n_zero > 0:
        print(f"  ⚠ {n_zero} tracts still have zero sidewalk coverage after proxy")
    n_proxy = (result['sidewalk_data_source'] == 'road_class_proxy').sum()
    n_direct = (result['sidewalk_data_source'] == 'osm_direct').sum()
    print(f"  Data sources: {n_direct} direct OSM, {n_proxy} road-class proxy")

    return result[['GEOID', 'sidewalk_index', 'sidewalk_data_source']]


# ============================================================
# NOTE: CPTA score computation has been moved to 02d_compute_cpta.py
# This script (02) now only saves raw supply metrics.
# ============================================================


# ============================================================
# Main Pipeline
# ============================================================

def main():
    """Main function to compute all supply metrics."""
    print("=" * 60)
    print("Transit Desert Pipeline: Supply Metrics (CPTA)")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Paths
    gtfs_path = PROJECT_ROOT / "data" / "raw" / "gtfs"
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "-" * 40)
    print("Loading data...")
    print("-" * 40)
    
    feed = load_gtfs(gtfs_path)
    print_gtfs_summary(feed)
    
    tracts_gdf = load_tracts(config)
    stops_gdf = get_stops_gdf(feed, config)
    
    print(f"\n  Tracts: {len(tracts_gdf)}")
    
    # ---- Transit Service Metrics ----
    print("\n" + "-" * 40)
    print("Computing transit service metrics...")
    print("-" * 40)
    
    stop_metrics = compute_stop_routes_frequency(feed, config)
    stop_metrics = compute_stop_connectivity_welch(stop_metrics)
    
    span = compute_span_of_service(feed, stop_metrics)
    stop_metrics = stop_metrics.merge(span, on='stop_id', how='left')
    
    weekend = compute_weekend_ratio(feed)
    stop_metrics = stop_metrics.merge(weekend, on='stop_id', how='left')
    
    stop_metrics.to_csv(output_dir / "stop_metrics.csv", index=False)
    print(f"\n  ✓ Saved stop metrics to {output_dir / 'stop_metrics.csv'}")
    
    # Aggregate to tracts
    print("\n" + "-" * 40)
    print("Aggregating to tract level...")
    print("-" * 40)
    
    tract_metrics = aggregate_stops_to_tracts(stops_gdf, stop_metrics, tracts_gdf, config)
    
    # Route coverage (with mode-specific thresholds)
    route_coverage = compute_route_coverage(feed, stops_gdf, tracts_gdf, config)
    tract_metrics = tract_metrics.merge(route_coverage, on='GEOID', how='left')
    
    # Walking access (with bus/rail differentiation)
    walking_access = compute_walking_access(stops_gdf, tracts_gdf, config)
    tract_metrics = tract_metrics.merge(walking_access, on='GEOID', how='left')
    
    # Aggregate span and weekend ratio to tract level
    stops_gdf_metrics = stops_gdf.merge(stop_metrics[['stop_id', 'span_hours', 'weekend_ratio']], on='stop_id')
    stops_in_tracts = gpd.sjoin(stops_gdf_metrics, tracts_gdf[['GEOID', 'geometry']], how='left', predicate='within')
    
    tract_span = stops_in_tracts.groupby('GEOID').agg({
        'span_hours': 'mean',
        'weekend_ratio': 'mean'
    }).reset_index()
    
    tract_metrics = tract_metrics.merge(tract_span, on='GEOID', how='left')
    
    # ---- Built Environment Metrics ----
    print("\n" + "-" * 40)
    print("Computing built environment metrics...")
    print("-" * 40)
    
    # Population density
    pop_density = compute_population_density(tracts_gdf, config)
    tract_metrics = tract_metrics.merge(pop_density, on='GEOID', how='left')
    
    # Intersection density
    int_density = compute_intersection_density(tracts_gdf, config)
    tract_metrics = tract_metrics.merge(int_density, on='GEOID', how='left')
    
    # Land use mix — always use OSM Shannon entropy
    # (EPA SLD removed: incomplete city coverage, e.g. Philadelphia returns zeros)
    lum = compute_land_use_mix(tracts_gdf, config)
    lum['land_use_source'] = 'OSM_Shannon'
    
    tract_metrics = tract_metrics.merge(
        lum[['GEOID', 'land_use_mix']], on='GEOID', how='left'
    )
    
    # Sidewalk index
    sw_config = config.get('built_environment', {}).get('sidewalk_index', None)
    if sw_config:
        sidewalk = compute_sidewalk_index(tracts_gdf, config)
        tract_metrics = tract_metrics.merge(sidewalk, on='GEOID', how='left')
    else:
        print("  ⚠ No sidewalk_index config — skipping")
        tract_metrics['sidewalk_index'] = 0.0
    
    # ---- Finalize ----
    tract_metrics = tract_metrics.fillna(0)
    
    # Ensure all tracts are included
    all_tracts = tracts_gdf[['GEOID']].copy()
    tract_metrics = all_tracts.merge(tract_metrics, on='GEOID', how='left').fillna(0)
    
    # NOTE: CPTA score is NOT computed here.
    # It is computed in 02d_compute_cpta.py after optional 02b/02c steps
    # have appended their columns to supply_metrics.csv.
    
    # ---- Save Results ----
    print("\n" + "-" * 40)
    print("Saving results...")
    print("-" * 40)
    
    tract_metrics.to_csv(output_dir / "supply_metrics.csv", index=False)
    print(f"  ✓ Saved to {output_dir / 'supply_metrics.csv'}")
    
    tract_metrics_gdf = tracts_gdf.merge(tract_metrics, on='GEOID')
    tract_metrics_gdf.to_file(output_dir / "supply_metrics.gpkg", driver='GPKG')
    print(f"  ✓ Saved to {output_dir / 'supply_metrics.gpkg'}")
    
    # Summary
    print("\n" + "-" * 40)
    print("Summary Statistics")
    print("-" * 40)
    
    print(f"\n  Tracts analyzed: {len(tract_metrics)}")
    
    # Print individual metric stats
    print(f"\n  Individual metric means:")
    for col in ['connectivity_density', 'route_coverage', 'freq_am_peak', 
                'freq_midday', 'freq_pm_peak', 'freq_evening',
                'walking_access_pct', 'span_hours', 'weekend_ratio',
                'population_density', 'intersection_density', 'land_use_mix',
                'sidewalk_index']:
        if col in tract_metrics.columns:
            print(f"    {col}: {tract_metrics[col].mean():.2f}")
    
    print("\n" + "=" * 60)
    print("Supply metrics complete! (raw metrics only — no CPTA yet)")
    print("Next steps:")
    print("  python src/02b_compute_jobs_accessibility.py  (optional)")
    print("  python src/02c_compute_poi_accessibility.py   (optional)")
    print("  python src/02d_compute_cpta.py                (computes CPTA score)")
    print("=" * 60)


if __name__ == "__main__":
    main()