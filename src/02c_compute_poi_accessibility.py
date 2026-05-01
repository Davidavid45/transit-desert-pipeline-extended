"""
02c_compute_poi_accessibility.py
Compute POI (Point of Interest) accessibility using category-specific decay models.

For each POI category, accessibility is computed as a gravity model:

    A_i^k = Σ exp(-β_k × t_ij)

Where β_k is the category-specific decay parameter:
  - Healthcare/grocery: steep β (~0.12) — only nearby POIs matter (~15-20 min)
  - Parks/clinics: moderate β (~0.08) — ~30 min effective range
  - Education/hospitals: gentle β (~0.05) — ~45 min effective range

The final POI accessibility score is the weighted sum across all categories:

    POI_i = Σ_k  w_k × A_i^k

8 POI categories (configurable in Config.yaml):
  healthcare, grocery, education, parks, food_drink,
  shopping, entertainment, personal_services

Prerequisites:
- r5py installed: pip install r5py
- Java 11+ installed
- GTFS data downloaded (run 01_download_data.py first)
- OpenStreetMap PBF file for the region
- osmnx installed: pip install osmnx

Usage:
    python src/02c_compute_poi_accessibility.py

Outputs:
    - data/processed/poi_accessibility.csv  (per-category + composite scores)
    - data/processed/pois.gpkg              (downloaded POI locations)
    - Updates supply_metrics.csv with poi_accessibility column
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
from state_osm import get_osm_pbf_path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gtfs_utils import is_valid_gtfs_zip

try:
    import r5py
    R5PY_AVAILABLE = True
except ImportError:
    R5PY_AVAILABLE = False
    print("⚠ r5py not installed. Run: pip install r5py")

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    print("⚠ osmnx not installed. Run: pip install osmnx")


def load_config():
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_poi_categories(config):
    """Get POI category definitions from config with fallback defaults."""
    poi_config = config.get('accessibility', {}).get('poi_accessibility', {})
    categories = poi_config.get('categories', None)
    if categories:
        return categories
    print("  ⚠ No POI categories in config — using 3-category defaults")
    return {
        'healthcare': {
            'osm_tags': {'amenity': ['hospital', 'clinic', 'doctors', 'pharmacy']},
            'decay_beta': 0.12, 'weight': 0.40, 'label': 'Healthcare'
        },
        'grocery': {
            'osm_tags': {'shop': ['supermarket', 'grocery', 'convenience']},
            'decay_beta': 0.12, 'weight': 0.35, 'label': 'Grocery'
        },
        'education': {
            'osm_tags': {'amenity': ['school', 'college', 'university', 'library']},
            'decay_beta': 0.05, 'weight': 0.25, 'label': 'Education'
        },
    }


def download_pois(config, tracts_gdf):
    """Download POIs from OpenStreetMap for all configured categories."""
    print("Downloading POIs from OpenStreetMap...")
    if not OSMNX_AVAILABLE:
        raise ImportError("osmnx is required for POI download")

    study_area = tracts_gdf.dissolve().to_crs('EPSG:4326')
    boundary = study_area.geometry.iloc[0]
    categories = get_poi_categories(config)
    all_pois = []

    for cat_name, cat_def in categories.items():
        osm_tags = cat_def.get('osm_tags', {})
        label = cat_def.get('label', cat_name)
        print(f"  Downloading {label} POIs...")
        cat_count = 0

        for tag_key, tag_values in osm_tags.items():
            for tag_value in tag_values:
                try:
                    features = ox.features_from_polygon(boundary, tags={tag_key: tag_value})
                    if len(features) > 0:
                        points = features[['geometry']].copy()
                        points['geometry'] = points.geometry.representative_point()
                        points['category'] = cat_name
                        all_pois.append(points)
                        cat_count += len(features)
                        print(f"    ✓ {tag_key}={tag_value}: {len(features)}")
                except Exception as e:
                    print(f"    ⚠ {tag_key}={tag_value}: {e}")

        if cat_count == 0:
            print(f"    ⚠ No POIs found for {label}")

    if not all_pois:
        print("  ✗ No POIs found!")
        return gpd.GeoDataFrame(columns=['geometry', 'category', 'poi_id'], crs='EPSG:4326')

    pois_gdf = pd.concat(all_pois, ignore_index=True)
    pois_gdf = gpd.GeoDataFrame(pois_gdf, geometry='geometry', crs='EPSG:4326')

    before = len(pois_gdf)
    pois_gdf = pois_gdf.drop_duplicates(subset=['geometry', 'category'])
    after = len(pois_gdf)
    if before > after:
        print(f"\n  Deduplicated: {before} → {after} POIs")

    pois_gdf['poi_id'] = range(len(pois_gdf))
    pois_gdf['id'] = pois_gdf['poi_id']

    print(f"\n  Total POIs: {len(pois_gdf)}")
    for cat, count in pois_gdf['category'].value_counts().items():
        cat_def = categories.get(cat, {})
        print(f"    {cat}: {count} (β={cat_def.get('decay_beta', '?')}, w={cat_def.get('weight', '?')})")

    return pois_gdf


def load_tract_centroids():
    """Load tract centroids for travel time calculation."""
    tracts_path = PROJECT_ROOT / "data" / "raw" / "census" / "study_area_tracts.shp"
    if not tracts_path.exists():
        raise FileNotFoundError(f"Tract shapefile not found: {tracts_path}")

    gdf = gpd.read_file(tracts_path).to_crs('EPSG:4326')
    centroids = gdf.copy()
    centroids['geometry'] = gdf.geometry.centroid
    centroids['lat'] = centroids.geometry.y
    centroids['lon'] = centroids.geometry.x
    centroids['id'] = centroids['GEOID']

    print(f"  ✓ Loaded {len(centroids)} tract centroids")
    return centroids[['GEOID', 'geometry', 'lat', 'lon', 'id']]


def compute_travel_time_matrix(origins, destinations, gtfs_path, osm_path, config):
    """Compute travel time matrix from tract centroids to all POIs."""
    if not R5PY_AVAILABLE:
        raise ImportError("r5py is required")

    print("  Building transport network...")

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

    poi_config = config.get('accessibility', {}).get('poi_accessibility', {})
    dep_time_str = poi_config.get('departure_time', '10:00:00')
    dep_h, dep_m, dep_s = map(int, dep_time_str.split(':'))

    analysis_date_str = config.get('gtfs', {}).get('analysis_date', '20241015')
    try:
        dep_date = datetime.strptime(analysis_date_str, '%Y%m%d').date()
    except ValueError:
        dep_date = date(2024, 10, 15)

    departure_time = datetime.combine(dep_date, time(dep_h, dep_m, dep_s))
    max_minutes = poi_config.get('max_travel_time_minutes', 60)
    max_time = timedelta(minutes=max_minutes)

    print(f"  Departure: {departure_time}")
    print(f"  Max travel time: {max_minutes} min")
    print(f"  Origins: {len(origins)}, Destinations: {len(destinations)}")

    travel_time_matrix = r5py.TravelTimeMatrix(
        transport_network,
        origins=origins,
        destinations=destinations,
        departure=departure_time,
        departure_time_window=timedelta(hours=1),
        transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
        max_time=max_time
    )

    result = pd.DataFrame(travel_time_matrix)
    print(f"  ✓ Computed {len(result)} OD pairs")
    return result


def calculate_poi_accessibility(travel_times, pois_gdf, config):
    """
    Calculate category-specific decay-weighted POI accessibility.

    For each category k:  A_i^k = Σ exp(-β_k × t_ij)
    Final:                POI_i = Σ_k  w_k × A_i^k
    """
    categories = get_poi_categories(config)

    valid_times = travel_times.dropna(subset=['travel_time']).copy()
    poi_lookup = pois_gdf[['poi_id', 'category']].rename(columns={'poi_id': 'to_id'})
    valid_times = valid_times.merge(poi_lookup, on='to_id', how='left')

    all_origins = travel_times['from_id'].unique()
    category_scores = {}

    print(f"\n  Category-specific decay scores:")
    for cat_name, cat_def in categories.items():
        beta = cat_def.get('decay_beta', 0.08)
        weight = cat_def.get('weight', 0.10)
        label = cat_def.get('label', cat_name)

        cat_times = valid_times[valid_times['category'] == cat_name].copy()

        if len(cat_times) == 0:
            print(f"    {label}: no reachable POIs")
            category_scores[cat_name] = pd.Series(0.0, index=all_origins)
            continue

        cat_times['decay_weight'] = np.exp(-beta * cat_times['travel_time'])
        cat_score = cat_times.groupby('from_id')['decay_weight'].sum()
        cat_score = cat_score.reindex(all_origins, fill_value=0.0)
        category_scores[cat_name] = cat_score

        print(f"    {label} (β={beta}, w={weight}): mean={cat_score.mean():.2f}, max={cat_score.max():.2f}")

    # Build result
    result = pd.DataFrame({'GEOID': all_origins})
    for cat_name in categories.keys():
        result[f'poi_{cat_name}'] = category_scores.get(
            cat_name, pd.Series(0.0, index=all_origins)
        ).values

    # Weighted sum
    weighted_sum = np.zeros(len(result))
    total_weight = 0
    for cat_name, cat_def in categories.items():
        w = cat_def.get('weight', 0.10)
        weighted_sum += w * result[f'poi_{cat_name}'].values
        total_weight += w

    result['poi_accessibility'] = weighted_sum / total_weight if total_weight > 0 else 0.0

    weight_sum = sum(c.get('weight', 0) for c in categories.values())
    if abs(weight_sum - 1.0) > 0.01:
        print(f"\n  ⚠ Category weights sum to {weight_sum:.2f}, not 1.0")

    print(f"\n  Final POI accessibility (weighted composite):")
    print(f"    Mean:   {result['poi_accessibility'].mean():.2f}")
    print(f"    Median: {result['poi_accessibility'].median():.2f}")
    print(f"    Max:    {result['poi_accessibility'].max():.2f}")

    # Show decay at key times for the most/least steep categories
    betas = {cat_def.get('label', k): cat_def.get('decay_beta', 0.08)
             for k, cat_def in categories.items()}
    print(f"\n  Decay weights at key travel times:")
    print(f"    {'Time':>6s}", end="")
    for label in betas:
        print(f"  {label[:12]:>12s}", end="")
    print()
    for t in [5, 10, 15, 20, 30, 45, 60]:
        print(f"    {t:4d} min", end="")
        for label, beta in betas.items():
            print(f"  {np.exp(-beta * t):12.3f}", end="")
        print()

    return result


def update_supply_metrics(poi_results):
    """Update supply_metrics.csv with composite poi_accessibility column."""
    supply_path = PROJECT_ROOT / "data" / "processed" / "supply_metrics.csv"
    if not supply_path.exists():
        print("  ⚠ supply_metrics.csv not found. Run 02_compute_supply.py first.")
        return

    supply_df = pd.read_csv(supply_path, dtype={'GEOID': str})
    if 'poi_accessibility' in supply_df.columns:
        supply_df = supply_df.drop(columns=['poi_accessibility'])

    supply_df = supply_df.merge(
        poi_results[['GEOID', 'poi_accessibility']], on='GEOID', how='left'
    )
    supply_df['poi_accessibility'] = supply_df['poi_accessibility'].fillna(0)

    supply_df.to_csv(supply_path, index=False)
    print(f"  ✓ Updated supply_metrics.csv with poi_accessibility column")
    print(f"  → Run 02d_compute_cpta.py to incorporate into CPTA score")


def main():
    print("=" * 60)
    print("Transit Desert Pipeline: POI Accessibility (r5py)")
    print("=" * 60)

    if not R5PY_AVAILABLE:
        print("\n⚠ r5py not installed. pip install r5py (+ Java 11+)")
        sys.exit(1)
    if not OSMNX_AVAILABLE:
        print("\n⚠ osmnx not installed. pip install osmnx")
        sys.exit(1)

    config = load_config()
    raw_dir = PROJECT_ROOT / "data" / "raw"
    external_dir = PROJECT_ROOT / "data" / "external"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    external_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download POIs
    print("\n" + "-" * 40)
    print("Step 1: Download POIs from OSM")
    print("-" * 40)

    tracts_path = raw_dir / "census" / "study_area_tracts.shp"
    if not tracts_path.exists():
        print(f"  ✗ Tracts not found: {tracts_path}")
        sys.exit(1)
    tracts_gdf = gpd.read_file(tracts_path)
    pois_gdf = download_pois(config, tracts_gdf)
    if len(pois_gdf) == 0:
        print("  ✗ No POIs. Cannot proceed.")
        sys.exit(1)
    pois_gdf.to_file(processed_dir / "pois.gpkg", driver='GPKG')
    print(f"  ✓ Saved POIs to {processed_dir / 'pois.gpkg'}")

    # Step 2: Check OSM PBF
    print("\n" + "-" * 40)
    print("Step 2: Check OSM Data")
    print("-" * 40)
    osm_path = get_osm_pbf_path(config, external_dir / "osm")
    if not osm_path.exists():
        print(f"  ✗ OSM PBF not found: {osm_path}")
        print("  → Run 02b first or download from Geofabrik")
        sys.exit(1)
    print(f"  ✓ OSM PBF: {osm_path}")

    # Step 3: Load centroids
    print("\n" + "-" * 40)
    print("Step 3: Load Tract Centroids")
    print("-" * 40)
    centroids = load_tract_centroids()

    # Step 4: Compute travel times
    print("\n" + "-" * 40)
    print("Step 4: Compute Travel Times to POIs")
    print("-" * 40)
    gtfs_path = raw_dir / "gtfs"
    gtfs_zips = list(gtfs_path.glob("*.zip")) if gtfs_path.is_dir() else []
    if not gtfs_zips:
        print(f"  ✗ No GTFS zips in {gtfs_path}")
        sys.exit(1)
    print(f"  Found {len(gtfs_zips)} GTFS file(s)")

    try:
        travel_times = compute_travel_time_matrix(
            centroids, pois_gdf, gtfs_path, osm_path, config
        )
    except Exception as e:
        print(f"  ✗ Error: {e}")
        sys.exit(1)

    # Step 5: Calculate decay-weighted accessibility
    print("\n" + "-" * 40)
    print("Step 5: Category-Specific POI Accessibility")
    print("-" * 40)
    poi_results = calculate_poi_accessibility(travel_times, pois_gdf, config)
    poi_results.to_csv(processed_dir / "poi_accessibility.csv", index=False)
    print(f"\n  ✓ Saved to {processed_dir / 'poi_accessibility.csv'}")

    # Step 6: Update supply metrics
    print("\n" + "-" * 40)
    print("Step 6: Update Supply Metrics")
    print("-" * 40)
    update_supply_metrics(poi_results)

    print("\n" + "=" * 60)
    print("POI accessibility complete!")
    print("Next: python src/02d_compute_cpta.py")
    print("=" * 60)


if __name__ == "__main__":
    main()