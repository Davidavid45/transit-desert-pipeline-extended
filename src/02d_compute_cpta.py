"""
02d_compute_cpta.py
Compute the Composite Public Transit Accessibility (CPTA) score.

This script runs AFTER 02, 02b (optional), and 02c (optional).
It reads the supply_metrics.csv file, which may contain columns added by:
  - 02_compute_supply.py:  transit service + built environment metrics
  - 02b_compute_jobs_accessibility.py:  jobs_accessibility column
  - 02c_compute_poi_accessibility.py:   poi_accessibility column

The CPTA score is a category-weighted composite:
  Transit Service (50%):  stop connectivity, route coverage, 4× frequency,
                          span of service, weekend ratio
  Accessibility (25%):    walking access, jobs accessibility, POI accessibility
  Built Environment (25%): population density, intersection density,
                           land use mix, sidewalk index

If optional columns (jobs_accessibility, poi_accessibility) are missing,
the category weights are renormalized across available categories.

Usage:
    python src/02d_compute_cpta.py

Inputs:
    - data/processed/supply_metrics.csv

Outputs:
    - data/processed/supply_metrics.csv       (updated with CPTA columns)
    - data/processed/supply_metrics.gpkg       (updated GeoPackage)
    - data/processed/cpta_correlation_matrix.csv
"""

import sys
import yaml
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import zscore

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    """Load configuration from YAML file."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_cpta_score(tract_metrics, config):
    """
    Compute final CPTA score using category-weighted z-scores.
    
    Steps:
      1. Z-score each available metric
      2. Compute mean z-score within each category
      3. Weighted sum of category means (50/25/25 default)
      4. Normalize to 0-100 scale
    
    If a category has no available metrics (e.g. accessibility without
    02b/02c), its weight is redistributed proportionally to the
    remaining categories.
    
    Parameters
    ----------
    tract_metrics : DataFrame
        Tract-level supply metrics from supply_metrics.csv
    config : dict
        Configuration dictionary
    
    Returns
    -------
    DataFrame
        Tract metrics with CPTA, CPTA_normalized, z-score, and category columns
    """
    print("Computing CPTA scores...")
    
    # Define metric categories — read from config, with hardcoded fallback.
    # Config key names (e.g. 'stop_connectivity') map to CSV column names
    # (e.g. 'connectivity_density') via this lookup.
    config_to_col = {
        'stop_connectivity': 'connectivity_density',
        'route_coverage': 'route_coverage',
        'freq_composite': 'freq_composite',       # NEW: consolidated frequency
        'freq_am_peak': 'freq_am_peak',            # kept for fallback
        'freq_midday': 'freq_midday',
        'freq_pm_peak': 'freq_pm_peak',
        'freq_evening': 'freq_evening',
        'span_of_service': 'span_hours',
        'weekend_ratio': 'weekend_ratio',
        'walking_access': 'walking_access_pct',
        'jobs_accessibility': 'jobs_accessibility',
        'poi_accessibility': 'poi_accessibility',
        'population_density': 'population_density',
        'intersection_density': 'intersection_density',
        'land_use_mix': 'land_use_mix',
        'sidewalk_index': 'sidewalk_index',
    }

    # ── Step 0: Compute frequency composite if enabled ──
    # Collapses 4 time-period frequencies into 1 weighted metric to avoid
    # over-representing frequency in the transit service category.
    freq_config = config.get('transit_service', {}).get('frequency_composite', {})
    if freq_config.get('enabled', False):
        freq_weights = freq_config.get('weights', {
            'freq_am_peak': 0.30, 'freq_pm_peak': 0.30,
            'freq_midday': 0.20, 'freq_evening': 0.20
        })
        # Map config keys to CSV column names
        freq_cols_available = {k: v for k, v in freq_weights.items()
                               if k in tract_metrics.columns}
        if freq_cols_available:
            tract_metrics['freq_composite'] = sum(
                tract_metrics[col].fillna(0) * weight
                for col, weight in freq_cols_available.items()
            )
            print(f"\n  Frequency composite: {len(freq_cols_available)} periods → 1 metric")
            for col, w in freq_cols_available.items():
                print(f"    {col}: {w:.0%}")
        else:
            print(f"\n  ⚠ No frequency columns found for composite — skipping")
    else:
        # If composite disabled, check if config still has individual freqs
        # in category_members — if so, they'll be used directly
        print(f"\n  Frequency composite: disabled (using individual periods)")

    cpta_config = config.get('cpta', {})
    cat_members = cpta_config.get('category_members', None)

    if cat_members:
        # Config-driven: read category members and map to column names
        transit_service_cols = [config_to_col.get(m, m)
                                for m in cat_members.get('transit_service', [])]
        accessibility_cols = [config_to_col.get(m, m)
                              for m in cat_members.get('accessibility', [])]
        built_env_cols = [config_to_col.get(m, m)
                          for m in cat_members.get('built_environment', [])]
        print(f"  Using config-defined category members")
    else:
        # Hardcoded fallback
        transit_service_cols = [
            'connectivity_density', 'route_coverage',
            'freq_am_peak', 'freq_midday', 'freq_pm_peak', 'freq_evening',
            'span_hours', 'weekend_ratio'
        ]
        accessibility_cols = [
            'walking_access_pct',
            'jobs_accessibility',
            'poi_accessibility'
        ]
        built_env_cols = [
            'population_density',
            'intersection_density',
            'land_use_mix',
            'sidewalk_index'
        ]
        print(f"  Using hardcoded category members (no cpta.category_members in config)")
    
    # Check which columns actually exist and have variance
    available_transit = [col for col in transit_service_cols
                         if col in tract_metrics.columns and tract_metrics[col].std() > 0]
    available_access = [col for col in accessibility_cols
                        if col in tract_metrics.columns and tract_metrics[col].std() > 0]
    available_built = [col for col in built_env_cols
                       if col in tract_metrics.columns and tract_metrics[col].std() > 0]
    
    available_cols = available_transit + available_access + available_built
    
    total_possible = len(transit_service_cols) + len(accessibility_cols) + len(built_env_cols)
    print(f"\n  Available metrics by category:")
    print(f"    Transit Service ({len(available_transit)}/{len(transit_service_cols)}): {available_transit}")
    print(f"    Accessibility   ({len(available_access)}/{len(accessibility_cols)}): {available_access}")
    print(f"    Built Env       ({len(available_built)}/{len(built_env_cols)}): {available_built}")
    print(f"    Total: {len(available_cols)}/{total_possible} metrics")
    
    if len(available_cols) == 0:
        print("  ✗ No metrics available — cannot compute CPTA")
        tract_metrics['CPTA'] = 0
        tract_metrics['CPTA_normalized'] = 50
        return tract_metrics
    
    # Step 1: Compute z-scores for each metric
    for col in available_cols:
        z_col = f'z_{col}'
        values = tract_metrics[col].fillna(0)
        if values.std() > 0:
            tract_metrics[z_col] = zscore(values)
        else:
            tract_metrics[z_col] = 0
    
    # Step 2 & 3: Category-weighted CPTA
    weighting_mode = cpta_config.get('aggregation_method', 'category_weighted')
    
    use_category = (weighting_mode in ('category', 'category_weighted')
                    and len(available_cols) > 1)
    
    if use_category:
        cat_weights = cpta_config.get('category_weights', {
            'transit_service': 0.50,
            'accessibility': 0.25,
            'built_environment': 0.25
        })
        
        w_transit = cat_weights.get('transit_service', 0.50)
        w_access = cat_weights.get('accessibility', 0.25)
        w_built = cat_weights.get('built_environment', 0.25)
        
        # Compute category means (only for categories with metrics)
        has_transit = len(available_transit) > 0
        has_access = len(available_access) > 0
        has_built = len(available_built) > 0
        
        # ── Within-category weighting ──
        # "correlation_penalty": w_m = 1 / Σ_k |r_mk|, normalized within category.
        # Metrics highly correlated with category-mates get downweighted.
        # "equal": simple mean (backward compatible).
        within_mode = cpta_config.get('within_category_weighting', 'equal')
        
        def compute_category_score(metric_cols, cat_name):
            """Compute weighted mean z-score for a category."""
            z_cols = [f'z_{col}' for col in metric_cols]
            
            if len(metric_cols) <= 1:
                # Single metric — no weighting needed
                return tract_metrics[z_cols[0]] if z_cols else 0
            
            if within_mode == 'correlation_penalty':
                # Compute Spearman correlation among category metrics
                raw_cols_data = tract_metrics[metric_cols].fillna(0)
                corr_mat = raw_cols_data.corr(method='spearman').abs()
                
                # w_m = 1 / sum of absolute correlations with other metrics
                raw_weights = 1.0 / corr_mat.sum(axis=1)
                norm_weights = raw_weights / raw_weights.sum()
                
                print(f"    {cat_name} correlation-penalty weights:")
                for col, w in zip(metric_cols, norm_weights):
                    print(f"      {col}: {w:.3f}")
                
                # Weighted mean of z-scores
                result = sum(
                    norm_weights.iloc[i] * tract_metrics[z_cols[i]]
                    for i in range(len(z_cols))
                )
                return result
            else:
                # Equal-weighted mean (original behavior)
                return tract_metrics[z_cols].mean(axis=1)
        
        if has_transit:
            tract_metrics['cat_transit'] = compute_category_score(
                available_transit, 'Transit Service')
        
        if has_access:
            tract_metrics['cat_accessibility'] = compute_category_score(
                available_access, 'Accessibility')
        
        if has_built:
            tract_metrics['cat_built_env'] = compute_category_score(
                available_built, 'Built Environment')
        
        # Renormalize weights to account for missing categories
        total_weight = 0
        if has_transit: total_weight += w_transit
        if has_access:  total_weight += w_access
        if has_built:   total_weight += w_built
        
        print(f"\n  Weighting mode: category-weighted")
        print(f"    Within-category: {within_mode}")
        print(f"    Transit Service:  {w_transit:.0%}" + (" (available)" if has_transit else " (MISSING — redistributed)"))
        print(f"    Accessibility:    {w_access:.0%}" + (" (available)" if has_access else " (MISSING — redistributed)"))
        print(f"    Built Environment: {w_built:.0%}" + (" (available)" if has_built else " (MISSING — redistributed)"))
        
        if total_weight > 0:
            tract_metrics['CPTA'] = (
                ((w_transit * tract_metrics['cat_transit']) if has_transit else 0) +
                ((w_access * tract_metrics['cat_accessibility']) if has_access else 0) +
                ((w_built * tract_metrics['cat_built_env']) if has_built else 0)
            ) / total_weight
        else:
            tract_metrics['CPTA'] = 0
    
    else:
        # Equal-weighted fallback
        print(f"\n  Weighting mode: equal (mean of {len(available_cols)} z-scores)")
        z_cols = [f'z_{col}' for col in available_cols]
        tract_metrics['CPTA'] = tract_metrics[z_cols].mean(axis=1)
    
    # Step 4: Normalize to 0-100
    cpta_min = tract_metrics['CPTA'].min()
    cpta_max = tract_metrics['CPTA'].max()
    if cpta_max > cpta_min:
        tract_metrics['CPTA_normalized'] = (
            (tract_metrics['CPTA'] - cpta_min) / (cpta_max - cpta_min) * 100
        )
    else:
        tract_metrics['CPTA_normalized'] = 50
    
    return tract_metrics


def compute_correlation_matrix(tract_metrics, output_dir):
    """
    Compute and save Pearson correlation matrix between CPTA component z-scores.
    
    High correlations (|r| > 0.85) are flagged — these may indicate
    redundant metrics that inflate one category's influence.
    
    Parameters
    ----------
    tract_metrics : DataFrame
        Tract-level metrics with z-score columns
    output_dir : Path
        Directory to save correlation matrix CSV
    """
    print("\nComputing CPTA correlation matrix (Spearman)...")
    
    z_cols = [col for col in tract_metrics.columns if col.startswith('z_')]
    
    if len(z_cols) < 2:
        print("  ⚠ Not enough z-score columns for correlation matrix")
        return
    
    # Clean column names for readability
    corr_df = tract_metrics[z_cols].copy()
    corr_df.columns = [col.replace('z_', '') for col in z_cols]
    
    # Compute correlation (Spearman — robust to right-skewed transit metrics)
    corr_matrix = corr_df.corr(method='spearman')
    
    # Save
    output_path = output_dir / "cpta_correlation_matrix.csv"
    corr_matrix.to_csv(output_path)
    print(f"  ✓ Saved to {output_path}")
    
    # Check for high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.85:
                high_corr_pairs.append(
                    (corr_matrix.columns[i], corr_matrix.columns[j], r)
                )
    
    if high_corr_pairs:
        print(f"  ⚠ High correlations detected (|r| > 0.85):")
        for col1, col2, r in high_corr_pairs:
            print(f"    {col1} ↔ {col2}: r={r:.3f}")
    else:
        print(f"  ✓ No high correlations (|r| > 0.85) between metrics")


def main():
    """Compute CPTA score from supply_metrics.csv."""
    print("=" * 60)
    print("Transit Desert Pipeline: CPTA Score Computation")
    print("=" * 60)
    
    config = load_config()
    
    output_dir = PROJECT_ROOT / "data" / "processed"
    supply_csv = output_dir / "supply_metrics.csv"
    
    if not supply_csv.exists():
        print(f"  ✗ supply_metrics.csv not found at {supply_csv}")
        print("  → Run 02_compute_supply.py first")
        sys.exit(1)
    
    # Load supply metrics
    print("\nLoading supply metrics...")
    tract_metrics = pd.read_csv(supply_csv, dtype={'GEOID': str})
    print(f"  ✓ Loaded {len(tract_metrics)} tracts, {len(tract_metrics.columns)} columns")
    
    # Show which optional columns are present
    optional_cols = {
        'jobs_accessibility': '02b_compute_jobs_accessibility.py',
        'poi_accessibility': '02c_compute_poi_accessibility.py'
    }
    for col, script in optional_cols.items():
        if col in tract_metrics.columns:
            print(f"  ✓ {col} found (from {script})")
        else:
            print(f"  ⚠ {col} not found — {script} was not run (optional)")
    
    # Compute CPTA
    print("\n" + "-" * 40)
    print("Computing CPTA score...")
    print("-" * 40)
    
    tract_metrics = compute_cpta_score(tract_metrics, config)
    
    # Correlation matrix
    compute_correlation_matrix(tract_metrics, output_dir)
    
    # Save updated CSV
    print("\n" + "-" * 40)
    print("Saving results...")
    print("-" * 40)
    
    tract_metrics.to_csv(supply_csv, index=False)
    print(f"  ✓ Updated {supply_csv}")
    
    # Also update GeoPackage if tracts shapefile exists
    tracts_path = PROJECT_ROOT / "data" / "raw" / "census" / "study_area_tracts.shp"
    if tracts_path.exists():
        crs_projected = config.get('spatial', {}).get('crs_projected', 'EPSG:2248')
        tracts_gdf = gpd.read_file(tracts_path).to_crs(crs_projected)
        merged = tracts_gdf.merge(tract_metrics, on='GEOID')
        merged.to_file(output_dir / "supply_metrics.gpkg", driver='GPKG')
        print(f"  ✓ Updated {output_dir / 'supply_metrics.gpkg'}")
    
    # Summary
    print("\n" + "-" * 40)
    print("CPTA Summary")
    print("-" * 40)
    
    print(f"\n  CPTA Score (raw z-score composite):")
    print(f"    Mean: {tract_metrics['CPTA'].mean():.2f}")
    print(f"    Std:  {tract_metrics['CPTA'].std():.2f}")
    print(f"    Min:  {tract_metrics['CPTA'].min():.2f}")
    print(f"    Max:  {tract_metrics['CPTA'].max():.2f}")
    
    print(f"\n  CPTA Normalized (0-100):")
    print(f"    Mean:   {tract_metrics['CPTA_normalized'].mean():.1f}")
    print(f"    Median: {tract_metrics['CPTA_normalized'].median():.1f}")
    print(f"    Min:    {tract_metrics['CPTA_normalized'].min():.1f}")
    print(f"    Max:    {tract_metrics['CPTA_normalized'].max():.1f}")
    
    # Category means if available
    for cat_col, label in [('cat_transit', 'Transit Service'),
                           ('cat_accessibility', 'Accessibility'),
                           ('cat_built_env', 'Built Environment')]:
        if cat_col in tract_metrics.columns:
            print(f"\n  {label} category mean: {tract_metrics[cat_col].mean():.3f}")
    
    print("\n" + "=" * 60)
    print("CPTA computation complete!")
    print("Next step: python src/03_compute_demand.py")
    print("=" * 60)


if __name__ == "__main__":
    main()