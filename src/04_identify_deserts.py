"""
04_identify_deserts.py
Identify transit deserts using LISA spatial clustering + absolute thresholds.

Methodology:
1. Compute Transit Gap = Normalized TVI - Normalized CPTA
2. Compute Global Moran's I to confirm spatial autocorrelation
3. Apply LISA (Local Moran's I) to identify spatial clusters
4. Classify tracts: Transit Desert, Transit Stressed, Underserved, Well-Served
5. Apply absolute threshold layer (cross-city comparable, no normalization)
6. Run TVI weight sensitivity analysis
7. Compute equity profile with statistical significance tests

The absolute threshold layer (Step 5) complements LISA by applying fixed
minimum service criteria uniformly across all cities — addressing the issue
that within-city normalization can mask real differences in transit quality
between cities (Karner, Pereira & Farber, 2024).

Usage:
    python src/04_identify_deserts.py

Outputs:
    - data/processed/transit_deserts.csv
    - data/processed/transit_deserts.gpkg
    - data/processed/equity_profile.csv
    - data/processed/global_morans.csv
    - data/processed/sensitivity_analysis.csv
    - data/processed/absolute_thresholds.csv
"""

import sys
import yaml
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import zscore, mannwhitneyu

# PySAL for spatial analysis
try:
    import libpysal
    from esda.moran import Moran, Moran_Local
except ImportError:
    print("PySAL libraries not installed. Run: pip install pysal esda libpysal")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_config():
    """Load configuration from YAML file."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_supply_demand_data():
    """Load supply (CPTA) and demand (TVI) metrics."""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    supply_path = processed_dir / "supply_metrics.csv"
    demand_path = processed_dir / "demand_metrics.csv"
    
    if not supply_path.exists():
        raise FileNotFoundError(f"Supply metrics not found: {supply_path}")
    if not demand_path.exists():
        raise FileNotFoundError(f"Demand metrics not found: {demand_path}")
    
    supply_df = pd.read_csv(supply_path, dtype={'GEOID': str})
    demand_df = pd.read_csv(demand_path, dtype={'GEOID': str})
    
    print(f"  ✓ Loaded supply metrics for {len(supply_df)} tracts")
    print(f"  ✓ Loaded demand metrics for {len(demand_df)} tracts")
    
    return supply_df, demand_df

def load_tracts(config):
    """Load Census tract boundaries."""
    tracts_path = PROJECT_ROOT / "data" / "raw" / "census" / "study_area_tracts.shp"
    
    if not tracts_path.exists():
        raise FileNotFoundError(f"Tract shapefile not found: {tracts_path}")
    
    gdf = gpd.read_file(tracts_path)
    gdf['GEOID'] = gdf['GEOID'].astype(str)
    gdf = gdf.to_crs(config['spatial']['crs_projected'])
    
    print(f"  ✓ Loaded {len(gdf)} tract boundaries")
    return gdf

def merge_supply_demand(supply_df, demand_df, tracts_gdf):
    """Merge supply and demand data with tract geometries."""
    print("Merging supply and demand data...")
    
    merged = tracts_gdf[['GEOID', 'geometry']].copy()
    
    # Add supply metrics
    supply_cols = ['GEOID', 'CPTA', 'CPTA_normalized']
    # Also bring in raw metric columns needed for absolute threshold evaluation
    raw_supply_cols = [
        'route_coverage', 'freq_am_peak', 'walking_access_pct',
        'span_hours', 'jobs_accessibility'
    ]
    for col in raw_supply_cols:
        if col in supply_df.columns:
            supply_cols.append(col)
    # Bring in z-score columns for deficit profiling
    z_cols = [col for col in supply_df.columns if col.startswith('z_')]
    supply_cols.extend(z_cols)
    supply_cols = list(dict.fromkeys(supply_cols))  # deduplicate, preserve order

    if all(col in supply_df.columns for col in ['GEOID', 'CPTA', 'CPTA_normalized']):
        merged = merged.merge(supply_df[supply_cols], on='GEOID', how='left')
    else:
        merged['CPTA'] = 0
        merged['CPTA_normalized'] = 50
    
    # Add demand metrics
    demand_cols = ['GEOID', 'TVI', 'TVI_normalized', 
                   'TVI_latent', 'TVI_latent_normalized', 'tvi_latent_shift',
                   'pct_low_income_car_commute', 'n_low_income_car_commuters', 'n_low_income_workers',
                   'pct_zero_vehicle', 'pct_poverty', 'pct_minority', 'pct_elderly', 'pct_youth']
    available_demand_cols = [col for col in demand_cols if col in demand_df.columns]
    merged = merged.merge(demand_df[available_demand_cols], on='GEOID', how='left')
    
    merged = merged.fillna({'CPTA': 0, 'CPTA_normalized': 0, 'TVI': 0, 'TVI_normalized': 0})
    # Fill latent columns too if present
    if 'TVI_latent_normalized' in merged.columns:
        merged['TVI_latent_normalized'] = merged['TVI_latent_normalized'].fillna(0)
        merged['TVI_latent'] = merged['TVI_latent'].fillna(0)
    
    print(f"  ✓ Merged {len(merged)} tracts")
    return merged

def compute_transit_gap(gdf):
    """
    Compute Transit Gap = Demand - Supply using percentile ranks.
    
    Percentile ranks ensure TVI and CPTA are on equivalent distributional
    scales before subtraction. A score of 80 means "higher than 80% of tracts"
    for both indices, making the gap interpretable and comparable across metrics.
    
    Also computes a positive-only gap for LISA analysis, which focuses
    spatial clustering on underservice rather than collapsing well-served
    and low-need tracts into the same Low-Low cluster.
    """
    print("Computing transit gap (percentile rank method)...")
    
    n = len(gdf)
    
    # Percentile ranks: rank / n × 100 (0-100 scale, distributionally equivalent)
    gdf['TVI_pct_rank'] = gdf['TVI'].rank(method='average') / n * 100
    gdf['CPTA_pct_rank'] = gdf['CPTA'].rank(method='average') / n * 100
    
    # Primary gap: percentile rank difference
    gdf['transit_gap'] = gdf['TVI_pct_rank'] - gdf['CPTA_pct_rank']
    
    # Keep the old normalized gap for backward compatibility / comparison
    gdf['transit_gap_raw'] = gdf['TVI_normalized'] - gdf['CPTA_normalized']
    
    # Z-score based gap (kept for spatial lag computation)
    gdf['z_TVI'] = zscore(gdf['TVI'].fillna(0))
    gdf['z_CPTA'] = zscore(gdf['CPTA'].fillna(0))
    gdf['transit_gap_z'] = gdf['z_TVI'] - gdf['z_CPTA']
    
    # Positive-only gap for LISA: set negative gaps to zero
    # This focuses spatial clustering on underservice only.
    # Negative-gap tracts (supply > demand) are reported separately.
    gdf['transit_gap_positive'] = gdf['transit_gap'].clip(lower=0)
    
    n_positive = (gdf['transit_gap'] > 0).sum()
    n_negative = (gdf['transit_gap'] < 0).sum()
    
    print(f"\n  Transit Gap Statistics (percentile rank):")
    print(f"    Mean: {gdf['transit_gap'].mean():.1f}")
    print(f"    Std:  {gdf['transit_gap'].std():.1f}")
    print(f"    Min:  {gdf['transit_gap'].min():.1f}")
    print(f"    Max:  {gdf['transit_gap'].max():.1f}")
    print(f"    Positive gap (demand > supply): {n_positive} tracts")
    print(f"    Negative gap (supply > demand): {n_negative} tracts")
    
    return gdf

def create_spatial_weights(gdf):
    """Create spatial weights matrix for LISA (Queen contiguity)."""
    print("Creating spatial weights matrix...")
    
    w = libpysal.weights.Queen.from_dataframe(gdf)
    w.transform = 'r'  # Row-standardize
    
    islands = w.islands
    if islands:
        print(f"  ⚠ Found {len(islands)} island tracts (no neighbors)")
    
    print(f"  ✓ Created weights for {w.n} tracts")
    print(f"  ✓ Mean neighbors: {w.mean_neighbors:.1f}")
    
    return w

def compute_lisa_univariate(gdf, w, variable='transit_gap'):
    """
    Compute univariate Local Moran's I for transit gap.
    
    Also computes and stores:
    - Global Moran's I with p-value
    - Spatial lag values for Moran scatterplot
    
    Returns
    -------
    gdf : GeoDataFrame
        Updated with LISA columns and spatial lag
    lisa : Moran_Local
        LISA results object
    global_moran : dict
        Global Moran's I statistics
    """
    print(f"Computing univariate LISA for {variable}...")
    
    y = gdf[variable].values
    
    # Global Moran's I
    global_result = Moran(y, w, permutations=999)
    global_moran = {
        'variable': variable,
        'I': global_result.I,
        'EI': global_result.EI,
        'p_value': global_result.p_sim,
        'z_score': global_result.z_sim
    }
    
    print(f"\n  Global Moran's I: {global_moran['I']:.4f}")
    print(f"  Expected I:       {global_moran['EI']:.4f}")
    print(f"  p-value:          {global_moran['p_value']:.4f}")
    print(f"  z-score:          {global_moran['z_score']:.4f}")
    
    if global_moran['p_value'] < 0.05:
        print("  → Significant spatial autocorrelation confirmed")
    else:
        print("  ⚠ Spatial autocorrelation not significant at p<0.05")
    
    # Local Moran's I (LISA)
    lisa = Moran_Local(y, w, permutations=999)
    
    gdf['lisa_I'] = lisa.Is
    gdf['lisa_p'] = lisa.p_sim
    gdf['lisa_q'] = lisa.q  # Quadrant (1=HH, 2=LH, 3=LL, 4=HL)
    
    # Spatial lag for Moran scatterplot
    # Standardize the variable and compute spatial lag
    y_standardized = (y - y.mean()) / y.std()
    gdf['gap_standardized'] = y_standardized
    gdf['gap_spatial_lag'] = libpysal.weights.lag_spatial(w, y_standardized)
    
    alpha = 0.05
    gdf['lisa_sig'] = gdf['lisa_p'] < alpha
    
    cluster_labels = {1: 'High-High', 2: 'Low-High', 3: 'Low-Low', 4: 'High-Low'}
    gdf['lisa_cluster'] = gdf.apply(
        lambda row: cluster_labels.get(row['lisa_q'], 'Not Significant') 
        if row['lisa_sig'] else 'Not Significant',
        axis=1
    )
    
    print(f"\n  LISA Cluster counts:")
    for cluster, count in gdf['lisa_cluster'].value_counts().items():
        print(f"    {cluster}: {count}")
    
    return gdf, lisa, global_moran

def classify_transit_deserts(gdf, config=None):
    """
    Create final transit desert classification using hybrid two-pathway approach.

    A tract is classified as a transit desert if EITHER:
      Pathway 1 (Relative/Spatial): LISA High-High cluster OR median-based
                                     high-demand/low-supply quadrant
      Pathway 2 (Absolute/Equity):  Fails ≥N absolute service thresholds
                                     AND has above-median TVI (equity gate)

    The equity gate in Pathway 2 prevents wealthy, low-dependency suburbs
    from being classified as deserts just because they lack bus service
    that their residents don't need.
    """
    print("Classifying tracts (hybrid: relative + absolute)...")
    
    demand_median = gdf['TVI_normalized'].median()
    supply_median = gdf['CPTA_normalized'].median()
    
    # --- Pathway 1: Relative (median quadrant) ---
    def classify_tract(row):
        high_demand = row['TVI_normalized'] >= demand_median
        high_supply = row['CPTA_normalized'] >= supply_median
        
        if high_demand and not high_supply:
            return 'Transit Desert'
        elif high_demand and high_supply:
            return 'Transit Stressed'
        elif not high_demand and not high_supply:
            return 'Underserved'
        else:
            return 'Well-Served'
    
    gdf['classification_simple'] = gdf.apply(classify_tract, axis=1)
    
    # --- Pathway 1b: LISA overlay ---
    def classify_tract_lisa(row):
        if row['lisa_cluster'] == 'High-High':
            return 'Transit Desert (LISA)'
        elif row['lisa_cluster'] == 'High-Low':
            return 'Isolated High Gap'
        elif row['lisa_cluster'] == 'Low-Low':
            return 'Well-Served (LISA)'
        elif row['lisa_cluster'] == 'Low-High':
            return 'Isolated Well-Served'
        else:
            return row['classification_simple']
    
    gdf['classification_lisa'] = gdf.apply(classify_tract_lisa, axis=1)
    
    # --- Pathway 1 result ---
    relative_desert = (
        (gdf['classification_simple'] == 'Transit Desert') | 
        (gdf['lisa_cluster'] == 'High-High')
    )
    
    # --- Pathway 2: Absolute thresholds + equity gate ---
    # Use TVI_latent for the equity gate when available. TVI_latent includes
    # forced car ownership, so low-income car-dependent tracts that would be
    # filtered by observed TVI (zero-vehicle undercount) can now pass through.
    has_absolute = 'abs_transit_desert' in gdf.columns
    if has_absolute:
        has_latent = 'TVI_latent_normalized' in gdf.columns and gdf['TVI_latent_normalized'].std() > 0
        
        if has_latent:
            equity_col = 'TVI_latent_normalized'
            equity_median = gdf[equity_col].median()
            equity_label = f"TVI-latent ≥ median ({equity_median:.1f})"
        else:
            equity_col = 'TVI_normalized'
            equity_median = demand_median
            equity_label = f"TVI ≥ median ({equity_median:.1f})"
        
        high_demand_latent = gdf[equity_col] >= equity_median
        absolute_desert_equity = gdf['abs_transit_desert'] & high_demand_latent
        
        n_abs_raw = gdf['abs_transit_desert'].sum()
        n_abs_gated = absolute_desert_equity.sum()
        n_filtered = n_abs_raw - n_abs_gated
        print(f"\n  Pathway 2 (Absolute + Equity Gate):")
        print(f"    Equity gate metric: {equity_col}")
        print(f"    Absolute threshold failures: {n_abs_raw}")
        print(f"    After equity gate ({equity_label}): {n_abs_gated}")
        print(f"    Filtered out (low demand): {n_filtered}")
        
        # Show what latent gate changes vs observed gate
        if has_latent:
            observed_gate = gdf['abs_transit_desert'] & (gdf['TVI_normalized'] >= demand_median)
            n_observed_gated = observed_gate.sum()
            n_new_from_latent = n_abs_gated - n_observed_gated
            if n_new_from_latent > 0:
                print(f"    ── Latent demand effect: +{n_new_from_latent} tracts rescued ──")
                rescued = absolute_desert_equity & ~observed_gate
                if rescued.any():
                    rescued_df = gdf[rescued][['GEOID', 'TVI_normalized', 'TVI_latent_normalized', 
                                               'pct_low_income_car_commute']].head(5)
                    for _, row in rescued_df.iterrows():
                        print(f"      {row['GEOID']}: TVI={row['TVI_normalized']:.1f}, "
                              f"TVI_latent={row['TVI_latent_normalized']:.1f}, "
                              f"car_commute={row.get('pct_low_income_car_commute', 0):.0%}")
            else:
                print(f"    ── Latent demand effect: no additional tracts (same as observed gate) ──")
    else:
        absolute_desert_equity = pd.Series(False, index=gdf.index)
    
    # --- Hybrid classification ---
    gdf['transit_desert'] = relative_desert | absolute_desert_equity
    
    # Track which pathway identified each desert
    gdf['desert_pathway'] = 'Not Desert'
    gdf.loc[relative_desert & absolute_desert_equity, 'desert_pathway'] = 'Both Pathways'
    gdf.loc[relative_desert & ~absolute_desert_equity, 'desert_pathway'] = 'Relative Only'
    gdf.loc[~relative_desert & absolute_desert_equity, 'desert_pathway'] = 'Absolute + Equity'
    
    # --- Summary ---
    print(f"\n  Simple Classification:")
    for cat, count in gdf['classification_simple'].value_counts().items():
        print(f"    {cat}: {count}")
    
    n_total = gdf['transit_desert'].sum()
    print(f"\n  Hybrid Transit Deserts: {n_total} ({n_total/len(gdf):.1%})")
    
    pathway_counts = gdf[gdf['transit_desert']]['desert_pathway'].value_counts()
    for pathway, count in pathway_counts.items():
        print(f"    {pathway}: {count}")
    
    return gdf

def compute_equity_profile(gdf):
    """
    Compute demographic profile of transit deserts vs non-deserts.
    
    Includes Mann-Whitney U tests for statistical significance.
    """
    print("\nComputing equity profile of transit deserts...")
    
    deserts = gdf[gdf['transit_desert'] == True]
    non_deserts = gdf[gdf['transit_desert'] == False]
    
    metrics = ['pct_zero_vehicle', 'pct_poverty', 'pct_minority', 'pct_elderly', 'pct_youth']
    
    profile = []
    for metric in metrics:
        if metric in gdf.columns:
            desert_vals = deserts[metric].dropna()
            non_desert_vals = non_deserts[metric].dropna()
            
            desert_mean = desert_vals.mean()
            non_desert_mean = non_desert_vals.mean()
            diff = desert_mean - non_desert_mean
            
            # Mann-Whitney U test
            if len(desert_vals) > 0 and len(non_desert_vals) > 0:
                stat, p_value = mannwhitneyu(
                    desert_vals, non_desert_vals, alternative='two-sided'
                )
                sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            else:
                p_value = np.nan
                sig = 'n/a'
            
            profile.append({
                'Metric': metric,
                'Transit Deserts': f"{desert_mean:.1%}",
                'Non-Deserts': f"{non_desert_mean:.1%}",
                'Difference': f"{diff:+.1%}",
                'U_statistic': stat if p_value is not np.nan else np.nan,
                'p_value': p_value,
                'Significance': sig
            })
    
    profile_df = pd.DataFrame(profile)
    print("\n  Demographic comparison:")
    print(profile_df[['Metric', 'Transit Deserts', 'Non-Deserts', 'Difference', 'Significance']].to_string(index=False))
    
    return profile_df

def run_sensitivity_analysis(gdf, w, config):
    """
    Run transit desert identification under multiple TVI weight scenarios.
    
    Recomputes TVI from raw percentages using alternative weights,
    recalculates the gap, and re-runs LISA for each scenario.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain pct_ columns and CPTA_normalized
    w : libpysal.weights.W
        Spatial weights matrix
    config : dict
        Configuration with tvi_weight_scenarios
    
    Returns
    -------
    DataFrame
        Comparison of desert counts across scenarios
    """
    print("Running TVI weight sensitivity analysis...")
    
    # Read scenarios from config — support both old and new paths
    tvi_config = config.get('tvi', {})
    scenarios_config = tvi_config.get('sensitivity_scenarios', {})
    
    if not scenarios_config:
        # Try old path
        scenarios_config = config.get('census', {}).get('tvi_weight_scenarios', {})
    
    # Get base weights
    components_config = tvi_config.get('components', {})
    if components_config and all('weight' in v for v in components_config.values()):
        base_weights = {k: v['weight'] for k, v in components_config.items()}
    else:
        base_weights = config.get('census', {}).get('tvi_weights', {
            'zero_vehicle': 0.30, 'poverty': 0.25,
            'minority': 0.20, 'elderly': 0.15, 'youth': 0.10
        })
    
    # Build scenarios — extract just the weights dict from each scenario
    all_scenarios = {'base': base_weights}
    for name, scenario_def in scenarios_config.items():
        if name == 'base':
            continue  # Already added
        if isinstance(scenario_def, dict) and 'weights' in scenario_def:
            all_scenarios[name] = scenario_def['weights']
        elif isinstance(scenario_def, dict):
            all_scenarios[name] = scenario_def
    
    tvi_cols = ['pct_zero_vehicle', 'pct_poverty', 'pct_minority', 'pct_elderly', 'pct_youth']
    available_cols = [col for col in tvi_cols if col in gdf.columns]
    
    results = []
    
    for scenario_name, weights in all_scenarios.items():
        print(f"\n  Scenario: {scenario_name}")
        print(f"    Weights: {weights}")
        
        # Compute z-scores for each component
        z_scores = {}
        for col in available_cols:
            values = gdf[col].fillna(0)
            if values.std() > 0:
                z_scores[col] = zscore(values)
            else:
                z_scores[col] = np.zeros(len(gdf))
        
        # Map column names to weight keys
        col_to_key = {
            'pct_zero_vehicle': 'zero_vehicle',
            'pct_poverty': 'poverty',
            'pct_minority': 'minority',
            'pct_elderly': 'elderly',
            'pct_youth': 'youth'
        }
        
        # Compute weighted TVI
        tvi_values = np.zeros(len(gdf))
        for col in available_cols:
            key = col_to_key[col]
            weight = weights.get(key, 0)
            tvi_values += weight * z_scores[col]
        
        # Normalize to 0-100
        tvi_min, tvi_max = tvi_values.min(), tvi_values.max()
        if tvi_max > tvi_min:
            tvi_normalized = (tvi_values - tvi_min) / (tvi_max - tvi_min) * 100
        else:
            tvi_normalized = np.full(len(gdf), 50.0)
        
        # Compute gap using percentile ranks (consistent with main analysis)
        n = len(gdf)
        tvi_pct_rank = pd.Series(tvi_normalized).rank(method='average').values / n * 100
        cpta_pct_rank = gdf['CPTA'].rank(method='average').values / n * 100
        gap = tvi_pct_rank - cpta_pct_rank
        
        # Positive-only gap for LISA (consistent with main analysis)
        gap_positive = np.clip(gap, 0, None)
        
        # Median-based classification
        demand_median = np.median(tvi_normalized)
        supply_median = gdf['CPTA_normalized'].median()
        
        desert_count = np.sum(
            (tvi_normalized >= demand_median) & (gdf['CPTA_normalized'].values < supply_median)
        )
        desert_pct = desert_count / len(gdf) * 100
        
        # LISA on the positive-only gap (consistent with main analysis)
        lisa_result = Moran_Local(gap_positive, w, permutations=999)
        hh_count = np.sum((lisa_result.q == 1) & (lisa_result.p_sim < 0.05))
        
        results.append({
            'Scenario': scenario_name,
            'Weights': str(weights),
            'Desert_Count_Simple': int(desert_count),
            'Desert_Pct_Simple': f"{desert_pct:.1f}%",
            'LISA_HH_Count': int(hh_count),
            'TVI_Mean': f"{tvi_values.mean():.3f}",
            'Gap_Mean': f"{gap.mean():.1f}"
        })
        
        print(f"    Transit Deserts (simple): {desert_count} ({desert_pct:.1f}%)")
        print(f"    LISA High-High clusters:  {hh_count}")
    
    results_df = pd.DataFrame(results)
    
    # Check robustness
    desert_counts = [r['Desert_Count_Simple'] for r in results]
    count_range = max(desert_counts) - min(desert_counts)
    pct_range = count_range / len(gdf) * 100
    print(f"\n  Robustness check:")
    print(f"    Desert count range across scenarios: {min(desert_counts)}-{max(desert_counts)}")
    print(f"    Percentage range: {pct_range:.1f} pp")
    if pct_range <= 3:
        print(f"    → Results are ROBUST to weight assumptions (≤3 pp variation)")
    else:
        print(f"    → Results show MODERATE sensitivity to weight assumptions (>{3} pp variation)")
    
    return results_df


def compute_deficit_profile(gdf, config, n_barriers=3):
    """
    For each transit desert tract, identify the top supply-side barriers
    by ranking CPTA component z-scores from worst (most negative) to best.

    This tells policymakers *why* a tract is a desert — is it frequency,
    walking access, job accessibility, etc.? — enabling targeted intervention.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain z_* columns from supply_metrics and transit_desert flag
    config : dict
        Configuration (used for metric labels)
    n_barriers : int
        Number of top barriers to store per tract (default 3)

    Returns
    -------
    gdf : GeoDataFrame
        Updated with barrier_1, barrier_2, barrier_3, barrier_1_zscore, etc.
    barrier_summary : DataFrame
        Aggregate frequency of each metric appearing as a top barrier
    """
    print("Computing deficit profiles for transit desert tracts...")

    # Find z-score columns in the data
    z_cols = sorted([col for col in gdf.columns if col.startswith('z_')])

    # Exclude z_TVI and z_CPTA (those are composites, not actionable components)
    z_cols = [col for col in z_cols if col not in ('z_TVI', 'z_CPTA')]

    if not z_cols:
        print("  ⚠ No z-score columns found — run 02d_compute_cpta.py first")
        return gdf, pd.DataFrame()

    # Clean metric names for readability
    label_map = {
        'z_connectivity_density': 'Stop Connectivity',
        'z_route_coverage': 'Route Coverage',
        'z_freq_am_peak': 'Frequency (AM Peak)',
        'z_freq_midday': 'Frequency (Midday)',
        'z_freq_pm_peak': 'Frequency (PM Peak)',
        'z_freq_evening': 'Frequency (Evening)',
        'z_span_hours': 'Span of Service',
        'z_weekend_ratio': 'Weekend Service',
        'z_walking_access_pct': 'Walking Access',
        'z_jobs_accessibility': 'Job Accessibility',
        'z_poi_accessibility': 'POI Accessibility',
        'z_population_density': 'Population Density',
        'z_intersection_density': 'Intersection Density',
        'z_land_use_mix': 'Land Use Mix',
        'z_sidewalk_index': 'Sidewalk Infrastructure',
    }

    print(f"  Analyzing {len(z_cols)} supply metrics: {[col.replace('z_', '') for col in z_cols]}")

    # Initialize barrier columns
    for i in range(1, n_barriers + 1):
        gdf[f'barrier_{i}'] = ''
        gdf[f'barrier_{i}_zscore'] = np.nan

    desert_mask = gdf['transit_desert'] == True
    n_deserts = desert_mask.sum()

    if n_deserts == 0:
        print("  ⚠ No transit deserts identified — skipping deficit profile")
        return gdf, pd.DataFrame()

    # For each desert tract, rank z-scores (most negative = worst barrier)
    barrier_counter = {}  # metric -> {rank_1: count, rank_2: count, ...}

    for idx in gdf[desert_mask].index:
        scores = {col: gdf.loc[idx, col] for col in z_cols
                  if pd.notna(gdf.loc[idx, col])}

        if not scores:
            continue

        # Sort ascending (most negative first = worst performing)
        ranked = sorted(scores.items(), key=lambda x: x[1])

        for rank, (col, zscore_val) in enumerate(ranked[:n_barriers], 1):
            clean_name = label_map.get(col, col.replace('z_', '').replace('_', ' ').title())
            gdf.loc[idx, f'barrier_{rank}'] = clean_name
            gdf.loc[idx, f'barrier_{rank}_zscore'] = round(zscore_val, 2)

            # Track aggregate counts
            if clean_name not in barrier_counter:
                barrier_counter[clean_name] = {f'rank_{r}': 0 for r in range(1, n_barriers + 1)}
                barrier_counter[clean_name]['total_top3'] = 0
            barrier_counter[clean_name][f'rank_{rank}'] += 1
            barrier_counter[clean_name]['total_top3'] += 1

    # Build summary table
    summary_rows = []
    for metric, counts in barrier_counter.items():
        row = {'metric': metric}
        row.update(counts)
        row['pct_of_deserts'] = round(counts['total_top3'] / n_deserts * 100, 1)
        summary_rows.append(row)

    barrier_summary = pd.DataFrame(summary_rows).sort_values('total_top3', ascending=False)

    print(f"\n  Top barriers across {n_deserts} transit desert tracts:")
    print(f"  {'Metric':<25s} {'#1':>5s} {'#2':>5s} {'#3':>5s} {'Total':>6s} {'% Deserts':>10s}")
    print(f"  {'-'*60}")
    for _, row in barrier_summary.head(8).iterrows():
        print(f"  {row['metric']:<25s} "
              f"{row.get('rank_1', 0):>5.0f} "
              f"{row.get('rank_2', 0):>5.0f} "
              f"{row.get('rank_3', 0):>5.0f} "
              f"{row['total_top3']:>6.0f} "
              f"{row['pct_of_deserts']:>9.1f}%")

    # Also fill non-desert tracts with their worst metrics (useful for planners
    # doing preventive analysis — "which tracts are close to becoming deserts?")
    non_desert_mask = ~desert_mask
    for idx in gdf[non_desert_mask].index:
        scores = {col: gdf.loc[idx, col] for col in z_cols
                  if pd.notna(gdf.loc[idx, col])}
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda x: x[1])
        for rank, (col, zscore_val) in enumerate(ranked[:n_barriers], 1):
            clean_name = label_map.get(col, col.replace('z_', '').replace('_', ' ').title())
            gdf.loc[idx, f'barrier_{rank}'] = clean_name
            gdf.loc[idx, f'barrier_{rank}_zscore'] = round(zscore_val, 2)

    print(f"\n  ✓ Deficit profile computed for all {len(gdf)} tracts")
    print(f"    (barrier_1/2/3 columns added — worst metrics per tract)")

    return gdf, barrier_summary


def apply_absolute_thresholds(gdf, config):
    """
    Apply absolute minimum-service thresholds to flag tracts as absolute transit deserts.

    Unlike the relative LISA/median approach, these thresholds are fixed values
    applied uniformly across all cities — a tract either meets a minimum standard
    or it doesn't, regardless of how other tracts in the same city perform.

    This addresses the normalization problem: within-city z-scoring forces ~18-21%
    desert rates everywhere, even in cities with objectively poor transit.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain raw supply metric columns (route_coverage, freq_am_peak, etc.)
    config : dict
        Configuration with identification.absolute_thresholds section

    Returns
    -------
    gdf : GeoDataFrame
        Updated with absolute threshold columns:
        - abs_fail_<criterion>: bool per criterion
        - abs_fail_count: number of criteria failed
        - abs_transit_desert: True if fail_count >= fail_threshold
    thresholds_df : DataFrame
        Summary of criteria pass/fail rates
    """
    abs_config = config.get('identification', {}).get('absolute_thresholds', {})

    if not abs_config.get('enabled', False):
        print("  Absolute thresholds disabled in config — skipping")
        gdf['abs_transit_desert'] = False
        gdf['abs_fail_count'] = 0
        return gdf, pd.DataFrame()

    fail_threshold = abs_config.get('fail_threshold', 3)
    criteria = abs_config.get('criteria', {})

    if not criteria:
        print("  ⚠ No criteria defined in absolute_thresholds — skipping")
        gdf['abs_transit_desert'] = False
        gdf['abs_fail_count'] = 0
        return gdf, pd.DataFrame()

    print(f"  Fail threshold: {fail_threshold} of {len(criteria)} criteria")

    # Map config metric names to actual CSV column names
    metric_to_col = {
        'route_coverage': 'route_coverage',
        'freq_am_peak': 'freq_am_peak',
        'walking_access': 'walking_access_pct',
        'span_of_service': 'span_hours',
        'jobs_accessibility': 'jobs_accessibility',
    }

    summary_rows = []

    for crit_name, crit_def in criteria.items():
        metric_key = crit_def.get('metric', crit_name)
        col_name = metric_to_col.get(metric_key, metric_key)
        operator = crit_def.get('operator', '>=')
        threshold_value = crit_def.get('value', 0)
        label = crit_def.get('label', crit_name)

        fail_col = f'abs_fail_{crit_name}'

        if col_name not in gdf.columns:
            print(f"    ⚠ {label}: column '{col_name}' not found — skipping")
            gdf[fail_col] = False
            summary_rows.append({
                'criterion': crit_name, 'label': label,
                'threshold': f"{operator} {threshold_value}",
                'column': col_name, 'status': 'MISSING',
                'fail_count': 0, 'fail_pct': 0
            })
            continue

        values = gdf[col_name].fillna(0)

        # Evaluate: a tract FAILS if it does NOT meet the criterion
        if operator == '>=':
            passes = values >= threshold_value
        elif operator == '>':
            passes = values > threshold_value
        elif operator == '<=':
            passes = values <= threshold_value
        elif operator == '<':
            passes = values < threshold_value
        else:
            print(f"    ⚠ Unknown operator '{operator}' for {crit_name}")
            passes = pd.Series(True, index=gdf.index)

        gdf[fail_col] = ~passes
        n_fail = (~passes).sum()
        pct_fail = n_fail / len(gdf) * 100

        print(f"    {label}: {n_fail} tracts fail ({pct_fail:.1f}%) "
              f"[{col_name} {operator} {threshold_value}]")

        summary_rows.append({
            'criterion': crit_name, 'label': label,
            'threshold': f"{operator} {threshold_value}",
            'column': col_name, 'status': 'OK',
            'fail_count': int(n_fail), 'fail_pct': round(pct_fail, 1)
        })

    # Count total failures per tract
    fail_cols = [f'abs_fail_{name}' for name in criteria.keys()
                 if f'abs_fail_{name}' in gdf.columns]
    gdf['abs_fail_count'] = gdf[fail_cols].sum(axis=1)
    gdf['abs_transit_desert'] = gdf['abs_fail_count'] >= fail_threshold

    n_abs_desert = gdf['abs_transit_desert'].sum()
    pct_abs_desert = n_abs_desert / len(gdf) * 100

    print(f"\n  Absolute transit deserts (≥{fail_threshold} failures): "
          f"{n_abs_desert} tracts ({pct_abs_desert:.1f}%)")

    # Compare with relative classification
    if 'transit_desert' in gdf.columns:
        both = (gdf['transit_desert'] & gdf['abs_transit_desert']).sum()
        relative_only = (gdf['transit_desert'] & ~gdf['abs_transit_desert']).sum()
        absolute_only = (~gdf['transit_desert'] & gdf['abs_transit_desert']).sum()
        print(f"\n  Overlap with relative (LISA/median) deserts:")
        print(f"    Both:          {both}")
        print(f"    Relative only: {relative_only}")
        print(f"    Absolute only: {absolute_only}")

    # Failure count distribution
    print(f"\n  Failure count distribution:")
    for n in range(len(criteria) + 1):
        count = (gdf['abs_fail_count'] == n).sum()
        if count > 0:
            marker = " ← desert threshold" if n == fail_threshold else ""
            print(f"    {n} failures: {count} tracts{marker}")

    thresholds_df = pd.DataFrame(summary_rows)
    return gdf, thresholds_df


def main():
    """Main function to identify transit deserts."""
    print("=" * 60)
    print("Transit Desert Pipeline: Desert Identification (LISA)")
    print("=" * 60)
    
    config = load_config()
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "-" * 40)
    print("Loading data...")
    print("-" * 40)
    
    supply_df, demand_df = load_supply_demand_data()
    tracts_gdf = load_tracts(config)
    gdf = merge_supply_demand(supply_df, demand_df, tracts_gdf)
    
    # Compute transit gap
    print("\n" + "-" * 40)
    print("Computing transit gap...")
    print("-" * 40)
    gdf = compute_transit_gap(gdf)
    
    # Spatial analysis
    print("\n" + "-" * 40)
    print("Spatial analysis...")
    print("-" * 40)
    w = create_spatial_weights(gdf)
    
    # Run LISA on positive-only gap (focuses clustering on underservice)
    # Negative gaps (supply > demand) are zeroed out so Low-Low clusters
    # represent genuine low-need areas, not a mix of low-need and well-served.
    gdf, lisa_uni, global_moran = compute_lisa_univariate(gdf, w, variable='transit_gap_positive')
    
    # Report well-served tracts (negative gap) separately
    well_served_surplus = gdf[gdf['transit_gap'] < -10]
    if len(well_served_surplus) > 0:
        print(f"\n  Well-served surplus tracts (gap < -10): {len(well_served_surplus)}")
        print(f"    Mean CPTA: {well_served_surplus['CPTA_normalized'].mean():.1f}")
        print(f"    Mean TVI:  {well_served_surplus['TVI_normalized'].mean():.1f}")
    
    # Absolute thresholds (must run before classification for hybrid integration)
    print("\n" + "-" * 40)
    print("Absolute threshold analysis...")
    print("-" * 40)
    gdf, thresholds_df = apply_absolute_thresholds(gdf, config)
    
    # Classify (hybrid: LISA + absolute thresholds gated by equity)
    print("\n" + "-" * 40)
    print("Classifying tracts...")
    print("-" * 40)
    gdf = classify_transit_deserts(gdf, config)
    
    # Deficit profile (per-tract barrier identification)
    print("\n" + "-" * 40)
    print("Deficit profile analysis...")
    print("-" * 40)
    gdf, barrier_summary = compute_deficit_profile(gdf, config)
    
    # Equity profile with statistical tests
    print("\n" + "-" * 40)
    print("Equity analysis...")
    print("-" * 40)
    profile_df = compute_equity_profile(gdf)
    
    # Sensitivity analysis
    print("\n" + "-" * 40)
    print("Sensitivity analysis...")
    print("-" * 40)
    sensitivity_df = run_sensitivity_analysis(gdf, w, config)
    
    # Save results
    print("\n" + "-" * 40)
    print("Saving results...")
    print("-" * 40)
    
    output_cols = [
        'GEOID', 'geometry',
        'TVI', 'TVI_normalized', 'CPTA', 'CPTA_normalized',
        'transit_gap', 'transit_gap_raw', 'transit_gap_positive', 'transit_gap_z',
        'TVI_pct_rank', 'CPTA_pct_rank',
        'gap_standardized', 'gap_spatial_lag',
        'lisa_I', 'lisa_p', 'lisa_cluster', 'lisa_sig',
        'classification_simple', 'classification_lisa', 'transit_desert',
        'desert_pathway',
        'abs_fail_count', 'abs_transit_desert'
    ]
    
    # Add individual absolute failure columns if present
    abs_fail_cols = [col for col in gdf.columns if col.startswith('abs_fail_') 
                     and col not in ('abs_fail_count',)]
    output_cols.extend(abs_fail_cols)
    
    # Add deficit profile barrier columns
    barrier_cols = [col for col in gdf.columns if col.startswith('barrier_')]
    output_cols.extend(barrier_cols)
    
    demo_cols = ['pct_zero_vehicle', 'pct_poverty', 'pct_minority', 'pct_elderly', 'pct_youth']
    output_cols.extend([col for col in demo_cols if col in gdf.columns])
    
    # Add TVI-latent and forced car ownership columns
    latent_cols = ['TVI_latent', 'TVI_latent_normalized', 'tvi_latent_shift',
                   'pct_low_income_car_commute', 'n_low_income_car_commuters', 'n_low_income_workers']
    output_cols.extend([col for col in latent_cols if col in gdf.columns])
    
    output_cols = [col for col in output_cols if col in gdf.columns]
    
    output_gdf = gdf[output_cols].copy()
    
    output_df = output_gdf.drop(columns=['geometry'])
    output_df.to_csv(output_dir / "transit_deserts.csv", index=False)
    print(f"  ✓ Saved to {output_dir / 'transit_deserts.csv'}")
    
    output_gdf.to_file(output_dir / "transit_deserts.gpkg", driver='GPKG')
    print(f"  ✓ Saved to {output_dir / 'transit_deserts.gpkg'}")
    
    profile_df.to_csv(output_dir / "equity_profile.csv", index=False)
    print(f"  ✓ Saved equity profile to {output_dir / 'equity_profile.csv'}")
    
    # Save Global Moran's I
    global_moran_df = pd.DataFrame([global_moran])
    global_moran_df.to_csv(output_dir / "global_morans.csv", index=False)
    print(f"  ✓ Saved Global Moran's I to {output_dir / 'global_morans.csv'}")
    
    # Save sensitivity analysis
    sensitivity_df.to_csv(output_dir / "sensitivity_analysis.csv", index=False)
    print(f"  ✓ Saved sensitivity analysis to {output_dir / 'sensitivity_analysis.csv'}")
    
    # Save absolute thresholds summary
    if len(thresholds_df) > 0:
        thresholds_df.to_csv(output_dir / "absolute_thresholds.csv", index=False)
        print(f"  ✓ Saved absolute thresholds to {output_dir / 'absolute_thresholds.csv'}")
    
    # Save barrier summary
    if len(barrier_summary) > 0:
        barrier_summary.to_csv(output_dir / "barrier_summary.csv", index=False)
        print(f"  ✓ Saved barrier summary to {output_dir / 'barrier_summary.csv'}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\n  Total tracts analyzed: {len(gdf)}")
    print(f"  Transit Deserts (hybrid): {gdf['transit_desert'].sum()}")
    print(f"  Percentage: {gdf['transit_desert'].mean():.1%}")
    
    if 'desert_pathway' in gdf.columns:
        pathway_counts = gdf[gdf['transit_desert']]['desert_pathway'].value_counts()
        for pathway, count in pathway_counts.items():
            print(f"    {pathway}: {count}")
    
    if 'abs_transit_desert' in gdf.columns:
        n_abs = gdf['abs_transit_desert'].sum()
        print(f"  Absolute threshold failures (pre-equity gate): {n_abs} ({n_abs/len(gdf):.1%})")
    
    print(f"\n  Global Moran's I: {global_moran['I']:.4f} (p={global_moran['p_value']:.4f})")
    
    if gdf['transit_desert'].sum() > 0:
        desert_tracts = gdf[gdf['transit_desert']]
        print(f"\n  Transit Desert characteristics:")
        print(f"    Average TVI: {desert_tracts['TVI_normalized'].mean():.1f}")
        print(f"    Average CPTA: {desert_tracts['CPTA_normalized'].mean():.1f}")
        print(f"    Average Gap: {desert_tracts['transit_gap'].mean():.1f}")
    
    print("\n" + "=" * 60)
    print("Transit desert identification complete!")
    print("Next step: python src/05_visualize.py")
    print("=" * 60)

if __name__ == "__main__":
    main()