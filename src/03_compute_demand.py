"""
03_compute_demand.py
Compute Transit Vulnerability Index (TVI) for each Census tract.

Computes two TVI variants:
  TVI (observed):  Traditional demand based on observed demographics
  TVI-latent:      Demand adjusted for forced car ownership (Allen & Farber 2021)

TVI Components (observed):
1. Zero-vehicle households (30%)
2. Population below poverty line (25%)
3. Minority population (20%)
4. Elderly population 65+ (15%)
5. Youth population ages 10-17 (10%)

TVI-latent adds:
6. Low-income car commuters (15%) — forced car ownership proxy from B08122
   Zero-vehicle weight reduced from 30% to 15% to compensate

Usage:
    python src/03_compute_demand.py

Outputs:
    - data/processed/demand_metrics.csv
    - data/processed/demand_metrics.gpkg
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

def load_acs_data():
    """
    Load ACS demographic data.
    
    Returns
    -------
    DataFrame
        ACS data with TVI component variables
    """
    acs_path = PROJECT_ROOT / "data" / "raw" / "acs" / "acs_tvi_variables.csv"
    
    if not acs_path.exists():
        raise FileNotFoundError(
            f"ACS data not found: {acs_path}\n"
            "Run 01_download_data.py first, or manually place ACS data in this location."
        )
    
    df = pd.read_csv(acs_path, dtype={'GEOID': str})
    
    print(f"  ✓ Loaded ACS data for {len(df)} tracts")
    
    return df

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
        Census tracts with geometry
    """
    tracts_path = PROJECT_ROOT / "data" / "raw" / "census" / "study_area_tracts.shp"
    
    if not tracts_path.exists():
        raise FileNotFoundError(f"Tract shapefile not found: {tracts_path}")
    
    gdf = gpd.read_file(tracts_path)
    
    # Ensure GEOID is string
    gdf['GEOID'] = gdf['GEOID'].astype(str)
    
    # Reproject
    crs_projected = config['spatial']['crs_projected']
    gdf = gdf.to_crs(crs_projected)
    
    print(f"  ✓ Loaded {len(gdf)} tract boundaries")
    
    return gdf

def compute_tvi_components(acs_df):
    """
    Compute individual TVI components from ACS data.
    
    Parameters
    ----------
    acs_df : DataFrame
        ACS data with raw variables
    
    Returns
    -------
    DataFrame
        TVI component percentages
    """
    print("Computing TVI components...")
    
    # Check if percentages are already computed
    required_cols = ['pct_zero_vehicle', 'pct_poverty', 'pct_minority', 'pct_elderly', 'pct_youth']
    # Optional forced car ownership column from B08122 (computed in 01_download_data)
    has_forced_car = 'pct_low_income_car_commute' in acs_df.columns
    
    if all(col in acs_df.columns for col in required_cols):
        print("  ✓ TVI percentages already computed in ACS data")
        keep_cols = ['GEOID'] + required_cols
        if has_forced_car:
            keep_cols += ['pct_low_income_car_commute', 
                          'n_low_income_car_commuters', 'n_low_income_workers',
                          'n_low_income_transit_commuters']
            print("  ✓ Forced car ownership proxy (B08122) found")
        else:
            print("  ⚠ Forced car ownership proxy not found — TVI-latent will not be computed")
        available = [c for c in keep_cols if c in acs_df.columns]
        return acs_df[available].copy()
    
    # Otherwise, compute from raw ACS variables
    tvi = pd.DataFrame()
    tvi['GEOID'] = acs_df['GEOID']
    
    # Zero-vehicle households
    if 'B08201_001E' in acs_df.columns and 'B08201_002E' in acs_df.columns:
        tvi['pct_zero_vehicle'] = np.where(
            acs_df['B08201_001E'] > 0,
            acs_df['B08201_002E'] / acs_df['B08201_001E'],
            0
        )
    else:
        print("  ⚠ Vehicle data not found, using placeholder")
        tvi['pct_zero_vehicle'] = 0
    
    # Poverty rate
    if 'B17001_001E' in acs_df.columns and 'B17001_002E' in acs_df.columns:
        tvi['pct_poverty'] = np.where(
            acs_df['B17001_001E'] > 0,
            acs_df['B17001_002E'] / acs_df['B17001_001E'],
            0
        )
    else:
        print("  ⚠ Poverty data not found, using placeholder")
        tvi['pct_poverty'] = 0
    
    # Minority population (non-white, non-Hispanic)
    if 'B03002_001E' in acs_df.columns and 'B03002_003E' in acs_df.columns:
        tvi['pct_minority'] = np.where(
            acs_df['B03002_001E'] > 0,
            1 - (acs_df['B03002_003E'] / acs_df['B03002_001E']),
            0
        )
    else:
        print("  ⚠ Race/ethnicity data not found, using placeholder")
        tvi['pct_minority'] = 0
    
    # Elderly population (65+)
    elderly_male_cols = ['B01001_020E', 'B01001_021E', 'B01001_022E', 
                         'B01001_023E', 'B01001_024E', 'B01001_025E']
    elderly_female_cols = ['B01001_044E', 'B01001_045E', 'B01001_046E',
                           'B01001_047E', 'B01001_048E', 'B01001_049E']
    
    if 'B01001_001E' in acs_df.columns and all(col in acs_df.columns for col in elderly_male_cols):
        elderly_pop = (
            acs_df[elderly_male_cols].sum(axis=1) + 
            acs_df[elderly_female_cols].sum(axis=1)
        )
        tvi['pct_elderly'] = np.where(
            acs_df['B01001_001E'] > 0,
            elderly_pop / acs_df['B01001_001E'],
            0
        )
    else:
        print("  ⚠ Age data not found, using placeholder")
        tvi['pct_elderly'] = 0
    
    # Youth population (ages 10-17 — independent transit users)
    # Male 10-14 (005E) + Male 15-17 (006E)
    # Female 10-14 (029E) + Female 15-17 (030E)
    youth_male_cols = ['B01001_005E', 'B01001_006E']
    youth_female_cols = ['B01001_029E', 'B01001_030E']
    
    if 'B01001_001E' in acs_df.columns and all(col in acs_df.columns for col in youth_male_cols):
        youth_pop = (
            acs_df[youth_male_cols].sum(axis=1) + 
            acs_df[youth_female_cols].sum(axis=1)
        )
        tvi['pct_youth'] = np.where(
            acs_df['B01001_001E'] > 0,
            youth_pop / acs_df['B01001_001E'],
            0
        )
    else:
        print("  ⚠ Youth age data not found, using placeholder")
        tvi['pct_youth'] = 0
    
    # Cap percentages at 1.0
    for col in required_cols:
        tvi[col] = tvi[col].clip(0, 1)
    
    return tvi

def compute_tvi_score(tvi_components, config):
    """
    Compute Transit Vulnerability Index — both observed and latent variants.
    
    TVI (observed): Traditional weighted z-score composite
    TVI_latent:     Adds forced car ownership proxy, reduces zero-vehicle weight
    
    Parameters
    ----------
    tvi_components : DataFrame
        Individual TVI component percentages
    config : dict
        Configuration dictionary with weights
    
    Returns
    -------
    DataFrame
        TVI components with both TVI and TVI_latent scores
    """
    print("Computing TVI scores...")
    
    # ── Get observed TVI weights from config ──
    tvi_config = config.get('tvi', {})
    components_config = tvi_config.get('components', {})
    
    if components_config and all('weight' in v for v in components_config.values()):
        weights = {k: v['weight'] for k, v in components_config.items()}
        print(f"\n  TVI Weights (from tvi.components):")
    else:
        weights = config.get('census', {}).get('tvi_weights', {
            'zero_vehicle': 0.30,
            'poverty': 0.25,
            'minority': 0.20,
            'elderly': 0.15,
            'youth': 0.10
        })
        print(f"\n  TVI Weights (from census.tvi_weights):")
    for component, weight in weights.items():
        print(f"    {component}: {weight}")
    
    # ── Map column names to weights (observed TVI) ──
    weight_mapping = {
        'pct_zero_vehicle': weights['zero_vehicle'],
        'pct_poverty': weights['poverty'],
        'pct_minority': weights['minority'],
        'pct_elderly': weights['elderly'],
        'pct_youth': weights.get('youth', 0)
    }
    
    # ── Compute z-scores for base components ──
    base_cols = ['pct_zero_vehicle', 'pct_poverty', 'pct_minority', 'pct_elderly', 'pct_youth']
    for col in base_cols:
        z_col = f'z_{col}'
        if col in tvi_components.columns:
            values = tvi_components[col].fillna(0)
            tvi_components[z_col] = zscore(values) if values.std() > 0 else 0
        else:
            tvi_components[z_col] = 0
    
    # ── Compute observed TVI ──
    tvi_components['TVI'] = sum(
        weight_mapping[col] * tvi_components[f'z_{col}']
        for col in base_cols
    )
    
    # Normalize 0-100
    tvi_min = tvi_components['TVI'].min()
    tvi_max = tvi_components['TVI'].max()
    if tvi_max > tvi_min:
        tvi_components['TVI_normalized'] = (
            (tvi_components['TVI'] - tvi_min) / (tvi_max - tvi_min) * 100
        )
    else:
        tvi_components['TVI_normalized'] = 50
    
    # ── Compute TVI-latent (with forced car ownership) ──
    has_forced_car = 'pct_low_income_car_commute' in tvi_components.columns
    fc_config = tvi_config.get('forced_car_ownership', {})
    fc_enabled = fc_config.get('enabled', True)  # default enabled if data exists
    
    if has_forced_car and fc_enabled:
        print(f"\n  Computing TVI-latent (forced car ownership adjusted)...")
        
        # Z-score the forced car ownership proxy
        fc_values = tvi_components['pct_low_income_car_commute'].fillna(0)
        tvi_components['z_pct_low_income_car_commute'] = (
            zscore(fc_values) if fc_values.std() > 0 else 0
        )
        
        # Latent weights: read from config if available, else split zero-vehicle
        fc_config = tvi_config.get('forced_car_ownership', {})
        latent_weights_config = fc_config.get('latent_weights', {})
        
        if latent_weights_config:
            latent_weights = {
                'pct_zero_vehicle': latent_weights_config['zero_vehicle'],
                'pct_low_income_car_commute': latent_weights_config['forced_car'],
                'pct_poverty': latent_weights_config['poverty'],
                'pct_minority': latent_weights_config['minority'],
                'pct_elderly': latent_weights_config['elderly'],
                'pct_youth': latent_weights_config.get('youth', 0)
            }
        else:
            # Fallback: split zero-vehicle weight in half
            latent_weights = {
                'pct_zero_vehicle': weights['zero_vehicle'] / 2,
                'pct_low_income_car_commute': weights['zero_vehicle'] / 2,
                'pct_poverty': weights['poverty'],
                'pct_minority': weights['minority'],
                'pct_elderly': weights['elderly'],
                'pct_youth': weights.get('youth', 0)
            }
        
        print(f"\n  TVI-latent Weights:")
        for component, weight in latent_weights.items():
            print(f"    {component}: {weight}")
        print(f"    Sum: {sum(latent_weights.values()):.2f}")
        
        # Compute TVI-latent
        tvi_components['TVI_latent'] = sum(
            latent_weights[col] * tvi_components[f'z_{col}']
            for col in latent_weights
        )
        
        # Normalize 0-100
        lat_min = tvi_components['TVI_latent'].min()
        lat_max = tvi_components['TVI_latent'].max()
        if lat_max > lat_min:
            tvi_components['TVI_latent_normalized'] = (
                (tvi_components['TVI_latent'] - lat_min) / (lat_max - lat_min) * 100
            )
        else:
            tvi_components['TVI_latent_normalized'] = 50
        
        # ── Diagnostics: compare observed vs latent ──
        corr = tvi_components['TVI_normalized'].corr(tvi_components['TVI_latent_normalized'])
        print(f"\n  TVI vs TVI-latent correlation: r = {corr:.3f}")
        
        # Tracts that shift most between observed and latent
        tvi_components['tvi_latent_shift'] = (
            tvi_components['TVI_latent_normalized'] - tvi_components['TVI_normalized']
        )
        top_gainers = tvi_components.nlargest(5, 'tvi_latent_shift')
        print(f"\n  Top 5 tracts gaining demand under TVI-latent:")
        for _, row in top_gainers.iterrows():
            print(f"    {row['GEOID']}: TVI={row['TVI_normalized']:.1f} → "
                  f"TVI_latent={row['TVI_latent_normalized']:.1f} "
                  f"(+{row['tvi_latent_shift']:.1f}, "
                  f"car_commute={row['pct_low_income_car_commute']:.0%})")
    else:
        print(f"\n  ⚠ Forced car ownership data not available — TVI-latent = TVI (observed)")
        tvi_components['TVI_latent'] = tvi_components['TVI']
        tvi_components['TVI_latent_normalized'] = tvi_components['TVI_normalized']
        tvi_components['tvi_latent_shift'] = 0
    
    return tvi_components

def create_demand_summary(tvi_df):
    """
    Create summary statistics for demand analysis.
    
    Parameters
    ----------
    tvi_df : DataFrame
        TVI data
    
    Returns
    -------
    dict
        Summary statistics
    """
    summary = {
        'n_tracts': len(tvi_df),
        'tvi_mean': tvi_df['TVI'].mean(),
        'tvi_std': tvi_df['TVI'].std(),
        'tvi_min': tvi_df['TVI'].min(),
        'tvi_max': tvi_df['TVI'].max(),
        'components': {}
    }
    
    component_cols = ['pct_zero_vehicle', 'pct_poverty', 'pct_minority', 'pct_elderly', 'pct_youth']
    if 'pct_low_income_car_commute' in tvi_df.columns:
        component_cols.append('pct_low_income_car_commute')
    
    for col in component_cols:
        if col in tvi_df.columns:
            summary['components'][col] = {
                'mean': tvi_df[col].mean(),
                'std': tvi_df[col].std(),
                'min': tvi_df[col].min(),
                'max': tvi_df[col].max()
            }
    
    return summary

def main():
    """Main function to compute demand metrics."""
    print("=" * 60)
    print("Transit Desert Pipeline: Demand Metrics (TVI)")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Paths
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "-" * 40)
    print("Loading data...")
    print("-" * 40)
    
    acs_df = load_acs_data()
    tracts_gdf = load_tracts(config)
    
    # Compute TVI components
    print("\n" + "-" * 40)
    print("Computing TVI components...")
    print("-" * 40)
    
    tvi_components = compute_tvi_components(acs_df)
    
    # Compute TVI score
    print("\n" + "-" * 40)
    print("Computing TVI score...")
    print("-" * 40)
    
    tvi_df = compute_tvi_score(tvi_components, config)
    
    # Ensure all tracts are included
    all_tracts = tracts_gdf[['GEOID']].copy()
    all_tracts['GEOID'] = all_tracts['GEOID'].astype(str)
    tvi_df['GEOID'] = tvi_df['GEOID'].astype(str)
    tvi_df = all_tracts.merge(tvi_df, on='GEOID', how='left').fillna(0)
    
    # Save results
    print("\n" + "-" * 40)
    print("Saving results...")
    print("-" * 40)
    
    # CSV
    tvi_df.to_csv(output_dir / "demand_metrics.csv", index=False)
    print(f"  ✓ Saved to {output_dir / 'demand_metrics.csv'}")
    
    # GeoPackage (with geometry)
    tvi_gdf = tracts_gdf.merge(tvi_df, on='GEOID')
    tvi_gdf.to_file(output_dir / "demand_metrics.gpkg", driver='GPKG')
    print(f"  ✓ Saved to {output_dir / 'demand_metrics.gpkg'}")
    
    # Summary statistics
    print("\n" + "-" * 40)
    print("Summary Statistics")
    print("-" * 40)
    
    summary = create_demand_summary(tvi_df)
    
    print(f"\n  Tracts analyzed: {summary['n_tracts']}")
    
    print(f"\n  TVI Score:")
    print(f"    Mean: {summary['tvi_mean']:.3f}")
    print(f"    Std:  {summary['tvi_std']:.3f}")
    print(f"    Min:  {summary['tvi_min']:.3f}")
    print(f"    Max:  {summary['tvi_max']:.3f}")
    
    print(f"\n  TVI Normalized (0-100):")
    print(f"    Mean: {tvi_df['TVI_normalized'].mean():.1f}")
    print(f"    Min:  {tvi_df['TVI_normalized'].min():.1f}")
    print(f"    Max:  {tvi_df['TVI_normalized'].max():.1f}")
    
    if 'TVI_latent_normalized' in tvi_df.columns:
        print(f"\n  TVI-latent Normalized (0-100):")
        print(f"    Mean: {tvi_df['TVI_latent_normalized'].mean():.1f}")
        print(f"    Min:  {tvi_df['TVI_latent_normalized'].min():.1f}")
        print(f"    Max:  {tvi_df['TVI_latent_normalized'].max():.1f}")
        shift = tvi_df['tvi_latent_shift']
        print(f"    Mean shift from observed: {shift.mean():+.1f}")
        print(f"    Max positive shift:       {shift.max():+.1f}")
        print(f"    Max negative shift:       {shift.min():+.1f}")
    
    print(f"\n  Component Statistics:")
    for col, stats in summary['components'].items():
        print(f"\n    {col}:")
        print(f"      Mean: {stats['mean']:.1%}")
        print(f"      Max:  {stats['max']:.1%}")
    
    # Identify high-demand tracts (top quartile)
    q75 = tvi_df['TVI'].quantile(0.75)
    high_demand = tvi_df[tvi_df['TVI'] >= q75]
    print(f"\n  High-demand tracts (TVI >= {q75:.2f}): {len(high_demand)}")
    
    print("\n" + "=" * 60)
    print("Demand metrics complete!")
    print("Next step: python src/04_identify_deserts.py")
    print("=" * 60)

if __name__ == "__main__":
    main()