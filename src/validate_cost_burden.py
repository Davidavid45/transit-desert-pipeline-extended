"""
validate_cost_burden.py
Validate TVI and TVI-latent against CNT H+T Index transportation cost burden.

Downloads tract-level data from CNT's Housing + Transportation Affordability Index
and computes correlations against pipeline demand/supply metrics for both cities.

Validates TVI-latent against external cost burden data: tracts flagged as
high-latent-need should also show high transportation cost burden, confirming
the approach without including cost burden in the gap equation directly
(avoiding circularity).

Usage:
    1. Download H+T Index CSV files manually from: https://htaindex.cnt.org/download/
       - Select "Census Tract" geography
       - Download for Maryland (FIPS 24) and/or Tennessee (FIPS 47)
       - Save as: data/validation/htaindex2022_data_tracts_24.csv
                   data/validation/htaindex2022_data_tracts_47.csv
    
    2. Run: python src/validate_cost_burden.py

    OR: If you have the files already, just run the script and it will guide you.

Outputs:
    - data/validation/validation_results.csv
    - data/validation/fig_cost_burden_correlation.png
    - data/validation/fig_cost_burden_maps.png
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_pipeline_data(city_dir):
    """Load transit_deserts.csv or demand_metrics.csv from a pipeline run."""
    # Try transit_deserts first (has everything merged)
    deserts_path = city_dir / "data" / "processed" / "transit_deserts.csv"
    if deserts_path.exists():
        df = pd.read_csv(deserts_path, dtype={'GEOID': str})
        print(f"  ✓ Loaded {len(df)} tracts from {deserts_path.name}")
        return df
    
    # Fall back to demand_metrics
    demand_path = city_dir / "data" / "processed" / "demand_metrics.csv"
    if demand_path.exists():
        df = pd.read_csv(demand_path, dtype={'GEOID': str})
        print(f"  ✓ Loaded {len(df)} tracts from {demand_path.name}")
        return df
    
    print(f"  ✗ No pipeline data found in {city_dir}")
    return None


def load_ht_index(ht_path):
    """
    Load CNT H+T Index data.
    
    Key variables we need (CNT uses _ami / _80ami / _nmi income-tier suffixes):
    - t_ami: Transportation costs as % of AMI income (the cost burden)
    - t_cost_ami: Annual transportation costs ($)
    - autos_per_hh_ami: Average autos per household
    - vmt_per_hh_ami: Annual vehicle miles traveled per household
    - transit_trips_ami: Annual transit trips per household
    
    We default to the _ami (Area Median Income) tier.
    """
    if not ht_path.exists():
        print(f"  ✗ H+T Index file not found: {ht_path}")
        print(f"    → Download from https://htaindex.cnt.org/download/")
        print(f"    → Select 'Census Tract', then your county")
        return None
    
    df = pd.read_csv(ht_path, dtype={'tract': str, 'GEOID': str, 'geoid': str})
    print(f"  ✓ Loaded {len(df)} tracts from {ht_path.name}")
    print(f"    Columns: {list(df.columns[:15])}...")
    
    # Normalize GEOID column name
    geoid_candidates = ['tract', 'GEOID', 'geoid', 'tractid', 'TRACTID', 'geo_id']
    geoid_col = None
    for col in geoid_candidates:
        if col in df.columns:
            geoid_col = col
            break
    
    if geoid_col is None:
        # Try to find a column that looks like a GEOID (11-digit string)
        for col in df.columns:
            sample = df[col].dropna().astype(str).iloc[0] if len(df) > 0 else ''
            cleaned = sample.strip('"')
            if len(cleaned) == 11 and cleaned.isdigit():
                geoid_col = col
                break
    
    if geoid_col:
        # Strip embedded quotes from CNT download format and zero-pad
        df['GEOID'] = (df[geoid_col].astype(str)
                        .str.strip('"')
                        .str.strip()
                        .str.zfill(11))
        print(f"    Using '{geoid_col}' as GEOID (cleaned quotes)")
    else:
        print(f"    ⚠ Could not identify GEOID column. Available: {list(df.columns)}")
        return None
    
    # ── Map H+T Index income-tier columns to standard names ──
    # CNT H+T Index 2022 uses _ami / _80ami / _nmi suffixes for income tiers.
    # We use the _ami (Area Median Income) tier as default.
    tier_mapping = {
        't_costs_pct':  ['t_ami', 't_80ami', 't_nmi'],
        't_costs':      ['t_cost_ami', 't_cost_80ami', 't_cost_nmi'],
        'autos_per_hh': ['autos_per_hh_ami', 'autos_per_hh_80ami', 'autos_per_hh_nmi'],
        'vmt_per_hh':   ['vmt_per_hh_ami', 'vmt_per_hh_80ami', 'vmt_per_hh_nmi'],
        'transit_trips': ['transit_trips_ami', 'transit_trips_80ami', 'transit_trips_nmi'],
    }
    
    mapped = {}
    for target, tier_cols in tier_mapping.items():
        # Prefer _ami tier, fall back to others
        for col in tier_cols:
            if col in df.columns:
                mapped[target] = col
                break
    
    print(f"    Mapped variables: {mapped}")
    
    # Rename to standard names
    rename_map = {v: k for k, v in mapped.items()}
    df = df.rename(columns=rename_map)
    
    # Convert mapped columns to numeric (they may be read as strings)
    for std_name in mapped:
        if std_name in df.columns:
            df[std_name] = pd.to_numeric(df[std_name], errors='coerce')
    
    return df


def compute_correlations(pipeline_df, ht_df, city_name):
    """Compute correlations between pipeline metrics and H+T cost burden."""
    # Merge on GEOID
    merged = pipeline_df.merge(ht_df, on='GEOID', how='inner', suffixes=('', '_ht'))
    print(f"\n  Matched {len(merged)} tracts for {city_name}")
    
    if len(merged) < 10:
        print(f"  ⚠ Too few matches — check GEOID format")
        print(f"    Pipeline GEOIDs sample: {pipeline_df['GEOID'].head(3).tolist()}")
        print(f"    H+T GEOIDs sample: {ht_df['GEOID'].head(3).tolist()}")
        return None
    
    # Define correlation pairs
    pipeline_vars = ['TVI_normalized', 'TVI_latent_normalized', 'CPTA_normalized',
                     'transit_gap', 'pct_zero_vehicle', 'pct_poverty',
                     'pct_low_income_car_commute']
    ht_vars = ['t_costs_pct', 't_costs', 'autos_per_hh', 'vmt_per_hh', 'transit_trips']
    
    results = []
    for pvar in pipeline_vars:
        if pvar not in merged.columns:
            continue
        for hvar in ht_vars:
            if hvar not in merged.columns:
                continue
            
            valid = merged[[pvar, hvar]].dropna()
            if len(valid) < 10:
                continue
            
            r_pearson, p_pearson = pearsonr(valid[pvar], valid[hvar])
            r_spearman, p_spearman = spearmanr(valid[pvar], valid[hvar])
            
            results.append({
                'city': city_name,
                'pipeline_var': pvar,
                'ht_var': hvar,
                'n': len(valid),
                'r_pearson': r_pearson,
                'p_pearson': p_pearson,
                'r_spearman': r_spearman,
                'p_spearman': p_spearman
            })
    
    results_df = pd.DataFrame(results)
    
    # Print key results
    print(f"\n  Key correlations for {city_name}:")
    print(f"  {'Pipeline Variable':<30} {'H+T Variable':<20} {'r (Pearson)':<12} {'p-value':<10}")
    print(f"  {'-'*72}")
    
    for _, row in results_df.iterrows():
        sig = '***' if row['p_pearson'] < 0.001 else '**' if row['p_pearson'] < 0.01 else '*' if row['p_pearson'] < 0.05 else ''
        print(f"  {row['pipeline_var']:<30} {row['ht_var']:<20} {row['r_pearson']:>8.3f}{sig:<4} {row['p_pearson']:.2e}")
    
    return results_df, merged


def create_validation_figure(merged_baltimore, merged_nashville, output_path):
    """Create key validation scatter plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    cities = [
        ('Baltimore', merged_baltimore, axes[0]),
        ('Nashville', merged_nashville, axes[1])
    ]
    
    for city_name, df, axrow in cities:
        if df is None:
            for ax in axrow:
                ax.text(0.5, 0.5, f'{city_name}\nNo data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=14)
                ax.axis('off')
            continue
        
        # Panel 1: TVI vs cost burden
        if 'TVI_normalized' in df.columns and 't_costs_pct' in df.columns:
            ax = axrow[0]
            valid = df[['TVI_normalized', 't_costs_pct']].dropna()
            ax.scatter(valid['TVI_normalized'], valid['t_costs_pct'], alpha=0.5, s=30, c='#d73027')
            r, p = pearsonr(valid['TVI_normalized'], valid['t_costs_pct'])
            ax.set_xlabel('TVI (Observed)')
            ax.set_ylabel('Transportation Cost Burden (%)')
            ax.set_title(f'{city_name}: TVI vs Cost Burden\nr={r:.3f}, p={p:.2e}')
            ax.grid(alpha=0.2)
        
        # Panel 2: TVI-latent vs cost burden
        if 'TVI_latent_normalized' in df.columns and 't_costs_pct' in df.columns:
            ax = axrow[1]
            valid = df[['TVI_latent_normalized', 't_costs_pct']].dropna()
            ax.scatter(valid['TVI_latent_normalized'], valid['t_costs_pct'], alpha=0.5, s=30, c='#4575b4')
            r, p = pearsonr(valid['TVI_latent_normalized'], valid['t_costs_pct'])
            ax.set_xlabel('TVI-latent (Forced Car Adjusted)')
            ax.set_ylabel('Transportation Cost Burden (%)')
            ax.set_title(f'{city_name}: TVI-latent vs Cost Burden\nr={r:.3f}, p={p:.2e}')
            ax.grid(alpha=0.2)
        
        # Panel 3: CPTA vs cost burden (expect negative correlation)
        if 'CPTA_normalized' in df.columns and 't_costs_pct' in df.columns:
            ax = axrow[2]
            valid = df[['CPTA_normalized', 't_costs_pct']].dropna()
            ax.scatter(valid['CPTA_normalized'], valid['t_costs_pct'], alpha=0.5, s=30, c='#1a9850')
            r, p = pearsonr(valid['CPTA_normalized'], valid['t_costs_pct'])
            ax.set_xlabel('CPTA (Transit Supply)')
            ax.set_ylabel('Transportation Cost Burden (%)')
            ax.set_title(f'{city_name}: CPTA vs Cost Burden\nr={r:.3f}, p={p:.2e}')
            ax.grid(alpha=0.2)
    
    fig.suptitle('Option 5 Validation: Pipeline Metrics vs. CNT Transportation Cost Burden',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n  ✓ Saved: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Validation: TVI/TVI-latent vs. Transportation Cost Burden")
    print("Data source: CNT Housing + Transportation Affordability Index")
    print("=" * 60)
    
    val_dir = PROJECT_ROOT / "data" / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Load H+T Index data ──
    print("\n" + "-" * 40)
    print("Step 1: Load H+T Index data")
    print("-" * 40)
    print("\n  If you haven't downloaded the data yet:")
    print("  1. Go to https://htaindex.cnt.org/download/")
    print("  2. Select 'Census Tract' as geography level")
    print("  3. Download for Baltimore City county and Davidson County")
    print("  4. Save CSV files to data/validation/")
    
    # Try multiple filename patterns
    ht_baltimore = None
    ht_nashville = None
    
    for pattern in ['htaindex2022_data_tracts_24.csv', 'htaindex*tracts*24*.csv',
                     'ht_baltimore.csv', 'baltimore_ht.csv',
                     'htaindex_tract_24510*.csv', '24510*.csv']:
        matches = list(val_dir.glob(pattern))
        if matches:
            ht_baltimore = load_ht_index(matches[0])
            break
    
    for pattern in ['htaindex2022_data_tracts_47.csv', 'htaindex*tracts*47*.csv',
                     'ht_nashville.csv', 'ht_index_nashville.csv', 'nashville_ht.csv',
                     'htaindex_tract_47037*.csv', '47037*.csv']:
        matches = list(val_dir.glob(pattern))
        if matches:
            ht_nashville = load_ht_index(matches[0])
            break
    
    if ht_baltimore is None and ht_nashville is None:
        # List what's in val_dir
        existing = list(val_dir.glob('*.csv'))
        if existing:
            print(f"\n  Found CSV files in {val_dir}:")
            for f in existing:
                print(f"    - {f.name}")
            print(f"\n  Expected naming: 'htaindex2022_data_tracts_24.csv' (Baltimore)")
            print(f"                   'htaindex2022_data_tracts_47.csv' (Nashville)")
        else:
            print(f"\n  No CSV files found in {val_dir}/")
            print(f"  Please download from https://htaindex.cnt.org/download/")
        return
    
    # ── Load pipeline data ──
    print("\n" + "-" * 40)
    print("Step 2: Load pipeline output data")
    print("-" * 40)
    
    # Load pipeline outputs from this project's data/processed/
    pipeline_df = load_pipeline_data(PROJECT_ROOT)
    
    # Split by state FIPS to match each H+T file
    pipeline_baltimore = None
    pipeline_nashville = None
    if pipeline_df is not None:
        pipeline_df['state_fips'] = pipeline_df['GEOID'].str[:2]
        balt_mask = pipeline_df['state_fips'] == '24'
        nash_mask = pipeline_df['state_fips'] == '47'
        if balt_mask.any():
            pipeline_baltimore = pipeline_df[balt_mask].copy()
            print(f"  → {len(pipeline_baltimore)} Baltimore tracts")
        if nash_mask.any():
            pipeline_nashville = pipeline_df[nash_mask].copy()
            print(f"  → {len(pipeline_nashville)} Nashville tracts")
        if pipeline_baltimore is None and pipeline_nashville is None:
            # All tracts belong to the same state — assign to whichever H+T file exists
            state = pipeline_df['state_fips'].iloc[0]
            if state == '24':
                pipeline_baltimore = pipeline_df
            elif state == '47':
                pipeline_nashville = pipeline_df
            else:
                # Fall back: use for whichever H+T dataset we have
                if ht_nashville is not None:
                    pipeline_nashville = pipeline_df
                elif ht_baltimore is not None:
                    pipeline_baltimore = pipeline_df
        pipeline_df.drop(columns=['state_fips'], inplace=True)
    
    # ── Compute correlations ──
    print("\n" + "-" * 40)
    print("Step 3: Compute correlations")
    print("-" * 40)
    
    all_results = []
    merged_balt = None
    merged_nash = None
    
    if ht_baltimore is not None and pipeline_baltimore is not None:
        results_balt, merged_balt = compute_correlations(pipeline_baltimore, ht_baltimore, "Baltimore")
        if results_balt is not None:
            all_results.append(results_balt)
    
    if ht_nashville is not None and pipeline_nashville is not None:
        results_nash, merged_nash = compute_correlations(pipeline_nashville, ht_nashville, "Nashville")
        if results_nash is not None:
            all_results.append(results_nash)
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(val_dir / "validation_results.csv", index=False)
        print(f"\n  ✓ Saved correlation results to {val_dir / 'validation_results.csv'}")
        
        # ── Key finding ──
        print("\n" + "=" * 60)
        print("KEY VALIDATION FINDINGS")
        print("=" * 60)
        
        for city in combined['city'].unique():
            city_df = combined[city == combined['city']]
            
            # TVI vs cost burden
            tvi_row = city_df[(city_df['pipeline_var'] == 'TVI_normalized') & 
                              (city_df['ht_var'] == 't_costs_pct')]
            latent_row = city_df[(city_df['pipeline_var'] == 'TVI_latent_normalized') & 
                                 (city_df['ht_var'] == 't_costs_pct')]
            cpta_row = city_df[(city_df['pipeline_var'] == 'CPTA_normalized') & 
                                (city_df['ht_var'] == 't_costs_pct')]
            
            print(f"\n  {city}:")
            if len(tvi_row) > 0:
                r = tvi_row.iloc[0]['r_pearson']
                print(f"    TVI vs cost burden:        r = {r:+.3f}")
            if len(latent_row) > 0:
                r = latent_row.iloc[0]['r_pearson']
                print(f"    TVI-latent vs cost burden:  r = {r:+.3f}")
            if len(cpta_row) > 0:
                r = cpta_row.iloc[0]['r_pearson']
                print(f"    CPTA vs cost burden:        r = {r:+.3f} (expect negative)")
    
    # ── Create figure ──
    if merged_balt is not None or merged_nash is not None:
        create_validation_figure(merged_balt, merged_nash,
                                  output_path=val_dir / "fig_cost_burden_validation.png")
    
    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()