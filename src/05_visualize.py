"""
05_visualize.py
Generate maps and visualizations for transit desert analysis.

Outputs:
- Maps: CPTA, TVI, Transit Gap, LISA clusters, Classification,
        Absolute vs Relative desert comparison
- Figures: Distribution plots, scatter plots, equity comparison,
           CPTA correlation heatmap, Moran scatterplot, sensitivity analysis,
           Absolute threshold failure heatmap, desert rate comparison
- Tables: Summary statistics

Usage:
    python src/05_visualize.py
"""

import sys
import yaml
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

try:
    import contextily as ctx
    CTX_AVAILABLE = True
except ImportError:
    ctx = None
    CTX_AVAILABLE = False
    print("contextily not installed. Basemaps will not be added.")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_results():
    results_path = PROJECT_ROOT / "data" / "processed" / "transit_deserts.gpkg"
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    gdf = gpd.read_file(results_path)
    print(f"  ✓ Loaded results for {len(gdf)} tracts")
    return gdf


def setup_plot_style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10


def create_choropleth_map(gdf, column, title, cmap='RdYlBu_r', output_path=None, 
                          legend_title=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    gdf_plot = gdf.to_crs(epsg=3857)
    gdf_plot.plot(
        column=column, cmap=cmap, linewidth=0.5, edgecolor='white',
        legend=True,
        legend_kwds={'label': legend_title or column, 'orientation': 'horizontal',
                     'shrink': 0.6, 'pad': 0.05},
        ax=ax
    )
    if CTX_AVAILABLE:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.5)
        except:
            pass
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_axis_off()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_classification_map(gdf, output_path=None, city_name=""):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    gdf_plot = gdf.to_crs(epsg=3857)
    colors = {
        'Transit Desert': '#d73027', 'Transit Stressed': '#fc8d59',
        'Underserved': '#fee090', 'Well-Served': '#91bfdb'
    }
    for category, color in colors.items():
        subset = gdf_plot[gdf_plot['classification_simple'] == category]
        if len(subset) > 0:
            subset.plot(color=color, linewidth=0.5, edgecolor='white', ax=ax)
    if CTX_AVAILABLE:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.5)
        except:
            pass
    patches = [mpatches.Patch(color=color, label=f"{cat} ({len(gdf_plot[gdf_plot['classification_simple']==cat])})") 
               for cat, color in colors.items()]
    ax.legend(handles=patches, loc='lower right', frameon=True, fontsize=10, title='Classification')
    title = 'Transit Desert Classification'
    if city_name:
        title += f'\n{city_name}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_axis_off()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_lisa_map(gdf, output_path=None, city_name=""):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    gdf_plot = gdf.to_crs(epsg=3857)
    colors = {
        'High-High': '#d7191c', 'Low-Low': '#2c7bb6',
        'High-Low': '#fdae61', 'Low-High': '#abd9e9',
        'Not Significant': '#eeeeee'
    }
    for cluster, color in colors.items():
        subset = gdf_plot[gdf_plot['lisa_cluster'] == cluster]
        if len(subset) > 0:
            subset.plot(color=color, linewidth=0.5, edgecolor='white', ax=ax)
    if CTX_AVAILABLE:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.5)
        except:
            pass
    patches = [mpatches.Patch(color=color, label=f"{cat} ({len(gdf_plot[gdf_plot['lisa_cluster']==cat])})") 
               for cat, color in colors.items()]
    ax.legend(handles=patches, loc='lower right', frameon=True, fontsize=10, title='LISA Clusters (p<0.05)')
    title = "Local Moran's I Clusters (Transit Gap)"
    if city_name:
        title += f'\n{city_name}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_axis_off()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_scatter_plot(gdf, output_path=None, city_name=""):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = {
        'Transit Desert': '#d73027', 'Transit Stressed': '#fc8d59',
        'Underserved': '#fee090', 'Well-Served': '#91bfdb'
    }
    for category, color in colors.items():
        subset = gdf[gdf['classification_simple'] == category]
        ax.scatter(subset['CPTA_normalized'], subset['TVI_normalized'],
                   c=color, label=f"{category} ({len(subset)})", alpha=0.7, s=50,
                   edgecolors='white', linewidth=0.5)
    cpta_median = gdf['CPTA_normalized'].median()
    tvi_median = gdf['TVI_normalized'].median()
    ax.axvline(x=cpta_median, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=tvi_median, color='gray', linestyle='--', alpha=0.5)
    ax.text(cpta_median/2, tvi_median + (100-tvi_median)/2, 'Transit\nDesert', 
            ha='center', va='center', fontsize=12, alpha=0.4, fontweight='bold')
    ax.text(cpta_median + (100-cpta_median)/2, tvi_median + (100-tvi_median)/2, 'Transit\nStressed', 
            ha='center', va='center', fontsize=12, alpha=0.4, fontweight='bold')
    ax.text(cpta_median/2, tvi_median/2, 'Under-\nserved', 
            ha='center', va='center', fontsize=12, alpha=0.4, fontweight='bold')
    ax.text(cpta_median + (100-cpta_median)/2, tvi_median/2, 'Well-\nServed', 
            ha='center', va='center', fontsize=12, alpha=0.4, fontweight='bold')
    ax.set_xlabel('Transit Supply (CPTA Score)', fontsize=12)
    ax.set_ylabel('Transit Demand (TVI Score)', fontsize=12)
    title = 'Transit Supply vs. Demand by Census Tract'
    if city_name:
        title += f'\n{city_name}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_distribution_plots(gdf, output_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1 = axes[0, 0]
    sns.histplot(gdf['CPTA_normalized'], kde=True, ax=ax1, color='steelblue', bins=20)
    ax1.axvline(x=gdf['CPTA_normalized'].median(), color='red', linestyle='--', linewidth=2, 
                label=f'Median: {gdf["CPTA_normalized"].median():.1f}')
    ax1.set_xlabel('CPTA Score (0-100)')
    ax1.set_title('Transit Supply Distribution', fontweight='bold')
    ax1.legend()
    
    ax2 = axes[0, 1]
    sns.histplot(gdf['TVI_normalized'], kde=True, ax=ax2, color='coral', bins=20)
    ax2.axvline(x=gdf['TVI_normalized'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {gdf["TVI_normalized"].median():.1f}')
    ax2.set_xlabel('TVI Score (0-100)')
    ax2.set_title('Transit Demand Distribution', fontweight='bold')
    ax2.legend()
    
    ax3 = axes[1, 0]
    sns.histplot(gdf['transit_gap'], kde=True, ax=ax3, color='purple', bins=20)
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax3.axvline(x=gdf['transit_gap'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {gdf["transit_gap"].median():.1f}')
    ax3.set_xlabel('Transit Gap (TVI - CPTA)')
    ax3.set_title('Transit Gap Distribution', fontweight='bold')
    ax3.legend()
    
    ax4 = axes[1, 1]
    class_counts = gdf['classification_simple'].value_counts()
    colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb']
    class_order = ['Transit Desert', 'Transit Stressed', 'Underserved', 'Well-Served']
    class_counts = class_counts.reindex(class_order).fillna(0)
    bars = ax4.bar(range(len(class_counts)), class_counts.values, color=colors)
    ax4.set_xticks(range(len(class_counts)))
    ax4.set_xticklabels(class_counts.index, rotation=45, ha='right')
    ax4.set_ylabel('Number of Tracts')
    ax4.set_title('Transit Desert Classification', fontweight='bold')
    for bar, count in zip(bars, class_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(int(count)), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_equity_comparison(gdf, output_path=None):
    """Create equity comparison bar chart with significance stars."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    deserts = gdf[gdf['transit_desert'] == True]
    non_deserts = gdf[gdf['transit_desert'] == False]
    
    metrics = {
        'pct_zero_vehicle': 'Zero-Vehicle HH',
        'pct_poverty': 'Below Poverty',
        'pct_minority': 'Minority Pop.',
        'pct_elderly': 'Elderly (65+)',
        'pct_youth': 'Youth (10-17)'
    }
    
    available_metrics = {k: v for k, v in metrics.items() if k in gdf.columns}
    
    if not available_metrics:
        print("  ⚠ No demographic columns found")
        plt.close()
        return
    
    x = np.arange(len(available_metrics))
    width = 0.35
    
    desert_vals = [deserts[m].mean() * 100 for m in available_metrics.keys()]
    non_desert_vals = [non_deserts[m].mean() * 100 for m in available_metrics.keys()]
    
    # Mann-Whitney U significance stars
    from scipy.stats import mannwhitneyu
    sig_stars = []
    for m in available_metrics.keys():
        d_vals = deserts[m].dropna()
        nd_vals = non_deserts[m].dropna()
        if len(d_vals) > 0 and len(nd_vals) > 0:
            _, p = mannwhitneyu(d_vals, nd_vals, alternative='two-sided')
            if p < 0.001:
                sig_stars.append('***')
            elif p < 0.01:
                sig_stars.append('**')
            elif p < 0.05:
                sig_stars.append('*')
            else:
                sig_stars.append('')
        else:
            sig_stars.append('')
    
    bars1 = ax.bar(x - width/2, desert_vals, width, label='Transit Deserts', color='#d73027')
    bars2 = ax.bar(x + width/2, non_desert_vals, width, label='Non-Deserts', color='#91bfdb')
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Demographic Comparison: Transit Deserts vs. Non-Deserts', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(available_metrics.values(), fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars1, desert_vals)):
        label = f'{val:.1f}%'
        if sig_stars[i]:
            label += f' {sig_stars[i]}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, label,
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, non_desert_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Significance legend
    ax.text(0.98, 0.02, '* p<0.05  ** p<0.01  *** p<0.001\n(Mann-Whitney U test)',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            color='gray', style='italic')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


# ============================================================
# NEW VISUALIZATIONS
# ============================================================

def create_correlation_heatmap(output_path=None):
    """Create heatmap of CPTA component correlations."""
    corr_path = PROJECT_ROOT / "data" / "processed" / "cpta_correlation_matrix.csv"
    
    if not corr_path.exists():
        print("  ⚠ Correlation matrix not found, skipping")
        return
    
    corr_matrix = pd.read_csv(corr_path, index_col=0)
    
    # Clean up column names for display
    name_map = {
        'connectivity_density': 'Connectivity',
        'route_coverage': 'Route Coverage',
        'freq_composite': 'Frequency (Composite)',
        'freq_am_peak': 'Freq (AM)',
        'freq_midday': 'Freq (Mid)',
        'freq_pm_peak': 'Freq (PM)',
        'freq_evening': 'freq_evening',
        'walking_access_pct': 'Walking Access',
        'span_hours': 'Span of Service',
        'weekend_ratio': 'Weekend Ratio',
        'jobs_accessibility': 'Job Access',
        'poi_accessibility': 'POI Access',
        'population_density': 'Pop. Density',
        'intersection_density': 'Intersection Den.',
        'land_use_mix': 'Land Use Mix',
        'sidewalk_index': 'Sidewalk Index'
    }
    
    corr_matrix = corr_matrix.rename(index=name_map, columns=name_map)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, vmin=-1, vmax=1, square=True,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Spearman Correlation', 'shrink': 0.8},
        ax=ax
    )
    
    ax.set_title('CPTA Component Correlation Matrix (Spearman)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_moran_scatterplot(gdf, output_path=None, city_name=""):
    """Create Moran scatterplot (standardized gap vs spatial lag)."""
    
    if 'gap_standardized' not in gdf.columns or 'gap_spatial_lag' not in gdf.columns:
        print("  ⚠ Spatial lag columns not found, skipping Moran scatterplot")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    x = gdf['gap_standardized'].values
    y = gdf['gap_spatial_lag'].values
    
    # Color by LISA cluster
    colors_map = {
        'High-High': '#d7191c', 'Low-Low': '#2c7bb6',
        'High-Low': '#fdae61', 'Low-High': '#abd9e9',
        'Not Significant': '#cccccc'
    }
    
    for cluster, color in colors_map.items():
        mask = gdf['lisa_cluster'] == cluster
        ax.scatter(x[mask], y[mask], c=color, label=cluster, alpha=0.7, s=40,
                   edgecolors='white', linewidth=0.5)
    
    # Add regression line
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, 'k-', linewidth=1.5, alpha=0.5,
            label=f'Slope = {m:.3f}')
    
    # Add quadrant lines
    ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)
    
    # Quadrant labels
    xlim = max(abs(x.min()), abs(x.max())) * 1.1
    ylim = max(abs(y.min()), abs(y.max())) * 1.1
    
    ax.text(xlim * 0.5, ylim * 0.7, 'HH', fontsize=20, alpha=0.2, ha='center', fontweight='bold', color='red')
    ax.text(-xlim * 0.5, -ylim * 0.7, 'LL', fontsize=20, alpha=0.2, ha='center', fontweight='bold', color='blue')
    ax.text(xlim * 0.5, -ylim * 0.7, 'HL', fontsize=20, alpha=0.2, ha='center', fontweight='bold', color='orange')
    ax.text(-xlim * 0.5, ylim * 0.7, 'LH', fontsize=20, alpha=0.2, ha='center', fontweight='bold', color='lightblue')
    
    ax.set_xlabel('Transit Gap (standardized)', fontsize=12)
    ax.set_ylabel('Spatial Lag of Transit Gap', fontsize=12)
    
    title = "Moran Scatterplot"
    if city_name:
        title += f'\n{city_name}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_sensitivity_chart(output_path=None):
    """Create bar chart comparing desert counts across weight scenarios."""
    sens_path = PROJECT_ROOT / "data" / "processed" / "sensitivity_analysis.csv"
    
    if not sens_path.exists():
        print("  ⚠ Sensitivity analysis not found, skipping")
        return
    
    sens_df = pd.read_csv(sens_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(sens_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sens_df['Desert_Count_Simple'], width, 
                   label='Simple Classification', color='#d73027', alpha=0.8)
    bars2 = ax.bar(x + width/2, sens_df['LISA_HH_Count'], width,
                   label='LISA High-High Clusters', color='#2c7bb6', alpha=0.8)
    
    ax.set_xlabel('Weight Scenario', fontsize=12)
    ax.set_ylabel('Number of Transit Desert Tracts', fontsize=12)
    ax.set_title('Sensitivity Analysis: Transit Deserts Under Different TVI Weights', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sens_df['Scenario'].str.title(), fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(int(height)),
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(int(height)),
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_cpta_category_chart(gdf, output_path=None):
    """
    Show CPTA category contributions (transit service / accessibility / built env)
    across transit desert classification groups.
    
    Grouped bar chart: one group per classification, three bars per group.
    """
    cat_cols = {
        'cat_transit': 'Transit Service (50%)',
        'cat_accessibility': 'Accessibility (25%)',
        'cat_built_env': 'Built Environment (25%)'
    }
    
    # Category columns live in supply_metrics.csv, not transit_deserts.gpkg
    # Merge them in if missing
    available = {k: v for k, v in cat_cols.items() if k in gdf.columns}
    
    if not available:
        supply_path = PROJECT_ROOT / "data" / "processed" / "supply_metrics.csv"
        if supply_path.exists():
            supply_df = pd.read_csv(supply_path, dtype={'GEOID': str})
            merge_cols = [c for c in cat_cols.keys() if c in supply_df.columns]
            if merge_cols:
                gdf = gdf.merge(supply_df[['GEOID'] + merge_cols], on='GEOID', how='left')
                available = {k: v for k, v in cat_cols.items() if k in gdf.columns}
    
    if not available:
        print("  ⚠ CPTA category columns not found — run 02d first")
        return
    
    classifications = ['Transit Desert', 'Transit Stressed', 'Underserved', 'Well-Served']
    class_col = 'classification_simple'
    
    if class_col not in gdf.columns:
        print("  ⚠ Classification column not found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(classifications))
    n_cats = len(available)
    width = 0.8 / n_cats
    colors = ['#2c7bb6', '#7fbc41', '#fc8d59']
    
    for i, (col, label) in enumerate(available.items()):
        means = []
        for cls in classifications:
            subset = gdf[gdf[class_col] == cls]
            means.append(subset[col].mean() if len(subset) > 0 else 0)
        
        offset = (i - n_cats / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=label, color=colors[i % len(colors)], alpha=0.85)
        
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Classification', fontsize=12)
    ax.set_ylabel('Mean Category Z-Score', fontsize=12)
    ax.set_title('CPTA Category Contributions by Classification', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classifications, fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_poi_category_chart(gdf, output_path=None):
    """
    Compare per-category POI accessibility scores between
    transit deserts and non-deserts.
    
    Reads from poi_accessibility.csv which has per-category columns.
    """
    poi_path = PROJECT_ROOT / "data" / "processed" / "poi_accessibility.csv"
    
    if not poi_path.exists():
        print("  ⚠ POI accessibility data not found (02c not run) — skipping")
        return
    
    poi_df = pd.read_csv(poi_path, dtype={'GEOID': str})
    
    # Find per-category columns (poi_healthcare, poi_grocery, etc.)
    poi_cat_cols = [col for col in poi_df.columns if col.startswith('poi_') and col != 'poi_accessibility']
    
    if not poi_cat_cols:
        print("  ⚠ No per-category POI columns found — skipping")
        return
    
    # Merge with desert classification
    if 'transit_desert' not in gdf.columns:
        print("  ⚠ transit_desert column not found — skipping")
        return
    
    merged = poi_df.merge(gdf[['GEOID', 'transit_desert']], on='GEOID', how='inner')
    deserts = merged[merged['transit_desert'] == True]
    non_deserts = merged[merged['transit_desert'] == False]
    
    # Clean up labels
    labels = [col.replace('poi_', '').replace('_', ' ').title() for col in poi_cat_cols]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = np.arange(len(poi_cat_cols))
    width = 0.35
    
    desert_vals = [deserts[col].mean() for col in poi_cat_cols]
    non_desert_vals = [non_deserts[col].mean() for col in poi_cat_cols]
    
    bars1 = ax.bar(x - width/2, desert_vals, width, label='Transit Deserts', color='#d73027', alpha=0.85)
    bars2 = ax.bar(x + width/2, non_desert_vals, width, label='Non-Deserts', color='#91bfdb', alpha=0.85)
    
    # Significance stars
    from scipy.stats import mannwhitneyu
    for i, col in enumerate(poi_cat_cols):
        d = deserts[col].dropna()
        nd = non_deserts[col].dropna()
        if len(d) > 0 and len(nd) > 0:
            _, p = mannwhitneyu(d, nd, alternative='two-sided')
            star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            if star:
                max_val = max(desert_vals[i], non_desert_vals[i])
                ax.text(x[i], max_val * 1.08, star, ha='center', fontsize=11, fontweight='bold', color='#333')
    
    ax.set_ylabel('Mean Decay-Weighted Accessibility Score', fontsize=11)
    ax.set_title('POI Accessibility by Category: Transit Deserts vs. Non-Deserts',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    ax.text(0.98, 0.02, '* p<0.05  ** p<0.01  *** p<0.001',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            color='gray', style='italic')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_absolute_vs_relative_map(gdf, output_path=None, city_name=""):
    """
    Side-by-side map comparing relative (LISA/median) and absolute threshold
    desert classifications. A third panel shows agreement/disagreement.
    """
    if 'abs_transit_desert' not in gdf.columns:
        print("  ⚠ abs_transit_desert not found — skipping comparison map")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    gdf_plot = gdf.to_crs(epsg=3857)

    # Panel 1: Relative deserts
    ax = axes[0]
    gdf_plot.plot(color='#eeeeee', linewidth=0.3, edgecolor='white', ax=ax)
    rel_deserts = gdf_plot[gdf_plot['transit_desert'] == True]
    if len(rel_deserts) > 0:
        rel_deserts.plot(color='#d73027', linewidth=0.3, edgecolor='white', ax=ax)
    if CTX_AVAILABLE:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.4)
        except:
            pass
    n_rel = gdf['transit_desert'].sum()
    ax.set_title(f'Relative Deserts (LISA/Median)\n{n_rel} tracts ({n_rel/len(gdf):.1%})',
                 fontsize=12, fontweight='bold')
    ax.set_axis_off()

    # Panel 2: Absolute deserts
    ax = axes[1]
    gdf_plot.plot(color='#eeeeee', linewidth=0.3, edgecolor='white', ax=ax)
    abs_deserts = gdf_plot[gdf_plot['abs_transit_desert'] == True]
    if len(abs_deserts) > 0:
        abs_deserts.plot(color='#7b3294', linewidth=0.3, edgecolor='white', ax=ax)
    if CTX_AVAILABLE:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.4)
        except:
            pass
    n_abs = gdf['abs_transit_desert'].sum()
    ax.set_title(f'Absolute Deserts (Threshold)\n{n_abs} tracts ({n_abs/len(gdf):.1%})',
                 fontsize=12, fontweight='bold')
    ax.set_axis_off()

    # Panel 3: Agreement map
    ax = axes[2]
    gdf_plot = gdf_plot.copy()
    conditions = [
        (gdf_plot['transit_desert'] == True) & (gdf_plot['abs_transit_desert'] == True),
        (gdf_plot['transit_desert'] == True) & (gdf_plot['abs_transit_desert'] == False),
        (gdf_plot['transit_desert'] == False) & (gdf_plot['abs_transit_desert'] == True),
    ]
    labels = ['Both', 'Relative Only', 'Absolute Only']
    gdf_plot['agreement'] = 'Neither'
    for cond, label in zip(conditions, labels):
        gdf_plot.loc[cond, 'agreement'] = label

    agree_colors = {
        'Both': '#d73027', 'Relative Only': '#fc8d59',
        'Absolute Only': '#7b3294', 'Neither': '#eeeeee'
    }
    for cat, color in agree_colors.items():
        subset = gdf_plot[gdf_plot['agreement'] == cat]
        if len(subset) > 0:
            subset.plot(color=color, linewidth=0.3, edgecolor='white', ax=ax)
    if CTX_AVAILABLE:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.4)
        except:
            pass

    counts = gdf_plot['agreement'].value_counts()
    patches = [mpatches.Patch(color=color, label=f"{cat} ({counts.get(cat, 0)})")
               for cat, color in agree_colors.items() if cat != 'Neither']
    ax.legend(handles=patches, loc='lower right', frameon=True, fontsize=9)
    ax.set_title('Agreement: Relative vs. Absolute', fontsize=12, fontweight='bold')
    ax.set_axis_off()

    suptitle = 'Relative vs. Absolute Transit Desert Identification'
    if city_name:
        suptitle += f' — {city_name}'
    fig.suptitle(suptitle, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_threshold_failure_chart(gdf, output_path=None, city_name=""):
    """
    Horizontal bar chart showing what % of tracts fail each absolute criterion.
    Makes it immediately obvious which service dimensions are weakest.
    """
    fail_cols = sorted([col for col in gdf.columns
                        if col.startswith('abs_fail_') and col != 'abs_fail_count'])

    if not fail_cols:
        print("  ⚠ No abs_fail_ columns found — skipping threshold failure chart")
        return

    # Try to load labels from the thresholds summary CSV
    thresh_path = PROJECT_ROOT / "data" / "processed" / "absolute_thresholds.csv"
    label_map = {}
    if thresh_path.exists():
        thresh_df = pd.read_csv(thresh_path)
        for _, row in thresh_df.iterrows():
            col_name = f"abs_fail_{row['criterion']}"
            label_map[col_name] = row.get('label', row['criterion'])

    labels = [label_map.get(col, col.replace('abs_fail_', '').replace('_', ' ').title())
              for col in fail_cols]
    fail_pcts = [(gdf[col].sum() / len(gdf) * 100) for col in fail_cols]

    # Sort by failure rate
    sorted_idx = np.argsort(fail_pcts)[::-1]
    labels = [labels[i] for i in sorted_idx]
    fail_pcts = [fail_pcts[i] for i in sorted_idx]

    fig, ax = plt.subplots(1, 1, figsize=(10, max(4, len(labels) * 0.7)))

    y = np.arange(len(labels))
    bars = ax.barh(y, fail_pcts, color='#d73027', alpha=0.85, edgecolor='white')

    for bar, pct in zip(bars, fail_pcts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Tracts Failing Criterion (%)', fontsize=12)
    ax.set_xlim(0, max(fail_pcts) * 1.2 if fail_pcts else 100)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    title = 'Absolute Threshold: Failure Rate by Criterion'
    if city_name:
        title += f'\n{city_name}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add fail-count distribution as text annotation
    if 'abs_fail_count' in gdf.columns:
        dist_text = 'Failure count: '
        for n in range(int(gdf['abs_fail_count'].max()) + 1):
            count = (gdf['abs_fail_count'] == n).sum()
            if count > 0:
                dist_text += f'{n}→{count}  '
        ax.text(0.98, 0.02, dist_text.strip(), transform=ax.transAxes,
                fontsize=8, ha='right', va='bottom', color='gray', style='italic')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_absolute_desert_map(gdf, output_path=None, city_name=""):
    """
    Choropleth map showing abs_fail_count (0-5) per tract.
    Tracts at/above the fail threshold get a bold outline.
    """
    if 'abs_fail_count' not in gdf.columns:
        print("  ⚠ abs_fail_count not found — skipping absolute desert map")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    gdf_plot = gdf.to_crs(epsg=3857)

    max_fail = int(gdf_plot['abs_fail_count'].max())
    gdf_plot.plot(
        column='abs_fail_count', cmap='YlOrRd', linewidth=0.3, edgecolor='white',
        legend=True, vmin=0, vmax=max(max_fail, 5),
        legend_kwds={'label': 'Criteria Failed', 'orientation': 'horizontal',
                     'shrink': 0.6, 'pad': 0.05},
        ax=ax
    )

    # Bold outline on absolute deserts
    abs_deserts = gdf_plot[gdf_plot['abs_transit_desert'] == True]
    if len(abs_deserts) > 0:
        abs_deserts.boundary.plot(ax=ax, color='black', linewidth=1.2, alpha=0.7)

    if CTX_AVAILABLE:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.4)
        except:
            pass

    n_abs = gdf['abs_transit_desert'].sum()
    title = f'Absolute Threshold Failures'
    if city_name:
        title += f'\n{city_name}'
    title += f'\n{n_abs} tracts classified as absolute transit deserts (bold outline)'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_axis_off()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_barrier_frequency_chart(gdf, output_path=None, city_name=""):
    """
    Stacked horizontal bar chart showing how often each supply metric
    appears as the #1, #2, or #3 barrier across transit desert tracts.

    Reads barrier_summary.csv if available, otherwise computes from
    barrier_1/2/3 columns in the GeoDataFrame.
    """
    # Try loading pre-computed summary first
    summary_path = PROJECT_ROOT / "data" / "processed" / "barrier_summary.csv"
    if summary_path.exists():
        barrier_df = pd.read_csv(summary_path)
    elif 'barrier_1' in gdf.columns:
        # Compute from tract-level data
        desert_gdf = gdf[gdf['transit_desert'] == True]
        if len(desert_gdf) == 0:
            print("  ⚠ No transit deserts — skipping barrier chart")
            return

        counts = {}
        for rank in [1, 2, 3]:
            col = f'barrier_{rank}'
            if col in desert_gdf.columns:
                for metric in desert_gdf[col].dropna():
                    if metric == '':
                        continue
                    if metric not in counts:
                        counts[metric] = {'rank_1': 0, 'rank_2': 0, 'rank_3': 0}
                    counts[metric][f'rank_{rank}'] += 1

        if not counts:
            print("  ⚠ No barrier data — skipping")
            return

        barrier_df = pd.DataFrame([
            {'metric': m, **c, 'total_top3': sum(c.values())}
            for m, c in counts.items()
        ]).sort_values('total_top3', ascending=False)
    else:
        print("  ⚠ No barrier columns or summary file — skipping barrier chart")
        return

    if len(barrier_df) == 0:
        return

    # Sort by total frequency
    barrier_df = barrier_df.sort_values('total_top3', ascending=True)

    fig, ax = plt.subplots(1, 1, figsize=(11, max(5, len(barrier_df) * 0.55)))

    y = np.arange(len(barrier_df))
    rank_colors = {'rank_1': '#d73027', 'rank_2': '#fc8d59', 'rank_3': '#fee08b'}
    rank_labels = {'rank_1': '#1 Barrier', 'rank_2': '#2 Barrier', 'rank_3': '#3 Barrier'}

    left = np.zeros(len(barrier_df))
    for rank_col, color in rank_colors.items():
        if rank_col in barrier_df.columns:
            vals = barrier_df[rank_col].fillna(0).values
            ax.barh(y, vals, left=left, color=color, label=rank_labels[rank_col],
                    edgecolor='white', linewidth=0.5)
            left += vals

    # Total count labels
    for i, total in enumerate(barrier_df['total_top3'].values):
        ax.text(total + 0.5, i, str(int(total)), va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(barrier_df['metric'].values, fontsize=10)
    ax.set_xlabel('Number of Transit Desert Tracts', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    n_deserts = gdf['transit_desert'].sum() if 'transit_desert' in gdf.columns else '?'
    title = f'Top Supply-Side Barriers in Transit Deserts (n={n_deserts})'
    if city_name:
        title += f'\n{city_name}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.text(0.98, 0.02,
            'For each desert tract, CPTA component z-scores are ranked.\n'
            'The most negative z-scores identify the weakest service dimensions.',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            color='gray', style='italic')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


# ============================================================
# TVI-LATENT / FORCED CAR OWNERSHIP VISUALIZATIONS
# ============================================================

def create_tvi_latent_comparison_map(gdf, output_path=None, city_name=""):
    """Side-by-side choropleth: TVI observed vs TVI-latent vs shift."""
    if 'TVI_latent_normalized' not in gdf.columns:
        print("  ⚠ TVI_latent not available, skipping comparison map")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    gdf.plot(column='TVI_normalized', cmap='RdYlBu_r', ax=axes[0],
             legend=True, legend_kwds={'shrink': 0.6, 'label': 'TVI (0-100)'})
    axes[0].set_title('TVI (Observed)', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    gdf.plot(column='TVI_latent_normalized', cmap='RdYlBu_r', ax=axes[1],
             legend=True, legend_kwds={'shrink': 0.6, 'label': 'TVI-latent (0-100)'})
    axes[1].set_title('TVI-latent\n(Forced Car Adjusted)', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    shift = gdf['tvi_latent_shift']
    vmax = max(abs(shift.min()), abs(shift.max()), 1)
    gdf.plot(column='tvi_latent_shift', cmap='RdBu_r', ax=axes[2],
             legend=True, vmin=-vmax, vmax=vmax,
             legend_kwds={'shrink': 0.6, 'label': 'Shift (latent − observed)'})
    axes[2].set_title('Demand Shift\n(Forced Car Effect)', fontsize=13, fontweight='bold')
    axes[2].axis('off')
    
    fig.suptitle(f'TVI vs TVI-latent — {city_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_latent_demand_scatter(gdf, output_path=None, city_name=""):
    """Scatter: TVI observed (x) vs TVI-latent (y), colored by classification."""
    if 'TVI_latent_normalized' not in gdf.columns:
        print("  ⚠ TVI_latent not available, skipping latent scatter")
        return
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    colors = {
        'Transit Desert': '#d73027',
        'Transit Stressed': '#fc8d59',
        'Underserved': '#fee090',
        'Well-Served': '#91bfdb'
    }
    
    for cat, color in colors.items():
        mask = gdf['classification_simple'] == cat
        if mask.any():
            ax.scatter(gdf.loc[mask, 'TVI_normalized'],
                      gdf.loc[mask, 'TVI_latent_normalized'],
                      c=color, label=f'{cat} ({mask.sum()})',
                      alpha=0.6, s=40, edgecolors='white', linewidth=0.3)
    
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.4, linewidth=1, label='No change')
    
    ax.fill_between([0, 100], [0, 100], [100, 100], alpha=0.04, color='red')
    ax.fill_between([0, 100], [0, 0], [0, 100], alpha=0.04, color='blue')
    ax.text(15, 85, '↑ Latent demand higher\n(forced car ownership)',
            fontsize=9, color='#d73027', alpha=0.7, ha='center')
    ax.text(85, 15, '↓ Observed demand\nhigher than latent',
            fontsize=9, color='#4575b4', alpha=0.7, ha='center')
    
    ax.set_xlabel('TVI (Observed) — Normalized 0-100', fontsize=12)
    ax.set_ylabel('TVI-latent (Forced Car Adjusted) — Normalized 0-100', fontsize=12)
    ax.set_title(f'Observed vs. Latent Transit Demand — {city_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    
    corr = gdf['TVI_normalized'].corr(gdf['TVI_latent_normalized'])
    ax.text(0.02, 0.98, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=11,
            va='top', ha='left', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_equity_gate_comparison(gdf, output_path=None, city_name=""):
    """Bar chart: equity gate pass rates under observed vs latent TVI."""
    if 'TVI_latent_normalized' not in gdf.columns or 'abs_transit_desert' not in gdf.columns:
        print("  ⚠ Missing TVI_latent or abs_transit_desert, skipping gate comparison")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    demand_median = gdf['TVI_normalized'].median()
    latent_median = gdf['TVI_latent_normalized'].median()
    
    n_abs_total = int(gdf['abs_transit_desert'].sum())
    n_observed_gate = int((gdf['abs_transit_desert'] & (gdf['TVI_normalized'] >= demand_median)).sum())
    n_latent_gate = int((gdf['abs_transit_desert'] & (gdf['TVI_latent_normalized'] >= latent_median)).sum())
    n_rescued = n_latent_gate - n_observed_gate
    
    categories = ['Absolute\nThreshold\nFailures',
                  'Pass Gate\n(Observed TVI)',
                  'Pass Gate\n(TVI-latent)',
                  'Rescued by\nLatent Demand']
    values = [n_abs_total, n_observed_gate, n_latent_gate, max(n_rescued, 0)]
    bar_colors = ['#969696', '#fc8d59', '#d73027', '#1a9850']
    
    bars = ax.bar(categories, values, color=bar_colors, edgecolor='white', linewidth=1.5)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Number of Tracts', fontsize=12)
    ax.set_title(f'Equity Gate Effect: Observed vs. Latent Demand — {city_name}',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    ax.text(0.98, 0.95,
            f'Observed TVI median: {demand_median:.1f}\n'
            f'TVI-latent median: {latent_median:.1f}\n'
            f'Tracts rescued: {n_rescued}',
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_tvi_correlation_heatmap(gdf, output_path=None):
    """Create Spearman correlation heatmap of TVI components.
    
    Flags correlations > 0.65, which motivate the sensitivity analysis
    and the discussion of minority population's role in the index.
    """
    tvi_cols = {
        'pct_zero_vehicle': 'Zero-Vehicle HH',
        'pct_poverty': 'Below Poverty',
        'pct_minority': 'Minority Pop.',
        'pct_elderly': 'Elderly (65+)',
        'pct_youth': 'Youth (10-17)'
    }
    
    available = {k: v for k, v in tvi_cols.items() if k in gdf.columns}
    if len(available) < 3:
        print("  ⚠ Not enough TVI components for correlation matrix")
        return
    
    data = gdf[list(available.keys())].rename(columns=available)
    corr = data.corr(method='spearman')
    
    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Spearman Correlation', 'shrink': 0.8},
                ax=ax)
    
    ax.set_title('TVI Component Correlation Matrix (Spearman)', fontsize=14, fontweight='bold', pad=15)
    
    # Flag high correlations in text
    high_pairs = []
    cols = list(available.values())
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.65:
                high_pairs.append(f"{cols[i]} ↔ {cols[j]}: {r:.2f}")
    
    if high_pairs:
        note = "High correlations (|r|>0.65):\n" + "\n".join(high_pairs)
        ax.text(0.98, 0.02, note, transform=ax.transAxes, fontsize=8,
                ha='right', va='bottom', color='red', style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    plt.close()
    
    # Also save the correlation matrix as CSV
    csv_path = output_path.parent.parent / 'tables' / 'tvi_correlation_matrix.csv' if output_path else None
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        corr.to_csv(csv_path)
        print(f"  ✓ Saved: {csv_path.name}")


def create_summary_table(gdf, output_path=None):
    overall = {
        'Metric': ['Total Tracts', 'Transit Deserts', 'Transit Stressed', 'Underserved', 'Well-Served'],
        'Count': [
            len(gdf),
            len(gdf[gdf['classification_simple'] == 'Transit Desert']),
            len(gdf[gdf['classification_simple'] == 'Transit Stressed']),
            len(gdf[gdf['classification_simple'] == 'Underserved']),
            len(gdf[gdf['classification_simple'] == 'Well-Served'])
        ]
    }
    
    # Add absolute desert row if available
    if 'abs_transit_desert' in gdf.columns:
        overall['Metric'].append('Absolute Transit Deserts')
        overall['Count'].append(int(gdf['abs_transit_desert'].sum()))
    
    overall_df = pd.DataFrame(overall)
    overall_df['Percentage'] = (overall_df['Count'] / len(gdf) * 100).round(1)
    overall_df.loc[0, 'Percentage'] = 100.0
    
    if output_path:
        overall_df.to_csv(output_path, index=False)
        print(f"  ✓ Saved: {output_path.name}")
    return overall_df


def main():
    print("=" * 60)
    print("Transit Desert Pipeline: Visualization")
    print("=" * 60)
    
    config = load_config()
    setup_plot_style()
    
    city_name = config['study_area'].get('name', '')
    
    maps_dir = PROJECT_ROOT / "outputs" / "maps"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    
    maps_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "-" * 40)
    print("Loading data...")
    print("-" * 40)
    gdf = load_results()
    
    # Maps
    print("\n" + "-" * 40)
    print("Creating maps...")
    print("-" * 40)
    
    create_choropleth_map(gdf, 'CPTA_normalized', f'Transit Supply (CPTA Score)\n{city_name}',
                          cmap='RdYlBu', output_path=maps_dir / 'map_cpta.png',
                          legend_title='CPTA Score (0-100)')
    
    create_choropleth_map(gdf, 'TVI_normalized', f'Transit Demand (TVI Score)\n{city_name}',
                          cmap='RdYlBu_r', output_path=maps_dir / 'map_tvi.png',
                          legend_title='TVI Score (0-100)')
    
    create_choropleth_map(gdf, 'transit_gap', f'Transit Gap (Demand - Supply)\n{city_name}',
                          cmap='RdYlBu_r', output_path=maps_dir / 'map_transit_gap.png',
                          legend_title='Transit Gap')
    
    create_classification_map(gdf, output_path=maps_dir / 'map_classification.png', city_name=city_name)
    create_lisa_map(gdf, output_path=maps_dir / 'map_lisa_clusters.png', city_name=city_name)
    
    # Absolute threshold maps (if data available)
    if 'abs_transit_desert' in gdf.columns and gdf['abs_transit_desert'].any():
        create_absolute_vs_relative_map(gdf, output_path=maps_dir / 'map_absolute_vs_relative.png',
                                        city_name=city_name)
        create_absolute_desert_map(gdf, output_path=maps_dir / 'map_absolute_failures.png',
                                   city_name=city_name)
    
    # Figures
    print("\n" + "-" * 40)
    print("Creating figures...")
    print("-" * 40)
    
    create_scatter_plot(gdf, output_path=figures_dir / 'fig_supply_vs_demand.png', city_name=city_name)
    create_distribution_plots(gdf, output_path=figures_dir / 'fig_distributions.png')
    create_equity_comparison(gdf, output_path=figures_dir / 'fig_equity_comparison.png')
    
    # New figures
    create_correlation_heatmap(output_path=figures_dir / 'fig_cpta_correlation.png')
    create_moran_scatterplot(gdf, output_path=figures_dir / 'fig_moran_scatterplot.png', city_name=city_name)
    create_sensitivity_chart(output_path=figures_dir / 'fig_sensitivity_analysis.png')
    create_cpta_category_chart(gdf, output_path=figures_dir / 'fig_cpta_categories.png')
    create_poi_category_chart(gdf, output_path=figures_dir / 'fig_poi_categories.png')
    
    # Absolute threshold figures
    if 'abs_transit_desert' in gdf.columns:
        create_threshold_failure_chart(gdf, output_path=figures_dir / 'fig_threshold_failures.png',
                                       city_name=city_name)
    
    # Deficit profile / barrier frequency chart
    if 'barrier_1' in gdf.columns:
        create_barrier_frequency_chart(gdf, output_path=figures_dir / 'fig_barrier_frequency.png',
                                       city_name=city_name)
    
    # TVI-latent / forced car ownership figures
    if 'TVI_latent_normalized' in gdf.columns:
        create_tvi_latent_comparison_map(gdf, output_path=maps_dir / 'map_tvi_latent_comparison.png',
                                          city_name=city_name)
        create_latent_demand_scatter(gdf, output_path=figures_dir / 'fig_latent_demand_scatter.png',
                                     city_name=city_name)
        create_equity_gate_comparison(gdf, output_path=figures_dir / 'fig_equity_gate_comparison.png',
                                      city_name=city_name)
    
    # TVI component correlation matrix (Spearman)
    create_tvi_correlation_heatmap(gdf, output_path=figures_dir / 'fig_tvi_correlation.png')
    
    # Tables
    print("\n" + "-" * 40)
    print("Creating tables...")
    print("-" * 40)
    
    summary_df = create_summary_table(gdf, output_path=tables_dir / 'summary_statistics.csv')
    
    print("\n  Summary:")
    print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"\nOutputs saved to:")
    print(f"  Maps:    {maps_dir}")
    print(f"  Figures: {figures_dir}")
    print(f"  Tables:  {tables_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()