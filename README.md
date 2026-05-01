# Transit Desert Identification Pipeline — Extended

An end-to-end Python pipeline that identifies **transit deserts** — Census tracts where demand for public transit significantly exceeds supply — using spatial statistics, accessibility modeling, and ACS demographic data.

---

## Overview

The pipeline computes two composite indices for each Census tract and uses their gap to classify transit deserts:

| Index | What it measures |
|---|---|
| **CPTA** — Composite Public Transit Accessibility | Transit supply: service frequency, stop coverage, built environment walkability, job/POI accessibility |
| **TVI** — Transit Vulnerability Index | Transit demand: zero-vehicle households, poverty, minority population, elderly, youth |

Tracts where **TVI > CPTA** (high demand, low supply) are classified as transit deserts using LISA (Local Moran's I) spatial clustering and absolute service thresholds.

---

## Pipeline Steps

```
1   Download Data          GTFS feeds, Census tract shapefiles, ACS demographics
2   Compute Supply         11 transit + built environment metrics
2b  Job Accessibility      Gravity-model jobs reachable by transit  [optional, requires r5py + Java]
2c  POI Accessibility      Category-weighted essential service access [optional, requires r5py + Java]
2d  Compute CPTA Score     Category-weighted composite accessibility index
3   Compute Demand (TVI)   5-component Transit Vulnerability Index + TVI-latent
4   Identify Deserts       LISA clustering, absolute thresholds, sensitivity analysis, equity tests
5   Visualize              Maps, distribution plots, Moran scatterplot, correlation heatmaps
```

---

## Project Structure

```
transit-desert-pipeline-extended/
├── run_pipeline.py              # Orchestrator: runs all steps in sequence
├── requirements.txt
├── config/
│   ├── Config.yaml              # Active configuration (currently Dallas, TX)
│   ├── Config_Baltimore.yaml
│   ├── Config_Dallas.yaml
│   ├── Config_Nashville.yaml
│   └── Config_Philadelphia.yaml
├── src/
│   ├── 01_download_data.py      # GTFS, Census, ACS download
│   ├── 02_compute_supply.py     # Transit service + built environment metrics
│   ├── 02b_compute_jobs_accessibility.py   # r5py gravity-model job access
│   ├── 02c_compute_poi_accessibility.py    # r5py POI access (8 categories)
│   ├── 02d_compute_cpta.py      # Final CPTA composite score
│   ├── 03_compute_demand.py     # TVI + TVI-latent (forced car ownership)
│   ├── 04_identify_deserts.py   # LISA, gap classification, equity tests
│   ├── 05_visualize.py          # Maps and figures
│   ├── gtfs_utils.py            # GTFS loading and multi-feed merging
│   ├── gtfs_date_picker.py      # Automatic analysis date selection
│   ├── state_osm.py             # Geofabrik OSM PBF download + county clipping
│   └── validate_cost_burden.py  # External validation against CNT H+T Index
├── data/
│   ├── raw/                     # Downloaded source data (gitignored)
│   ├── processed/               # Intermediate and final CSVs/GeoPackages
│   └── external/                # OSM PBFs, LODES job data
├── outputs/
│   ├── maps/
│   ├── figures/
│   └── tables/
├── notebooks/
│   └── baltimore_analysis.ipynb
└── logs/
```

---

## Quickstart

### 1. Install dependencies

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

For job and POI accessibility (steps 2b/2c), Java 11+ must be installed and `r5py` must be available (`pip install r5py`).

### 2. Configure your study area

Copy and edit one of the provided configs, or edit `config/Config.yaml` directly:

```yaml
study_area:
  name: "Dallas County"
  state_fips: "48"
  county_fips: "113"

gtfs:
  url:
    - "https://www.dart.org/transitdata/latest/google_transit.zip"
  analysis_date: "20260310"     # Must be a valid date in the GTFS calendar

census:
  acs_year: 2022
```

You need a free Census API key. Set it as an environment variable:

```bash
export CENSUS_API_KEY="your_key_here"
```

### 3. Run the pipeline

```bash
# Full pipeline (steps 1 → 2 → 2b → 2c → 2d → 3 → 4 → 5)
python run_pipeline.py

# Use a specific config file
python run_pipeline.py --config config/Config_Baltimore.yaml

# Skip optional accessibility steps (no Java required)
python run_pipeline.py --steps 1,2,2d,3,4,5

# Re-run analysis and visualization only (skip downloads)
python run_pipeline.py --steps 4,5 --no-clean

# Quiet mode (suppress terminal output)
python run_pipeline.py --quiet
```

---

## CPTA Metric Weights

| Category | Weight | Metrics |
|---|---|---|
| Transit Service | 50% | Stop connectivity, route coverage, AM/midday/PM/evening frequency, span of service, weekend ratio |
| Accessibility | 25% | Walking access (400 m bus / 800 m rail buffers), job accessibility\*, POI accessibility\* |
| Built Environment | 25% | Population density, intersection density, land use mix (Shannon entropy), sidewalk index |

\* Optional metrics from steps 2b/2c. If omitted, weights are renormalized across available metrics.

## TVI Component Weights

| Component | Weight | ACS Table |
|---|---|---|
| Zero-vehicle households | 30% | B08201 |
| Below poverty line | 25% | B17001 |
| Minority population | 20% | B03002 |
| Elderly (65+) | 15% | B01001 |
| Youth (10–17) | 10% | B01001 |

**TVI-latent** additionally incorporates a forced car ownership proxy (share of low-income workers commuting by private vehicle, from B08122), reducing the zero-vehicle weight to 15% and adding a 15% forced-car component.

---

## Classification Method

1. **Transit Gap** = percentile rank(TVI) − percentile rank(CPTA)
2. **Global Moran's I** confirms spatial autocorrelation
3. **LISA (Local Moran's I)** identifies High-High clusters (transit deserts) and Low-Low clusters (well-served areas)
4. **Absolute thresholds** layer applies fixed minimum service criteria uniformly across cities, independent of within-city normalization
5. **Hybrid classification**: tracts are labeled *Transit Desert*, *Transit Stressed*, *Underserved*, or *Well-Served*

A **sensitivity analysis** evaluates how classification stability changes under TVI weight perturbations.

---

## Supported Cities

Pre-built configs are included for:

- **Baltimore, MD** — MTA Maryland (bus + light rail + metro, multi-feed merge)
- **Dallas, TX** — DART
- **Nashville, TN** — WeGo Public Transit
- **Philadelphia, PA** — SEPTA

To add a new city, create a new YAML config with the city's state/county FIPS codes and GTFS feed URL(s).

---

## Outputs

| File | Description |
|---|---|
| `data/processed/supply_metrics.csv` | Per-tract CPTA component metrics |
| `data/processed/demand_metrics.csv` | Per-tract TVI and TVI-latent |
| `data/processed/transit_deserts.csv` | Final classification with gap scores |
| `data/processed/transit_deserts.gpkg` | GeoPackage with all results + geometries |
| `data/processed/sensitivity_analysis.csv` | TVI weight sensitivity results |
| `data/processed/equity_profile.csv` | Demographic profile of desert tracts |
| `outputs/maps/` | Choropleth maps (CPTA, TVI, gap, LISA clusters, classification) |
| `outputs/figures/` | Distribution plots, scatter plots, Moran scatterplot, heatmaps |
| `outputs/tables/` | Summary statistics |

---

## Validation

`src/validate_cost_burden.py` validates TVI-latent against the CNT Housing + Transportation (H+T) Affordability Index. Tracts flagged as high-latent-need should correlate with high transportation cost burden, confirming the demand estimates without including cost burden in the gap calculation.

Download H+T Index CSV files from [htaindex.cnt.org/download](https://htaindex.cnt.org/download/) and place them in `data/validation/` before running.

---

## Requirements

- Python 3.10+
- Java 11+ (for steps 2b/2c only)
- Census API key (`CENSUS_API_KEY` environment variable)
- osmium-tool (optional, for clipping state OSM PBFs to county extent — falls back to full-state PBF if unavailable)
