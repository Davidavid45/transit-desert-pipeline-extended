# Transit Desert Identification Pipeline (Extended)

An end-to-end Python pipeline that finds the neighborhoods where people need public transit most and get the least of it, using only public data.

Developed at the [Smart Cities and Civic Technologies (SC&CT) Research Center](https://ischool.syracuse.edu/research/smart-grid-research-center/), School of Information Studies, Syracuse University.

---

## What this answers

Most transit equity analysis asks whether service is distributed *fairly*: does a change hurt minority and low-income neighborhoods more than everyone else? That is the question federal Title VI analysis is built to answer, and it is the right question in a city where most people ride.

It is the wrong question on its own in a car-dependent city. If service is thin everywhere and almost everyone who could buy a car already has, the gap between neighborhoods shrinks, and a fairness measure comes back clean while the bus still does not run at 11 p.m.

This pipeline reports both things side by side:

1. **Relative gap.** Which neighborhoods are worse off than the rest of their own region.
2. **Minimum service.** Whether a neighborhood gets a basic level of service at all, measured against fixed thresholds that do not move from city to city.

A neighborhood can look fine on the first and fail the second. That is the case this tool is built to catch.

---

## The minimum-service screen

Five criteria, applied identically in every city, with no within-city normalization:

| # | Criterion | Threshold | Underlying metric |
|---|---|---|---|
| 1 | A route you can walk to | at least 1 route within the buffer | `route_coverage` |
| 2 | A bus at least every 30 minutes at morning peak | at least 2 departures per hour | `freq_am_peak` |
| 3 | Enough of the neighborhood near a stop | at least 25% of tract area within the buffer | `walking_access` |
| 4 | Service through the day | at least 12 hours of daily span | `span_of_service` |
| 5 | Somewhere to get to | at least 5,000 jobs reachable in 45 minutes | `jobs_accessibility` |

Walk buffers are 400 m for bus stops and 800 m for rail.

**A tract fails the screen if it fails at least 3 of the 5.**

Failing the screen is then gated on need: a tract is flagged only if its Transit Vulnerability Index is at or above the regional median. This stops a wealthy low-density suburb from being called a transit desert because it lacks a bus its residents do not want.

All five thresholds and the fail count live in `config/Config.yaml` under `identification.absolute_thresholds`. Change them and rerun. They are inputs, not constants of nature.

### Why this matters in practice

In the four-city study behind this pipeline (Baltimore, Philadelphia, Nashville, Dallas), the relative measure flagged neighborhood after neighborhood in Philadelphia, exactly as an equity analysis is designed to do. Not one of those flagged Philadelphia tracts failed the minimum-service screen. In Dallas roughly four in ten flagged tracts failed it, and in Nashville more than half. The two measures disagree, and the disagreement is the finding.

See the paper for the full results: [arxiv.org/abs/2609.21956](https://arxiv.org/abs/2609.21956).

---

## What this does not do

Stated plainly, because the limits matter as much as the outputs.

- **The five thresholds are a judgment, not a derived standard.** They are defensible and drawn from practice, but another analyst could justify different numbers and would get a different count. Using an absolute floor removes the distortion that comes from normalizing within a city; it does not remove the need to choose the floor. Karner, Pereira and Farber (2024), [*Advances and pitfalls in measuring transportation equity*](https://doi.org/10.1007/s11116-023-10460-7), make this point directly and it stands.
- **The relative gap is normalized within a city, so it can move the wrong way.** If a region cuts service everywhere, percentile ranks shift and the measured gap can shrink even though every rider is worse off. This is why the minimum-service screen exists alongside it, and why the two are reported separately rather than combined into one number.
- **Supply and demand are composite indices built from z-scores**, which makes them comparable but not natural units. A Foster-Greer-Thorbecke accessibility-poverty module is maintained alongside this pipeline for cases where a shortfall against a stated absolute threshold is the better framing.
- **It measures scheduled service, not service as delivered.** GTFS says what the timetable promises. It does not know about a bus that did not come.
- **Tract-level results are tract-level.** A tract can pass on average while a corner of it has nothing.
- **Classification is sensitive to demand weights.** `src/04_identify_deserts.py` runs a sensitivity analysis over TVI weight perturbations and writes the result. Read it before quoting a count.

---

## If you are not going to run the code

That is most people, and the useful things do not require Python.

- **Read the study:** [arxiv.org/abs/2609.21956](https://arxiv.org/abs/2609.21956) (four cities), and the methods paper, [preprint on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7315618).
- **Look up your own agency's Title VI Program**, usually on its website. Check whether it commits to *when* buses run (span, headway) or only to *where* routes go. If it is only route geography, there is nothing in it to hold the agency to on frequency or span.
- **Ask your agency to run this.** It takes public data, open code and a free Census API key. Point staff at this repository.
- **Or ask me.** Open an issue, or contact me through the research center above.

---

## What it measures

| Index | What it captures |
|---|---|
| **CPTA**, Composite Public Transit Accessibility | Supply: service frequency, stop coverage, span, built environment walkability, job and POI accessibility |
| **TVI**, Transit Vulnerability Index | Need: zero-vehicle households, poverty, minority population, elderly, youth |

Tracts where TVI outruns CPTA are candidate transit deserts. They are confirmed through LISA (Local Moran's I) spatial clustering for the relative pathway, and through the minimum-service screen for the absolute pathway.

### CPTA metric weights

| Category | Weight | Metrics |
|---|---|---|
| Transit Service | 50% | Stop connectivity, route coverage, AM/midday/PM/evening frequency, span of service, weekend ratio |
| Accessibility | 25% | Walking access (400 m bus / 800 m rail buffers), job accessibility\*, POI accessibility\* |
| Built Environment | 25% | Population density, intersection density, land use mix (Shannon entropy), sidewalk index |

\* Optional metrics from steps 2b and 2c. If omitted, weights are renormalized across the available metrics. Note that skipping them also disables criterion 5 of the minimum-service screen.

### TVI component weights

| Component | Weight | ACS Table |
|---|---|---|
| Zero-vehicle households | 30% | B08201 |
| Below poverty line | 25% | B17001 |
| Minority population | 20% | B03002 |
| Elderly (65+) | 15% | B01001 |
| Youth (10 to 17) | 10% | B01001 |

**TVI-latent** adds a forced car ownership proxy (share of low-income workers commuting by private vehicle, from B08122), reducing the zero-vehicle weight to 15% and adding a 15% forced-car component. This is the variant that matters in car-dependent regions, where a household owning a car is not evidence that it did not need transit.

---

## Classification method

1. **Transit Gap** = percentile rank(TVI) minus percentile rank(CPTA)
2. **Global Moran's I** confirms spatial autocorrelation
3. **LISA (Local Moran's I)** identifies High-High clusters (relative transit deserts) and Low-Low clusters (well-served areas)
4. **Minimum-service screen** applies the five fixed criteria, gated on above-median need
5. **Hybrid classification** labels each tract *Transit Desert*, *Transit Stressed*, *Underserved* or *Well-Served*, and records which pathway flagged it in `desert_pathway`

A tract flagged by only one pathway is the interesting case. `desert_pathway` distinguishes *Relative Only*, *Absolute Only* and *Both Pathways* so the two findings never get collapsed.

A **sensitivity analysis** reports how the classification shifts under TVI weight perturbations.

---

## Pipeline steps

```
1   Download Data          GTFS feeds, Census tract shapefiles, ACS demographics
2   Compute Supply         12 transit + accessibility + built environment metrics
2b  Job Accessibility      Gravity-model jobs reachable by transit  [optional, requires r5py + Java]
2c  POI Accessibility      Category-weighted essential service access [optional, requires r5py + Java]
2d  Compute CPTA Score     Category-weighted composite accessibility index
3   Compute Demand (TVI)   5-component Transit Vulnerability Index + TVI-latent
4   Identify Deserts       LISA clustering, minimum-service screen, sensitivity analysis, equity tests
5   Visualize              Maps, distribution plots, Moran scatterplot, correlation heatmaps
```

---

## Quickstart

### 1. Install

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

For job and POI accessibility (steps 2b and 2c), Java 11+ must be installed and `r5py` available (`pip install r5py`).

### 2. Configure your study area

Copy one of the provided configs and edit it, or edit `config/Config.yaml`:

```yaml
study_area:
  name: "Dallas County"
  state_fips: "48"
  county_fips: "113"

gtfs:
  url:
    - "https://www.dart.org/transitdata/latest/google_transit.zip"
  analysis_date: "20260310"     # must be a valid date in the GTFS calendar

census:
  acs_year: 2022
```

A free Census API key is required:

```bash
export CENSUS_API_KEY="your_key_here"
```

Find your agency's GTFS feed at [Mobility Database](https://mobilitydatabase.org/) or on the agency's developer page.

### 3. Run

```bash
# Full pipeline
python run_pipeline.py

# A specific config
python run_pipeline.py --config config/Config_Baltimore.yaml

# Skip the optional accessibility steps (no Java needed, disables screen criterion 5)
python run_pipeline.py --steps 1,2,2d,3,4,5

# Re-run analysis and visualization only
python run_pipeline.py --steps 4,5 --no-clean

# Quiet mode
python run_pipeline.py --quiet
```

---

## Included configurations

| Config | Region | Agency |
|---|---|---|
| `Config_Baltimore.yaml` | Baltimore, MD | MTA Maryland (bus, light rail, metro; multi-feed merge) |
| `Config_Dallas.yaml` | Dallas County, TX | DART |
| `Config_Nashville.yaml` | Nashville, TN | WeGo Public Transit |
| `Config_Philadelphia.yaml` | Philadelphia, PA | SEPTA |

`config/Config.yaml` is the active working configuration and is edited in place as regions are analyzed, so it will not always match any of the four above. Pass `--config` explicitly for a reproducible run.

To add a region, copy a config and set the state and county FIPS codes, the GTFS feed URL or URLs, a projected CRS in metres or feet, and an analysis date valid in that feed's calendar.

---

## Project structure

```
transit-desert-pipeline-extended/
├── run_pipeline.py              # orchestrator
├── requirements.txt
├── config/                      # study area configurations
├── src/
│   ├── 01_download_data.py      # GTFS, Census, ACS download
│   ├── 02_compute_supply.py     # transit service + built environment metrics
│   ├── 02b_compute_jobs_accessibility.py   # r5py gravity-model job access
│   ├── 02c_compute_poi_accessibility.py    # r5py POI access (8 categories)
│   ├── 02d_compute_cpta.py      # CPTA composite
│   ├── 03_compute_demand.py     # TVI and TVI-latent
│   ├── 04_identify_deserts.py   # LISA, minimum-service screen, classification
│   ├── 05_visualize.py          # maps and figures
│   ├── gtfs_utils.py            # GTFS loading and multi-feed merging
│   ├── gtfs_date_picker.py      # automatic analysis date selection
│   ├── state_osm.py             # Geofabrik OSM download and county clipping
│   └── validate_cost_burden.py  # external validation against CNT H+T Index
├── data/                        # raw, processed, external (not committed)
├── outputs/                     # maps, figures, tables (not committed)
├── notebooks/
└── logs/
```

Generated data and outputs are produced locally and are not tracked, so that nothing in this repository is a stale artifact of an older version of the code. Run the pipeline to produce them.

---

## Outputs

| File | Contents |
|---|---|
| `data/processed/supply_metrics.csv` | Per-tract CPTA component metrics |
| `data/processed/demand_metrics.csv` | Per-tract TVI and TVI-latent |
| `data/processed/absolute_thresholds.csv` | Per-criterion failure counts and rates |
| `data/processed/transit_deserts.csv` | Final classification, gap scores, pathway |
| `data/processed/transit_deserts.gpkg` | GeoPackage with results and geometries |
| `data/processed/sensitivity_analysis.csv` | TVI weight sensitivity |
| `data/processed/equity_profile.csv` | Demographic profile of flagged tracts, with Mann-Whitney U tests |
| `data/processed/barrier_summary.csv` | Which metric is the binding constraint in each flagged tract |
| `outputs/maps/` | Choropleth maps: CPTA, TVI, gap, LISA clusters, classification, absolute vs relative |
| `outputs/figures/` | Distributions, scatter plots, Moran scatterplot, correlation heatmaps |
| `outputs/tables/` | Summary statistics |

`barrier_summary.csv` is the one an agency planner usually wants first. It says *what* is failing in each flagged tract, span, walking access, frequency, so the finding points at an intervention instead of just a label.

---

## Validation

`src/validate_cost_burden.py` checks TVI-latent against the CNT Housing and Transportation (H+T) Affordability Index. Tracts flagged as high latent need should show high transportation cost burden, which tests the demand estimate without letting cost burden into the gap calculation.

Download the H+T Index CSVs from [htaindex.cnt.org/download](https://htaindex.cnt.org/download/) and place them in `data/validation/` before running.

---

## Requirements

- Python 3.10+
- Java 11+ (steps 2b and 2c only)
- Census API key in `CENSUS_API_KEY`
- osmium-tool (optional, for clipping state OSM extracts to county extent; falls back to the full-state extract)

---

## Citation

If you use this pipeline, please cite:

> Adegoke, O., Erdogan, S., and Adepitan, A. A. (2026). A Reproducible Geospatial
> Framework for Equity-Focused Transit Service Gap Analysis. IEEE International
> Conference on Intelligent Transportation Systems (ITSC), Naples.
> Preprint: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7315618

A `CITATION.cff` file is included, so GitHub's "Cite this repository" button generates BibTeX and APA automatically.

---

## Affiliation

Carried out at the
[Smart Cities and Civic Technologies (SC&CT) Research Center](https://ischool.syracuse.edu/research/smart-grid-research-center/),
School of Information Studies, Syracuse University.

---

## License

MIT. See [LICENSE](LICENSE).
