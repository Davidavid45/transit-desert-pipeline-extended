"""
state_osm.py — Map state FIPS codes to Geofabrik OSM PBF download URLs.

Downloads the full state PBF, then clips it to the study-area bounding box
(+ ~10 km buffer) so that r5py/R5 stays under its 975 000 km² geographic
extent limit.  This is essential for large states like Texas and California.

Usage in 02b / 02c:
    from state_osm import get_osm_pbf_path

    pbf_path = get_osm_pbf_path(config, output_dir)
"""

from pathlib import Path
import shutil
import subprocess
import requests
import geopandas as gpd

# FIPS code → Geofabrik state name
FIPS_TO_STATE = {
    "01": "alabama", "02": "alaska", "04": "arizona", "05": "arkansas",
    "06": "california", "08": "colorado", "09": "connecticut", "10": "delaware",
    "11": "district-of-columbia", "12": "florida", "13": "georgia", "15": "hawaii",
    "16": "idaho", "17": "illinois", "18": "indiana", "19": "iowa",
    "20": "kansas", "21": "kentucky", "22": "louisiana", "23": "maine",
    "24": "maryland", "25": "massachusetts", "26": "michigan", "27": "minnesota",
    "28": "mississippi", "29": "missouri", "30": "montana", "31": "nebraska",
    "32": "nevada", "33": "new-hampshire", "34": "new-jersey", "35": "new-mexico",
    "36": "new-york", "37": "north-carolina", "38": "north-dakota", "39": "ohio",
    "40": "oklahoma", "41": "oregon", "42": "pennsylvania", "44": "rhode-island",
    "45": "south-carolina", "46": "south-dakota", "47": "tennessee", "48": "texas",
    "49": "utah", "50": "vermont", "51": "virginia", "53": "washington",
    "54": "west-virginia", "55": "wisconsin", "56": "wyoming",
}


def get_state_name(config):
    """Get Geofabrik state name from config's state FIPS code."""
    state_fips = config['study_area']['state_fips']
    state_name = FIPS_TO_STATE.get(state_fips)
    if not state_name:
        raise ValueError(f"Unknown state FIPS: {state_fips}. Add it to FIPS_TO_STATE.")
    return state_name


def get_osm_pbf_path(config, output_dir):
    """
    Get (and download / clip if needed) the OSM PBF file for the study area.

    1. Downloads the full-state PBF from Geofabrik (cached).
    2. Clips it to the study-area bounding box + ~10 km buffer using osmium.

    Parameters
    ----------
    config : dict
        Pipeline config with study_area.state_fips / county_fips
    output_dir : Path
        Base data directory (e.g., data/external/osm)

    Returns
    -------
    Path
        Path to the clipped PBF file (or full-state if clipping unavailable)
    """
    state_name = get_state_name(config)
    state_filename = f"{state_name}-latest.osm.pbf"
    county_fips = config['study_area']['county_fips']

    osm_dir = Path(output_dir) / "osm"
    osm_dir.mkdir(parents=True, exist_ok=True)

    state_pbf = osm_dir / state_filename
    clipped_filename = f"{state_name}-county-{county_fips}.osm.pbf"
    clipped_pbf = osm_dir / clipped_filename

    # If clipped PBF already exists, use it directly
    if clipped_pbf.exists():
        print(f"  ✓ Clipped OSM PBF already exists: {clipped_pbf}")
        return clipped_pbf

    # --- Step 1: ensure state PBF is downloaded ---
    if not state_pbf.exists():
        # Remove stale PBFs from a different state
        for old_pbf in osm_dir.glob("*-latest.osm.pbf"):
            if old_pbf.name != state_filename:
                print(f"  ⚠ Removing stale OSM PBF: {old_pbf.name}")
                old_pbf.unlink()

        url = f"https://download.geofabrik.de/north-america/us/{state_filename}"
        print(f"  Downloading OSM data from: {url}")
        print(f"  (This may take a few minutes...)")

        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(state_pbf, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        mb = downloaded / 1024 / 1024
                        print(f"\r  Downloaded {mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

            print(f"\n  ✓ Downloaded OSM PBF to {state_pbf}")

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error downloading OSM data: {e}")
            return None
    else:
        print(f"  ✓ State OSM PBF already exists: {state_pbf}")

    # --- Step 2: clip to study-area bbox + buffer ---
    clipped = _clip_to_study_area(config, state_pbf, clipped_pbf)
    if clipped and clipped.exists():
        return clipped

    # Fallback to full state PBF if clipping failed
    print(f"  ⚠ Using full state PBF (clipping unavailable)")
    return state_pbf


def _clip_to_study_area(config, state_pbf, clipped_pbf):
    """
    Clip a state PBF to the study-area bounding box + ~10 km buffer.

    Requires:
    - osmium-tool installed (brew install osmium-tool / apt install osmium-tool)
    - study_area_tracts.shp already downloaded by step 01

    Returns clipped_pbf Path on success, None on failure.
    """
    # Check for osmium
    if not shutil.which("osmium"):
        print("  ⚠ osmium-tool not installed — cannot clip OSM data")
        print("    Install with: brew install osmium-tool  (macOS)")
        print("              or: apt install osmium-tool   (Linux)")
        return None

    # Locate study-area tracts shapefile
    project_root = Path(__file__).parent.parent
    tracts_path = project_root / "data" / "raw" / "census" / "study_area_tracts.shp"

    if not tracts_path.exists():
        print(f"  ⚠ Cannot clip: tracts not found at {tracts_path}")
        print(f"    Run step 01 first to download Census tracts")
        return None

    # Compute bounding box from tracts (in WGS84)
    tracts = gpd.read_file(tracts_path).to_crs("EPSG:4326")
    bounds = tracts.total_bounds  # [minx, miny, maxx, maxy]

    # Add ~10 km buffer (≈ 0.1 degrees at mid-latitudes)
    buffer_deg = 0.1
    west  = bounds[0] - buffer_deg
    south = bounds[1] - buffer_deg
    east  = bounds[2] + buffer_deg
    north = bounds[3] + buffer_deg

    bbox_str = f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}"
    print(f"  Clipping OSM PBF to study area bbox (+ ~10 km buffer)...")
    print(f"    Bbox: {bbox_str}")

    cmd = [
        "osmium", "extract",
        "--bbox", bbox_str,
        "--strategy", "smart",
        "--overwrite",
        "-o", str(clipped_pbf),
        str(state_pbf),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            state_mb = state_pbf.stat().st_size / 1024 / 1024
            clipped_mb = clipped_pbf.stat().st_size / 1024 / 1024
            print(f"  ✓ Clipped: {state_mb:.1f} MB → {clipped_mb:.1f} MB")
            return clipped_pbf
        else:
            print(f"  ✗ osmium extract failed: {result.stderr.strip()}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  ✗ osmium extract timed out")
        return None
    except Exception as e:
        print(f"  ✗ Error clipping OSM data: {e}")
        return None
