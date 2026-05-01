#!/usr/bin/env python3
"""
run_pipeline.py
================
Transit Desert Identification Pipeline (Extended Version)

Pipeline Steps:
  1   Download Data (GTFS, Census, ACS, OSM)
  2   Compute Supply Metrics (CPTA) — 11 built-in metrics
  2b  Compute Job Accessibility (optional, requires r5py + Java)
  2c  Compute POI Accessibility (optional, requires r5py + Java)
  2d  Compute CPTA Score (composite accessibility index)
  3   Compute Demand Metrics (TVI) — 5 components
  4   Identify Transit Deserts (LISA clustering + sensitivity analysis)
  5   Generate Visualizations (maps, figures, tables)

Usage examples:
  python run_pipeline.py                                  # Run all steps (1→2→2b→2c→2d→3→4→5)
  python run_pipeline.py --config config/MyCity.yaml      # Use a specific config file
  python run_pipeline.py --steps 1,2,3,4,5                # Skip 2b/2c/2d (no accessibility)
  python run_pipeline.py --steps 3,4,5 --no-clean         # Re-run from demand onward
  python run_pipeline.py --no-clean --steps 4,5            # Re-run analysis + viz only
  python run_pipeline.py --quiet                           # Suppress terminal output
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List


# ---------------------------
# Project paths
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"


# ---------------------------
# Step registry
# ---------------------------
@dataclass(frozen=True)
class StepDef:
    id: str
    name: str
    script: str
    description: str
    optional: bool = False


STEPS = {
    "1": StepDef(
        id="1",
        name="Download Data",
        script="src/01_download_data.py",
        description="Download GTFS feeds, Census tracts, ACS demographics",
    ),
    "2": StepDef(
        id="2",
        name="Compute Supply (CPTA)",
        script="src/02_compute_supply.py",
        description="Calculate 11 transit + built environment metrics, CPTA score",
    ),
    "2b": StepDef(
        id="2b",
        name="Job Accessibility",
        script="src/02b_compute_jobs_accessibility.py",
        description="Jobs reachable within 45 min by transit (requires r5py + Java)",
        optional=True,
    ),
    "2c": StepDef(
        id="2c",
        name="POI Accessibility",
        script="src/02c_compute_poi_accessibility.py",
        description="Essential service POIs reachable within 30 min (requires r5py + Java)",
        optional=True,
    ),
    "2d": StepDef(
        id="2d",
        name="Compute CPTA Score",
        script="src/02d_compute_cpta.py",
        description="Category-weighted composite transit accessibility index",
    ),
    "3": StepDef(
        id="3",
        name="Compute Demand (TVI)",
        script="src/03_compute_demand.py",
        description="Calculate 5-component Transit Vulnerability Index",
    ),
    "4": StepDef(
        id="4",
        name="Identify Transit Deserts",
        script="src/04_identify_deserts.py",
        description="LISA clustering, sensitivity analysis, equity tests",
    ),
    "5": StepDef(
        id="5",
        name="Generate Visualizations",
        script="src/05_visualize.py",
        description="Maps, figures, correlation heatmap, Moran scatterplot",
    ),
}

# Canonical step order
STEP_ORDER = ["1", "2", "2b", "2c", "2d", "3", "4", "5"]


# ---------------------------
# Printing helpers
# ---------------------------
def print_banner(steps_to_run: List[str]) -> None:
    print()
    print("=" * 70)
    print("  TRANSIT DESERT IDENTIFICATION PIPELINE (Extended)")
    print("  Equity-Focused Service Gap Analysis")
    print("=" * 70)
    print()
    print("  Pipeline Steps:")
    print("  " + "-" * 55)
    for k in STEP_ORDER:
        s = STEPS[k]
        marker = "→" if k in steps_to_run else " "
        opt = " (optional)" if s.optional else ""
        skip = " [SKIP]" if k not in steps_to_run and s.optional else ""
        print(f"  {marker} {s.id:>3}. {s.name}{opt}{skip}")
        print(f"        {s.description}")
    print("  " + "-" * 55)
    print()


# ---------------------------
# Cleaning
# ---------------------------
def _empty_dir(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for item in dir_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            try:
                item.unlink()
            except FileNotFoundError:
                pass


def clean_run_artifacts() -> None:
    targets = [
        PROJECT_ROOT / "data" / "raw" / "gtfs",
        PROJECT_ROOT / "data" / "raw" / "acs",
        PROJECT_ROOT / "data" / "raw" / "census",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "external",
        PROJECT_ROOT / "outputs" / "maps",
        PROJECT_ROOT / "outputs" / "figures",
        PROJECT_ROOT / "outputs" / "tables",
    ]

    print("\n  Cleaning previous run artifacts...")
    for t in targets:
        _empty_dir(t)
        print(f"    Emptied: {t.relative_to(PROJECT_ROOT)}")
    print()


# ---------------------------
# Logging
# ---------------------------
def get_run_log_path() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"pipeline_run_{ts}.log"


def log_write(log_path: Optional[Path], text: str) -> None:
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


# ---------------------------
# Step execution
# ---------------------------
def run_step(step_id: str, *, verbose: bool, log_path: Optional[Path]) -> bool:
    if step_id not in STEPS:
        raise ValueError(f"Unknown step: {step_id}")

    step = STEPS[step_id]
    script_path = PROJECT_ROOT / step.script

    if not script_path.exists():
        msg = f"  Script not found: {script_path}"
        print(msg)
        log_write(log_path, msg)
        return False

    header = f"STEP {step.id}: {step.name}"
    print("\n" + "=" * 60)
    print(f"  {header}")
    print("=" * 60)

    start_time = time.time()

    log_write(log_path, f"\n{'=' * 60}")
    log_write(log_path, f"  {header}")
    log_write(log_path, f"  Started: {datetime.now().isoformat(timespec='seconds')}")
    log_write(log_path, f"{'=' * 60}\n")

    cmd = [sys.executable, str(script_path)]

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            if verbose:
                print(line, end="")
            log_write(log_path, line.rstrip())

        rc = proc.wait()
        elapsed = time.time() - start_time

        if rc == 0:
            msg = f"\n  Step {step.id} completed in {elapsed:.1f}s"
            print(msg)
            log_write(log_path, msg)
            return True
        else:
            msg = f"\n  Step {step.id} FAILED (exit code {rc})"
            print(msg)
            log_write(log_path, msg)
            return False

    except Exception as e:
        msg = f"\n  Error running step {step.id}: {e}"
        print(msg)
        log_write(log_path, msg)
        return False


# ---------------------------
# Pipeline orchestration
# ---------------------------
def parse_steps_arg(steps_str: Optional[str]) -> List[str]:
    if steps_str:
        steps = [s.strip().lower() for s in steps_str.split(",") if s.strip()]
    else:
        steps = ["1", "2", "2b", "2c", "2d", "3", "4", "5"]

    for s in steps:
        if s not in STEPS:
            raise ValueError(f"Invalid step '{s}'. Valid: {', '.join(STEPS.keys())}")
    return steps


def run_pipeline(
    steps: List[str],
    *,
    verbose: bool,
    clean: bool,
    log_path: Optional[Path],
    config_path: Optional[Path] = None,
) -> bool:
    print_banner(steps)

    if config_path:
        dest = PROJECT_ROOT / "config" / "config.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, dest)
        print(f"  Using config: {config_path}")

    if clean and "1" in steps:
        clean_run_artifacts()

    if log_path:
        print(f"  Log file: {log_path.relative_to(PROJECT_ROOT)}")
        log_write(log_path, "=" * 70)
        log_write(log_path, "TRANSIT DESERT PIPELINE RUN LOG (Extended)")
        log_write(log_path, f"Started: {datetime.now().isoformat(timespec='seconds')}")
        log_write(log_path, f"Steps: {', '.join(steps)}")
        log_write(log_path, "=" * 70)

    print(f"\n  Running steps: {' → '.join(steps)}")

    pipeline_start = time.time()
    step_status = {}
    failed = False

    for step_id in steps:
        step = STEPS[step_id]

        ok = run_step(step_id, verbose=verbose, log_path=log_path)
        step_status[step_id] = ok

        if not ok:
            if step.optional:
                print(f"\n  Optional step {step_id} failed — continuing pipeline")
                step_status[step_id] = "skipped"
            else:
                print(f"\n  Pipeline stopped at step {step_id}")
                failed = True
                break

    total_elapsed = time.time() - pipeline_start

    # Summary
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)

    for s in steps:
        status = step_status.get(s)
        if status is True:
            icon = "✓"
            label = "Complete"
        elif status == "skipped":
            icon = "⚠"
            label = "Failed (optional, skipped)"
        elif status is False:
            icon = "✗"
            label = "Failed"
        else:
            icon = "—"
            label = "Not reached"
        print(f"    {icon} Step {s}: {label}")

    minutes = total_elapsed / 60
    print(f"\n  Total time: {total_elapsed:.1f}s ({minutes:.1f} min)")

    all_ok = not failed

    if log_path:
        log_write(log_path, "\n" + "=" * 70)
        log_write(log_path, f"Finished: {datetime.now().isoformat(timespec='seconds')}")
        log_write(log_path, f"Total time: {total_elapsed:.1f}s")
        log_write(log_path, f"Status: {'SUCCESS' if all_ok else 'FAILED'}")
        log_write(log_path, "=" * 70)

    if all_ok:
        print("\n  Pipeline completed successfully!")
        print(f"\n  Results: data/processed/")
        print(f"  Outputs: outputs/maps/, outputs/figures/, outputs/tables/")
    else:
        print("\n  Pipeline completed with errors.")

    print()
    return all_ok


# ---------------------------
# CLI
# ---------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transit Desert Identification Pipeline (Extended)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                                  Run all steps (1→2→2b→2c→2d→3→4→5)
  python run_pipeline.py --config config/MyCity.yaml      Use a specific config file
  python run_pipeline.py --steps 1,2,3,4,5                Skip accessibility steps
  python run_pipeline.py --steps 3,4,5 --no-clean         Re-run from demand onward
  python run_pipeline.py --no-clean --steps 4,5            Re-run analysis + viz only
        """
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: config/Config.yaml)",
    )
    parser.add_argument(
        "--steps", type=str, default=None,
        help="Comma-separated steps: 1,2,2b,2c,2d,3,4,5 (default: all)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress terminal output (still logs to file)",
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="Keep data/outputs from previous run",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Do not write a run log file",
    )

    args = parser.parse_args()

    steps = parse_steps_arg(args.steps)
    log_path = None if args.no_log else get_run_log_path()
    verbose = not args.quiet
    clean = not args.no_clean
    config_path = Path(args.config) if args.config else None

    ok = run_pipeline(steps, verbose=verbose, clean=clean, log_path=log_path, config_path=config_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()