"""
gtfs_date_picker.py — Automatically select the best analysis date from GTFS feeds.

Usage:
    from gtfs_date_picker import pick_best_date

    best_date = pick_best_date("/path/to/gtfs/directory")
    # Returns: "20260310" (string in GTFS date format)

Logic:
    1. Scan all GTFS feeds (zips) in the directory
    2. Read calendar.txt and calendar_dates.txt from each feed
    3. Find the intersection of valid date ranges across all feeds
    4. For each candidate date (Tue/Wed/Thu only), count active trips
    5. Return the date with the most trips

Can be called from 02_compute_supply.py to replace the hardcoded analysis_date,
or from run_pipeline.py to set it once for the entire run.
"""

import zipfile
import csv
import io
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict


def _read_csv_from_zip(zip_path, filename):
    """Read a CSV file from inside a GTFS zip, return list of dicts."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        # Handle nested directories inside zip
        match = [n for n in names if n.endswith(filename)]
        if not match:
            return []
        with zf.open(match[0]) as f:
            text = io.TextIOWrapper(f, encoding='utf-8-sig')
            reader = csv.DictReader(text)
            return list(reader)


def _parse_date(date_str):
    """Parse GTFS date string (YYYYMMDD) to datetime.date."""
    return datetime.strptime(date_str.strip(), "%Y%m%d").date()


def _get_feed_info(zip_path):
    """
    Extract calendar info from a single GTFS feed.
    
    Returns:
        dict with:
            'start_date': earliest service start
            'end_date': latest service end
            'service_days': {date: set_of_service_ids} for all valid dates
            'trips_per_service': {service_id: trip_count}
    """
    zip_path = Path(zip_path)
    
    # --- Read calendar.txt ---
    calendar_rows = _read_csv_from_zip(zip_path, 'calendar.txt')
    
    # --- Read calendar_dates.txt ---
    calendar_dates_rows = _read_csv_from_zip(zip_path, 'calendar_dates.txt')
    
    # --- Read trips.txt to count trips per service_id ---
    trips_rows = _read_csv_from_zip(zip_path, 'trips.txt')
    trips_per_service = defaultdict(int)
    for row in trips_rows:
        sid = row.get('service_id', '').strip()
        if sid:
            trips_per_service[sid] += 1
    
    # --- Build service_days: {date -> set of active service_ids} ---
    service_days = defaultdict(set)
    
    # From calendar.txt: regular weekly service
    day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    
    for row in calendar_rows:
        sid = row.get('service_id', '').strip()
        try:
            start = _parse_date(row['start_date'])
            end = _parse_date(row['end_date'])
        except (KeyError, ValueError):
            continue
        
        # Which days of the week is this service active?
        active_days = set()
        for i, day in enumerate(day_names):
            if row.get(day, '0').strip() == '1':
                active_days.add(i)  # 0=Monday, 6=Sunday
        
        # Enumerate all dates in range
        current = start
        while current <= end:
            if current.weekday() in active_days:
                service_days[current].add(sid)
            current += timedelta(days=1)
    
    # From calendar_dates.txt: exceptions
    for row in calendar_dates_rows:
        sid = row.get('service_id', '').strip()
        try:
            date = _parse_date(row['date'])
            exception_type = row.get('exception_type', '').strip()
        except (KeyError, ValueError):
            continue
        
        if exception_type == '1':
            # Service added
            service_days[date].add(sid)
        elif exception_type == '2':
            # Service removed
            service_days[date].discard(sid)
    
    # Find date range
    if service_days:
        all_dates = sorted(service_days.keys())
        start_date = all_dates[0]
        end_date = all_dates[-1]
    else:
        start_date = None
        end_date = None
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'service_days': dict(service_days),
        'trips_per_service': dict(trips_per_service),
        'zip_name': zip_path.name,
    }


def pick_best_date(gtfs_dir, prefer_weekdays=(1, 2, 3), verbose=True):
    """
    Automatically select the best analysis date from GTFS feeds.
    
    Parameters
    ----------
    gtfs_dir : str or Path
        Directory containing GTFS zip file(s)
    prefer_weekdays : tuple of int
        Preferred days of week (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri)
        Default: (1, 2, 3) = Tuesday, Wednesday, Thursday
    verbose : bool
        Print diagnostic info
        
    Returns
    -------
    str
        Best date in GTFS format (YYYYMMDD), or None if no valid date found
    """
    gtfs_dir = Path(gtfs_dir)
    
    # Find all GTFS zips
    zip_files = sorted(gtfs_dir.glob("*.zip"))
    if not zip_files:
        if verbose:
            print("  ⚠ No GTFS zip files found")
        return None
    
    if verbose:
        print(f"  Scanning {len(zip_files)} GTFS feed(s) for best analysis date...")
    
    # Get calendar info from each feed
    feeds = []
    for zf in zip_files:
        try:
            info = _get_feed_info(zf)
            feeds.append(info)
            if verbose:
                if info['start_date'] and info['end_date']:
                    n_services = len(info['trips_per_service'])
                    total_trips = sum(info['trips_per_service'].values())
                    print(f"    {info['zip_name']}: "
                          f"{info['start_date']} to {info['end_date']}, "
                          f"{n_services} services, {total_trips} trips")
                else:
                    print(f"    {info['zip_name']}: no calendar data found")
        except Exception as e:
            if verbose:
                print(f"    {zf.name}: error reading calendar — {e}")
    
    if not feeds:
        return None
    
    # Find date range that works across ALL feeds
    valid_starts = [f['start_date'] for f in feeds if f['start_date']]
    valid_ends = [f['end_date'] for f in feeds if f['end_date']]
    
    if not valid_starts or not valid_ends:
        if verbose:
            print("  ⚠ No valid date ranges found in any feed")
        return None
    
    # Intersection: latest start to earliest end
    range_start = max(valid_starts)
    range_end = min(valid_ends)
    
    if range_start > range_end:
        if verbose:
            print(f"  ⚠ No overlapping date range across feeds")
            print(f"    Starts: {valid_starts}")
            print(f"    Ends: {valid_ends}")
        return None
    
    if verbose:
        print(f"  Valid range across all feeds: {range_start} to {range_end}")
    
    # Score each candidate date
    # Prefer recent dates (closer to range_end), on Tue/Wed/Thu
    candidates = []
    current = range_start
    while current <= range_end:
        if current.weekday() in prefer_weekdays:
            # Count total trips across all feeds on this date
            total_trips = 0
            all_feeds_active = True
            
            for feed in feeds:
                service_ids = feed['service_days'].get(current, set())
                if not service_ids:
                    all_feeds_active = False
                    break
                feed_trips = sum(
                    feed['trips_per_service'].get(sid, 0) 
                    for sid in service_ids
                )
                total_trips += feed_trips
            
            if all_feeds_active and total_trips > 0:
                candidates.append((current, total_trips))
        
        current += timedelta(days=1)
    
    if not candidates:
        if verbose:
            print("  ⚠ No valid Tue/Wed/Thu dates with active service across all feeds")
            # Fall back to any weekday
            print("  Trying any weekday...")
            current = range_start
            while current <= range_end:
                if current.weekday() < 5:  # Mon-Fri
                    total_trips = 0
                    all_active = True
                    for feed in feeds:
                        sids = feed['service_days'].get(current, set())
                        if not sids:
                            all_active = False
                            break
                        total_trips += sum(feed['trips_per_service'].get(s, 0) for s in sids)
                    if all_active and total_trips > 0:
                        candidates.append((current, total_trips))
                current += timedelta(days=1)
    
    if not candidates:
        if verbose:
            print("  ⚠ No valid weekday dates found")
        return None
    
    # Sort by trip count (descending), then by recency (descending)
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    
    best_date, best_trips = candidates[0]
    date_str = best_date.strftime("%Y%m%d")
    
    if verbose:
        # Show top 5 candidates
        print(f"\n  Top candidate dates:")
        for date, trips in candidates[:5]:
            marker = " ← SELECTED" if date == best_date else ""
            print(f"    {date} ({date.strftime('%A')}): {trips:,} trips{marker}")
        
        # Also show the worst to give context
        if len(candidates) > 5:
            worst_date, worst_trips = candidates[-1]
            print(f"    ...")
            print(f"    {worst_date} ({worst_date.strftime('%A')}): {worst_trips:,} trips (worst)")
        
        print(f"\n  ✓ Best date: {date_str} ({best_date.strftime('%A')}) — {best_trips:,} trips")
    
    return date_str


def validate_date(gtfs_dir, date_str, verbose=True):
    """
    Check whether a given date is valid across all GTFS feeds.
    
    Parameters
    ----------
    gtfs_dir : str or Path
        Directory containing GTFS zip file(s)
    date_str : str
        Date to validate in YYYYMMDD format
    verbose : bool
        Print diagnostic info
        
    Returns
    -------
    bool
        True if the date has active service in all feeds
    """
    gtfs_dir = Path(gtfs_dir)
    target = _parse_date(date_str)
    
    zip_files = sorted(gtfs_dir.glob("*.zip"))
    if not zip_files:
        return False
    
    all_valid = True
    for zf in zip_files:
        try:
            info = _get_feed_info(zf)
            service_ids = info['service_days'].get(target, set())
            trips = sum(info['trips_per_service'].get(sid, 0) for sid in service_ids)
            
            if verbose:
                if service_ids:
                    print(f"    {zf.name}: ✓ {len(service_ids)} services, {trips} trips")
                else:
                    print(f"    {zf.name}: ✗ NO active service on {date_str}")
                    if info['start_date'] and info['end_date']:
                        print(f"      Valid range: {info['start_date']} to {info['end_date']}")
            
            if not service_ids:
                all_valid = False
                
        except Exception as e:
            if verbose:
                print(f"    {zf.name}: error — {e}")
            all_valid = False
    
    return all_valid


# --- CLI usage ---
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gtfs_date_picker.py <gtfs_directory> [date_to_validate]")
        print()
        print("Examples:")
        print("  python gtfs_date_picker.py data/raw/gtfs")
        print("  python gtfs_date_picker.py data/raw/gtfs 20260310")
        sys.exit(1)
    
    gtfs_dir = sys.argv[1]
    
    if len(sys.argv) >= 3:
        # Validate a specific date
        date_str = sys.argv[2]
        print(f"Validating date {date_str} against GTFS feeds in {gtfs_dir}:")
        valid = validate_date(gtfs_dir, date_str)
        if valid:
            print(f"\n  ✓ Date {date_str} is valid across all feeds")
        else:
            print(f"\n  ✗ Date {date_str} is NOT valid — picking best alternative...")
            best = pick_best_date(gtfs_dir)
            if best:
                print(f"\n  Suggested alternative: {best}")
    else:
        # Auto-pick best date
        print(f"Scanning GTFS feeds in {gtfs_dir}:")
        best = pick_best_date(gtfs_dir)
        if best:
            print(f"\nTo use this date, update Config.yaml:")
            print(f'  analysis_date: "{best}"')
        else:
            print("\nCould not determine a valid analysis date.")
