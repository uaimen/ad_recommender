"""
Ad-switch cadence simulation (Update Frequency)

Simulates how often the ad shown on screen would actually need to switch,
using the REAL frame-by-frame audience composition 

Method:
  1. For every frame, find which group is "dominant" and has the most
     members currently visible in that frame. This approximates "who the
     screen is showing an ad to" at that instant.
  2. Track how often that dominant group changes across the whole clip, both
     RAW (no debounce, switches on every single frame-level change) and
     DEBOUNCED at several hold windows (a new group must stay dominant for
     N consecutive frames before it's counted as a real switch).
  3. Compare the resulting "average time between switches" against published
     digital-signage attention-span research.

LIVE GROUPING:
  Groups are re-detected fresh per rolling window, reusing pipeline_runner.py's
  own WINDOW_SECONDS/STEP_SECONDS cadence and compute_all_window_groups()
 so this script and pipeline_runner.py's
  ad-category scoring answer their respective cadence questions on the same
  basis. Each frame's dominant group is looked up from whichever window had
  most recently COMPLETED as of that frame's timestamp -- what a live system
  would actually have on hand, including the real startup latency of having
  NO group knowledge for the first WINDOW_SECONDS of a deployment.

  Group identity across windows is compared by MEMBER SET (frozenset),  two independently re-detected
  windows assign unrelated ids to the same real-world group, so comparing
  raw ids would over-count spurious "switches" from renumbering alone, not
  real audience change. See pipeline_runner.detect_window_groups() for
  where this identity scheme is built.

Requires: frames_bbox_only.jsonl, pipeline_runner.py (for windowing
          constants + compute_all_window_groups)
"""

import json
import os
import sys
from collections import Counter
import bisect
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_runner import compute_all_window_groups

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
GENERATED_DIR = os.path.join(BASE_DIR, "data", "generated")
FRAMES_PATH = os.path.join(RAW_DIR, "frames_bbox_only.jsonl")
METADATA_PATH = os.path.join(RAW_DIR, "metadata.json")
OUTPUT_CSV_PATH = os.path.join(GENERATED_DIR, "ad_switch_cadence_results.csv")

FALLBACK_FPS = 29.97 


def get_source_fps():
    """Read the real source FPS from metadata.json so this script works
    correctly on any video's output, not just the one it was built against.
    Falls back to the current video's known FPS if metadata.json is missing,
    so the script never silently breaks -- but prints a warning either way
    so it's obvious which path was taken."""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        fps = meta.get("source_fps")
        if fps:
            print(f"Read source_fps={fps:.3f} from metadata.json")
            return fps
    print(f"WARNING: metadata.json not found or missing source_fps -- "
          f"falling back to {FALLBACK_FPS} FPS. Place metadata.json in this "
          f"folder to get the correct value automatically for a new video.")
    return FALLBACK_FPS


SOURCE_FPS = get_source_fps()


def load_frames():
    with open(FRAMES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dominant_group(frame: dict, windows: list, window_ends: list):
    """Which group has the most members visible in this frame, using ONLY
    the group mapping from the most recently COMPLETED rolling window as of
    this frame's timestamp -- what a live system re-running every
    STEP_SECONDS would actually have on hand right now. Returns None before
    the first window has completed (real startup latency) or if none of the
    present tracks appear in that window's mapping.

    `windows` / `window_ends` come from pipeline_runner.compute_all_window_groups()
    -- window_ends is the parallel, pre-sorted list of each window's end
    time, used for the bisect lookup."""
    t = frame["timestamp_ms"] / 1000.0
    idx = bisect.bisect_right(window_ends, t) - 1
    if idx < 0:
        return None
    _, _, track_to_members = windows[idx]
    present_tracks = [p["track_id"] for p in frame["persons"]]
    member_sets = [track_to_members[tid] for tid in present_tracks if tid in track_to_members]
    if not member_sets:
        return None
    return Counter(member_sets).most_common(1)[0][0]


def debounced_sequence(seq: list, min_hold: int):
    """A new dominant group only 'takes over' once it's held for min_hold
    consecutive frames -- same stability idea used in group_detection.py."""
    result = []
    current = seq[0]
    hold_count = 0
    pending = None
    for val in seq:
        if val == current:
            hold_count += 1
            pending = None
        else:
            if pending == val:
                hold_count += 1
            else:
                pending = val
                hold_count = 1
            if hold_count >= min_hold:
                current = val
                hold_count = 0
                pending = None
        result.append(current)
    return result


def count_switches(seq: list) -> int:
    return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])


def switch_table(sequence, n_frames):
    """Raw + debounced switch counts/avg-hold-time for the dominant-group sequence."""
    rows = []
    raw_switches = count_switches(sequence)
    raw_avg_frames = n_frames / max(raw_switches, 1)
    rows.append({
        "debounce_frames": 0, "debounce_seconds": 0.0,
        "real_switches": raw_switches,
        "avg_seconds_between_switches": round(raw_avg_frames / SOURCE_FPS, 2),
    })
    for min_hold in [15, 30, 45, 90]:  # 0.5s, 1s, 1.5s, 3s at ~30fps
        deb = debounced_sequence(sequence, min_hold=min_hold)
        switches = count_switches(deb)
        avg_frames = n_frames / max(switches, 1)
        rows.append({
            "debounce_frames": min_hold, "debounce_seconds": round(min_hold / SOURCE_FPS, 2),
            "real_switches": switches,
            "avg_seconds_between_switches": round(avg_frames / SOURCE_FPS, 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    frames = load_frames()
    os.makedirs(GENERATED_DIR, exist_ok=True)

    windows = compute_all_window_groups(frames, SOURCE_FPS)
    window_ends = [w_end for _, w_end, _ in windows]
    sequence = [dominant_group(f, windows, window_ends) for f in frames]
    df = switch_table(sequence, len(frames))
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    recommended = df[df["debounce_frames"] == 15].iloc[0]
    hold_time = recommended["avg_seconds_between_switches"]
    print(f"Computed switch cadence across {len(windows)} rolling windows: "
          f"{hold_time}s average hold at 0.5s debounce")
    if hold_time < 4.6:
        print("Warning: this is below the 4.6s reference attention span; "
              "check the wider debounce settings in the saved results")
    print("Saved ad_switch_cadence_results.csv")
