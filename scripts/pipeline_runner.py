"""

it re-calls existing functions on a rolling slice of frames, so
there is exactly one place each rule lives (same principle as
ATTRIBUTE_TO_GROUP in ad_selection.py).

CADENCE (derived from this project's real data, not guessed):

  WINDOW_SECONDS = 30   STEP_SECONDS = 15   (50% overlap)

  Real multi-member group durations in stable_groups.json range from 4.0s
  to 108.3s; median 24.6s, mean 30.4s, p75 34.6s (measured directly --
  see docs/PROGRESS_RUNBOOK.md Stage 6). A 30s window fully captures the
  median real group in a single pass. Groups longer than the window just
  get re-detected as "still present" in the next window -- for a live
  system that's the correct behavior (its membership is being reconfirmed,
  not lost), not a bug.

  15s step means every group gets checked against TWO overlapping windows
  before either one fully "expires," so a single unlucky window boundary
  can't silently drop a real group the way a hard non-overlapping cut could.

"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from group_detection import build_cooccurrence, stable_groups
from group_profiles import build_group_profiles
from ad_selection import (
    strategy_majority_class, strategy_weighted_cascade, strategy_kmodes,
    build_cluster_to_category_map,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
GENERATED_DIR = os.path.join(BASE_DIR, "data", "generated")
OUTPUT_CSV = os.path.join(GENERATED_DIR, "windowed_pipeline_results.csv")
ATTR_HISTORY_PATH = os.path.join(GENERATED_DIR, "track_attribute_history.json")

FALLBACK_FPS = 29.97  

WINDOW_SECONDS = 30
STEP_SECONDS = 15
EPS_RATIO = 0.9             
CO_OCCURRENCE_THRESHOLD = 0.6
MIN_SHARED_SECONDS = 0.5     

def get_source_fps():
    metadata_path = os.path.join(RAW_DIR, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        fps = meta.get("source_fps")
        if fps:
            print(f"Read source_fps={fps:.3f} from metadata.json")
            return fps
    print(f"WARNING: metadata.json not found/missing source_fps -- falling back to "
          f"{FALLBACK_FPS} FPS. Place metadata.json in data/raw/ for a new video to "
          f"get this right automatically.")
    return FALLBACK_FPS


def make_windows(frames, window_s, step_s):
    """Yield (window_start_s, window_end_s, frames_in_window) slices by
    timestamp_ms, so this is robust to any real FPS variation """
    if not frames:
        return
    t0 = frames[0]["timestamp_ms"] / 1000.0
    t_end = frames[-1]["timestamp_ms"] / 1000.0
    start = t0
    while start <= t_end:
        stop = start + window_s
        chunk = [f for f in frames if start <= f["timestamp_ms"] / 1000.0 < stop]
        yield start, stop, chunk
        start += step_s


def load_attribute_history():
  
    if not os.path.exists(ATTR_HISTORY_PATH):
        return None
    with open(ATTR_HISTORY_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    # sort each track's observations by time once, up front
    return {tid: sorted(obs, key=lambda o: o["timestamp_ms"]) for tid, obs in raw.items()}


def attributes_as_of(history, track_id, cutoff_ms):
    """The latest attribute observation known for this track by cutoff_ms
    i.e. what a live system would actually have on hand at that moment,
    not the track's whole-video-final reading. Returns {} (no data yet) if
    no observation has happened by cutoff_ms."""
    obs_list = history.get(str(track_id))
    if not obs_list:
        return {}
    latest = None
    for obs in obs_list:
        if obs["timestamp_ms"] > cutoff_ms:
            break
        latest = obs
    return dict(latest["attributes"]) if latest else {}


def child_ever_seen_as_of(history, track_id, cutoff_ms):
    """STICKY safety check: the safety-relevant Child flag is
    intentionally NOT allowed to un-set itself once observed true."""
    for obs in history.get(str(track_id), []):
        if obs["timestamp_ms"] > cutoff_ms:
            break
        if obs["attributes"].get("Child", {}).get("value") is True:
            return True
    return False


def build_asof_tracks_by_id(track_ids, history, cutoff_ms):
    """Build a synthetic tracks_by_id using only what was known as of cutoff_ms
    latest observation only."""
    out = {}
    for tid in track_ids:
        out[tid] = {"track_id": tid, "attributes": attributes_as_of(history, tid, cutoff_ms)}
    return out


def detect_window_groups(chunk, fps):
    """Group detection ONLY, for a single window's frames, returns
    {track_id: frozenset(all_members_of_its_group)}.
 """
    min_shared_frames = round(MIN_SHARED_SECONDS * fps)
    pair_counts = build_cooccurrence(chunk, eps_ratio=EPS_RATIO)
    track_to_group, components = stable_groups(
        pair_counts, min_shared_frames=min_shared_frames,
        co_occurrence_threshold=CO_OCCURRENCE_THRESHOLD,
    )
 
    seen_tracks = {p["track_id"] for f in chunk for p in f["persons"]}
    grouped = set(track_to_group.keys())
    solo_groups = [frozenset([t]) for t in seen_tracks - grouped]
    member_sets = [frozenset(c) for c in components] + solo_groups

    track_to_members = {}
    for members in member_sets:
        for t in members:
            track_to_members[t] = members
    return track_to_members


def compute_all_window_groups(frames, fps, window_s=WINDOW_SECONDS, step_s=STEP_SECONDS):
    """Run detect_window_groups() over every rolling window in the video,
    returning [(window_start_s, window_end_s, track_to_members), ...] in
    time order. Shared by this script's __main__ and ad_switch_cadence.py
    so both answer their respective cadence questions against the SAME
    windowed group-detection basis, instead of ad_switch_cadence.py using
    the whole-video."""
    result = []
    for w_start, w_end, chunk in make_windows(frames, window_s, step_s):
        if not chunk or not any(f["persons"] for f in chunk):
            continue
        result.append((w_start, w_end, detect_window_groups(chunk, fps)))
    return result


def run_window(chunk, tracks_by_id, fps, track_cluster_map, cluster_to_category, centroids_df,
               history=None, window_end_s=None):
    """Re-run group detection + profiles + ad-selection scoring on one
    window's frames only. Reuses the exact same functions the full-batch
    scripts use -- no re-implemented business logic.
"""
    track_to_members = detect_window_groups(chunk, fps)
    all_groups = [sorted(m) for m in set(track_to_members.values())]

    if history is not None:
        cutoff_ms = window_end_s * 1000.0
        tracks_by_id = build_asof_tracks_by_id(track_to_members.keys(), history, cutoff_ms)

    profiles = build_group_profiles(tracks_by_id, all_groups)

    if history is not None:

        for g in profiles:
            g["child_present"] = g["child_present"] or any(
                child_ever_seen_as_of(history, tid, cutoff_ms) for tid in g["member_track_ids"]
            )

    rows = []
    for g in profiles:
        rows.append({
            "group_id": g["group_id"],
            "member_track_ids": g["member_track_ids"],
            "size": g["size"],
            "child_present": g["child_present"],
            "coverage": g["coverage"],
            "majority_class": strategy_majority_class(g),
            "weighted_cascade": strategy_weighted_cascade(g, tracks_by_id),
            "kmodes": strategy_kmodes(g, track_cluster_map, cluster_to_category, centroids_df),
        })
    return rows


if __name__ == "__main__":
    os.makedirs(GENERATED_DIR, exist_ok=True)
    fps = get_source_fps()

    with open(os.path.join(RAW_DIR, "frames_bbox_only.jsonl"), encoding="utf-8") as f:
        frames = [json.loads(line) for line in f]
    with open(os.path.join(RAW_DIR, "tracks.json"), encoding="utf-8") as f:
        tracks = json.load(f)
    tracks_by_id = {t["track_id"]: t for t in tracks}

    track_clusters = pd.read_csv(os.path.join(GENERATED_DIR, "track_clusters.csv"))
    track_cluster_map = dict(zip(track_clusters["track_id"], track_clusters["cluster"]))
    centroids_df = pd.read_csv(os.path.join(GENERATED_DIR, "cluster_centroids.csv"))
    cluster_to_category = build_cluster_to_category_map(
        os.path.join(GENERATED_DIR, "cluster_centroids.csv")
    )

    attr_history = load_attribute_history()

    all_rows = []
    for w_start, w_end, chunk in make_windows(frames, WINDOW_SECONDS, STEP_SECONDS):
        n_people = sum(len(f["persons"]) for f in chunk)
        if not chunk or n_people == 0:
            continue
        rows = run_window(chunk, tracks_by_id, fps, track_cluster_map,
                           cluster_to_category, centroids_df,
                           history=attr_history, window_end_s=w_end)
        for r in rows:
            r["window_start_s"] = round(w_start, 1)
            r["window_end_s"] = round(w_end, 1)
            all_rows.append(r)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    n_windows = df["window_start_s"].nunique() if not df.empty else 0
    print(f"Re-scored {len(df)} group observations across {n_windows} rolling windows")
    print("Saved windowed_pipeline_results.csv")
