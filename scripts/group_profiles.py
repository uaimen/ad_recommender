"""
Joins Group Detection output (stable_groups.json) with resolved per-track
attributes (from cluster_audience.py's resolve_track logic) to produce a
single group-level table: this is the direct input to the ad-selection
strategies (majority-class / weighted-matching / k-modes).

For each group, produces:
  - member track IDs
  - each member's resolved attribute profile
  - a `child_present` flag (hard safety-filter input, computed independent
    of any voting/clustering logic)
  - coverage: how many members actually have attribute data vs. none
"""

import json
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))
from cluster_audience import resolve_track  # reuse the exact preprocessing already built


def load_data():
    with open(os.path.join(RAW_DIR, "tracks.json"), encoding="utf-8") as f:
        tracks = json.load(f)
    with open(os.path.join(RAW_DIR, "stable_groups.json"), encoding="utf-8") as f:
        groups_data = json.load(f)
    return {t["track_id"]: t for t in tracks}, groups_data["groups"]


def build_group_profiles(tracks_by_id, groups):
    group_profiles = []
    for gid, member_ids in enumerate(groups):
        members = []
        for tid in member_ids:
            track = tracks_by_id.get(tid)
            if track is None:
                continue
            resolved = resolve_track(track)
            members.append(resolved)

        n_with_attrs = sum(1 for m in members if m["has_attributes"])
        child_present = any(m.get("Age") == "Child" for m in members)

        group_profiles.append({
            "group_id": gid,
            "size": len(members),
            "member_track_ids": member_ids,
            "members_with_attributes": n_with_attrs,
            "coverage": round(n_with_attrs / len(members), 2) if members else 0,
            "child_present": child_present,
            "members": members,
        })
    return group_profiles


if __name__ == "__main__":
    tracks_by_id, groups = load_data()
    profiles = build_group_profiles(tracks_by_id, groups)

    child_present = sum(1 for p in profiles if p["child_present"])

    os.makedirs(RAW_DIR, exist_ok=True)
    with open(os.path.join(RAW_DIR, "group_profiles.json"), "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    print(f"Built profiles for {len(profiles)} groups ({child_present} with a child present)")
    print("Saved group_profiles.json")
