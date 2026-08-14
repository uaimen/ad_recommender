"""
Missing-Data Handling Test Suite.

The ~44% no-attribute-data track rate is the single biggest data-quality issue
in this pipeline (Section 3.3 of the report). This script EXPLICITLY tests
the required behavior, rather than leaving it as an implicit side effect of
ad_selection.py's code:

    Partial missing (some members have data, some don't):
        -> use the members that HAVE data, ignore the ones that don't
        -> group still gets a real category

    Total missing (every member has no data):
        -> the group has no usable signal
        -> group MUST get Neutral-Fallback, not a guessed category

Tests run against:
  (a) REAL cases pulled directly from group_profiles.json wherever they exist
  (b) CONSTRUCTED cases for scenarios real data doesn't happen to contain


"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from ad_selection import (
    strategy_majority_class,
    strategy_weighted_cascade,
    strategy_kmodes,
    build_cluster_to_category_map,
    BASE_DIR,
    RAW_DIR,
    GENERATED_DIR,
)
import pandas as pd

FALLBACK = "Neutral-Fallback"

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
        line = f"FAIL  {name}"
        if detail:
            line += f": {detail}"
        print(line)
    return condition


def make_member(track_id, has_attributes, **attrs):
    """Build a resolved-member dict in the same shape resolve_track() produces."""
    base = {
        "track_id": track_id,
        "has_attributes": has_attributes,
        "Gender": "Unknown", "Age": "Unknown", "Body size": "Unknown",
        "Viewpoint": "Unknown", "Upper-body clothing": "Unknown",
        "Lower-body clothing": "Unknown", "Footwear": "Unknown",
    }
    base.update(attrs)
    return base


def make_group(group_id, members, child_present=False):
    n_with = sum(1 for m in members if m["has_attributes"])
    return {
        "group_id": group_id,
        "size": len(members),
        "member_track_ids": [m["track_id"] for m in members],
        "members_with_attributes": n_with,
        "coverage": round(n_with / len(members), 2) if members else 0,
        "child_present": child_present,
        "members": members,
    }


def make_fake_track(track_id, has_attributes, attr_dict=None):
    """A minimal tracks.json-style record, needed by strategy_weighted_cascade
    (it re-derives confidence values from the raw track, not the resolved member)."""
    return {
        "track_id": track_id,
        "attributes": attr_dict if has_attributes else {},
    }


if __name__ == "__main__":
    with open(os.path.join(RAW_DIR, "group_profiles.json"), encoding="utf-8") as f:
        real_groups = json.load(f)
    with open(os.path.join(RAW_DIR, "tracks.json"), encoding="utf-8") as f:
        real_tracks = json.load(f)
    real_tracks_by_id = {t["track_id"]: t for t in real_tracks}

    track_clusters = pd.read_csv(os.path.join(GENERATED_DIR, "track_clusters.csv"))
    track_cluster_map = dict(zip(track_clusters["track_id"], track_clusters["cluster"]))
    cluster_to_category = build_cluster_to_category_map(
        os.path.join(GENERATED_DIR, "cluster_centroids.csv")
    )
    centroids_df = pd.read_csv(os.path.join(GENERATED_DIR, "cluster_centroids.csv"))

    # --- Real case: partial coverage (Group 55, coverage 0.33) ---
    g55 = next((g for g in real_groups if g["group_id"] == 55), None)
    if g55:
        result = strategy_weighted_cascade(g55, real_tracks_by_id)
        check(
            "Real Group 55 (1 of 3 members has data) does NOT fall back to Neutral",
            result != FALLBACK,
            f"got '{result}' using only the member(s) with data, as required",
        )

    # --- Real case: total-missing SOLO groups (the only real all-missing case
    #     that exists in this dataset -- multi-member all-missing doesn't occur) ---
    solo_missing = [g for g in real_groups if g["size"] == 1 and g["coverage"] == 0]
    if solo_missing:
        sample = solo_missing[:20]
        all_fallback = all(
            strategy_weighted_cascade(g, real_tracks_by_id) == FALLBACK for g in sample
        )
        check(
            f"Real solo no-data groups ({len(solo_missing)} total in dataset, "
            f"checked {len(sample)}) all resolve to Neutral-Fallback",
            all_fallback,
        )

    # --- Constructed: Group of 4, exactly 2 missing -- the scenario from the
    #     original spec (use A+B, ignore C+D, still produce a real decision) ---
    person_a = make_member(9001, True, Age="Adult", **{"Upper-body clothing": "Suit"})
    person_b = make_member(9002, True, Age="Adult", **{"Upper-body clothing": "Suit"})
    person_c = make_member(9003, False)
    person_d = make_member(9004, False)
    group_4_partial = make_group(9000, [person_a, person_b, person_c, person_d])

    fake_tracks = {
        9001: make_fake_track(9001, True, {
            "Adult": {"value": True, "confidence": 0.9},
            "Suit": {"value": True, "confidence": 0.9},
        }),
        9002: make_fake_track(9002, True, {
            "Adult": {"value": True, "confidence": 0.9},
            "Suit": {"value": True, "confidence": 0.9},
        }),
        9003: make_fake_track(9003, False),
        9004: make_fake_track(9004, False),
    }

    result = strategy_weighted_cascade(group_4_partial, fake_tracks)
    check(
        "Constructed group of 4 (A,B have data; C,D missing) uses A+B, "
        "does NOT fall back to Neutral",
        result != FALLBACK,
        f"got '{result}' -- both present members wear Suit, so Formal-Adult is expected",
    )
    check(
        "  ...and specifically resolves to Formal-Adult "
        "(the category both present members' data supports)",
        result == "Formal-Adult",
        f"got '{result}'",
    )

    # --- Constructed: same group, but flip so ALL 4 are missing ---
    person_a2 = make_member(9001, False)
    person_b2 = make_member(9002, False)
    person_c2 = make_member(9003, False)
    person_d2 = make_member(9004, False)
    group_4_all_missing = make_group(9010, [person_a2, person_b2, person_c2, person_d2])
    fake_tracks_all_missing = {
        9001: make_fake_track(9001, False),
        9002: make_fake_track(9002, False),
        9003: make_fake_track(9003, False),
        9004: make_fake_track(9004, False),
    }
    result = strategy_weighted_cascade(group_4_all_missing, fake_tracks_all_missing)
    check(
        "Constructed group of 4 (ALL members missing) falls back to Neutral-Fallback",
        result == FALLBACK,
        f"got '{result}'",
    )

    # --- Same two constructed cases, run through majority_class and kmodes too,
    #     so the behavior is confirmed consistent across all three strategies ---
    for strat_name, strat_fn, extra_args in [
        ("majority_class", strategy_majority_class, ()),
        ("kmodes", strategy_kmodes, (track_cluster_map, cluster_to_category, centroids_df)),
    ]:
        r_partial = strat_fn(group_4_partial, *extra_args)
        r_all_missing = strat_fn(group_4_all_missing, *extra_args)
        check(
            f"{strat_name}: partial-missing group does not force Neutral-Fallback",
            r_partial != FALLBACK,
            f"got '{r_partial}'",
        )
        check(
            f"{strat_name}: all-missing group DOES force Neutral-Fallback",
            r_all_missing == FALLBACK,
            f"got '{r_all_missing}'",
        )

    # --- Constructed: child present but missing attributes themselves --
    #     safety override must still fire on child_present flag alone,
    #     independent of whether the child's own attributes came through ---
    child_missing_attrs = make_member(9005, False)
    other_present = make_member(9006, True, **{"Lower-body clothing": "Shorts"})
    group_child_flagged = make_group(
        9020, [child_missing_attrs, other_present], child_present=True
    )
    fake_tracks_child = {
        9005: make_fake_track(9005, False),
        9006: make_fake_track(9006, True, {"Shorts": {"value": True, "confidence": 0.9}}),
    }
    result = strategy_weighted_cascade(group_child_flagged, fake_tracks_child)
    check(
        "Constructed group with child_present=True overrides to Family/Child-safe "
        "even when the child's own attribute data is missing",
        result == "Family/Child-safe",
        f"got '{result}'",
    )

    print(f"Missing-data test suite: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    if FAIL_COUNT > 0:
        sys.exit(1)