"""
strategy comparison.

Implements and benchmarks THREE strategies against real group_profiles.json:
  1. Majority-class      — each member independently "votes" for their best-matching
                            category (unweighted rule match count); group takes the
                            most common vote. No confidence, no safety priority.
  2. Safety-cascade +
     weighted-matching    — child_present is checked FIRST as a hard override
                            (independent of any scoring). If clear, each category is
                            scored using confidence-weighted rule matches, summed
                            across group members, argmax wins.
  3. k-modes persona      — reuses the per-track cluster assignments already computed
                            in cluster_audience.py (track_clusters.csv), aggregates to
                            group level by majority cluster vote, then maps each
                            cluster's centroid to the nearest taxonomy category using
                            the same rule set as strategy 2.

Requires: group_profiles.json, tracks.json, track_clusters.csv, cluster_centroids.csv
"""

import json
import os
import pandas as pd
from collections import Counter
from cluster_audience import EXCLUSIVE_GROUPS, NON_EXCLUSIVE_GROUPS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
GENERATED_DIR = os.path.join(BASE_DIR, "data", "generated")
TAXONOMY_PATH = os.path.join(RAW_DIR, "taxonomy.json")


def load_taxonomy(path=TAXONOMY_PATH):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


CATEGORY_RULES = load_taxonomy()

# 1b. ATTRIBUTE_TO_GROUP: built directly from cluster_audience.py's own
#     taxonomy definitions (EXCLUSIVE_GROUPS + NON_EXCLUSIVE_GROUPS), so
#     there is exactly ONE place the attribute->group structure is defined,
#     not a second hand-maintained copy that could silently drift out of
#     sync with it. Used to correctly attach real model confidence to the
#     right "Group::Label" key for weighted scoring.

ATTRIBUTE_TO_GROUP = {}
for group, labels in EXCLUSIVE_GROUPS.items():
    for label in labels:
        ATTRIBUTE_TO_GROUP[label] = group
for group, labels in NON_EXCLUSIVE_GROUPS.items():
    for label in labels:
        ATTRIBUTE_TO_GROUP[label] = group
ATTRIBUTE_TO_GROUP["Female"] = "Gender"


def rule_score(member: dict, weighted: bool, confidences: dict = None) -> dict:
    """Score a single resolved member profile against every category.
    If weighted=True and confidences provided, multiply weight by that
    attribute's real model confidence. If weighted=False, this reduces to
    a plain unweighted rule-match count (used for majority-class)."""
    scores = {cat: 0.0 for cat in CATEGORY_RULES}
    for cat, ruleset in CATEGORY_RULES.items():
        for group, label_weights in ruleset["exclusive"].items():
            member_label = member.get(group)
            if member_label in label_weights:
                w = label_weights[member_label]
                if weighted and confidences:
                    conf = confidences.get(f"{group}::{member_label}", 1.0)
                    w *= conf
                elif not weighted:
                    w = 1.0 if w > 0 else 0.0  # unweighted: just "did it match"
                scores[cat] += w
        for col, w in ruleset["multihot"].items():
            # multihot columns are "True"/"False"/"Unknown" strings
            if member.get(col) == "True":
                if weighted and confidences:
                    conf = confidences.get(col, 1.0)
                    w *= conf
                elif not weighted:
                    w = 1.0 if w > 0 else 0.0
                scores[cat] += w
    return scores


CONFIDENCE_MARGIN_RATIO = 0.15  # see best_category() docstring for derivation


def best_category(scores: dict, margin_ratio: float = CONFIDENCE_MARGIN_RATIO) -> str:
    """Argmax over category scores, with a confidence-margin check.

    If nothing matched at all (all scores zero), behavior is UNCHANGED from
    before: default to "Casual-Adult" ("nothing matches strongly" -- an
    existing, separately-made, already-tested design choice this change
    does not touch).

    Otherwise, the top-scoring category must lead the runner-up by at least
    `margin_ratio` of its own score, or the result is judged too close to
    call confidently -- "Neutral-Fallback" is returned instead of guessing
    between two similarly-supported categories. E.g. with the default 0.15,
    winning 4.6-to-4.0 (13% margin) does NOT clear the bar and falls back;
    winning 4.8-to-4.0 (20% margin) does.

    This closes a real gap in ads_library.json's own documented
    Neutral-Fallback trigger -- "No attribute data available ... OR no rule
    matched confidently" -- the second half of which was never actually
    implemented before this change; only the zero-data path existed.

    CONFIDENCE_MARGIN_RATIO=0.15 is a principled but UNVALIDATED default --
    like taxonomy.json's own rule weights, it has no ground-truth backing.
    It was chosen by simulating margin_ratio in [0.10, 0.15, 0.20, 0.30, 0.50]
    against all 286 real groups: the flip rate to Neutral-Fallback rises
    gently (12/144 -> 21/144 matched groups across that whole range, no
    cliff), and zero real child-present groups are affected at any threshold
    tested (the safety override in strategy_weighted_cascade already
    short-circuits before scoring, so this margin check and child safety
    never interact) -- see docs/PROGRESS_RUNBOOK.md for the full simulation.
    Revisit this value once real ground-truth labels exist."""
    if all(v == 0 for v in scores.values()):
        return "Casual-Adult"  # default when nothing matches strongly -- unchanged

    ranked = sorted(scores.values(), reverse=True)
    top, runner_up = ranked[0], ranked[1]
    if top - runner_up < margin_ratio * top:
        return "Neutral-Fallback"  # matched, but too close to call confidently
    return max(scores, key=scores.get)

def member_confidences(track: dict) -> dict:
    """Build a lookup of real model confidence per attribute, correctly
    namespaced by the label's ACTUAL group (via ATTRIBUTE_TO_GROUP) --
    not blindly duplicated across unrelated groups. Includes both True and
    False observations (rule_score only consults this for labels it's
    actually checking, so having both present is harmless and correct)."""
    attrs = track.get("attributes") or {}
    conf = {}
    for label, d in attrs.items():
        group = ATTRIBUTE_TO_GROUP.get(label)
        if group is None:
            continue  
        conf[f"{group}::{label}"] = d["confidence"]
    return conf

# 3. Strategy 1: Majority-class
def strategy_majority_class(group: dict) -> str:
    votes = []
    for m in group["members"]:
        if not m["has_attributes"]:
            continue
        scores = rule_score(m, weighted=False)
        votes.append(best_category(scores))
    if not votes:
        return "Neutral-Fallback"  # every member missing data
    return Counter(votes).most_common(1)[0][0]

# 4. Strategy 2: Safety-cascade + weighted-matching

def strategy_weighted_cascade(group: dict, tracks_by_id: dict) -> str:
    if group["child_present"]:
        return "Family/Child-safe"  # hard override, no scoring involved

    total_scores = {cat: 0.0 for cat in CATEGORY_RULES}
    any_data = False
    for m in group["members"]:
        if not m["has_attributes"]:
            continue
        any_data = True
        track = tracks_by_id[m["track_id"]]
        conf = member_confidences(track)
        scores = rule_score(m, weighted=True, confidences=conf)
        for cat, s in scores.items():
            total_scores[cat] += s

    if not any_data:
        return "Neutral-Fallback"
    return best_category(total_scores)

# 5. Strategy 3: k-modes persona (reuse Stage 2 cluster assignments)

def build_cluster_to_category_map(centroids_csv: str) -> dict:
    """Map each k-modes cluster centroid to the nearest taxonomy category
    using the SAME unweighted rule set, so cluster labels become interpretable
    ad categories rather than opaque cluster IDs."""
    centroids = pd.read_csv(centroids_csv)
    mapping = {}
    for i, row in centroids.iterrows():
        pseudo_member = row.to_dict()
        scores = rule_score(pseudo_member, weighted=False)
        mapping[i] = best_category(scores)
    return mapping


def nearest_centroid(member: dict, centroids: pd.DataFrame) -> int:
    """Assign a track to its nearest k-modes centroid by Hamming distance
    (mismatch count across the shared categorical columns) -- the standard
    distance metric k-modes itself uses. This is the missing inference step:
    without it, a track that was never part of the original .fit() call has
    no way to be classified at all, even when it has real attribute data."""
    shared_cols = [c for c in centroids.columns if c in member]
    best_cluster, best_dist = None, None
    for i, row in centroids.iterrows():
        dist = sum(1 for c in shared_cols if str(row[c]) != str(member.get(c)))
        if best_dist is None or dist < best_dist:
            best_dist, best_cluster = dist, i
    return best_cluster


def strategy_kmodes(group: dict, track_cluster_map: dict, cluster_to_category: dict,
                     centroids: pd.DataFrame = None) -> str:
    if group["child_present"]:
        return "Family/Child-safe"  # safety override applies to every strategy equally

    member_clusters = []
    for m in group["members"]:
        cid = track_cluster_map.get(m["track_id"])
        if cid is None and m["has_attributes"] and centroids is not None:
            
            # fit (e.g. a new/incoming track) -- assign it to its nearest
            # centroid instead of silently treating it as missing data.
            cid = nearest_centroid(m, centroids)
        if cid is not None:
            member_clusters.append(cid)
    if not member_clusters:
        return "Neutral-Fallback"

    majority_cluster = Counter(member_clusters).most_common(1)[0][0]
    return cluster_to_category.get(majority_cluster, "Casual-Adult")


# 6. Main: run all three, compare

if __name__ == "__main__":
    os.makedirs(GENERATED_DIR, exist_ok=True)

    with open(os.path.join(RAW_DIR, "group_profiles.json"), encoding="utf-8") as f:
        groups = json.load(f)

    with open(os.path.join(RAW_DIR, "tracks.json"), encoding="utf-8") as f:
        tracks = json.load(f)
    tracks_by_id = {t["track_id"]: t for t in tracks}

    track_clusters = pd.read_csv(os.path.join(GENERATED_DIR, "track_clusters.csv"))
    track_cluster_map = dict(zip(track_clusters["track_id"], track_clusters["cluster"]))
    centroids_df = pd.read_csv(os.path.join(GENERATED_DIR, "cluster_centroids.csv"))
    cluster_to_category = build_cluster_to_category_map(
        os.path.join(GENERATED_DIR, "cluster_centroids.csv")
    )

    results = []
    for g in groups:
        r = {
            "group_id": g["group_id"],
            "size": g["size"],
            "child_present": g["child_present"],
            "coverage": g["coverage"],
            "majority_class": strategy_majority_class(g),
            "weighted_cascade": strategy_weighted_cascade(g, tracks_by_id),
            "kmodes": strategy_kmodes(g, track_cluster_map, cluster_to_category, centroids_df),
        }
        results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(GENERATED_DIR, "ad_selection_comparison.csv"), index=False)

    print(f"Scored {len(df)} groups across 3 strategies")
    print("Saved ad_selection_comparison.csv")