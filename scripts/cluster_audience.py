"""

Input:  tracks.json  
Output: a resolved, group-level categorical profile per track, clustered with k-modes
        into audience personas.

  - Resolved mutually-exclusive attribute groups (Age, Body size, Viewpoint,
    Upper-body clothing, Lower-body clothing, Footwear) to a single label
    using highest confidence among the True flags. If none are True, label
    the group "Unknown" rather than defaulting to False.
  - Non-exclusive groups (Head/accessories, Bags, Activity, Posture) are kept
    as multi-hot flags.
  - Gender is kept as-is (binary Female flag; False covers Male/Unknown per
    the taxonomy's own attribute list, which only names "Female").
  - Tracks with zero attribute observations (attribute_observations == 0) are
    EXCLUDED from the k-modes fit (they'd distort cluster centers with an
    all-"Unknown" row) but are kept in a separate list to be assigned to a
    fallback "Unknown/Neutral" persona at inference time.
  - The child-safety rule is NOT learned by clustering. It's applied as a
    hard override, independent of whatever cluster a track lands in.
"""

import json
import os
import pandas as pd
from kmodes.kmodes import KModes

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
GENERATED_DIR = os.path.join(BASE_DIR, "data", "generated")

# 1. Attribute taxonomy

EXCLUSIVE_GROUPS = {
    "Age": ["Child", "Adult", "Elderly"],
    "Body size": ["Fat", "Normal", "Thin"],
    "Viewpoint": ["Front", "Back", "Side"],
    "Upper-body clothing": [
        "Short sleeves", "Long sleeves", "Shirt", "Jacket", "Suit", "Vest",
        "Cotton-padded coat", "Coat", "Graduation gown", "Chef uniform",
    ],
    "Lower-body clothing": [
        "Trousers", "Shorts", "Jeans", "Long skirt", "Short skirt", "Dress",
    ],
    "Footwear": ["Leather shoes", "Casual shoes", "Boots", "Sandals", "Other shoes"],
}

NON_EXCLUSIVE_GROUPS = {
    "Head and accessories": [
        "Bald", "Long hair", "Black hair", "Hat", "Glasses", "Mask",
        "Helmet", "Scarf", "Gloves",
    ],
    "Bags and carried objects": [
        "Backpack", "Shoulder bag", "Handbag", "Plastic bag", "Paper bag",
        "Suitcase", "Others",
    ],
    "Activity": ["Making a phone call", "Smoking", "Hands behind back", "Arms crossed"],
    "Posture and Movement": [
        "Walking", "Running", "Standing", "Riding a bicycle",
        "Riding an scooter", "Riding a skateboard",
    ],
}

GENDER_ATTR = "Female"


# 2. Load + resolve a single track's attributes into a group-level profile

def resolve_track(track: dict) -> dict:
    """Turn one track's raw 57-attribute dict into a resolved, group-level
    categorical profile. Returns a flat dict suitable for a DataFrame row."""
    attrs = track.get("attributes") or {}
    row = {"track_id": track["track_id"], "has_attributes": bool(attrs)}

    # Gender: keep as-is
    if attrs:
        row["Gender"] = "Female" if attrs.get(GENDER_ATTR, {}).get("value") else "Male/Unknown"
    else:
        row["Gender"] = "Unknown"

    # Mutually-exclusive groups: pick highest-confidence True label, else "Unknown"
    for group, labels in EXCLUSIVE_GROUPS.items():
        if not attrs:
            row[group] = "Unknown"
            continue
        true_labels = [
            (l, attrs[l]["confidence"]) for l in labels
            if l in attrs and attrs[l]["value"] is True
        ]
        if true_labels:
            best_label = max(true_labels, key=lambda x: x[1])[0]
            row[group] = best_label
        else:
            row[group] = "Unknown"

    # Non-exclusive groups: keep as individual flags, but preserve the
    # distinction between "confidently False" and "no observation at all"
   
    for group, labels in NON_EXCLUSIVE_GROUPS.items():
        for l in labels:
            col = f"{group}::{l}"
            if attrs and l in attrs:
                row[col] = "True" if attrs[l]["value"] else "False"
            else:
                row[col] = "Unknown"

    return row


def build_dataframe(tracks: list) -> pd.DataFrame:
    rows = [resolve_track(t) for t in tracks]
    return pd.DataFrame(rows)

# 3. Cluster (k-modes) on tracks that actually have attribute data

def run_kmodes(df: pd.DataFrame, n_clusters: int, random_state: int = 42):
    fit_df = df[df["has_attributes"]].drop(columns=["track_id", "has_attributes"])
    categorical_cols = list(range(fit_df.shape[1]))  # all columns are categorical

    km = KModes(n_clusters=n_clusters, init="Huang", n_init=10, random_state=random_state, verbose=0)
    clusters = km.fit_predict(fit_df, categorical=categorical_cols)

    result = df[df["has_attributes"]].copy()
    result["cluster"] = clusters
    return km, result


def elbow_scan(df: pd.DataFrame, k_range=range(2, 11), random_state: int = 42):
    fit_df = df[df["has_attributes"]].drop(columns=["track_id", "has_attributes"])
    categorical_cols = list(range(fit_df.shape[1]))
    costs = []
    for k in k_range:
        km = KModes(n_clusters=k, init="Huang", n_init=5, random_state=random_state, verbose=0)
        km.fit(fit_df, categorical=categorical_cols)
        costs.append((k, km.cost_))
    return costs

# 4. Main

if __name__ == "__main__":
    os.makedirs(GENERATED_DIR, exist_ok=True)

    with open(os.path.join(RAW_DIR, "tracks.json"), encoding="utf-8") as f:
        tracks = json.load(f)

    df = build_dataframe(tracks)

    taxonomy_path = os.path.join(RAW_DIR, "taxonomy.json")
    with open(taxonomy_path, encoding="utf-8") as f:
        _taxonomy = json.load(f)
    _rule_scored_categories = [k for k in _taxonomy if not k.startswith("_")]
    N_PERSONAS = len(_rule_scored_categories) + 2
    km, result = run_kmodes(df, n_clusters=N_PERSONAS)

    centroid_cols = df.drop(columns=["track_id", "has_attributes"]).columns
    centroids = pd.DataFrame(km.cluster_centroids_, columns=centroid_cols)

    result.to_csv(os.path.join(GENERATED_DIR, "track_clusters.csv"), index=False)
    centroids.to_csv(os.path.join(GENERATED_DIR, "cluster_centroids.csv"), index=False)
    print(f"Clustered {df['has_attributes'].sum()} tracks into {N_PERSONAS} personas")
    print("Saved track_clusters.csv, cluster_centroids.csv")