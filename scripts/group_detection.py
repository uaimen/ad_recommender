"""
Group Detection module

Input:  frames_bbox_only.jsonl  (per-frame track_id + bbox_xyxy, from Intern 2's pipeline)
Output: stable group assignments per track, the input the ad-selection logic needs.

Method:
  1. Per-frame DBSCAN on depth-normalized bbox centers (eps scaled by bbox height,
     so "near" means the same thing regardless of distance from camera).
  2. Build a pairwise co-occurrence score across all frames: for every pair of tracks
     that appear together in at least one frame, what fraction of their shared frames
     were they DBSCAN-clustered together?
  3. Threshold that co-occurrence score, then take connected components on the
     resulting graph for stable groups. 
"""
import json
import os
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import networkx as nx


def bbox_center_and_height(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2, (y2 - y1)


def cluster_frame(frame, eps_ratio=0.9, min_samples=1):
    persons = frame["persons"]
    if not persons:
        return {}
    data = [bbox_center_and_height(p["bbox_xyxy"]) for p in persons]
    cx = np.array([d[0] for d in data])
    cy = np.array([d[1] for d in data])
    h = np.array([d[2] for d in data])
    scaled = np.stack([cx / h, cy / h], axis=1)
    db = DBSCAN(eps=eps_ratio, min_samples=min_samples).fit(scaled)
    return {p["track_id"]: int(lbl) for p, lbl in zip(persons, db.labels_)}


def build_cooccurrence(frames, eps_ratio=0.9):
    
    pair_counts = defaultdict(lambda: [0, 0])  # (same_cluster_count, total_together_count)

    for frame in frames:
        labels = cluster_frame(frame, eps_ratio=eps_ratio)
        ids = list(labels.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = sorted([ids[i], ids[j]])
                key = (a, b)
                pair_counts[key][1] += 1  # seen together
                if labels[ids[i]] == labels[ids[j]]:
                    pair_counts[key][0] += 1  # same cluster this frame

    return pair_counts


def stable_groups(pair_counts, min_shared_frames=15, co_occurrence_threshold=0.6):
    g = nx.Graph()
    for (a, b), (same, total) in pair_counts.items():
        g.add_node(a)
        g.add_node(b)
        if total >= min_shared_frames and (same / total) >= co_occurrence_threshold:
            g.add_edge(a, b, weight=same / total)

    components = list(nx.connected_components(g))
    track_to_group = {}
    for gid, comp in enumerate(components):
        for t in comp:
            track_to_group[t] = gid
    return track_to_group, components


def calibrate_eps_ratio(frames, sweep=None):
  
    if sweep is None:
        sweep = [round(x, 2) for x in np.arange(0.3, 1.6, 0.1)]

    busiest = max(frames, key=lambda f: len(f["persons"]))
    print(f"Calibrating on busiest frame (frame_id={busiest.get('frame_id')}, "
          f"{len(busiest['persons'])} people)")

    n_groups = []
    for r in sweep:
        labels = cluster_frame(busiest, eps_ratio=r)
        n_groups.append(len(set(labels.values())) if labels else 0)
        print(f"  eps_ratio={r}: {n_groups[-1]} groups")

    # find the longest run of consecutive sweep values with identical group count
    best_start, best_len, cur_start, cur_len = 0, 1, 0, 1
    for i in range(1, len(n_groups)):
        if n_groups[i] == n_groups[i - 1]:
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_start, cur_len = i, 1
    plateau = sweep[best_start:best_start + best_len]
    suggestion = plateau[len(plateau) // 2]

    print(f"\nStable plateau detected: {plateau} (group count={n_groups[best_start]})")
    print(f"Suggested eps_ratio: {suggestion}")
    print("This is a SUGGESTION only -- manually inspect the sweep above (same way")
    print("0.9 was chosen for the current video) before updating the hardcoded value.")
    return suggestion, plateau


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

if __name__ == "__main__":
    FALLBACK_FPS = 29.97  
    MIN_SHARED_SECONDS = 0.5  

    metadata_path = os.path.join(RAW_DIR, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as f:
            _meta = json.load(f)
        fps = _meta.get("source_fps", FALLBACK_FPS)
    else:
        fps = FALLBACK_FPS

    min_shared_frames = round(MIN_SHARED_SECONDS * fps)

    with open(os.path.join(RAW_DIR, "frames_bbox_only.jsonl"), encoding="utf-8") as f:
        frames = [json.loads(line) for line in f]

    pair_counts = build_cooccurrence(frames, eps_ratio=0.9)

    track_to_group, components = stable_groups(
        pair_counts, min_shared_frames=min_shared_frames, co_occurrence_threshold=0.6
    )

    multi_member = sum(1 for c in components if len(c) > 1)

    os.makedirs(RAW_DIR, exist_ok=True)
    with open(os.path.join(RAW_DIR, "stable_groups.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "track_to_group": track_to_group,
                "groups": [sorted(c) for c in components],
            },
            f,
            indent=2,
        )
    print(f"Detected {len(components)} groups from {len(frames)} frames "
          f"({multi_member} multi-member)")
    print("Saved stable_groups.json")