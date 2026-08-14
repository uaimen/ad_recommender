# Retrain Policy — k-modes Persona Model

Two different cadences are involved in this system, on purpose kept
independent (see `scripts/pipeline_runner.py`):

| Layer | What re-runs | Cadence | Basis |
|---|---|---|---|
| Detection + scoring | `group_detection` → `group_profiles` → `ad_selection` scoring | every 15s, over a rolling 30s window | derived from real group-duration data (median 24.6s span) — see `pipeline_runner.py` docstring |
| Displayed ad | which category's ad is actually shown | debounced to a 0.5s minimum hold | `ad_switch_cadence.py`, benchmarked against published attention-span research |
| **Persona model** (`cluster_audience.py` k-modes centroids) | **retrain** (refit `KModes`, regenerate `track_clusters.csv` + `cluster_centroids.csv`) | **not per-window** — separate, slower schedule (this document) | population-level model, needs enough accumulated tracks to be statistically meaningful |

This document only covers the third row.

## Why the persona model can't share the 30s/15s cadence

`strategy_kmodes` in `ad_selection.py` already supports scoring *new* tracks
against an *existing*, unchanged model via `nearest_centroid()` (Hamming
distance to the current centroids). That's what lets `pipeline_runner.py`
run every 15s without refitting anything. Refitting `KModes` itself on a
15–30s window's worth of tracks (single digits to low tens of people) would
produce centroids that are noise-dominated, not real personas — the model
needs a large accumulated sample to be worth retraining at all.

## Current situation: we only have one video

Everything in `data/raw/` is one ~5-minute clip. There isn't yet a real
multi-session dataset to empirically derive an ideal retrain interval the
way `pipeline_runner.py`'s window size was derived from real group-duration
data. So the policy below is a **starting point to operate under and revisit
once real multi-session volume exists** — not a number pulled from this
project's data, and that distinction should be stated plainly whenever this
policy is cited.

## Recommended policy (trigger-based, not calendar-only)

Retrain when **any** of these fires, whichever comes first:

1. **Calendar backstop** — at least once every 7 days, so the model never
   silently goes indefinitely stale even if the other triggers don't fire.
2. **Volume trigger** — accumulated new tracks (since the last retrain)
   exceed 20% of the training-set size used for the current centroids.
   Prevents retraining on a trickle of new data that can't move the
   centroids meaningfully.
3. **Drift signal from `pipeline_runner.py`'s own stability check** —
   if the window-to-window category flip rate (currently 5.8% on the one
   available clip) rises materially above its established baseline, or the
   `Neutral-Fallback` / unmatched-cluster rate climbs, that's a sign the
   population the model was trained on no longer matches what's showing up
   live (new season, new venue, new camera angle, etc.) — retrain
   immediately rather than waiting for the calendar backstop.

## How to retrain (mechanically)

```bash
python3 scripts/cluster_audience.py   # refits KModes, overwrites
                                       # track_clusters.csv + cluster_centroids.csv
```

No other script needs to change: `ad_selection.py` and `pipeline_runner.py`
both read `cluster_centroids.csv` / `track_clusters.csv` from disk at
runtime, so a retrain takes effect the next time either script runs, with
no code changes and no restart logic beyond "the CSVs on disk changed."

## What to revisit once more data exists

Once footage from multiple sessions/days is available, replace the 7-day/
20% numbers above with an empirically-derived interval the same way
`pipeline_runner.py`'s 30s/15s window was derived — e.g. by tracking how
many days of accumulation it actually takes before centroids meaningfully
shift, rather than guessing a week.
