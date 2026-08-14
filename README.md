# Ad Recommender Pipeline

Turns tracked people in a video feed into audience-aware ad category assignments. Given per-frame bounding boxes and per-track attribute predictions from an upstream detection/tracking/attribute-recognition system, this pipeline groups people who are actually together, profiles each group, scores it against an ad-category taxonomy, and picks a category-appropriate ad — with a hard safety override whenever a child is present.

Built and validated against one ~5-minute test clip (`people_in_park.mp4`, 323 tracks, 286 detected groups). See `docs/PROGRESS_RUNBOOK.md` for the full write-up of what was tried, what worked, and what didn't.

## How it works

1. Group detection: per-frame DBSCAN on camera-normalized positions, then a co-occurrence threshold across frames to turn "stood near each other once" into "is actually a group."
2. Persona clustering: each track's 57 raw attribute flags get resolved into a group-level categorical profile (age, clothing, posture, etc.), then k-modes clusters tracks into personas.
3. Group profiles: group membership + resolved attributes + a `child_present` flag, computed independently of any scoring so the safety check can't be voted away.
4. Ad selection: three strategies are implemented and compared; the deployed one is a safety-cascade (child check first, no exceptions) followed by confidence-weighted rule matching against `data/raw/taxonomy.json`.
5. Ad assignment: maps the winning category to a specific ad from `data/raw/ads_library.json`.
6. Update frequency: two separate cadence questions, answered from real data rather than assumed: how often the *displayed* ad should switch (`ad_switch_cadence.py`), and how often detection/scoring should re-run on a live feed (`pipeline_runner.py`, rolling 30s window / 15s step).

## Repo layout

```
scripts/
  group_detection.py      per-frame DBSCAN + co-occurrence -> stable_groups.json
  cluster_audience.py     attribute resolution + k-modes persona clustering
  group_profiles.py       joins groups with resolved attributes -> group_profiles.json
  ad_selection.py         the three scoring strategies, run and compared side by side
  assign_dummy_ads.py     category -> concrete ad from ads_library.json
  ad_switch_cadence.py    how often the on-screen ad should actually change
  pipeline_runner.py      rolling-window re-scoring for a live/streaming deployment
  missing_data_tests.py   explicit tests for partial/total missing attribute data
  evaluation.py           strategy agreement, safety-case pass rate, test suite summary
  run_pipeline.py         orchestrates all of the above, skips stages that are up to date

data/
  raw/          inputs from the upstream tracking pipeline, plus taxonomy/ads config
  generated/    everything the scripts above produce (CSV/JSON, gitignored-sized)

docs/
  PROGRESS_RUNBOOK.md     stage-by-stage findings and the reasoning behind each decision
  RETRAIN_POLICY.md       when to refit the k-modes persona model
```


```bash
pip install numpy pandas scikit-learn networkx kmodes
```

## Running it

Drop the upstream outputs into `data/raw/`: `frames_bbox_only.jsonl`, `tracks.json`, `metadata.json`, `taxonomy.json`, `ads_library.json`. Then run everything through the orchestrator rather than calling scripts individually; it resolves dependency order and skips stages whose outputs are already newer than their inputs:

```bash
python scripts/run_pipeline.py            # run everything that's stale
python scripts/run_pipeline.py --force    # rerun every stage regardless
python scripts/run_pipeline.py --only ad_selection   # this stage + its dependencies
python scripts/run_pipeline.py --list     # print the stage DAG, don't run anything
```

Each script also runs standalone (`python scripts/group_detection.py`, etc.) if you only need one stage's output while iterating.

## Ad selection strategies

`ad_selection.py` implements and benchmarks three approaches against the same group data:


| Majority-class: Each member votes independently, no weighting, no safety check | Baseline only — fails the child-safety test outright, not deployable 

| Weighted cascade: Child check first as a hard override, then confidence-weighted rule matching ---deployed.

| K-modes persona: Reuses the persona clusters from `cluster_audience.py`, maps cluster centroid to nearest category | Matches the cascade on the safety test; kept for comparison 

A category has to beat the runner-up by at least 15% of its own score or the group falls back to `Neutral-Fallback` instead of guessing on a near-tie. Groups with no attribute data at all get the same fallback; groups with partial data use whichever members actually have it.

## Known limitations

- `taxonomy.json`'s rule weights are hand-set, not fit against any ground truth.
- Group-detection thresholds (DBSCAN eps, co-occurrence cutoff) are calibrated to this one video and need re-checking on new footage.
- `ads_library.json` is placeholder creative; assignment is a seeded random pick, not real rotation or frequency capping.
- The k-modes retrain cadence in `docs/RETRAIN_POLICY.md` is a starting policy — there's only one session's worth of data to derive it from so far.
