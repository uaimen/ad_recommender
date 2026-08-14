# Ad Recommender Pipeline Runbook

This document tracks how the pipeline was built, what each script does, and what was found. Run `scripts/run_pipeline.py` to reproduce everything; it runs every stage below in order and skips any stage that is already up to date.

## Data received

Input from the tracking and attribute recognition stage: `tracks.json` (323 tracks, 181 with attribute data), `metadata.json`, and the full per-frame data, later reduced to a smaller bounding box only file for upload size.

## Persona clustering

Script: `cluster_audience.py`

Resolves each track's raw attributes into a group level profile and runs k-modes clustering into 6 personas. Mutually exclusive attribute groups such as age, body size, clothing, and footwear resolve to the highest confidence label. Non-exclusive groups such as bags, accessories, and posture stay multi-hot.

Finding: no natural cluster count in this dataset. One recurring pattern, male or unknown gender paired with a dress, looked like a misclassification from the attribute model rather than a real persona. This was later confirmed when the same pattern showed up across several independent tracks.

## Group detection

Script: `group_detection.py`

Detects real groups in two stages: DBSCAN per frame on camera perspective normalized positions, then a temporal check requiring a pair to be clustered together for a minimum share of the frames they appear in together. This avoids counting a single frame coincidence as a group.

Finding: 323 tracks resolve to 286 stable groups, 263 solo and 23 multi-member.

## Group profiles

Script: `group_profiles.py`

Joins detected groups with resolved attributes and computes two group level fields: whether a child is present, which is a hard safety flag independent of any scoring, and coverage, the share of members with real attribute data.

Finding: 4 of 286 groups contain a child. No multi-member group has zero attribute coverage, so the rule of using whichever members have data instead of dropping the group is a real requirement, not a theoretical one.

## Ad selection strategies

Script: `ad_selection.py`

Three strategies were built and compared.

Majority class: each member votes on a category with no weighting and no safety override. Kept only as a baseline.

Weighted cascade: checks the child present flag first as a hard override, then scores categories using confidence weighted rule matching. This is the deployed strategy.

K-modes persona: maps each group's dominant persona cluster to a category using the same rules as weighted cascade.

Finding: on the 4 real child present groups, weighted cascade and k-modes both route correctly every time. Majority class has no safety override and fails this test completely, so it is disqualified from deployment regardless of its general agreement rate with the other two.

A later fix added a confidence margin check: a category must beat the next best option by at least 15 percent, or the group falls back to a neutral category instead of guessing on a near tie. This also exposed a silent bug where two categories sharing the same clothing rule were always resolved the same arbitrary way rather than by any real signal.

## Update frequency

Two separate questions were answered here.

How often should the displayed ad change: `ad_switch_cadence.py` simulates how often the dominant group in view actually changes and compares that against published attention span research. A half second debounce gives a real average hold time of about 14 seconds, well above the reference attention span, so ads are not switched faster than a viewer could register them.

How often should detection and scoring re-run: `pipeline_runner.py` re-runs detection and scoring on a rolling 30 second window, refreshed every 15 seconds, instead of requiring a full video first. The window size was derived from real group duration data in this dataset. The persona clustering model itself is not retrained on this schedule. Each window only scores against the existing cluster centroids, and retraining follows a separate, slower policy documented in `RETRAIN_POLICY.md`.

## Attribute timing fix

The windowed pipeline was originally scoring every window using each track's final, whole video attribute reading, which leaks future information into early windows. A one time extraction pulled the full per observation attribute history from the complete frame data, and `pipeline_runner.py` was updated to score each window using only what was known by that window's end time. The extracted history stays in the generated data even though the extraction script itself was removed once its job was done.

This also surfaced a safety related bug. Making the child flag sticky by patching the attribute value alone was not enough, since a later high confidence adult reading could still outrank an earlier low confidence child reading. This was fixed by applying the sticky override directly to the child present flag instead of through the normal scoring path.

`ad_switch_cadence.py` had the same class of problem, since it used a single whole video group assignment instead of the same rolling window approach. It was updated to share the same window based group detection as `pipeline_runner.py`, and the older whole video method was removed once the fix was confirmed not to change the recommendation.

## Pipeline orchestrator

Script: `run_pipeline.py`

Runs every stage in the correct order and skips any stage whose output is already newer than its inputs, so a full run only re-executes what changed. Supports forcing a full rerun, running a single stage plus its dependencies, and listing the stage order without running anything.

## Tier-1 attribute check

The original attribute recognition specification calls for gender, age group, bag type, and child presence as the inputs to ad selection. The current rules mostly use clothing, posture, and age instead. Checking this against the real data showed that adding gender and bag type would not change any group's current category outcome on this video, since bag type is almost never observed and gender does not map onto any of the existing categories. This is a dataset specific finding, not a permanent conclusion, and should be re-checked on new footage.

## Known limitations

- `taxonomy.json`'s rule weights have not been validated against any ground truth labels, only cross strategy agreement.
- The k-modes retrain interval is a starting policy, not derived from real multi-session data yet.
- The DBSCAN distance threshold and co-occurrence threshold in `group_detection.py` are calibrated to this one video and need manual recalibration for a new one.
- Ad creative in `ads_library.json` is placeholder content, and ad assignment is a seeded random pick rather than real rotation or frequency capping.
