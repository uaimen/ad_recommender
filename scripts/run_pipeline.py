"""
Single orchestrating entrypoint for the full ad-recommender pipeline.

Runs all stages in dependency order
STALENESS: a stage is SKIPPED if all of its declared outputs already exist
and are newer than all of its declared inputs 

"""

import argparse
import os
import subprocess
import sys
import time

# Without this, this script's own print()s can get buffered and flushed
# AFTER a subprocess stage's output (which writes to the same inherited

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPTS_DIR)
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
GENERATED_DIR = os.path.join(BASE_DIR, "data", "generated")


def raw(*p):
    return os.path.join(RAW_DIR, *p)


def gen(*p):
    return os.path.join(GENERATED_DIR, *p)


def script(name):
    return os.path.join(SCRIPTS_DIR, name)


STAGES = [
    {
        "name": "group_detection",
        "script": script("group_detection.py"),
        "inputs": [raw("frames_bbox_only.jsonl")],
   
        "optional_inputs": [raw("metadata.json")],
        "outputs": [raw("stable_groups.json")],
        "depends_on": [],
    },
    {
        "name": "cluster_audience",
        "script": script("cluster_audience.py"),
        "inputs": [raw("tracks.json"), raw("taxonomy.json")],
        "outputs": [gen("track_clusters.csv"), gen("cluster_centroids.csv")],
        "depends_on": [],
    },
    {
        "name": "group_profiles",
        "script": script("group_profiles.py"),
        "inputs": [raw("tracks.json"), raw("stable_groups.json")],
        "outputs": [raw("group_profiles.json")],
        "depends_on": ["group_detection"],
    },
    {
        "name": "ad_selection",
        "script": script("ad_selection.py"),
        "inputs": [raw("group_profiles.json"), raw("tracks.json"), raw("taxonomy.json"),
                   gen("track_clusters.csv"), gen("cluster_centroids.csv")],
        "outputs": [gen("ad_selection_comparison.csv")],
        "depends_on": ["group_profiles", "cluster_audience"],
    },
    {
        "name": "pipeline_runner",
        "script": script("pipeline_runner.py"),
        "inputs": [raw("frames_bbox_only.jsonl"), raw("tracks.json"),
                   gen("track_clusters.csv"), gen("cluster_centroids.csv")],
       
        "optional_inputs": [raw("metadata.json"), gen("track_attribute_history.json")],
        "outputs": [gen("windowed_pipeline_results.csv")],
   
        "depends_on": ["cluster_audience"],
    },
    {
        "name": "ad_switch_cadence",
        "script": script("ad_switch_cadence.py"),
        "inputs": [raw("frames_bbox_only.jsonl"),
                   gen("track_clusters.csv"), gen("cluster_centroids.csv")],
        "optional_inputs": [raw("metadata.json")],
        "outputs": [gen("ad_switch_cadence_results.csv")],
 
        "depends_on": ["cluster_audience"],
    },
    {
        "name": "missing_data_tests",
        "script": script("missing_data_tests.py"),
        "inputs": [raw("group_profiles.json"), raw("tracks.json"),
                   gen("track_clusters.csv"), gen("cluster_centroids.csv")],
        "outputs": [],  
        "depends_on": ["group_profiles", "cluster_audience"],
    },
    {
        "name": "evaluation",
        "script": script("evaluation.py"),
        "inputs": [raw("group_profiles.json"), raw("tracks.json"),
                   gen("track_clusters.csv"), gen("cluster_centroids.csv")],
        "outputs": [gen("evaluation_summary.csv")],
        "depends_on": ["group_profiles", "cluster_audience"],
    },
    {
        "name": "assign_dummy_ads",
        "script": script("assign_dummy_ads.py"),
        "inputs": [gen("ad_selection_comparison.csv"), raw("ads_library.json")],
        "outputs": [gen("final_ad_assignments.csv")],
        "depends_on": ["ad_selection"],
    },
]

STAGE_BY_NAME = {s["name"]: s for s in STAGES}


def all_inputs(stage):
    """Required + optional inputs combined -- for staleness purposes only,
    both count equally: if EITHER changes, the stage's output may no longer
    reflect reality. The distinction only matters for run_stage()'s missing-
    input handling (optional_inputs never block a run; inputs do)."""
    return stage["inputs"] + stage.get("optional_inputs", [])


def is_stale(stage):
    """True if this stage needs to (re)run: any declared output missing, or
    the newest present input (required OR optional) is younger than the
    oldest output. A stage with no declared outputs (a test suite) is always
    considered stale -- there's nothing on disk to compare against, and
    re-running a test suite every time is the correct default anyway."""
    outputs = stage["outputs"]
    if not outputs:
        return True
    if not all(os.path.exists(o) for o in outputs):
        return True
    oldest_output = min(os.path.getmtime(o) for o in outputs)
    present_inputs = [i for i in all_inputs(stage) if os.path.exists(i)]
    if not present_inputs:
        return False
    newest_input = max(os.path.getmtime(i) for i in present_inputs)
    return newest_input > oldest_output


STAGE_NAME_WIDTH = max(len(s["name"]) for s in STAGES)


def log(name, message):
    print(f"  {name:<{STAGE_NAME_WIDTH}}  {message}")


def run_stage(stage, force):
    name = stage["name"]
    missing_inputs = [i for i in stage["inputs"] if not os.path.exists(i)]

    if missing_inputs and stage.get("optional"):
        missing_names = ", ".join(os.path.basename(m) for m in missing_inputs)
        log(name, f"skipped, optional input not found ({missing_names})")
        return "skipped-optional"

    if missing_inputs:
        log(name, f"error, required input missing: {', '.join(missing_inputs)}")
        return "error"

    if not force and not is_stale(stage):
        log(name, "skipped, already up to date")
        return "skipped-fresh"

    log(name, "running")
    t0 = time.time()
    result = subprocess.run([sys.executable, stage["script"]])
    dt = time.time() - t0

    if result.returncode != 0:
        log(name, f"failed, exit code {result.returncode} ({dt:.1f}s)")
        return "failed"
    log(name, f"done ({dt:.1f}s)")
    return "ran"


def resolve_stage_set(only):
    """--only STAGE runs that stage plus every stage it (transitively)
    depends on, in DAG order -- not the whole pipeline."""
    if only is None:
        return list(STAGES)
    if only not in STAGE_BY_NAME:
        valid = ", ".join(STAGE_BY_NAME)
        raise SystemExit(f"Unknown stage '{only}'. Valid stages: {valid}")
    needed, seen = [], set()

    def visit(name):
        if name in seen:
            return
        seen.add(name)
        for dep in STAGE_BY_NAME[name]["depends_on"]:
            visit(dep)
        needed.append(STAGE_BY_NAME[name])

    visit(only)
    return needed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                         help="Rerun every stage regardless of staleness.")
    parser.add_argument("--only", metavar="STAGE",
                         help="Run only this stage plus its stale dependencies.")
    parser.add_argument("--list", action="store_true",
                         help="Print the stage DAG and exit without running anything.")
    args = parser.parse_args()

    if args.list:
        print("Pipeline stages, in run order:\n")
        for s in STAGES:
            deps = f"  (depends on {', '.join(s['depends_on'])})" if s["depends_on"] else ""
            tag = "  (optional)" if s.get("optional") else ""
            print(f"  {s['name']}{tag}{deps}")
            if s.get("optional_inputs"):
                names = ", ".join(os.path.basename(i) for i in s["optional_inputs"])
                print(f"      uses if present, never required: {names}")
        raise SystemExit(0)

    plan = resolve_stage_set(args.only)
    print(f"Running {len(plan)} stage(s): {', '.join(s['name'] for s in plan)}")
    print("Stages already up to date are skipped; use --force to rerun everything.\n")

    statuses = {}
    failed_upstream = set()
    for stage in plan:
        blocked_by = [d for d in stage["depends_on"]
                      if d in STAGE_BY_NAME and statuses.get(d) in ("failed", "error")]
        if blocked_by or stage["name"] in failed_upstream:
            reason = ", ".join(blocked_by) if blocked_by else "an earlier failure"
            log(stage["name"], f"skipped, dependency failed ({reason})")
            statuses[stage["name"]] = "skipped-blocked"
            failed_upstream.add(stage["name"])
            continue
        status = run_stage(stage, force=args.force)
        statuses[stage["name"]] = status
        if status in ("failed", "error"):
            failed_upstream.add(stage["name"])

    status_labels = {
        "ran": "completed",
        "skipped-fresh": "skipped, up to date",
        "skipped-optional": "skipped, optional input missing",
        "skipped-blocked": "skipped, dependency failed",
        "failed": "failed",
        "error": "error, required input missing",
    }

    print("\nSummary")
    for name, status in statuses.items():
        print(f"  {name:<{STAGE_NAME_WIDTH}}  {status_labels[status]}")

    if any(s in ("failed", "error") for s in statuses.values()):
        print("\nOne or more stages did not complete; see the log above for details.")
        sys.exit(1)
    print("\nAll stages complete.")
