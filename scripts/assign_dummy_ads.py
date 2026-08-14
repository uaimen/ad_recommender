import json
import pandas as pd
import random
import os

random.seed(42)  # reproducible ad rotation

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
GENERATED_DIR = os.path.join(BASE_DIR, "data", "generated")

ADS_LIBRARY_PATH = os.path.join(RAW_DIR, "ads_library.json")
AD_SELECTION_CSV_PATH = os.path.join(GENERATED_DIR, "ad_selection_comparison.csv")
OUTPUT_CSV_PATH = os.path.join(GENERATED_DIR, "final_ad_assignments.csv")


def load_ads_library(path=ADS_LIBRARY_PATH):
    with open(path) as f:
        return json.load(f)


def assign_ad(category: str, ads_library: dict) -> dict:
    """Pick one ad from the category's pool. In production this would rotate
    or use frequency-capping; here we just pick deterministically at random
    (seeded) since we only have dummy creative."""
    pool = ads_library.get(category, ads_library["Neutral-Fallback"])
    return random.choice(pool)


if __name__ == "__main__":
    os.makedirs(GENERATED_DIR, exist_ok=True)
    ads_library = load_ads_library()

    df = pd.read_csv(AD_SELECTION_CSV_PATH)

    assignments = []
    for _, row in df.iterrows():
        category = row["weighted_cascade"]  # the safety-validated strategy
        ad = assign_ad(category, ads_library)
        assignments.append({
            "group_id": row["group_id"],
            "size": row["size"],
            "child_present": row["child_present"],
            "coverage": row["coverage"],
            "category": category,
            "ad_id": ad["ad_id"],
            "ad_title": ad["title"],
            "ad_tagline": ad["tagline"],
        })

    result_df = pd.DataFrame(assignments)
    result_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"Assigned ads to {len(result_df)} groups")
    print("Saved final_ad_assignments.csv")