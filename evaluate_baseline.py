from pathlib import Path
import pandas as pd
import numpy as np
import re


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\ashig\OneDrive\Documents\vs code\Hackathon\amazon-business-entity-resolution"
)

TRAIN_DIR = (
    BASE_DIR
    / "dataset"
    / "extracted"
    / "student_resource"
    / "dataset"
    / "train"
)


# ============================================================
# NORMALIZATION
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()
    value = re.sub(r"\s+", " ", value)

    return value


def preprocess(df):
    df = df.copy()

    df["name_clean"] = df["business_name"].map(clean_text)
    df["address_clean"] = df["business_address"].map(clean_text)
    df["country_clean"] = df["country"].map(clean_text)

    return df


# ============================================================
# LOOKUP
# ============================================================

def build_lookup(df, column):
    lookup = {}

    for entity_id, country, value in df[
        ["entity_id", "country_clean", column]
    ].itertuples(index=False, name=None):

        if not value:
            continue

        key = (country, value)

        if key not in lookup:
            lookup[key] = []

        lookup[key].append(entity_id)

    return lookup


# ============================================================
# GROUND TRUTH
# ============================================================

def build_ground_truth(gt):
    """
    Convert:

        source1_entity_id | matched_entity_ids

    into:

        {
            S1-id: {
                "S2": {ids...},
                "S3": {ids...}
            }
        }
    """

    truth = {}

    for source1_id, matched_ids in gt[
        ["source1_entity_id", "matched_entity_ids"]
    ].itertuples(index=False, name=None):

        truth[source1_id] = {
            "S2": set(),
            "S3": set(),
        }

        if pd.isna(matched_ids):
            continue

        for entity_id in str(matched_ids).split(","):

            entity_id = entity_id.strip()

            if not entity_id:
                continue

            if entity_id.startswith("S2-"):
                truth[source1_id]["S2"].add(entity_id)

            elif entity_id.startswith("S3-"):
                truth[source1_id]["S3"].add(entity_id)

    return truth


# ============================================================
# EVALUATE ONE SOURCE
# ============================================================

def evaluate_source(
    source1,
    candidate_source,
    ground_truth,
    source_label,
):

    print()
    print("=" * 70)
    print(f"EVALUATING {source_label}")
    print("=" * 70)

    print()
    print("Building exact-name lookup...")

    name_lookup = build_lookup(
        candidate_source,
        "name_clean",
    )

    print("Building exact-address lookup...")

    address_lookup = build_lookup(
        candidate_source,
        "address_clean",
    )

    print()
    print("Evaluating S1 rows...")

    total = 0

    truth_nonempty = 0
    captured = 0

    candidate_counts = []
    false_candidate_counts = []

    exact_name_rows = 0
    unique_address_rows = 0
    no_candidate_rows = 0

    for (
        s1_id,
        country,
        name,
        address,
    ) in source1[
        [
            "entity_id",
            "country_clean",
            "name_clean",
            "address_clean",
        ]
    ].itertuples(index=False, name=None):

        # ----------------------------------------------------
        # Current fast_final.py blocking logic
        # ----------------------------------------------------

        name_matches = name_lookup.get(
            (country, name),
            [],
        )

        address_matches = address_lookup.get(
            (country, address),
            [],
        )

        # Exact name matches are candidates.
        if name_matches:

            candidates = list(
                dict.fromkeys(name_matches)
            )

            exact_name_rows += 1

        # If there is no exact-name match,
        # use a unique exact-address match.
        elif len(address_matches) == 1:

            candidates = list(address_matches)

            unique_address_rows += 1

        else:

            candidates = []

            no_candidate_rows += 1

        candidate_set = set(candidates)

        candidate_counts.append(
            len(candidate_set)
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        truth_for_s1 = ground_truth.get(
            s1_id,
            {
                "S2": set(),
                "S3": set(),
            },
        )

        true_ids = truth_for_s1.get(
            source_label,
            set(),
        )

        if true_ids:

            truth_nonempty += 1

            intersection = (
                candidate_set.intersection(
                    true_ids
                )
            )

            if intersection:
                captured += 1

            false_candidate_counts.append(
                len(
                    candidate_set - true_ids
                )
            )

        total += 1

        if total % 250000 == 0:

            print(
                f"Processed {total:,} "
                f"/ {len(source1):,}"
            )

    # ========================================================
    # STATISTICS
    # ========================================================

    counts = np.array(
        candidate_counts,
        dtype=np.int64,
    )

    false_counts = np.array(
        false_candidate_counts,
        dtype=np.int64,
    )

    recall = (
        captured / truth_nonempty
        if truth_nonempty
        else 0.0
    )

    print()
    print("=" * 70)
    print(f"{source_label} RESULTS")
    print("=" * 70)

    print()
    print("Blocking recall")
    print("-" * 70)

    print(
        f"S1 rows evaluated       : {total:,}"
    )

    print(
        f"Non-empty ground truth  : "
        f"{truth_nonempty:,}"
    )

    print(
        f"True matches captured   : "
        f"{captured:,}"
    )

    print(
        f"Blocking recall         : "
        f"{recall:.4%}"
    )

    print()
    print("Candidate count statistics")
    print("-" * 70)

    print(
        f"Total candidates        : "
        f"{counts.sum():,}"
    )

    print(
        f"Average candidates/S1   : "
        f"{counts.mean():.4f}"
    )

    print(
        f"Median candidates/S1    : "
        f"{np.median(counts):.0f}"
    )

    print(
        f"P90 candidates/S1       : "
        f"{np.percentile(counts, 90):.0f}"
    )

    print(
        f"P95 candidates/S1       : "
        f"{np.percentile(counts, 95):.0f}"
    )

    print(
        f"P99 candidates/S1       : "
        f"{np.percentile(counts, 99):.0f}"
    )

    print(
        f"Maximum candidates/S1   : "
        f"{counts.max():,}"
    )

    print()
    print("Blocking route")
    print("-" * 70)

    print(
        f"Exact-name rows         : "
        f"{exact_name_rows:,}"
    )

    print(
        f"Unique-address rows     : "
        f"{unique_address_rows:,}"
    )

    print(
        f"No-candidate rows       : "
        f"{no_candidate_rows:,}"
    )

    print()
    print("False candidates")
    print("-" * 70)

    if len(false_counts) > 0:

        print(
            f"Average false candidates: "
            f"{false_counts.mean():.4f}"
        )

        print(
            f"Median false candidates : "
            f"{np.median(false_counts):.0f}"
        )

        print(
            f"P95 false candidates    : "
            f"{np.percentile(false_counts, 95):.0f}"
        )

        print(
            f"Maximum false candidates: "
            f"{false_counts.max():,}"
        )

    return {
        "recall": recall,
        "candidate_total": int(
            counts.sum()
        ),
        "candidate_mean": float(
            counts.mean()
        ),
        "candidate_median": float(
            np.median(counts)
        ),
        "candidate_p95": float(
            np.percentile(counts, 95)
        ),
        "candidate_p99": float(
            np.percentile(counts, 99)
        ),
        "candidate_max": int(
            counts.max()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON BUSINESS ENTITY RESOLUTION")
    print("BASELINE BLOCKING EVALUATION")
    print("=" * 70)

    print()
    print("Loading training data...")

    s1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t",
    )

    s2 = pd.read_csv(
        TRAIN_DIR / "train_source2.tsv",
        sep="\t",
    )

    s3 = pd.read_csv(
        TRAIN_DIR / "train_source3.tsv",
        sep="\t",
    )

    gt = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t",
    )

    print(
        f"S1 rows            : {len(s1):,}"
    )

    print(
        f"S2 rows            : {len(s2):,}"
    )

    print(
        f"S3 rows            : {len(s3):,}"
    )

    print(
        f"Ground truth rows  : {len(gt):,}"
    )

    print()
    print("Preprocessing...")

    s1 = preprocess(s1)
    s2 = preprocess(s2)
    s3 = preprocess(s3)

    print("Building ground truth...")

    truth = build_ground_truth(gt)

    print(
        f"Ground-truth S1 IDs: "
        f"{len(truth):,}"
    )

    # ========================================================
    # S2
    # ========================================================

    result_s2 = evaluate_source(
        s1,
        s2,
        truth,
        "S2",
    )

    # ========================================================
    # S3
    # ========================================================

    result_s3 = evaluate_source(
        s1,
        s3,
        truth,
        "S3",
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL BASELINE SUMMARY")
    print("=" * 70)

    print()
    print(
        f"S2 blocking recall : "
        f"{result_s2['recall']:.4%}"
    )

    print(
        f"S3 blocking recall : "
        f"{result_s3['recall']:.4%}"
    )

    print()
    print(
        f"S2 average candidates/S1 : "
        f"{result_s2['candidate_mean']:.4f}"
    )

    print(
        f"S3 average candidates/S1 : "
        f"{result_s3['candidate_mean']:.4f}"
    )

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
    