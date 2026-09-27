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
# STOP TOKENS
# ============================================================

STOP_TOKENS = {
    "of", "the", "and", "for", "inc", "llp", "ltd", "lp",
    "llc", "corp", "co", "company", "care", "service",
    "services", "health", "clinic", "associates",
    "enterprise", "enterprises", "industries",
    "p", "d", "m"
}


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
# GROUND TRUTH
# ============================================================

def build_ground_truth(gt):

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

            if entity_id.startswith("S2-"):
                truth[source1_id]["S2"].add(entity_id)

            elif entity_id.startswith("S3-"):
                truth[source1_id]["S3"].add(entity_id)

    return truth


# ============================================================
# EXACT NAME BLOCK
# ============================================================

def block_by_exact_name(source1, source2):

    left = source1[
        ["entity_id", "name_clean", "country_clean"]
    ].copy()

    right = source2[
        ["entity_id", "name_clean", "country_clean"]
    ].copy()

    left = left[left["name_clean"] != ""]
    right = right[right["name_clean"] != ""]

    candidates = left.merge(
        right,
        on=["name_clean", "country_clean"],
        how="inner",
        suffixes=("_s1", "_candidate"),
    )

    return candidates[
        ["entity_id_s1", "entity_id_candidate"]
    ].rename(
        columns={
            "entity_id_s1": "source1_entity_id",
            "entity_id_candidate": "candidate_entity_id",
        }
    )


# ============================================================
# TOKEN BLOCK
# ============================================================

def block_by_name_tokens(source1, source2):

    left = source1[
        ["entity_id", "name_clean", "country_clean"]
    ].copy()

    right = source2[
        ["entity_id", "name_clean", "country_clean"]
    ].copy()

    left = left[left["name_clean"] != ""]
    right = right[right["name_clean"] != ""]

    left["name_token"] = left["name_clean"].str.split()
    right["name_token"] = right["name_clean"].str.split()

    left = left.explode("name_token")
    right = right.explode("name_token")

    # Remove short tokens
    left = left[
        left["name_token"].str.len() >= 3
    ]

    right = right[
        right["name_token"].str.len() >= 3
    ]

    # Remove weak/common tokens
    left = left[
        ~left["name_token"].isin(STOP_TOKENS)
    ]

    right = right[
        ~right["name_token"].isin(STOP_TOKENS)
    ]

    # Frequency in candidate source
    token_counts = (
        right["name_token"]
        .value_counts()
    )

    # Existing blocking.py rule:
    # candidate token must occur <= 20 times
    useful_tokens = token_counts[
        token_counts <= 20
    ].index

    left = left[
        left["name_token"].isin(useful_tokens)
    ]

    right = right[
        right["name_token"].isin(useful_tokens)
    ]

    candidates = left.merge(
        right,
        on=["name_token", "country_clean"],
        how="inner",
        suffixes=("_s1", "_candidate"),
    )

    # Count shared useful tokens per pair
    pair_token_counts = (
        candidates
        .groupby(
            [
                "entity_id_s1",
                "entity_id_candidate",
            ],
            as_index=False,
        )
        .agg(
            shared_token_count=(
                "name_token",
                "nunique",
            )
        )
    )

    # Two or more shared tokens
    strong_pairs = pair_token_counts[
        pair_token_counts["shared_token_count"] >= 2
    ][
        [
            "entity_id_s1",
            "entity_id_candidate",
        ]
    ]

    # One-token matches only when token frequency <= 3
    single_token = candidates[
        candidates["name_token"].map(
            token_counts
        ) <= 3
    ]

    rare_single_pairs = single_token[
        [
            "entity_id_s1",
            "entity_id_candidate",
        ]
    ].drop_duplicates()

    return pd.concat(
        [
            strong_pairs,
            rare_single_pairs,
        ],
        ignore_index=True,
    ).drop_duplicates().rename(
        columns={
            "entity_id_s1": "source1_entity_id",
            "entity_id_candidate": "candidate_entity_id",
        }
    )


# ============================================================
# EXACT ADDRESS BLOCK
# ============================================================

def block_by_address(source1, source2):

    left = source1[
        [
            "entity_id",
            "address_clean",
            "country_clean",
        ]
    ].copy()

    right = source2[
        [
            "entity_id",
            "address_clean",
            "country_clean",
        ]
    ].copy()

    left = left[
        left["address_clean"] != ""
    ]

    right = right[
        right["address_clean"] != ""
    ]

    candidates = left.merge(
        right,
        on=[
            "address_clean",
            "country_clean",
        ],
        how="inner",
        suffixes=("_s1", "_candidate"),
    )

    return candidates[
        [
            "entity_id_s1",
            "entity_id_candidate",
        ]
    ].rename(
        columns={
            "entity_id_s1": "source1_entity_id",
            "entity_id_candidate": "candidate_entity_id",
        }
    )


# ============================================================
# COMBINE
# ============================================================

def combine_candidates(*frames):

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .drop_duplicates(
            subset=[
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_source(
    source1,
    source2,
    truth,
    label,
):

    print()
    print("=" * 70)
    print(f"EVALUATING TOKEN BLOCKING: {label}")
    print("=" * 70)

    print()
    print("Generating exact-name candidates...")

    exact_name = block_by_exact_name(
        source1,
        source2,
    )

    print(
        f"Exact-name pairs: "
        f"{len(exact_name):,}"
    )

    print()
    print("Generating token candidates...")

    token_candidates = block_by_name_tokens(
        source1,
        source2,
    )

    print(
        f"Token pairs: "
        f"{len(token_candidates):,}"
    )

    print()
    print("Generating exact-address candidates...")

    address_candidates = block_by_address(
        source1,
        source2,
    )

    print(
        f"Address pairs: "
        f"{len(address_candidates):,}"
    )

    print()
    print("Combining candidates...")

    candidates = combine_candidates(
        exact_name,
        token_candidates,
        address_candidates,
    )

    print(
        f"Total unique candidate pairs: "
        f"{len(candidates):,}"
    )

    # --------------------------------------------------------
    # Group candidates by S1
    # --------------------------------------------------------

    candidate_groups = (
        candidates
        .groupby("source1_entity_id")
        ["candidate_entity_id"]
        .apply(set)
        .to_dict()
    )

    counts = []

    truth_nonempty = 0
    captured = 0

    false_counts = []

    print()
    print("Evaluating recall...")

    total = len(source1)

    for i, s1_id in enumerate(
        source1["entity_id"],
        start=1,
    ):

        candidate_set = candidate_groups.get(
            s1_id,
            set(),
        )

        counts.append(
            len(candidate_set)
        )

        true_ids = truth.get(
            s1_id,
            {
                "S2": set(),
                "S3": set(),
            },
        ).get(
            label,
            set(),
        )

        if true_ids:

            truth_nonempty += 1

            if candidate_set.intersection(
                true_ids
            ):
                captured += 1

            false_counts.append(
                len(
                    candidate_set - true_ids
                )
            )

        if i % 250000 == 0:

            print(
                f"Processed {i:,} / {total:,}"
            )

    counts = np.array(
        counts,
        dtype=np.int64,
    )

    false_counts = np.array(
        false_counts,
        dtype=np.int64,
    )

    recall = (
        captured / truth_nonempty
        if truth_nonempty
        else 0.0
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(f"{label} TOKEN-BLOCKING RESULTS")
    print("=" * 70)

    print()
    print(
        f"Non-empty ground truth : "
        f"{truth_nonempty:,}"
    )

    print(
        f"True matches captured  : "
        f"{captured:,}"
    )

    print(
        f"Blocking recall        : "
        f"{recall:.4%}"
    )

    print()
    print("Candidate statistics")
    print("-" * 70)

    print(
        f"Total candidates       : "
        f"{counts.sum():,}"
    )

    print(
        f"Average/S1             : "
        f"{counts.mean():.4f}"
    )

    print(
        f"Median/S1              : "
        f"{np.median(counts):.0f}"
    )

    print(
        f"P90/S1                 : "
        f"{np.percentile(counts, 90):.0f}"
    )

    print(
        f"P95/S1                 : "
        f"{np.percentile(counts, 95):.0f}"
    )

    print(
        f"P99/S1                 : "
        f"{np.percentile(counts, 99):.0f}"
    )

    print(
        f"Maximum/S1             : "
        f"{counts.max():,}"
    )

    print()
    print("False candidates")
    print("-" * 70)

    if len(false_counts):

        print(
            f"Average false/S1       : "
            f"{false_counts.mean():.4f}"
        )

        print(
            f"Median false/S1        : "
            f"{np.median(false_counts):.0f}"
        )

        print(
            f"P95 false/S1           : "
            f"{np.percentile(false_counts, 95):.0f}"
        )

        print(
            f"Maximum false/S1       : "
            f"{false_counts.max():,}"
        )

    return recall, counts


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TOKEN BLOCKING EVALUATION")
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
        f"S1: {len(s1):,}"
    )

    print(
        f"S2: {len(s2):,}"
    )

    print(
        f"S3: {len(s3):,}"
    )

    print()
    print("Preprocessing...")

    s1 = preprocess(s1)
    s2 = preprocess(s2)
    s3 = preprocess(s3)

    print("Building ground truth...")

    truth = build_ground_truth(gt)

    # --------------------------------------------------------
    # S2
    # --------------------------------------------------------

    s2_recall, s2_counts = evaluate_source(
        s1,
        s2,
        truth,
        "S2",
    )

    # --------------------------------------------------------
    # S3
    # --------------------------------------------------------

    s3_recall, s3_counts = evaluate_source(
        s1,
        s3,
        truth,
        "S3",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TOKEN BLOCKING SUMMARY")
    print("=" * 70)

    print()
    print(
        f"S2 recall       : "
        f"{s2_recall:.4%}"
    )

    print(
        f"S2 avg candidates: "
        f"{s2_counts.mean():.4f}"
    )

    print()
    print(
        f"S3 recall       : "
        f"{s3_recall:.4%}"
    )

    print(
        f"S3 avg candidates: "
        f"{s3_counts.mean():.4f}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()