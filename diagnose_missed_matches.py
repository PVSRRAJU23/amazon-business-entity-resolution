from pathlib import Path
import pandas as pd

from src.preprocessing import (
    preprocess_dataframe,
    preprocess_ground_truth,
)


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


def main():

    print("=" * 70)
    print("MISSED MATCH DIAGNOSTIC")
    print("=" * 70)

    print("\nLoading training data...")

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

    print("Preprocessing using project normalization...")

    s1 = preprocess_dataframe(s1)
    s2 = preprocess_dataframe(s2)
    s3 = preprocess_dataframe(s3)
    gt = preprocess_ground_truth(gt)

    # --------------------------------------------------------
    # Lookups
    # --------------------------------------------------------

    def make_lookup(df, column):

        result = {}

        for (
            entity_id,
            country,
            value,
        ) in df[
            [
                "entity_id",
                "country_clean",
                column,
            ]
        ].itertuples(index=False, name=None):

            if not value:
                continue

            key = (country, value)

            result.setdefault(
                key,
                set(),
            ).add(entity_id)

        return result

    print("\nBuilding S2 lookups...")

    s2_name = make_lookup(
        s2,
        "name_clean",
    )

    s2_address = make_lookup(
        s2,
        "address_clean",
    )

    print("Building S3 lookups...")

    s3_name = make_lookup(
        s3,
        "name_clean",
    )

    s3_address = make_lookup(
        s3,
        "address_clean",
    )

    # --------------------------------------------------------
    # Entity lookup for examining true matches
    # --------------------------------------------------------

    s2_rows = s2.set_index("entity_id")
    s3_rows = s3.set_index("entity_id")
    s1_rows = s1.set_index("entity_id")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = {
        "S2": {
            "total_true": 0,
            "exact_name": 0,
            "exact_address": 0,
            "either": 0,
            "neither": 0,
        },
        "S3": {
            "total_true": 0,
            "exact_name": 0,
            "exact_address": 0,
            "either": 0,
            "neither": 0,
        },
    }

    # --------------------------------------------------------
    # Examine each ground-truth pair
    # --------------------------------------------------------

    print("\nAnalyzing ground-truth pairs...")

    for row in gt.itertuples(index=False):

        s1_id = row.source1_entity_id
        matched_ids = row.matched_entity_ids

        if s1_id not in s1_rows.index:
            continue

        s1_row = s1_rows.loc[s1_id]

        s1_country = s1_row["country_clean"]
        s1_name = s1_row["name_clean"]
        s1_address = s1_row["address_clean"]

        for matched_id in matched_ids:

            matched_id = str(
                matched_id
            ).strip()

            if matched_id.startswith("S2-"):

                source = "S2"
                rows = s2_rows
                name_lookup = s2_name
                address_lookup = s2_address

            elif matched_id.startswith("S3-"):

                source = "S3"
                rows = s3_rows
                name_lookup = s3_name
                address_lookup = s3_address

            else:
                continue

            if matched_id not in rows.index:
                continue

            stats[source]["total_true"] += 1

            true_row = rows.loc[matched_id]

            true_name = true_row[
                "name_clean"
            ]

            true_address = true_row[
                "address_clean"
            ]

            true_country = true_row[
                "country_clean"
            ]

            exact_name = (
                s1_country == true_country
                and s1_name != ""
                and s1_name == true_name
            )

            exact_address = (
                s1_country == true_country
                and s1_address != ""
                and s1_address == true_address
            )

            if exact_name:
                stats[source]["exact_name"] += 1

            if exact_address:
                stats[source]["exact_address"] += 1

            if exact_name or exact_address:
                stats[source]["either"] += 1

            else:
                stats[source]["neither"] += 1

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    for source in ["S2", "S3"]:

        s = stats[source]

        total = s["total_true"]

        print()
        print("=" * 70)
        print(f"{source} TRUE-MATCH DIAGNOSTIC")
        print("=" * 70)

        print(
            f"Total true pairs       : "
            f"{total:,}"
        )

        print(
            f"Exact normalized name  : "
            f"{s['exact_name']:,}"
        )

        print(
            f"Exact normalized addr  : "
            f"{s['exact_address']:,}"
        )

        print(
            f"Name OR address exact  : "
            f"{s['either']:,}"
        )

        print(
            f"Neither exact          : "
            f"{s['neither']:,}"
        )

        if total:

            print()

            print(
                f"Name coverage          : "
                f"{s['exact_name'] / total:.4%}"
            )

            print(
                f"Address coverage       : "
                f"{s['exact_address'] / total:.4%}"
            )

            print(
                f"Name/address coverage  : "
                f"{s['either'] / total:.4%}"
            )

            print(
                f"Neither coverage       : "
                f"{s['neither'] / total:.4%}"
            )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()