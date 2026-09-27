import os
import pandas as pd

from src.data_loader import load_data
from src.preprocessing import preprocess_dataframe


OUTPUT_DIR = "output"


def build_lookup(df, column):
    lookup = {}

    for row in df[
        ["entity_id", "country_clean", column]
    ].itertuples(index=False):

        entity_id, country, value = row

        if not value:
            continue

        key = (str(country), str(value))

        if key not in lookup:
            lookup[key] = []

        lookup[key].append(str(entity_id))

    return lookup


def generate_fast_matches(source1, candidate_source, prefix):
    """
    Memory-safe exact matching.

    Results are written directly to disk instead of being
    accumulated in large Python lists.
    """

    name_lookup = build_lookup(
        candidate_source,
        "name_clean"
    )

    address_lookup = build_lookup(
        candidate_source,
        "address_clean"
    )

    candidate_file = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_candidates.tmp"
    )

    match_file = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_matches.tmp"
    )

    candidate_count = 0
    match_count = 0

    with open(
        candidate_file,
        "w",
        encoding="utf-8"
    ) as cf, open(
        match_file,
        "w",
        encoding="utf-8"
    ) as mf:

        cf.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )

        mf.write(
            "source1_entity_id\tcandidate_entity_id\n"
        )

        for row in source1[
            [
                "entity_id",
                "country_clean",
                "name_clean",
                "address_clean",
            ]
        ].itertuples(index=False):

            s1_id, country, name, address = row

            s1_id = str(s1_id)
            country = str(country)

            name_matches = []

            address_matches = []

            if name:
                name_matches = name_lookup.get(
                    (country, str(name)),
                    []
                )

            if address:
                address_matches = address_lookup.get(
                    (country, str(address)),
                    []
                )

            candidates = list(
                dict.fromkeys(
                    name_matches + address_matches
                )
            )

            # IMPORTANT:
            # Write every S1, including those with no candidates.
            cf.write(
                s1_id
                + "\t"
                + ",".join(candidates)
                + "\n"
            )

            candidate_count += 1

            # Exact name match
            matches = list(name_matches)

            # If no name match, accept unique exact address
            if (
                not matches
                and len(address_matches) == 1
            ):
                matches = list(address_matches)

            for candidate_id in dict.fromkeys(matches):

                mf.write(
                    s1_id
                    + "\t"
                    + str(candidate_id)
                    + "\n"
                )

                match_count += 1

    print(
        f"{prefix}: "
        f"{candidate_count:,} S1 rows, "
        f"{match_count:,} matches"
    )

    return candidate_file, match_file


def create_final_matching_results(
    source1,
    s2_match_file,
    s3_match_file
):
    """
    Create the required matching_results.tsv.

    Every S1 appears exactly once.
    """

    match_lookup = {}

    for filename in [
        s2_match_file,
        s3_match_file,
    ]:

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as f:

            next(f)  # skip header

            for line in f:

                line = line.rstrip("\n")

                if not line:
                    continue

                parts = line.split("\t")

                if len(parts) != 2:
                    continue

                s1_id, candidate_id = parts

                if s1_id not in match_lookup:
                    match_lookup[s1_id] = []

                match_lookup[s1_id].append(
                    candidate_id
                )

    output_file = os.path.join(
        OUTPUT_DIR,
        "matching_results.tsv"
    )

    row_count = 0

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "source1_entity_id\tmatched_entity_ids\n"
        )

        for s1_id in source1["entity_id"]:

            s1_id = str(s1_id)

            matches = list(
                dict.fromkeys(
                    match_lookup.get(
                        s1_id,
                        []
                    )
                )
            )

            f.write(
                s1_id
                + "\t"
                + ",".join(matches)
                + "\n"
            )

            row_count += 1

    return output_file, row_count


def create_final_candidate_pairs(
    source1,
    s2_candidate_file,
    s3_candidate_file
):
    """
    Merge S2 and S3 candidate files without
    loading the entire candidate set into RAM.
    """

    s2 = {}

    s3 = {}

    # Candidate files have one row per S1,
    # so these dictionaries contain at most
    # the number of S1 entities.
    for filename, target in [
        (s2_candidate_file, s2),
        (s3_candidate_file, s3),
    ]:

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as f:

            next(f)

            for line in f:

                line = line.rstrip("\n")

                parts = line.split("\t", 1)

                if len(parts) != 2:
                    continue

                s1_id, candidates = parts

                target[s1_id] = candidates

    output_file = os.path.join(
        OUTPUT_DIR,
        "candidate_pairs.tsv"
    )

    row_count = 0

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )

        for s1_id in source1["entity_id"]:

            s1_id = str(s1_id)

            ids = []

            if s1_id in s2:
                ids.extend(
                    x for x in s2[s1_id].split(",")
                    if x
                )

            if s1_id in s3:
                ids.extend(
                    x for x in s3[s1_id].split(",")
                    if x
                )

            ids = list(
                dict.fromkeys(ids)
            )

            f.write(
                s1_id
                + "\t"
                + ",".join(ids)
                + "\n"
            )

            row_count += 1

    return output_file, row_count


# ============================================================
# MAIN
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("Loading dataset...")

data = load_data()

print("Preprocessing test data...")

source1 = preprocess_dataframe(
    data["test_source1"]
)

source2 = preprocess_dataframe(
    data["test_source2"]
)

source3 = preprocess_dataframe(
    data["test_source3"]
)

print("Test data ready")

print("S1:", f"{len(source1):,}")
print("S2:", f"{len(source2):,}")
print("S3:", f"{len(source3):,}")


print("\nFast matching S2...")

s2_candidate_file, s2_match_file = (
    generate_fast_matches(
        source1,
        source2,
        "s2"
    )
)


print("\nFast matching S3...")

s3_candidate_file, s3_match_file = (
    generate_fast_matches(
        source1,
        source3,
        "s3"
    )
)


print("\nCreating matching_results.tsv...")

matching_file, matching_rows = (
    create_final_matching_results(
        source1,
        s2_match_file,
        s3_match_file
    )
)


print("\nCreating candidate_pairs.tsv...")

candidate_file, candidate_rows = (
    create_final_candidate_pairs(
        source1,
        s2_candidate_file,
        s3_candidate_file
    )
)


print("\nFINAL FILES CREATED")

print(
    "matching_results rows:",
    f"{matching_rows:,}"
)

print(
    "candidate_pairs rows:",
    f"{candidate_rows:,}"
)

print(
    "Expected S1 rows:",
    f"{len(source1):,}"
)

print("\nFiles:")

print(matching_file)

print(candidate_file)

print("\nDONE")