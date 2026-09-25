import pandas as pd


def block_by_exact_name(source1, source2):
    """Generate candidates where normalized business names are identical."""

    left = source1[["entity_id", "name_clean"]].copy()
    right = source2[["entity_id", "name_clean"]].copy()

    left = left[left["name_clean"] != ""]
    right = right[right["name_clean"] != ""]

    candidates = left.merge(
        right,
        on="name_clean",
        how="inner",
        suffixes=("_s1", "_candidate")
    )

    return candidates[
        ["entity_id_s1", "entity_id_candidate"]
    ].rename(
        columns={
            "entity_id_s1": "source1_entity_id",
            "entity_id_candidate": "candidate_entity_id"
        }
    )


def block_by_name_tokens(source1, source2):
    """Generate candidates when S1 and candidate share a meaningful name token."""

    left = source1[["entity_id", "name_clean"]].copy()
    right = source2[["entity_id", "name_clean"]].copy()

    left = left[left["name_clean"] != ""]
    right = right[right["name_clean"] != ""]

    left["name_token"] = left["name_clean"].str.split()
    right["name_token"] = right["name_clean"].str.split()

    left = left.explode("name_token")
    right = right.explode("name_token")

    # Ignore extremely short tokens such as "a", "of", "co"
    left = left[left["name_token"].str.len() >= 3]
    right = right[right["name_token"].str.len() >= 3]

    candidates = left.merge(
        right,
        on="name_token",
        how="inner",
        suffixes=("_s1", "_candidate")
    )

    return (
        candidates[
            ["entity_id_s1", "entity_id_candidate"]
        ]
        .rename(
            columns={
                "entity_id_s1": "source1_entity_id",
                "entity_id_candidate": "candidate_entity_id"
            }
        )
        .drop_duplicates()
    )


def block_by_address(source1, source2):
    """Generate candidates where normalized addresses are identical."""

    left = source1[["entity_id", "address_clean"]].copy()
    right = source2[["entity_id", "address_clean"]].copy()

    left = left[left["address_clean"] != ""]
    right = right[right["address_clean"] != ""]

    candidates = left.merge(
        right,
        on="address_clean",
        how="inner",
        suffixes=("_s1", "_candidate")
    )

    return candidates[
        ["entity_id_s1", "entity_id_candidate"]
    ].rename(
        columns={
            "entity_id_s1": "source1_entity_id",
            "entity_id_candidate": "candidate_entity_id"
        }
    )


def combine_candidates(*candidate_frames):
    """Combine candidate sets and remove duplicate pairs."""

    if not candidate_frames:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id"
            ]
        )

    candidates = pd.concat(
        candidate_frames,
        ignore_index=True
    )

    return (
        candidates
        .drop_duplicates(
            subset=[
                "source1_entity_id",
                "candidate_entity_id"
            ]
        )
        .reset_index(drop=True)
    )


def generate_candidates(source1, candidate_source):
    """
    Generate final candidate pairs between Source 1 and either
    Source 2 or Source 3.

    candidate_source can therefore be Source 2 or Source 3.
    """

    exact_name_candidates = block_by_exact_name(
        source1,
        candidate_source
    )

    token_candidates = block_by_name_tokens(
        source1,
        candidate_source
    )

    address_candidates = block_by_address(
        source1,
        candidate_source
    )

    return combine_candidates(
        exact_name_candidates,
        token_candidates,
        address_candidates
    )


def generate_all_candidates(source1, source2, source3):
    """
    Generate candidates from Source 1 against both Source 2
    and Source 3.
    """

    source2_candidates = generate_candidates(
        source1,
        source2
    )

    source3_candidates = generate_candidates(
        source1,
        source3
    )

    return combine_candidates(
        source2_candidates,
        source3_candidates
    )