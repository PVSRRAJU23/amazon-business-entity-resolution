import pandas as pd


# Tokens that are too common or too weak to be useful for blocking
STOP_TOKENS = {
    "of", "the", "and", "for", "inc", "llp", "ltd", "lp",
    "llc", "corp", "co", "company", "care", "service",
    "services", "health", "clinic", "associates",
    "enterprise", "enterprises", "industries",
    "p", "d", "m"
}


def block_by_exact_name(source1, source2):
    """Candidates where normalized business names are identical."""

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
    """
    Generate candidates using informative business-name tokens.

    Rules:
    - Token must have at least 3 characters.
    - Very common/weak tokens are ignored.
    - Pairs sharing 2 or more useful tokens are retained.
    - Pairs sharing only 1 token are retained only when that
      token occurs at most 3 times in the candidate source.
    """

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
    left = left[left["name_token"].str.len() >= 3]
    right = right[right["name_token"].str.len() >= 3]

    # Remove weak/common tokens
    left = left[~left["name_token"].isin(STOP_TOKENS)]
    right = right[~right["name_token"].isin(STOP_TOKENS)]

    # Count token frequency in candidate source
    token_counts = right["name_token"].value_counts()

    # Keep only useful tokens
    useful_tokens = token_counts[
        token_counts <= 20
    ].index

    left = left[left["name_token"].isin(useful_tokens)]
    right = right[right["name_token"].isin(useful_tokens)]

    # Match S1 and candidate records by token + country
    candidates = left.merge(
        right,
        on=["name_token", "country_clean"],
        how="inner",
        suffixes=("_s1", "_candidate")
    )

    # Count how many useful tokens each pair shares
    pair_token_counts = (
        candidates
        .groupby(
            ["entity_id_s1", "entity_id_candidate"],
            as_index=False
        )
        .agg(
            shared_token_count=("name_token", "nunique")
        )
    )

    # Keep strong multi-token matches
    strong_pairs = pair_token_counts[
        pair_token_counts["shared_token_count"] >= 2
    ]

    # For one-token matches, keep only very rare tokens
    single_token = candidates[
        candidates["name_token"].map(token_counts) <= 3
    ]

    rare_single_pairs = single_token[
        ["entity_id_s1", "entity_id_candidate"]
    ].drop_duplicates()

    strong_pairs = strong_pairs[
        ["entity_id_s1", "entity_id_candidate"]
    ]

    return pd.concat(
        [
            strong_pairs,
            rare_single_pairs
        ],
        ignore_index=True
    ).rename(
        columns={
            "entity_id_s1": "source1_entity_id",
            "entity_id_candidate": "candidate_entity_id"
        }
    ).drop_duplicates()

def block_by_address(source1, source2):
    """Candidates where normalized addresses are identical."""

    left = source1[
        ["entity_id", "address_clean", "country_clean"]
    ].copy()

    right = source2[
        ["entity_id", "address_clean", "country_clean"]
    ].copy()

    left = left[left["address_clean"] != ""]
    right = right[right["address_clean"] != ""]

    candidates = left.merge(
        right,
        on=["address_clean", "country_clean"],
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
    Generate candidates between Source 1 and Source 2 or Source 3.
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