import pandas as pd
from rapidfuzz import fuzz


def string_similarity(text1, text2):
    """
    Calculate different string similarity measures.
    Returns values between 0 and 1.
    """

    text1 = "" if pd.isna(text1) else str(text1)
    text2 = "" if pd.isna(text2) else str(text2)

    if not text1 or not text2:
        return {
            "ratio": 0.0,
            "token_sort": 0.0,
            "token_set": 0.0
        }

    return {
        "ratio": fuzz.ratio(text1, text2) / 100.0,
        "token_sort": fuzz.token_sort_ratio(text1, text2) / 100.0,
        "token_set": fuzz.token_set_ratio(text1, text2) / 100.0
    }


def create_pair_features(source1_row, candidate_row):
    """
    Create features for one Source1-Candidate pair.
    """

    name = string_similarity(
        source1_row["business_name_normalized"],
        candidate_row["business_name_normalized"]
    )

    address = string_similarity(
        source1_row["business_address_normalized"],
        candidate_row["business_address_normalized"]
    )

    country_match = int(
        str(source1_row["country"]).strip().lower()
        ==
        str(candidate_row["country"]).strip().lower()
    )

    features = {
        # Name features
        "name_ratio": name["ratio"],
        "name_token_sort": name["token_sort"],
        "name_token_set": name["token_set"],

        # Address features
        "address_ratio": address["ratio"],
        "address_token_sort": address["token_sort"],
        "address_token_set": address["token_set"],

        # Country
        "country_match": country_match,

        # Length differences
        "name_length_diff": abs(
            len(str(source1_row["business_name_normalized"]))
            -
            len(str(candidate_row["business_name_normalized"]))
        ),

        "address_length_diff": abs(
            len(str(source1_row["business_address_normalized"]))
            -
            len(str(candidate_row["business_address_normalized"]))
        )
    }

    return features


def build_feature_dataset(
    source1_df,
    candidate_df,
    candidate_pairs
):
    """
    Build ML features for all candidate pairs.

    Parameters
    ----------
    source1_df:
        Source 1 records.

    candidate_df:
        Combined Source 2 and Source 3 records.

    candidate_pairs:
        DataFrame containing:
            source1_entity_id
            candidate_entity_id
    """

    source1_lookup = source1_df.set_index("entity_id")
    candidate_lookup = candidate_df.set_index("entity_id")

    rows = []

    for _, pair in candidate_pairs.iterrows():

        source1_id = pair["source1_entity_id"]
        candidate_id = pair["candidate_entity_id"]

        if source1_id not in source1_lookup.index:
            continue

        if candidate_id not in candidate_lookup.index:
            continue

        source1_row = source1_lookup.loc[source1_id]
        candidate_row = candidate_lookup.loc[candidate_id]

        features = create_pair_features(
            source1_row,
            candidate_row
        )

        features["source1_entity_id"] = source1_id
        features["candidate_entity_id"] = candidate_id

        rows.append(features)

    return pd.DataFrame(rows)