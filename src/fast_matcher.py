import pandas as pd
from rapidfuzz import fuzz


def token_overlap(a, b):
    a_tokens = set(str(a).split())
    b_tokens = set(str(b).split())

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def calculate_score(s1_row, candidate_row):
    name1 = str(s1_row["name_clean"])
    name2 = str(candidate_row["name_clean"])

    address1 = str(s1_row["address_clean"])
    address2 = str(candidate_row["address_clean"])

    country1 = str(s1_row["country_clean"])
    country2 = str(candidate_row["country_clean"])

    country_match = country1 == country2

    exact_name = name1 != "" and name1 == name2
    exact_address = address1 != "" and address1 == address2

    name_overlap = token_overlap(name1, name2)
    address_overlap = token_overlap(address1, address2)

    name_fuzzy = 0.0

    if exact_name:
        name_fuzzy = 1.0
    elif name_overlap >= 0.5:
        name_fuzzy = fuzz.ratio(name1, name2) / 100.0

    score = 0.0

    if exact_name:
        score += 0.60

    if exact_address:
        score += 0.30

    score += 0.20 * name_overlap
    score += 0.10 * address_overlap

    if name_fuzzy >= 0.90:
        score += 0.15

    if country_match:
        score += 0.10

    return min(score,1.0)


def match_candidates(
    source1,
    candidate_source,
    candidate_pairs,
    threshold=0.70
):
    """
    Faster candidate matching.

    Uses dictionaries instead of repeatedly calling DataFrame .loc[]
    for every candidate pair.
    """

    # Keep only the columns required for matching
    s1_data = source1[
        ["entity_id", "name_clean", "address_clean", "country_clean"]
    ].copy()

    candidate_data = candidate_source[
        ["entity_id", "name_clean", "address_clean", "country_clean"]
    ].copy()

    # Convert rows to dictionaries once.
    s1_lookup = {
        str(row["entity_id"]): row
        for _, row in s1_data.iterrows()
    }

    candidate_lookup = {
        str(row["entity_id"]): row
        for _, row in candidate_data.iterrows()
    }

    results = []

    # Process candidates using tuples.
    for pair in candidate_pairs.itertuples(index=False):

        s1_id = str(pair.source1_entity_id)
        candidate_id = str(pair.candidate_entity_id)

        s1_row = s1_lookup.get(s1_id)
        candidate_row = candidate_lookup.get(candidate_id)

        if s1_row is None or candidate_row is None:
            continue

        score = calculate_score(s1_row, candidate_row)

        if score >= threshold:
            results.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id,
                    "score": score,
                }
            )

    return pd.DataFrame(
        results,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "score",
        ],
    )