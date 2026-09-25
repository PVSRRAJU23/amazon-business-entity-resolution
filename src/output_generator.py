import os
import pandas as pd


def create_matching_results(source1, matches):
    """
    Create one output row for every Source-1 entity.

    Unmatched Source-1 entities receive an empty match list.
    """

    match_lookup = {}

    for row in matches.itertuples(index=False):

        s1_id = row.source1_entity_id
        candidate_id = row.candidate_entity_id

        if s1_id not in match_lookup:
            match_lookup[s1_id] = []

        match_lookup[s1_id].append(candidate_id)

    results = []

    for s1_id in source1["entity_id"]:

        matched_ids = match_lookup.get(s1_id, [])

        # Remove duplicate IDs
        matched_ids = list(dict.fromkeys(matched_ids))

        results.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched_ids)
        })

    return pd.DataFrame(results)


def save_outputs(
    source1,
    matches,
    candidate_pairs,
    output_dir="output"
):

    os.makedirs(output_dir, exist_ok=True)

    matching_results = create_matching_results(
        source1,
        matches
    )

    matching_results.to_csv(
        f"{output_dir}/matching_results.tsv",
        sep="\t",
        index=False
    )

    # Validator expects:
    # source1_entity_id    candidate_entity_ids
    candidate_output = (
        candidate_pairs
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(
            lambda x: ",".join(
                dict.fromkeys(x.astype(str))
            )
        )
        .reset_index()
        .rename(
            columns={
                "candidate_entity_id": "candidate_entity_ids"
            }
        )
    )

    candidate_output.to_csv(
        f"{output_dir}/candidate_pairs.tsv",
        sep="\t",
        index=False
    )

    print("Saved:")
    print(f"{output_dir}/matching_results.tsv")
    print(f"{output_dir}/candidate_pairs.tsv")