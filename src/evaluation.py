import pandas as pd


def calculate_f05(precision, recall):
    """
    Calculate F0.5.
    """

    if precision == 0 and recall == 0:
        return 0.0

    return (
        1.25 * precision * recall
    ) / (
        0.25 * precision + recall
    )


def evaluate_predictions(
    predictions_df,
    ground_truth_df
):
    """
    Calculate macro-averaged F0.5
    across Source 1 entities.
    """

    ground_truth_lookup = {}

    for _, row in ground_truth_df.iterrows():

        source1_id = row["source1_entity_id"]

        matched_ids = row["matched_entity_ids"]

        if pd.isna(matched_ids) or str(matched_ids).strip() == "":
            ground_truth_lookup[source1_id] = set()
        else:
            ground_truth_lookup[source1_id] = set(
                str(matched_ids).split(",")
            )

    scores = []

    for _, row in predictions_df.iterrows():

        source1_id = row["source1_entity_id"]

        predicted_ids = row["matched_entity_ids"]

        if pd.isna(predicted_ids) or str(predicted_ids).strip() == "":
            predicted = set()
        else:
            predicted = set(
                str(predicted_ids).split(",")
            )

        actual = ground_truth_lookup.get(
            source1_id,
            set()
        )

        true_positive = len(
            predicted & actual
        )

        # Precision
        if len(predicted) == 0:
            precision = 1.0 if len(actual) == 0 else 0.0
        else:
            precision = (
                true_positive /
                len(predicted)
            )

        # Recall
        if len(actual) == 0:
            recall = 1.0 if len(predicted) == 0 else 0.0
        else:
            recall = (
                true_positive /
                len(actual)
            )

        score = calculate_f05(
            precision,
            recall
        )

        scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)