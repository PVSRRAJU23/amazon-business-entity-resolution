import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


FEATURE_COLUMNS = [
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "country_match",
    "name_length_diff",
    "address_length_diff"
]


def create_labels(feature_df, ground_truth):
    """
    Create 1/0 labels for candidate pairs.

    1 = actual match
    0 = not a match
    """

    ground_truth_lookup = {}

    for _, row in ground_truth.iterrows():

        source1_id = row["source1_entity_id"]

        matched_ids = row["matched_entity_ids"]

        if pd.isna(matched_ids) or str(matched_ids).strip() == "":
            ground_truth_lookup[source1_id] = set()
        else:
            ground_truth_lookup[source1_id] = set(
                str(matched_ids).split(",")
            )

    labels = []

    for _, row in feature_df.iterrows():

        source1_id = row["source1_entity_id"]
        candidate_id = row["candidate_entity_id"]

        actual_matches = ground_truth_lookup.get(
            source1_id,
            set()
        )

        if candidate_id in actual_matches:
            labels.append(1)
        else:
            labels.append(0)

    return labels


class MatchingModel:

    def __init__(self):

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                )
            )
        ])

    def train(self, feature_df, labels):

        X = feature_df[FEATURE_COLUMNS]

        self.pipeline.fit(X, labels)

    def predict_probability(self, feature_df):

        X = feature_df[FEATURE_COLUMNS]

        return self.pipeline.predict_proba(X)[:, 1]

    def predict(
        self,
        feature_df,
        threshold=0.70
    ):

        probabilities = self.predict_probability(
            feature_df
        )

        predictions = (
            probabilities >= threshold
        ).astype(int)

        return predictions, probabilities