import pandas as pd

from src.evaluation import evaluate_predictions
from src.features import build_feature_dataset
from src.model import MatchingModel, create_labels


# -----------------------------
# 1. Source 1
# -----------------------------

source1 = pd.DataFrame([
    {
        "entity_id": "S1-001",
        "business_name_normalized": "abc technologies",
        "business_address_normalized": "hyderabad road",
        "country": "India"
    },
    {
        "entity_id": "S1-002",
        "business_name_normalized": "xyz foods",
        "business_address_normalized": "chennai main road",
        "country": "India"
    },
    {
        "entity_id": "S1-003",
        "business_name_normalized": "pqr stores",
        "business_address_normalized": "mumbai road",
        "country": "India"
    }
])


# -----------------------------
# 2. Candidate records
# -----------------------------

candidate_df = pd.DataFrame([
    {
        "entity_id": "S2-001",
        "business_name_normalized": "abc technology",
        "business_address_normalized": "hyderabad rd",
        "country": "India"
    },
    {
        "entity_id": "S2-002",
        "business_name_normalized": "random shop",
        "business_address_normalized": "delhi road",
        "country": "India"
    },
    {
        "entity_id": "S2-003",
        "business_name_normalized": "xyz foods",
        "business_address_normalized": "chennai main rd",
        "country": "India"
    },
    {
        "entity_id": "S2-004",
        "business_name_normalized": "pqr store",
        "business_address_normalized": "mumbai rd",
        "country": "India"
    },
    {
        "entity_id": "S2-005",
        "business_name_normalized": "completely different",
        "business_address_normalized": "pune road",
        "country": "India"
    }
])


# -----------------------------
# 3. Candidate pairs
# -----------------------------

candidate_pairs = pd.DataFrame([
    {
        "source1_entity_id": "S1-001",
        "candidate_entity_id": "S2-001"
    },
    {
        "source1_entity_id": "S1-001",
        "candidate_entity_id": "S2-002"
    },
    {
        "source1_entity_id": "S1-002",
        "candidate_entity_id": "S2-003"
    },
    {
        "source1_entity_id": "S1-002",
        "candidate_entity_id": "S2-005"
    },
    {
        "source1_entity_id": "S1-003",
        "candidate_entity_id": "S2-004"
    },
    {
        "source1_entity_id": "S1-003",
        "candidate_entity_id": "S2-005"
    }
])


# -----------------------------
# 4. Ground truth
# -----------------------------

ground_truth = pd.DataFrame([
    {
        "source1_entity_id": "S1-001",
        "matched_entity_ids": "S2-001"
    },
    {
        "source1_entity_id": "S1-002",
        "matched_entity_ids": "S2-003"
    },
    {
        "source1_entity_id": "S1-003",
        "matched_entity_ids": "S2-004"
    }
])


# -----------------------------
# 5. Create features
# -----------------------------

feature_df = build_feature_dataset(
    source1,
    candidate_df,
    candidate_pairs
)

print("\nFEATURES:")
print(feature_df)


# -----------------------------
# 6. Create labels
# -----------------------------

labels = create_labels(
    feature_df,
    ground_truth
)

print("\nLABELS:")
print(labels)


# -----------------------------
# 7. Train model
# -----------------------------

model = MatchingModel()

model.train(
    feature_df,
    labels
)


# -----------------------------
# 8. Predict
# -----------------------------

predictions, probabilities = model.predict(
    feature_df,
    threshold=0.70
)

feature_df["prediction"] = predictions
feature_df["probability"] = probabilities


print("\nPREDICTIONS:")
print(
    feature_df[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "probability",
            "prediction"
        ]
    ]
)

# -----------------------------
# 9. Create prediction output
# -----------------------------

prediction_rows = []

for source1_id in feature_df["source1_entity_id"].unique():

    rows = feature_df[
        feature_df["source1_entity_id"] == source1_id
    ]

    matched = rows[
        rows["prediction"] == 1
    ]["candidate_entity_id"].tolist()

    prediction_rows.append({
        "source1_entity_id": source1_id,
        "matched_entity_ids": ",".join(matched)
    })


predictions_df = pd.DataFrame(prediction_rows)


# -----------------------------
# 10. Evaluate F0.5
# -----------------------------

score = evaluate_predictions(
    predictions_df,
    ground_truth
)

print("\nPREDICTION OUTPUT:")
print(predictions_df)

print("\nF0.5 SCORE:")
print(score)