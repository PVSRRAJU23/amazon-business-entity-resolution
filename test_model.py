from src.data_loader import load_data
from src.preprocessing import preprocess_dataframe, preprocess_ground_truth
from src.blocking import generate_candidates
from src.features import build_feature_dataset
from src.model import MatchingModel, create_labels

print("Loading data...")

data = load_data()

source1 = preprocess_dataframe(
    data["train_source1"].head(100000)
)

source2 = preprocess_dataframe(
    data["train_source2"]
)

ground_truth = preprocess_ground_truth(
    data["train_ground_truth"]
)

print("Generating candidates...")

candidate_pairs = generate_candidates(
    source1,
    source2
)

print("Candidates:", len(candidate_pairs))

print("Building features...")

feature_df = build_feature_dataset(
    source1,
    source2,
    candidate_pairs
)

print("Feature rows:", len(feature_df))

print("Creating labels...")

labels = create_labels(
    feature_df,
    ground_truth
)

print("Positive matches:", sum(labels))
print("Negative matches:", len(labels) - sum(labels))

print("Training model...")

model = MatchingModel()

model.train(
    feature_df,
    labels
)

predictions, probabilities = model.predict(
    feature_df,
    threshold=0.70
)

print("Predicted matches:", predictions.sum())
print("Model test completed!")