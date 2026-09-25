import pandas as pd

from src.data_loader import load_data
from src.preprocessing import preprocess_dataframe
from src.blocking import generate_candidates
from src.fast_matcher import match_candidates
from src.output_generator import save_outputs


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

print("Generating S2 candidates...")

s2_candidates = generate_candidates(
    source1,
    source2
)

print("S2 candidates:", len(s2_candidates))

print("Generating S3 candidates...")

s3_candidates = generate_candidates(
    source1,
    source3
)

print("S3 candidates:", len(s3_candidates))

print("Matching S2...")

s2_matches = match_candidates(
    source1,
    source2,
    s2_candidates,
    threshold=0.70
)

print("S2 matches:", len(s2_matches))

print("Matching S3...")

s3_matches = match_candidates(
    source1,
    source3,
    s3_candidates,
    threshold=0.70
)

print("S3 matches:", len(s3_matches))

all_matches = pd.concat(
    [s2_matches, s3_matches],
    ignore_index=True
)

all_candidates = pd.concat(
    [s2_candidates, s3_candidates],
    ignore_index=True
).drop_duplicates(
    subset=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
)

print("Total matches:", len(all_matches))

print("Total candidates:", len(all_candidates))

print("Saving final outputs...")

save_outputs(
    source1,
    all_matches,
    all_candidates
)

print("FINAL TEST PIPELINE COMPLETE")