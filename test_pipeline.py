from src.data_loader import load_data
from src.preprocessing import preprocess_dataframe
from src.blocking import generate_candidates
from src.fast_matcher import match_candidates

print("Loading data...")

data = load_data()

# Small test sizes so the pipeline finishes quickly
source1 = preprocess_dataframe(
    data["train_source1"].head(10000)
)

source2 = preprocess_dataframe(
    data["train_source2"].head(100000)
)

source3 = preprocess_dataframe(
    data["train_source3"].head(100000)
)

print("Data ready")

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

print("Matching S2 candidates...")

s2_matches = match_candidates(
    source1,
    source2,
    s2_candidates,
    threshold=0.70
)

print("S2 matches:", len(s2_matches))

print("Matching S3 candidates...")

s3_matches = match_candidates(
    source1,
    source3,
    s3_candidates,
    threshold=0.70
)

print("S3 matches:", len(s3_matches))

all_matches = __import__("pandas").concat(
    [s2_matches, s3_matches],
    ignore_index=True
)

print("Total matches:", len(all_matches))

print("\nSample matches:")
print(all_matches.head(10))

print("\nPipeline completed successfully!")
from src.output_generator import save_outputs

all_candidates = __import__("pandas").concat(
    [s2_candidates, s3_candidates],
    ignore_index=True
).drop_duplicates()

save_outputs(
    source1,
    all_matches,
    all_candidates
)

print("Output files created.")