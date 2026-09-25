from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT / DATASET PATHS
# ============================================================

# data_loader.py:
# E:\ML-Hackathon\amazon-business-entity-resolution\src\data_loader.py
#
# .parent
# -> E:\ML-Hackathon\amazon-business-entity-resolution\src
#
# .parent.parent
# -> E:\ML-Hackathon\amazon-business-entity-resolution
#
# .parent.parent.parent
# -> E:\ML-Hackathon
#
# student_resource is located directly inside E:\ML-Hackathon
# alongside amazon-business-entity-resolution.

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = WORKSPACE_ROOT / "student_resource" / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"


# ============================================================
# DATA LOADER
# ============================================================

def load_data():
    """
    Load all training and test TSV files.

    Returns
    -------
    dict
        Dictionary containing all seven dataset DataFrames.
    """

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    train_source1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t"
    )

    train_source2 = pd.read_csv(
        TRAIN_DIR / "train_source2.tsv",
        sep="\t"
    )

    train_source3 = pd.read_csv(
        TRAIN_DIR / "train_source3.tsv",
        sep="\t"
    )

    train_ground_truth = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t"
    )

    # --------------------------------------------------------
    # Test data
    # --------------------------------------------------------

    test_source1 = pd.read_csv(
        TEST_DIR / "test_source1.tsv",
        sep="\t"
    )

    test_source2 = pd.read_csv(
        TEST_DIR / "test_source2.tsv",
        sep="\t"
    )

    test_source3 = pd.read_csv(
        TEST_DIR / "test_source3.tsv",
        sep="\t"
    )

    # --------------------------------------------------------
    # Return all datasets
    # --------------------------------------------------------

    return {
        "train_source1": train_source1,
        "train_source2": train_source2,
        "train_source3": train_source3,
        "train_ground_truth": train_ground_truth,
        "test_source1": test_source1,
        "test_source2": test_source2,
        "test_source3": test_source3,
    }


# ============================================================
# SIMPLE DATASET SUMMARY
# ============================================================

def print_dataset_summary(data):
    """
    Print the shape and columns of every loaded dataset.
    """

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    for name, df in data.items():
        print(f"\n{name}")
        print(f"Rows    : {len(df):,}")
        print(f"Columns : {list(df.columns)}")
        print(f"Shape   : {df.shape}")

    print("\n" + "=" * 60)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("\nLoading dataset...")

    data = load_data()

    print("\nDataset loaded successfully!")

    print_dataset_summary(data)