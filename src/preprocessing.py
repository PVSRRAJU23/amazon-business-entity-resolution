import re
import pandas as pd


# ---------------------------------------------------------
# Text normalization
# ---------------------------------------------------------

def normalize_text(value):
    """
    Normalize a business-name/address/country string.

    Steps:
    - Handle missing values
    - Convert to lowercase
    - Remove accents/non-ASCII characters
    - Replace punctuation with spaces
    - Normalize whitespace
    """
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Remove accents / convert unicode characters
    value = (
        value.encode("ascii", "ignore")
        .decode("ascii")
    )

    # Replace punctuation/special characters with spaces
    value = re.sub(r"[^a-z0-9\s]", " ", value)

    # Collapse multiple spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ---------------------------------------------------------
# Business-name normalization
# ---------------------------------------------------------

def normalize_business_name(value):
    """
    Normalize a business name.
    """
    text = normalize_text(value)

    # Common business suffixes
    suffixes = [
        "incorporated",
        "corporation",
        "company",
        "limited",
        "llc",
        "inc",
        "ltd",
        "corp",
        "co",
    ]

    words = text.split()

    # Remove business suffixes from the end
    while words and words[-1] in suffixes:
        words.pop()

    return " ".join(words)


# ---------------------------------------------------------
# Address normalization
# ---------------------------------------------------------

def normalize_address(value):
    """
    Normalize a business address.
    """
    text = normalize_text(value)

    # Common address abbreviations
    replacements = {
        "street": "st",
        "road": "rd",
        "avenue": "ave",
        "boulevard": "blvd",
        "drive": "dr",
        "lane": "ln",
        "highway": "hwy",
        "parkway": "pkwy",
        "place": "pl",
        "court": "ct",
        "suite": "ste",
        "apartment": "apt",
    }

    words = text.split()

    words = [
        replacements.get(word, word)
        for word in words
    ]

    return " ".join(words)


# ---------------------------------------------------------
# Country normalization
# ---------------------------------------------------------

def normalize_country(value):
    """
    Normalize country codes/names.
    """
    text = normalize_text(value)

    country_map = {
        "us": "us",
        "usa": "us",
        "united states": "us",
        "united states of america": "us",

        "in": "india",
        "india": "india",

        "gb": "uk",
        "uk": "uk",
        "united kingdom": "uk",
    }

    return country_map.get(text, text)


# ---------------------------------------------------------
# DataFrame preprocessing
# ---------------------------------------------------------

def preprocess_dataframe(df):
    """
    Add normalized columns to a source dataframe.

    Expected columns:
        entity_id
        business_name
        business_address
        country
    """

    df = df.copy()

    # Make sure expected columns exist
    required_columns = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Normalize original fields
    df["name_clean"] = (
        df["business_name"]
        .map(normalize_business_name)
    )

    df["address_clean"] = (
        df["business_address"]
        .map(normalize_address)
    )

    df["country_clean"] = (
        df["country"]
        .map(normalize_country)
    )

    return df


# ---------------------------------------------------------
# Ground-truth preprocessing
# ---------------------------------------------------------

def preprocess_ground_truth(df):
    """
    Prepare train_ground_truth.tsv.

    Expected columns:
        source1_entity_id
        matched_entity_ids

    matched_entity_ids contains comma-separated IDs.
    """

    df = df.copy()

    if "source1_entity_id" not in df.columns:
        raise ValueError(
            "source1_entity_id column not found"
        )

    if "matched_entity_ids" not in df.columns:
        raise ValueError(
            "matched_entity_ids column not found"
        )

    df["source1_entity_id"] = (
        df["source1_entity_id"]
        .astype(str)
        .str.strip()
    )

    df["matched_entity_ids"] = (
        df["matched_entity_ids"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Convert comma-separated IDs into lists
    df["matched_entity_ids"] = (
        df["matched_entity_ids"]
        .apply(
            lambda x: [
                item.strip()
                for item in x.split(",")
                if item.strip()
            ]
        )
    )

    return df


# ---------------------------------------------------------
# Quick preprocessing test
# ---------------------------------------------------------

if __name__ == "__main__":
    from src.data_loader import load_data

    data = load_data()

    print("\nPreprocessing training source 1...")

    train_source1 = preprocess_dataframe(
        data["train_source1"]
    )

    print(train_source1.head())

    print("\nNew columns:")
    print(train_source1.columns.tolist())

    print("\nPreprocessing ground truth...")

    ground_truth = preprocess_ground_truth(
        data["train_ground_truth"]
    )

    print(ground_truth.head())

    print("\nPREPROCESSING SUCCESSFUL")