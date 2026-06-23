from pathlib import Path
from zipfile import ZipFile
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_ZIP = ROOT_DIR / "emotionClassification_train.zip"
TEST_ZIP = ROOT_DIR / "emotionClassification_test.zip"

PERSON_COLUMNS = ["사람문장1", "사람문장2", "사람문장3", "사람문장4"]
LABEL_COLUMN = "감정_대분류"


def list_zip_files(zip_path: Path) -> list[str]:
    """Return file names inside a zip file."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    with ZipFile(zip_path, "r") as zip_file:
        return zip_file.namelist()


def read_csv_from_zip(zip_path: Path, inner_file_name: str) -> pd.DataFrame:
    """Read a CSV file inside a zip file."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    with ZipFile(zip_path, "r") as zip_file:
        with zip_file.open(inner_file_name) as file:
            return pd.read_csv(file)


def combine_person_sentences(df: pd.DataFrame) -> pd.Series:
    """
    Combine 사람문장1~4 into one text field.
    Only human utterances are used because the target is emotion classification.
    """
    existing_columns = [col for col in PERSON_COLUMNS if col in df.columns]

    if not existing_columns:
        raise ValueError(f"No person sentence columns found. Expected one of: {PERSON_COLUMNS}")

    return (
        df[existing_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_train_data() -> pd.DataFrame:
    """Load train.csv and create text/label columns."""
    df = read_csv_from_zip(TRAIN_ZIP, "train.csv")

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column not found: {LABEL_COLUMN}")

    df = df.copy()
    df["text"] = combine_person_sentences(df)
    df["label"] = df[LABEL_COLUMN]

    return df[["text", "label"]]


def load_test_data() -> pd.DataFrame:
    """Load test.csv and create text column."""
    df = read_csv_from_zip(TEST_ZIP, "test.csv")

    if "ID" not in df.columns:
        raise ValueError("ID column not found in test.csv")

    df = df.copy()
    df["text"] = combine_person_sentences(df)

    return df[["ID", "text"]]


def inspect_dataset() -> None:
    """Print basic dataset information."""
    print("=== Train ZIP files ===")
    for name in list_zip_files(TRAIN_ZIP):
        print(name)

    print("\n=== Test ZIP files ===")
    for name in list_zip_files(TEST_ZIP):
        print(name)

    train_df = load_train_data()
    test_df = load_test_data()

    print("\n=== Train Data ===")
    print(train_df.head())
    print("\nTrain shape:", train_df.shape)
    print("\nLabel distribution:")
    print(train_df["label"].value_counts())

    print("\n=== Test Data ===")
    print(test_df.head())
    print("\nTest shape:", test_df.shape)


if __name__ == "__main__":
    inspect_dataset()