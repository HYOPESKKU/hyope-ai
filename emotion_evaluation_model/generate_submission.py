from pathlib import Path

import joblib
import pandas as pd

from data_loader import load_test_data


MODEL_PATH = Path(__file__).resolve().parent / "outputs" / "emotion_tfidf_logreg.joblib"
OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "submission.csv"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Please run train_emotion_classifier.py first."
        )

    return joblib.load(MODEL_PATH)


def generate_submission() -> None:
    model = load_model()
    test_df = load_test_data()

    predictions = model.predict(test_df["text"])

    submission_df = pd.DataFrame(
        {
            "ID": test_df["ID"],
            "감정_대분류": predictions,
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("=== Submission Generated ===")
    print(f"Rows: {len(submission_df)}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\n=== Preview ===")
    print(submission_df.head())


if __name__ == "__main__":
    generate_submission()