from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_loader import load_train_data


MODEL_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_PATH = MODEL_DIR / "emotion_tfidf_logreg.joblib"


def train_baseline_model() -> None:
    df = load_train_data()

    x_train, x_valid, y_train, y_valid = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=30000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("Training baseline emotion classifier...")
    model.fit(x_train, y_train)

    predictions = model.predict(x_valid)

    print("\n=== Validation Result ===")
    print("Accuracy:", accuracy_score(y_valid, predictions))
    print(classification_report(y_valid, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_baseline_model()