from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from data_loader import load_train_data


MODEL_PATH = Path(__file__).resolve().parent / "outputs" / "emotion_tfidf_logreg.joblib"
REPORT_PATH = Path(__file__).resolve().parent / "outputs" / "evaluation_report.txt"


def evaluate_saved_model() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Please run train_emotion_classifier.py first."
        )

    df = load_train_data()

    _, x_valid, _, y_valid = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(x_valid)

    accuracy = accuracy_score(y_valid, predictions)
    report = classification_report(y_valid, predictions)
    labels = sorted(df["label"].unique())
    matrix = confusion_matrix(y_valid, predictions, labels=labels)

    output_lines = []
    output_lines.append("=== HYOPE Emotion Evaluation Model Report ===")
    output_lines.append("")
    output_lines.append(f"Model path: {MODEL_PATH}")
    output_lines.append(f"Validation accuracy: {accuracy:.4f}")
    output_lines.append("")
    output_lines.append("=== Classification Report ===")
    output_lines.append(report)
    output_lines.append("")
    output_lines.append("=== Confusion Matrix ===")
    output_lines.append("Labels: " + ", ".join(labels))
    output_lines.append(str(matrix))

    output = "\n".join(output_lines)

    print(output)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(output, encoding="utf-8")

    print(f"\nEvaluation report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    evaluate_saved_model()