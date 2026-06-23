from pathlib import Path
import argparse
import joblib

from label_mapping import to_care_signals, describe_label


MODEL_PATH = Path(__file__).resolve().parent / "outputs" / "emotion_tfidf_logreg.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Please run train_emotion_classifier.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_emotion(text: str) -> dict:
    model = load_model()

    predicted_label = model.predict([text])[0]

    confidence = None
    probabilities = {}

    if hasattr(model, "predict_proba"):
        labels = model.classes_
        probs = model.predict_proba([text])[0]
        probabilities = {
            label: float(prob)
            for label, prob in zip(labels, probs)
        }
        confidence = float(max(probs))

    sorted_probs = sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    top_emotions = [
        {
            "label": label,
            "probability": prob,
            "description": describe_label(label),
            "careSignals": to_care_signals(label),
        }
        for label, prob in sorted_probs[:3]
    ]

    low_confidence = confidence is not None and confidence < 0.5

    merged_care_signals = []
    if low_confidence:
        for emotion in top_emotions:
            for signal in emotion["careSignals"]:
                if signal not in merged_care_signals:
                    merged_care_signals.append(signal)
    else:
        merged_care_signals = to_care_signals(predicted_label)

    return {
        "text": text,
        "primaryEmotion": predicted_label,
        "confidence": confidence,
        "lowConfidence": low_confidence,
        "description": describe_label(predicted_label),
        "careSignals": merged_care_signals,
        "topEmotions": top_emotions,
        "probabilities": probabilities,
        "disclaimer": "본 결과는 의료적 진단이 아니라 보호자 및 복지사의 확인을 돕기 위한 참고 감정 분류 결과입니다.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Input Korean sentence")
    args = parser.parse_args()

    result = predict_emotion(args.text)

    print("\n=== Emotion Prediction Result ===")
    print(f"Input text      : {result['text']}")
    print(f"Emotion         : {result['primaryEmotion']}")
    print(f"Confidence      : {result['confidence']:.4f}" if result["confidence"] else "Confidence      : N/A")
    print(f"Description     : {result['description']}")
    print(f"Care signals    : {result['careSignals']}")
    print(f"Disclaimer      : {result['disclaimer']}")
    print(f"Low confidence  : {result['lowConfidence']}")
    print(f"Top emotions    : {[item['label'] for item in result['topEmotions']]}")

    print("\n=== Class probabilities ===")
    for label, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
        print(f"{label}: {prob:.4f}")


if __name__ == "__main__":
    main()