"""학습된 모델로 감정을 분류한다. FastAPI 통합 시 이 클래스를 가져다 쓰면 된다."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import ID2LABEL, LABELS


class EmotionPredictor:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, text: str) -> dict:
        """단일 텍스트 감정 분류.

        Returns:
            {
              "emotion": "슬픔",
              "confidence": 0.87,
              "scores": {"분노": 0.02, "기쁨": 0.01, ..., "슬픔": 0.87, ...}
            }
        """
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=256,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        label_id = int(probs.argmax())

        return {
            "emotion": ID2LABEL[label_id],
            "confidence": round(float(probs[label_id]), 4),
            "scores": {LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
        }

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """다수 텍스트 배치 처리."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt",
                truncation=True, padding=True, max_length=256,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            for p in probs:
                label_id = int(p.argmax())
                results.append({
                    "emotion": ID2LABEL[label_id],
                    "confidence": round(float(p[label_id]), 4),
                    "scores": {LABELS[i]: round(float(v), 4) for i, v in enumerate(p)},
                })
        return results


# 직접 실행 시 간단한 테스트
if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "../model_output"
    predictor = EmotionPredictor(model_path)

    test_texts = [
        "요즘 너무 외롭고 힘들어요. 아무것도 하기 싫어요.",
        "오늘 손주가 놀러와서 정말 기분이 좋았어요.",
        "약을 먹어야 하는데 자꾸 깜빡해서 걱정돼요.",
    ]
    for text in test_texts:
        result = predictor.predict(text)
        print(f"입력: {text}")
        print(f"결과: {result['emotion']} (신뢰도: {result['confidence']})\n")
