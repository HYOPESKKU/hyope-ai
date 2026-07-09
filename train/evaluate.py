"""학습된 감정 분류 모델을 AI Hub 보류(held-out) 테스트셋으로 평가한다.

emotionClassification_test.zip의 test.csv는 레이블이 없는 제출용 분할이라 채점이 불가능하다.
그래서 레이블이 있는 train.csv에서 계층화(stratified) 방식으로 10%를 떼어낸 보류셋을
테스트셋으로 사용한다. 분할 로직은 노트북(notebooks/train_emotion_classifier.ipynb)과 공유한다.

주의: train.py의 train()은 AI Hub 전체를 학습에 사용하므로, train.py로 학습한 모델을
이 스크립트로 평가하면 보류셋이 학습에 섞여 점수가 부풀려진다.
정직한 평가는 노트북 방식(보류셋 제외 후 학습)으로 학습한 모델에만 유효하다.

사용법:
    python evaluate.py [모델경로]   # 기본: ../model_output
"""
import sys

import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import LABELS, TrainConfig
from data_loader import load_aihub_data
from predict import EmotionPredictor

TEST_RATIO = 0.1


def split_aihub_holdout(
    config: TrainConfig = TrainConfig(), test_ratio: float = TEST_RATIO
) -> tuple[list[dict], list[dict]]:
    """AI Hub 데이터를 (학습용, 보류 테스트용)으로 계층화 분할한다.

    보류셋은 노인 발화 도메인 성능을 재는 게 목적이므로 AI Hub 데이터에서만 뗀다.
    KOTE·KorEmpatheticDialogues는 학습 보강용이라 보류셋에 포함하지 않는다.
    """
    samples = load_aihub_data(config)
    rest, holdout = train_test_split(
        samples,
        test_size=test_ratio,
        random_state=config.seed,
        stratify=[s["label"] for s in samples],
    )
    return rest, holdout


def evaluate(model_path: str, config: TrainConfig = TrainConfig()) -> dict:
    """보류 테스트셋으로 모델을 평가하고 분류 리포트를 출력한다.

    Returns:
        {"texts", "true", "pred", "results", "confusion"} — 노트북의 오류 분석에서 재사용.
    """
    _, holdout = split_aihub_holdout(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = EmotionPredictor(model_path, device=device)

    texts = [s["text"] for s in holdout]
    true = [s["label"] for s in holdout]
    results = predictor.predict_batch(texts)
    pred = [LABELS.index(r["emotion"]) for r in results]

    print(f"보류 테스트셋 {len(holdout)}건 평가 (device={device})")
    print(classification_report(true, pred, target_names=LABELS, digits=4))

    return {
        "texts": texts,
        "true": true,
        "pred": pred,
        "results": results,
        "confusion": confusion_matrix(true, pred),
    }


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "../model_output"
    evaluate(model_path)
