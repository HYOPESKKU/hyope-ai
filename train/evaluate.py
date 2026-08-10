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
from data_loader import (
    expand_rows,
    load_aihub_rows,
    load_somatic_negatives,
    load_somatic_seed,
)
from predict import EmotionPredictor

TEST_RATIO = 0.1


def split_aihub_holdout(
    config: TrainConfig = TrainConfig(), test_ratio: float = TEST_RATIO
) -> tuple[list[dict], list[dict]]:
    """평가용 데이터를 (학습용, 보류 테스트용)으로 계층화 분할한다.

    보류셋은 노인 발화 도메인 성능을 재는 게 목적이라 AI Hub에서 떼는 것이
    원칙이다. KOTE·KorEmpatheticDialogues는 학습 보강용이라 제외한다.

    다만 신체불편은 AI Hub에 0건이라 AI Hub만 쓰면 보류셋에 그 라벨이 아예
    없게 된다. 그러면 5종 중 4종만 채점되는데, 정작 healthRisk와 직결되는
    라벨이 측정에서 빠지는 셈이다. 그래서 시드에서도 같은 비율로 떼어 넣는다.

    ⚠ 신체불편 점수는 시드 데이터 자기 자신에 대한 점수라 다른 라벨과 같은
      의미로 읽으면 안 된다. 시드는 작성된 문장이라 실제 통화 발화보다 쉽다.
    """
    # ⚠ 반드시 행 단위로 나눈 뒤 펼친다. AI Hub 한 대화가 합본과 문장 변형으로
    #   여러 샘플이 되는데, 펼친 뒤에 나누면 같은 대화의 합본이 학습에, 문장이
    #   보류셋에 들어가 데이터 누출이 된다.
    rows = load_aihub_rows(config)
    if config.use_somatic_seed:
        somatic = load_somatic_seed(config)
        # somatic_candidates는 AI Hub 문장에서 뽑은 것이라 문장 단위 분해를 켜면
        # 같은 문장이 두 라벨로 들어온다. 행 안에서 그 문장을 걷어낸다.
        somatic_texts = {s["text"] for s in somatic}
        for r in rows:
            r["texts"] = [t for t in r["texts"] if t not in somatic_texts]
        rows = [r for r in rows if r["texts"]]

        # 신체불편·정상응답은 이미 문장 단위라 한 건이 곧 한 행이다.
        # 신체 정상 응답을 보류셋에 넣는 이유: 이 구분 실패가 곧 healthRisk
        # 오작동이라 반드시 측정 대상이어야 한다.
        rows += [{"label": s["label"], "texts": [s["text"]]}
                 for s in somatic + load_somatic_negatives(config)]

    rest_rows, holdout_rows = train_test_split(
        rows,
        test_size=test_ratio,
        random_state=config.seed,
        stratify=[r["label"] for r in rows],
    )
    return _drop_cross_split_duplicates(expand_rows(rest_rows), expand_rows(holdout_rows))


def _drop_cross_split_duplicates(
    train: list[dict], holdout: list[dict]
) -> tuple[list[dict], list[dict]]:
    """학습셋과 보류셋에 똑같이 나타나는 문장을 정리한다.

    행 단위로 나눠도 서로 다른 대화에 같은 문장이 들어있는 경우가 있다
    ("너무 기쁘다.", "밤에 잠이 안 와." 같은 흔한 표현). 실측 8.9%.

    두 가지로 갈린다.
      - 라벨이 같으면: 학습에서 본 문장을 그대로 채점하는 셈이라 점수가 부풀려진다.
        보류셋에서만 뺀다.
      - 라벨이 다르면: 우리 라벨 체계 자체의 모호함이다. "밤에 잠이 안 와"가
        한 대화에서는 불안, 다른 대화에서는 신체불편으로 붙어 있다. 모델이
        무엇을 예측하든 틀리므로 학습·평가 양쪽에서 뺀다.
    """
    train_labels: dict[str, set[int]] = {}
    for s in train:
        train_labels.setdefault(s["text"], set()).add(s["label"])

    ambiguous = {
        s["text"] for s in holdout
        if s["text"] in train_labels and s["label"] not in train_labels[s["text"]]
    }
    kept_holdout = [
        s for s in holdout
        if s["text"] not in train_labels and s["text"] not in ambiguous
    ]
    kept_train = [s for s in train if s["text"] not in ambiguous]

    dropped_h = len(holdout) - len(kept_holdout)
    if dropped_h:
        print(f"[중복 정리] 보류셋에서 {dropped_h}건 제외 "
              f"(라벨 상충 {len(ambiguous)}종은 학습에서도 제외)")
    return kept_train, kept_holdout


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
