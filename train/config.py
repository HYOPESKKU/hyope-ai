from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# HYOPE 돌봄 라벨 5종.
#
# AI Hub 감성분류 6종(분노/기쁨/불안/당황/슬픔/상처)을 그대로 쓰지 않는 이유:
# 그 라벨은 일반 감성 분류용이라 우리가 실제로 필요한 신호와 어긋난다.
#   - "슬픔"과 "상처"를 나눠봐야 돌봄 조치가 같다 → 우울로 병합
#   - "당황"은 상황적 난처함이라 돌봄 위험 신호가 아니다 → 제외
#   - 정작 필요한 "신체불편"이 6종에 없다 → 신설
#
# 위험도 지표와의 연결:
#   우울 → depressionScore   불안·분노 → emotionalStability
#   신체불편 → healthRisk    긍정 → 전체 완충
LABELS = ["우울", "불안", "분노", "긍정", "신체불편"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

LABEL_DESCRIPTIONS = {
    "우울": "우울감, 외로움, 무기력, 소외감, 상실감",
    "불안": "불안, 걱정, 두려움, 초조함",
    "분노": "화남, 짜증, 억울함, 불만",
    "긍정": "기쁨, 안도, 만족, 편안함",
    "신체불편": "통증, 소화·수면 문제, 어지러움, 기력 저하 등 신체 증상 호소",
}

# 감정 라벨 → 보호자·복지사 확인이 필요한 돌봄 플래그.
# feature/evaluation-model 브랜치의 care_signal 개념을 5종 체계로 옮긴 것.
CARE_SIGNAL_MAP = {
    "우울": ["depression_related", "loneliness_check_recommended"],
    "불안": ["anxiety_related", "care_check_recommended"],
    "분노": ["negative_emotion", "care_check_recommended"],
    "긍정": ["positive_emotion"],
    "신체불편": ["health_related", "physical_check_recommended"],
}


@dataclass
class TrainConfig:
    model_name: str = "monologg/koelectra-base-v3-discriminator"

    # 데이터 경로
    aihub_train_zip: str = str(BASE_DIR / "emotionClassification_train.zip")
    somatic_seed_file: str = str(BASE_DIR / "data" / "somatic_seed.txt")
    # 자동 추출 후보. 파일 상단이 "# REVIEWED: yes"일 때만 학습에 들어간다.
    somatic_candidates_file: str = str(BASE_DIR / "data" / "somatic_candidates.txt")
    output_dir: str = str(BASE_DIR / "model_output")

    # 학습 하이퍼파라미터
    max_length: int = 256
    batch_size: int = 32
    learning_rate: float = 3e-5
    epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    early_stopping_patience: int = 2

    # 데이터 구성
    use_kote: bool = True                  # KOTE 44종 공개 데이터
    use_empathetic: bool = True            # KorEmpatheticDialogues 공개 데이터
    use_somatic_seed: bool = True          # 신체불편 시드 (직접 작성, 검수 필요)

    # 보조 소스(KOTE·KorEmpathetic)의 라벨당 상한.
    #
    # 상한이 없으면 KOTE 혼자 39,355건으로 전체의 62%를 차지하고, 그중 분노가
    # 19,911건이라 전체 분노의 82%가 온라인 댓글에서 온다. 반면 노인 페르소나가
    # 있는 유일한 소스인 AI Hub는 13.4%로 밀린다.
    # 평가는 AI Hub 보류셋으로 하므로 학습 분포와 평가 분포가 어긋나게 된다.
    #
    # None이면 상한 없음.
    max_per_label_per_aux_source: int | None = 2000

    val_ratio: float = 0.1
    seed: int = 42

    # 신체불편은 시드 데이터뿐이라 다른 라벨보다 압도적으로 적다.
    # 클래스 가중치 없이 학습하면 모델이 이 라벨을 거의 예측하지 않는다.
    use_class_weights: bool = True
