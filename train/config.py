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
    # "아픈 데는 없어" 같은 정상 응답 → 긍정 라벨.
    # 신체불편 데이터가 전부 신체 단어를 포함하는 반면 긍정에는 거의 없어서,
    # 이게 없으면 모델이 "신체 단어 = 신체불편"이라는 지름길을 학습한다.
    somatic_negative_file: str = str(BASE_DIR / "data" / "somatic_negative_candidates.txt")
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

    # AI Hub 대화를 합본(사람문장1~4 이어붙이기)과 문장 단위로 **둘 다** 학습에 넣는다.
    # 합본만 쓰면 AI Hub가 중앙값 104자, 신체불편 데이터가 26자라 모델이 감정이 아니라
    # "짧으면 신체불편"이라는 길이 지름길을 학습한다(실측: "요새 사는 게 재미가 없어"를
    # 신체불편 0.997로 오분류). 실제 추론 입력은 통화 한 턴의 짧은 답변이므로
    # 짧은 변형이 오히려 실전에 가깝다.
    use_sentence_split: bool = True

    # 보조 소스(KOTE·KorEmpathetic)의 라벨당 상한.
    #
    # 상한이 없으면 KOTE 혼자 39,355건으로 전체의 62%를 차지하고, 그중 분노가
    # 19,911건이라 전체 분노의 82%가 온라인 댓글에서 온다. 반면 노인 페르소나가
    # 있는 유일한 소스인 AI Hub는 13.4%로 밀린다.
    # 평가는 AI Hub 보류셋으로 하므로 학습 분포와 평가 분포가 어긋나게 된다.
    #
    # None이면 상한 없음.
    max_per_label_per_aux_source: int | None = 2000

    # 전체 합친 뒤 라벨당 상한. 문장 단위 분해로 AI Hub가 4배가 되면서 불균형이
    # 20:1에서 50:1로 나빠졌다(신체불편은 387건 그대로). 가중치가 27까지 오르면
    # 모델이 소수 클래스를 과대예측하는 쪽으로 쏠린다 — 가중치 13이던 이전
    # 학습에서도 길이 지름길을 세게 파고들었다.
    # 387 x 20 ≈ 8,000으로 잡아 불균형을 이전 수준으로 되돌린다.
    max_per_label: int | None = 8000

    val_ratio: float = 0.1
    seed: int = 42

    # 신체불편은 시드 데이터뿐이라 다른 라벨보다 압도적으로 적다.
    # 클래스 가중치 없이 학습하면 모델이 이 라벨을 거의 예측하지 않는다.
    use_class_weights: bool = True
