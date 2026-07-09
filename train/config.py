from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

LABELS = ["분노", "기쁨", "불안", "당황", "슬픔", "상처"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


@dataclass
class TrainConfig:
    model_name: str = "monologg/koelectra-base-v3-discriminator"

    # 데이터 경로
    aihub_train_zip: str = str(BASE_DIR / "emotionClassification_train.zip")
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
    val_ratio: float = 0.1
    seed: int = 42
