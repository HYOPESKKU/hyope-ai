import random
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import ID2LABEL, LABEL2ID, LABELS, TrainConfig
from data_loader import build_dataset


class WeightedTrainer(Trainer):
    """클래스 가중치를 적용한 Trainer.

    신체불편은 시드 데이터뿐이라 다른 라벨의 1/50 수준이다. 가중치 없이
    학습하면 그 라벨을 아예 예측하지 않는 게 손실 함수 관점에서 이득이라
    모델이 그렇게 수렴한다. healthRisk 신호가 통째로 죽는다는 뜻이다.
    """

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits,
            labels,
            weight=(
                self.class_weights.to(outputs.logits.device)
                if self.class_weights is not None
                else None
            ),
        )
        return (loss, outputs) if return_outputs else loss


class EmotionDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, max_length: int):
        self.encodings = tokenizer(
            [s["text"] for s in samples],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor([s["label"] for s in samples])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, preds, average="macro")}


def train(config: TrainConfig = TrainConfig()):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # 데이터 준비
    samples = build_dataset(config)
    train_samples, val_samples = train_test_split(
        samples,
        test_size=config.val_ratio,
        random_state=config.seed,
        stratify=[s["label"] for s in samples],
    )
    print(f"학습: {len(train_samples)}건 / 검증: {len(val_samples)}건")

    # 모델 & 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = EmotionDataset(train_samples, tokenizer, config.max_length)
    val_dataset = EmotionDataset(val_samples, tokenizer, config.max_length)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=f"{config.output_dir}/logs",
        seed=config.seed,
        report_to="none",
    )

    # 클래스 가중치 = 전체평균빈도 / 클래스빈도 (역빈도 가중)
    class_weights = None
    if config.use_class_weights:
        dist = Counter(s["label"] for s in train_samples)
        counts = torch.tensor([dist.get(i, 0) for i in range(len(LABELS))], dtype=torch.float)
        # 빈도 0인 클래스가 있으면 0으로 나눠 inf가 된다. 학습이 NaN으로 터지므로 막는다.
        if (counts == 0).any():
            missing = [LABELS[i] for i, c in enumerate(counts) if c == 0]
            raise ValueError(f"학습 데이터에 없는 라벨: {missing}")
        class_weights = counts.mean() / counts
        print("클래스 가중치: " + ", ".join(
            f"{name}={w:.2f}" for name, w in zip(LABELS, class_weights.tolist())
        ))

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"모델 저장 완료: {config.output_dir}")

    # 검증셋 최종 리포트
    result = trainer.predict(val_dataset)
    pred_labels = np.argmax(result.predictions, axis=-1)
    true_labels = [s["label"] for s in val_samples]
    print(classification_report(true_labels, pred_labels, target_names=LABELS))


if __name__ == "__main__":
    train()
