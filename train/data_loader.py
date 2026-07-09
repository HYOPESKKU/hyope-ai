import csv
import io
import zipfile

from datasets import load_dataset

from config import LABEL2ID, TrainConfig
from label_mapper import EMPATHETIC_TO_6, KOTE_TO_6


def load_aihub_data(config: TrainConfig) -> list[dict]:
    """AI Hub 노년층 감성 분류 CSV 로딩.

    CSV 구조: 성별, 감정_대분류, 사람문장1~4, 시스템응답1~4
    사람문장 1~4를 이어붙여 하나의 텍스트로 사용한다.
    """
    with zipfile.ZipFile(config.aihub_train_zip) as z:
        with z.open("train.csv") as f:
            content = f.read().decode("utf-8-sig")

    samples = []
    for row in csv.DictReader(io.StringIO(content)):
        sentences = [
            row[f"사람문장{i}"].strip()
            for i in range(1, 5)
            if row.get(f"사람문장{i}", "").strip()
        ]
        text = " ".join(sentences)
        label = row["감정_대분류"].strip()
        if text and label in LABEL2ID:
            samples.append({"text": text, "label": LABEL2ID[label]})

    print(f"[AI Hub] {len(samples)}건 로드")
    return samples


def load_kote_data() -> list[dict]:
    """KOTE 44종 감정 공개 데이터 로딩 → 6종 매핑.

    multi-label이므로 첫 번째로 매핑되는 레이블만 사용한다.
    HuggingFace: searle-j/kote
    """
    ds = load_dataset("searle-j/kote", split="train")
    samples = []
    for row in ds:
        text = row["input"].strip()
        raw_labels = row["output"] if isinstance(row["output"], list) else [row["output"]]
        for raw in raw_labels:
            mapped = KOTE_TO_6.get(raw)
            if mapped and text:
                samples.append({"text": text, "label": LABEL2ID[mapped]})
                break

    print(f"[KOTE] {len(samples)}건 로드")
    return samples


def load_empathetic_data() -> list[dict]:
    """KorEmpatheticDialogues 32종 감정 공개 데이터 로딩 → 6종 매핑.

    EmpatheticDialogues 번역본. context 컬럼이 감정 레이블.
    HuggingFace: passing2961/KorEmpatheticDialogues
    """
    ds = load_dataset("passing2961/KorEmpatheticDialogues", split="train")
    samples = []
    for row in ds:
        # 텍스트: prompt 또는 utterance 컬럼
        text = (row.get("prompt") or row.get("utterance") or "").strip()
        # 감정: context 컬럼 (영문 소문자)
        raw_label = (row.get("context") or "").strip().lower()
        mapped = EMPATHETIC_TO_6.get(raw_label)
        if mapped and text:
            samples.append({"text": text, "label": LABEL2ID[mapped]})

    print(f"[KorEmpathetic] {len(samples)}건 로드")
    return samples


def build_dataset(config: TrainConfig) -> list[dict]:
    """전체 학습 데이터 구성."""
    samples = load_aihub_data(config)

    if config.use_kote:
        samples += load_kote_data()

    if config.use_empathetic:
        samples += load_empathetic_data()

    print(f"[전체] {len(samples)}건")
    return samples
