import csv
import io
import urllib.request
import zipfile

from datasets import load_dataset

from config import LABEL2ID, TrainConfig
from label_mapper import EMPATHETIC_TO_6, KOTE_TO_6

# searle-j/kote는 HF Hub에 커스텀 로딩 스크립트(kote.py)로만 올라와 있고
# 최신 datasets 라이브러리는 스크립트 방식을 더 이상 지원하지 않는다.
# kote.py가 실제로 하는 일(GitHub에서 dichotomized TSV 받기)을 그대로 재현한다.
_KOTE_LABELS = [
    "불평/불만", "환영/호의", "감동/감탄", "지긋지긋", "고마움", "슬픔", "화남/분노",
    "존경", "기대감", "우쭐댐/무시함", "안타까움/실망", "비장함", "의심/불신", "뿌듯함",
    "편안/쾌적", "신기함/관심", "아껴주는", "부끄러움", "공포/무서움", "절망", "한심함",
    "역겨움/징그러움", "짜증", "어이없음", "없음", "패배/자기혐오", "귀찮음", "힘듦/지침",
    "즐거움/신남", "깨달음", "죄책감", "증오/혐오", "흐뭇함(귀여움/예쁨)", "당황/난처", "경악",
    "부담/안_내킴", "서러움", "재미없음", "불쌍함/연민", "놀람", "행복", "불안/걱정", "기쁨",
    "안심/신뢰",
]
_KOTE_TRAIN_URL = "https://raw.githubusercontent.com/searle-j/KOTE/main/train.tsv"


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
    GitHub: searle-j/KOTE (dichotomized train split)
    """
    with urllib.request.urlopen(_KOTE_TRAIN_URL) as resp:
        content = resp.read().decode("utf-8")

    samples = []
    reader = csv.DictReader(io.StringIO(content), delimiter="\t", fieldnames=["ID", "text", "labels"])
    for row in reader:
        text = row["text"].strip()
        if not text:
            continue
        for idx in row["labels"].split(","):
            mapped = KOTE_TO_6.get(_KOTE_LABELS[int(idx)])
            if mapped:
                samples.append({"text": text, "label": LABEL2ID[mapped]})
                break

    print(f"[KOTE] {len(samples)}건 로드")
    return samples


def load_empathetic_data() -> list[dict]:
    """KorEmpatheticDialogues 32종 감정 공개 데이터 로딩 → 6종 매핑.

    EmpatheticDialogues 번역본. emotion 컬럼이 감정 레이블(영문 32종).
    HuggingFace: passing2961/KorEmpatheticDialogues
    """
    ds = load_dataset("passing2961/KorEmpatheticDialogues", split="train")
    samples = []
    for row in ds:
        # 텍스트: 상황 설명(situation) 컬럼
        text = (row.get("situation") or "").strip()
        # 감정: emotion 컬럼 (영문 소문자)
        raw_label = (row.get("emotion") or "").strip().lower()
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
