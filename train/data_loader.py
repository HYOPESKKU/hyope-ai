import csv
import io
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from config import LABEL2ID, LABELS, TrainConfig
from label_mapper import AIHUB_TO_5, EMPATHETIC_TO_5, KOTE_TO_5

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
    """AI Hub 감성 분류 CSV 로딩 → 돌봄 라벨 5종 매핑.

    CSV 구조: 성별, 감정_대분류, 사람문장1~4, 시스템응답1~4
    사람문장 1~4를 이어붙여 하나의 텍스트로 사용한다.

    주의: 이 zip은 원본 감성대화 말뭉치(27만 문장, 소분류 60종, 세대별 분리)의
    축약본이라 감정_소분류도 연령 컬럼도 없다. 노년 발화만 골라낼 수 없다.
    """
    with zipfile.ZipFile(config.aihub_train_zip) as z:
        with z.open("train.csv") as f:
            content = f.read().decode("utf-8-sig")

    samples = []
    skipped = 0
    for row in csv.DictReader(io.StringIO(content)):
        sentences = [
            row[f"사람문장{i}"].strip()
            for i in range(1, 5)
            if row.get(f"사람문장{i}", "").strip()
        ]
        text = " ".join(sentences)
        mapped = AIHUB_TO_5.get(row["감정_대분류"].strip())
        if not text:
            continue
        if mapped is None:
            skipped += 1
            continue
        samples.append({"text": text, "label": LABEL2ID[mapped]})

    print(f"[AI Hub] {len(samples)}건 로드 (5종 매핑 제외 {skipped}건)")
    return samples


def load_kote_data() -> list[dict]:
    """KOTE 44종 감정 공개 데이터 로딩 → 5종 매핑.

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
            mapped = KOTE_TO_5.get(_KOTE_LABELS[int(idx)])
            if mapped:
                samples.append({"text": text, "label": LABEL2ID[mapped]})
                break

    print(f"[KOTE] {len(samples)}건 로드")
    return samples


def load_empathetic_data() -> list[dict]:
    """KorEmpatheticDialogues 32종 감정 공개 데이터 로딩 → 5종 매핑.

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
        mapped = EMPATHETIC_TO_5.get(raw_label)
        if mapped and text:
            samples.append({"text": text, "label": LABEL2ID[mapped]})

    print(f"[KorEmpathetic] {len(samples)}건 로드")
    return samples


def _read_utterance_file(path: Path) -> list[str]:
    """한 줄에 발화 하나. 빈 줄과 #으로 시작하는 줄은 무시한다.

    검수는 거부할 줄 앞에 #을 붙이는 방식이라, 주석 처리된 줄이 곧 탈락분이다.
    """
    return [
        line
        for line in (l.strip() for l in path.read_text(encoding="utf-8").splitlines())
        if line and not line.startswith("#")
    ]


def _is_reviewed(path: Path) -> bool:
    """파일 상단의 "# REVIEWED: yes" 마커를 확인한다.

    자동 추출 후보를 검수 없이 학습에 넣으면 절반가량이 오염 데이터다.
    실수로 섞이지 않도록 명시적 마커를 요구한다.
    """
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if line.strip().lower().replace(" ", "").startswith("#reviewed:"):
            return line.split(":", 1)[1].strip().lower() in ("yes", "y", "true", "완료")
    return False


def load_somatic_seed(config: TrainConfig) -> list[dict]:
    """신체불편 데이터 로딩 — 직접 작성한 시드 + 검수 완료된 자동 추출 후보.

    세 공개 소스의 감정 라벨 어디에도 신체 증상 호소가 없어서 만든 데이터다.
    실제 노인 발화가 아니므로 통화 데이터가 쌓이면 교체해야 한다.
    """
    texts: list[str] = []

    seed_path = Path(config.somatic_seed_file)
    if seed_path.exists():
        seed_texts = _read_utterance_file(seed_path)
        texts += seed_texts
        print(f"[신체불편] 시드 {len(seed_texts)}건")
    else:
        print(f"[신체불편] 시드 파일 없음: {seed_path}")

    cand_path = Path(config.somatic_candidates_file)
    if cand_path.exists():
        if _is_reviewed(cand_path):
            cand_texts = _read_utterance_file(cand_path)
            texts += cand_texts
            print(f"[신체불편] 검수 완료 후보 {len(cand_texts)}건")
        else:
            print(f"[신체불편] {cand_path.name}은 검수 전이라 제외 "
                  f'(검수 끝나면 상단을 "# REVIEWED: yes"로 바꿀 것)')

    # 다듬는 과정에서 서로 다른 문장이 같은 표현으로 수렴하기 쉽다.
    # near-duplicate가 쌓이면 모델이 그 표현 하나만 외우므로 완전 중복은 제거한다.
    unique = list(dict.fromkeys(texts))
    if len(unique) < len(texts):
        print(f"[신체불편] 중복 {len(texts) - len(unique)}건 제거")

    print(f"[신체불편] 최종 {len(unique)}건")
    return [{"text": t, "label": LABEL2ID["신체불편"]} for t in unique]


def build_dataset(config: TrainConfig) -> list[dict]:
    """전체 학습 데이터 구성."""
    samples = load_aihub_data(config)

    if config.use_kote:
        samples += load_kote_data()

    if config.use_empathetic:
        samples += load_empathetic_data()

    if config.use_somatic_seed:
        samples += load_somatic_seed(config)

    dist = Counter(s["label"] for s in samples)
    print(f"[전체] {len(samples)}건")
    for i, name in enumerate(LABELS):
        n = dist.get(i, 0)
        print(f"  {name:<6} {n:>7,}건 ({n / len(samples) * 100:5.2f}%)")

    # 신체불편은 시드뿐이라 극단적으로 적다. 조용히 넘어가면 모델이 이 라벨을
    # 사실상 예측하지 않는 채로 학습이 끝나므로 눈에 띄게 경고한다.
    rarest = min(dist.values()) if dist else 0
    if rarest and max(dist.values()) / rarest > 20:
        print(f"  ⚠ 클래스 불균형 {max(dist.values()) / rarest:.0f}:1 — "
              f"config.use_class_weights 확인 필요")

    return samples
