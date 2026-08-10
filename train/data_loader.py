import random
import csv
import re
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


# 이보다 짧은 텍스트는 어느 소스에서 왔든 버린다.
#
# 문장 단위 분해에서는 "응.", "네." 같은 맞장구를 걸러내고,
# KorEmpathetic에서는 번역이 깨진 행을 걸러낸다 — situation 컬럼에 "1", "2",
# "생식 얻기", "부모 통과" 같은 값이 들어간 게 33건 있다(영문 번역본의 잔재).
# 어느 쪽이든 감정 라벨을 지탱하지 못하면서 노이즈만 늘린다.
_MIN_TEXT_LEN = 6


def load_aihub_rows(config: TrainConfig) -> list[dict]:
    """AI Hub CSV를 **행 단위**로 읽는다 → [{"label", "texts": [...]}]

    행 하나가 대화 하나다. texts에는 그 대화에서 뽑은 학습 텍스트 변형이 들어간다.

    변형을 두 가지로 두는 이유(합본 + 문장 단위):
      합본만 쓰면 AI Hub 텍스트가 중앙값 104자인데 신체불편 데이터는 26자라,
      모델이 감정이 아니라 "짧으면 신체불편"이라는 길이 지름길을 학습한다.
      실제로 "요새 사는 게 재미가 없어"를 신체불편 0.997로 찍는 걸 확인했다.
      문장 단위만 쓰면 길이는 맞춰지지만 맥락이 사라져 라벨 노이즈가 커진다
      ("어제 병원에 다녀왔어"는 슬픔 대화의 일부지만 그 문장만 보면 중립이다).
      둘 다 넣으면 모든 라벨이 긴 것과 짧은 것을 함께 갖게 돼 길이가 라벨을
      알려주지 못하고, 합본이 남아 맥락 기반 라벨도 보존된다.

    ⚠ 반드시 행 단위로 분할한 뒤에 펼쳐야 한다. 펼친 다음 나누면 같은 대화의
      합본이 학습에, 문장이 보류셋에 들어가 데이터 누출이 된다.

    주의: 이 zip은 원본 감성대화 말뭉치(27만 문장, 소분류 60종, 세대별 분리)의
    축약본이라 감정_소분류도 연령 컬럼도 없다. 노년 발화만 골라낼 수 없다.
    """
    with zipfile.ZipFile(config.aihub_train_zip) as z:
        with z.open("train.csv") as f:
            content = f.read().decode("utf-8-sig")

    rows: list[dict] = []
    skipped = 0
    for row in csv.DictReader(io.StringIO(content)):
        mapped = AIHUB_TO_5.get(row["감정_대분류"].strip())
        sentences = [
            row[f"사람문장{i}"].strip()
            for i in range(1, 5)
            if row.get(f"사람문장{i}", "").strip()
        ]
        if not sentences:
            continue
        if mapped is None:
            skipped += 1
            continue

        texts = [" ".join(sentences)]
        if config.use_sentence_split:
            for s in sentences:
                for piece in re.split(r"(?<=[.!?])\s+", s):
                    piece = piece.strip()
                    if len(piece) >= _MIN_TEXT_LEN:
                        texts.append(piece)
        # 한 문장짜리 대화는 합본과 문장이 같은 문자열이 된다
        rows.append({"label": LABEL2ID[mapped], "texts": list(dict.fromkeys(texts))})

    print(f"[AI Hub] {len(rows)}행 로드 (5종 매핑 제외 {skipped}행)")
    return rows


def expand_rows(rows: list[dict]) -> list[dict]:
    """행 단위 데이터를 학습 샘플로 펼친다."""
    return [{"text": t, "label": r["label"]} for r in rows for t in r["texts"]]


def load_aihub_data(config: TrainConfig) -> list[dict]:
    """AI Hub를 학습 샘플로 바로 펼쳐 반환한다(행 분할이 필요 없을 때)."""
    samples = expand_rows(load_aihub_rows(config))
    print(f"[AI Hub] → {len(samples)}건으로 펼침")
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

    for attr, label in [("somatic_candidates_file", "추출 후보"),
                        ("somatic_generated_file", "LLM 생성분")]:
        path = Path(getattr(config, attr))
        if not path.exists():
            continue
        if _is_reviewed(path):
            more = _read_utterance_file(path)
            texts += more
            print(f"[신체불편] 검수 완료 {label} {len(more)}건")
        else:
            print(f"[신체불편] {path.name}은 검수 전이라 제외 "
                  f'(검수 끝나면 상단을 "# REVIEWED: yes"로 바꿀 것)')

    # 다듬는 과정에서 서로 다른 문장이 같은 표현으로 수렴하기 쉽다.
    # near-duplicate가 쌓이면 모델이 그 표현 하나만 외우므로 완전 중복은 제거한다.
    unique = list(dict.fromkeys(texts))
    if len(unique) < len(texts):
        print(f"[신체불편] 중복 {len(texts) - len(unique)}건 제거")

    print(f"[신체불편] 최종 {len(unique)}건")
    return [{"text": t, "label": LABEL2ID["신체불편"]} for t in unique]


def load_somatic_negatives(config: TrainConfig) -> list[dict]:
    """신체 질문에 대한 정상 응답 → 긍정 라벨.

    "당뇨가 있지만 크게 아픈 데는 없어" 같은 문장이다. 신체불편 데이터는
    전부 신체 단어를 포함하는데 긍정 라벨(AI Hub 기쁨)에는 신체 단어가 거의
    없다. 그대로 학습하면 모델이 "신체 단어가 보이면 신체불편"이라는 지름길을
    잡아서, 통화에서 "무릎은 괜찮아"라고 답해도 healthRisk를 올린다.

    ⚠ 이 파일은 오염되기 쉽다. 자동 추출 때 `없어`를 부정 표현으로 잡았다가
      "돈이 없어", "관심이 없어"까지 긁어와 68건 중 57건이 잘못 들어갔었다.
      아픈 문장이 긍정 라벨로 들어가면 학습이 정면으로 망가지므로,
      추가·수정할 때 부정 대상이 정말 "아픔"인지 매번 확인할 것.
    """
    texts: list[str] = []
    for attr in ("somatic_negative_file", "somatic_negative_generated_file"):
        path = Path(getattr(config, attr))
        if not path.exists():
            continue
        if not _is_reviewed(path):
            print(f"[긍정-신체정상] {path.name}은 검수 전이라 제외")
            continue
        texts += _read_utterance_file(path)
    texts = list(dict.fromkeys(texts))
    print(f"[긍정-신체정상] {len(texts)}건 로드")
    return [{"text": t, "label": LABEL2ID["긍정"]} for t in texts]


def _drop_too_short(samples: list[dict], name: str) -> list[dict]:
    """_MIN_TEXT_LEN 미만 텍스트 제거. 소스마다 깨진 행의 모양이 달라서
    로딩 시점이 아니라 합류 지점에서 공통으로 거른다."""
    kept = [s for s in samples if len(s["text"]) >= _MIN_TEXT_LEN]
    if len(kept) < len(samples):
        print(f"[{name}] {_MIN_TEXT_LEN}자 미만 {len(samples) - len(kept)}건 제외")
    return kept


def _cap_per_label(samples: list[dict], cap: int | None, seed: int, name: str) -> list[dict]:
    """보조 소스를 라벨당 cap건으로 제한한다.

    앞에서부터 자르지 않고 무작위 표집한다. KOTE는 원본 파일 순서에 편향이
    있을 수 있어 앞부분만 쓰면 표현이 한쪽으로 쏠린다. seed로 재현성은 유지한다.
    """
    if cap is None:
        return samples

    by_label: dict[int, list[dict]] = {}
    for s in samples:
        by_label.setdefault(s["label"], []).append(s)

    rng = random.Random(seed)
    capped: list[dict] = []
    trimmed = 0
    for label, group in by_label.items():
        if len(group) > cap:
            trimmed += len(group) - cap
            group = rng.sample(group, cap)
        capped += group

    if trimmed:
        print(f"[{name}] 라벨당 상한 {cap:,}건 적용 — {trimmed:,}건 제외")
    return capped


def resolve_somatic_conflicts(samples: list[dict], somatic: list[dict]) -> list[dict]:
    """신체불편으로 검수된 문장이 다른 라벨로도 들어와 있으면 그쪽을 제거한다.

    somatic_candidates.txt는 AI Hub 문장에서 뽑은 것이라, AI Hub를 문장 단위로도
    학습에 넣는 순간 같은 문장이 두 라벨을 갖게 된다(실측 360건 중 237건, 66%).
    "건강이 안 좋아."가 신체불편이면서 동시에 우울로 들어가면 두 라벨이 서로를
    지운다.

    신체불편을 살리는 이유: AI Hub 라벨은 **대화 전체**에 붙은 것이고, 신체불편은
    사람이 그 **문장 하나**를 보고 "신체 호소다"라고 판정한 것이다. 문장 단위
    학습에서는 문장 단위 판정이 맞다.
    """
    somatic_texts = {s["text"] for s in somatic}
    somatic_label = LABEL2ID["신체불편"]
    # 라벨까지 봐야 한다. 텍스트만 보고 지우면 samples 안에 이미 신체불편으로
    # 들어와 있던 샘플까지 날아간다(노트북 경로에서 실제로 그래서 신체불편이
    # 0건이 됐다).
    kept = [
        s for s in samples
        if s["label"] == somatic_label or s["text"] not in somatic_texts
    ]
    dropped = len(samples) - len(kept)
    if dropped:
        print(f"[충돌 정리] 신체불편으로 검수된 문장이 다른 라벨로도 있던 {dropped}건 제거")
    return kept


def build_dataset(config: TrainConfig) -> list[dict]:
    """전체 학습 데이터 구성."""
    samples = load_aihub_data(config)
    cap = config.max_per_label_per_aux_source

    if config.use_kote:
        kote = _drop_too_short(load_kote_data(), "KOTE")
        samples += _cap_per_label(kote, cap, config.seed, "KOTE")

    if config.use_empathetic:
        emp = _drop_too_short(load_empathetic_data(), "KorEmpathetic")
        samples += _cap_per_label(emp, cap, config.seed, "KorEmpathetic")

    if config.use_somatic_seed:
        somatic = load_somatic_seed(config)
        samples = resolve_somatic_conflicts(samples, somatic)
        samples += somatic
        samples += load_somatic_negatives(config)

    samples = _cap_per_label(samples, config.max_per_label, config.seed, "전체")

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
