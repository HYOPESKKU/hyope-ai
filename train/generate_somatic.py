"""신체불편 학습·평가 데이터를 LLM으로 생성한다.

왜 생성인가:
  신체불편은 AI Hub·KOTE·KorEmpatheticDialogues 어디에도 없어서 직접 만든
  387건이 전부다. 그 387건으로 잰 F1 0.76은 평가 표본이 39건뿐이라
  95% 신뢰구간이 [0.651, 0.851]로, 개선인지 노이즈인지 구분조차 안 된다.
  실제 미탐률이 18%면 독거노인 건강 모니터링으로는 못 쓴다.

  필터링으로는 이 문제를 못 고친다. 정서 3종은 문장 분해 노이즈가 원인이라
  걸러내면 되지만, 신체불편은 애초에 데이터가 없는 게 원인이라 늘려야 한다.

근거:
  WHO ICOPE 스크리닝 도메인(이동성·영양·시각·청각)과 FRAIL 척도 문항
  (피로·계단·보행·질환·체중감소), 낙상 게이트를 뼈대로 삼는다. 무작정
  생성하는 게 아니라 검증된 척도의 문항을 발화로 바꾸는 것이라 근거가 선다.
  questions.json의 physical_health 블록과 같은 출처다.

⚠ 생성 데이터는 실제 노인 발화가 아니다. 통화 데이터가 쌓이면 교체해야 한다.
⚠ 출력은 검수 전이다. REVIEWED 마커를 yes로 바꿔야 학습에 들어간다.

사용법:
    python generate_somatic.py                 # 기본 수량
    python generate_somatic.py --per-domain 40
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from config import TrainConfig

load_dotenv(Path(__file__).resolve().parent.parent.parent / "hyope-ai-server" / ".env")

MODEL = os.environ.get("GEN_MODEL", "gpt-5.6-terra")

# 도메인별로 나눠 생성한다. 한 프롬프트로 몰아서 뽑으면 표현이 한쪽으로 쏠려
# near-duplicate가 쌓이고, 그러면 모델이 그 표현만 외운다.
DOMAINS = {
    "이동성": "계단 오르내리기, 오래 걷기, 앉았다 일어서기, 지팡이·보행기 의존, 외출이 힘들어짐 (ICOPE 이동성 / FRAIL 계단·보행)",
    "피로": "기운 없음, 쉽게 지침, 아침에 일어나기 힘듦, 하루 종일 누워 있게 됨 (FRAIL 피로)",
    "영양": "입맛 없음, 체중이 줄어 옷이 헐렁함, 씹기·삼키기 어려움, 사레들림, 맛을 모르겠음 (ICOPE 영양 / FRAIL 체중감소)",
    "시각": "눈이 침침함, 글씨·자막이 안 보임, 눈부심, 비문증 (ICOPE 시각)",
    "청각": "잘 안 들려 되묻게 됨, 이명, 말이 웅얼거리게 들림 (ICOPE 청각)",
    "낙상": "넘어진 적 있음, 서 있거나 걸을 때 불안정함, 넘어질까 걱정됨 (CDC STEADI 낙상 게이트)",
    "통증": "무릎·허리·어깨·관절·목·발바닥 통증, 쑤심, 결림, 저림",
    "소화": "속쓰림, 체함, 더부룩함, 변비, 설사, 트림",
    "순환호흡": "어지러움, 숨참, 가슴 답답함, 두근거림, 손발 저림, 부종",
    "수면": "잠들기 어려움, 자꾸 깸, 새벽에 깨서 못 잠, 자도 개운하지 않음",
    "비뇨": "소변이 자주 마려움, 밤에 여러 번 일어남, 요실금, 기저귀",
    "만성질환": "고혈압·당뇨·관절염 등으로 약을 계속 먹음, 병원 정기 방문, 약이 잘 안 듣는 느낌",
    "구강피부": "이가 안 좋아 딱딱한 걸 못 씹음, 입 마름, 피부 가려움, 상처가 잘 안 아묾",
}

_STYLE = """
작성 규칙 (반드시 지킬 것):
- 70~90대 어르신이 전화로 말하는 **구어체**. 문어체 금지.
- 통화에서 "몸이 어디 불편하신 데 없으세요?"에 대한 답으로 자연스러울 것.
- 한 문장, 8~45자. 종결어미를 다양하게 (~어, ~아, ~네, ~구먼, ~요, ~습니다, 반말/존댓말 섞기).
- **표현을 최대한 다르게**. 같은 증상이라도 서술 방식을 바꿀 것.
- "아프다"라는 말만 반복하지 말 것. 증상을 묘사하는 문장도 많이 넣을 것
  (예: "계단이 힘들어" / "옷이 자꾸 흘러내려" / "자꾸 되묻게 돼").
- 감정 표현(외롭다, 우울하다, 서럽다, 화난다)은 넣지 말 것.
  신체 상태만 말할 것. 감정이 섞이면 다른 라벨과 충돌한다.
- 돈·가족·관계 이야기를 주제로 삼지 말 것. 신체가 문장의 주제여야 한다.
"""

SOMATIC_PROMPT = """독거노인 안부 전화에서 어르신이 신체 불편을 호소하는 발화를 만들어라.

증상 영역: {domain}
구체적으로: {desc}
{style}
JSON 배열로만 출력하라. 설명 금지. 형식: ["문장1", "문장2", ...]
{n}개를 만들어라."""

NEGATIVE_PROMPT = """독거노인 안부 전화에서 "몸이 어디 불편하신 데 없으세요?"라는 질문에
어르신이 **괜찮다고 답하는** 발화를 만들어라.

증상 영역: {domain} — 이 영역을 언급하되 **문제가 없다/나았다**고 답해야 한다.
구체적으로: {desc}
{style}
- 반드시 신체 부위나 증상 이름을 언급하되, 그게 괜찮다고 말해야 한다.
  (예: "무릎은 요새 좀 나아졌어", "당뇨는 있어도 크게 불편한 건 없어")
- 이중부정 금지. "안 아픈 데가 없어"는 다 아프다는 뜻이라 정반대다.

JSON 배열로만 출력하라. 설명 금지. 형식: ["문장1", "문장2", ...]
{n}개를 만들어라."""


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _ask(client: OpenAI, prompt: str, retries: int = 4) -> list[str]:
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4000,
            )
            txt = (r.choices[0].message.content or "").strip()
            m = re.search(r"\[.*\]", txt, re.S)
            if not m:
                continue
            return [s.strip() for s in json.loads(m.group(0)) if isinstance(s, str)]
        except Exception:
            time.sleep(2 ** attempt)
    return []


def generate(kind: str, per_domain: int) -> list[tuple[str, str]]:
    """(도메인, 문장) 목록을 만든다."""
    template = SOMATIC_PROMPT if kind == "somatic" else NEGATIVE_PROMPT
    client = _client()

    def one(item):
        domain, desc = item
        out = _ask(client, template.format(
            domain=domain, desc=desc, style=_STYLE, n=per_domain))
        print(f"  {kind}/{domain}: {len(out)}건", flush=True)
        return [(domain, s) for s in out]

    with ThreadPoolExecutor(max_workers=6) as ex:
        return [p for group in ex.map(one, DOMAINS.items()) for p in group]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=40)
    ap.add_argument("--neg-per-domain", type=int, default=12)
    ap.add_argument("--eval-ratio", type=float, default=0.25)
    args = ap.parse_args()

    cfg = TrainConfig()
    data_dir = Path(cfg.somatic_seed_file).parent

    # 기존 데이터와 겹치면 평가가 오염된다
    existing = set()
    for f in ("somatic_seed.txt", "somatic_candidates.txt", "somatic_negative_candidates.txt"):
        p = data_dir / f
        if p.exists():
            existing |= {l.strip() for l in p.read_text(encoding="utf-8").splitlines()
                         if l.strip() and not l.startswith("#")}

    for kind, per_domain, out_train, out_eval in [
        ("somatic", args.per_domain, "somatic_generated.txt", "somatic_eval.txt"),
        ("negative", args.neg_per_domain, "somatic_negative_generated.txt", None),
    ]:
        print(f"\n=== {kind} 생성 ===")
        pairs = generate(kind, per_domain)

        seen, uniq = set(), []
        for domain, s in pairs:
            if s in seen or s in existing or not (8 <= len(s) <= 60):
                continue
            seen.add(s)
            uniq.append((domain, s))
        print(f"  중복·기존중복·길이 제외 후 {len(uniq)}건")

        # 평가셋은 도메인 계층화로 떼어낸다. 학습셋과 같은 도메인 분포여야
        # 평가가 편향되지 않는다.
        by_domain: dict[str, list[str]] = {}
        for d, s in uniq:
            by_domain.setdefault(d, []).append(s)

        train_lines, eval_lines = [], []
        for d, ss in by_domain.items():
            k = int(len(ss) * args.eval_ratio) if out_eval else 0
            eval_lines += ss[:k]
            train_lines += ss[k:]

        _write(data_dir / out_train, kind, train_lines, is_eval=False)
        if out_eval:
            _write(data_dir / out_eval, kind, eval_lines, is_eval=True)


def _write(path: Path, kind: str, lines: list[str], is_eval: bool) -> None:
    role = ("평가 전용 — 학습에 절대 넣지 말 것" if is_eval
            else ("신체불편 라벨" if kind == "somatic" else "긍정 라벨 (신체 정상 응답)"))
    header = f"""# {role}
#
# REVIEWED: no
#   ↑ 검수를 마치면 yes로 바꾼다. 그전까지 사용되지 않는다.
#
# train/generate_somatic.py가 LLM으로 생성한 데이터다.
# WHO ICOPE 도메인과 FRAIL 척도 문항을 뼈대로 삼았다(questions.json의
# physical_health와 같은 출처).
#
# ⚠ 실제 노인 발화가 아니다. 통화 데이터가 쌓이면 교체해야 한다.
# ⚠ 생성물이라 어색하거나 중복에 가까운 문장이 섞인다. 거부는 줄 앞에 #.
#
# 총 {len(lines)}건
"""
    path.write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"  저장: {path.name} ({len(lines)}건)")


if __name__ == "__main__":
    main()
