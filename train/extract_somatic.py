"""AI Hub 감성분류 데이터에서 신체불편 후보 문장을 추출한다.

배경: 신체불편은 세 공개 소스의 감정 라벨 어디에도 없다. 하지만 AI Hub
데이터의 *텍스트 안에는* 신체 호소 문장이 섞여 있다. 감정 라벨만 보고
텍스트를 안 봤을 뿐이다. 그걸 문장 단위로 끌어내는 스크립트다.

KOTE로도 같은 시도를 했으나 실패했다. 40,000건 중 키워드 매치가 1,299건
나왔지만 온라인 댓글 코퍼스라 대부분 오탐이었다("아파트", "이명박",
"거미 다리", "제 입맛으로는"). 1인칭 신체 호소가 거의 없어 후보에서 뺐다.

왜 문장 단위인가:
  data_loader는 사람문장1~4를 이어붙여 하나의 텍스트로 쓴다. 그러면 신체
  문장이 정서 서사에 묻힌다. 반면 우리 통화 답변("몸이 어디 불편하세요?"의
  응답)은 짧은 단문이라, 문장 단위 쪽이 실제 추론 대상과 도메인이 맞는다.

출력 두 가지:
  somatic_candidates.txt   신체불편 라벨 후보
  somatic_negative_candidates.txt
      "크게 아픈 데는 없어", "이제 무릎이 많이 나아서" 같은 부정·회복 문장.
      신체불편이 아니라 긍정 라벨 후보다. 이게 없으면 모델이 신체 단어만
      보이면 전부 신체불편으로 찍는다 — 통화 13번 질문에 "괜찮아"라고
      답하는 정상 응답을 위험 신호로 오분류하게 된다.

⚠ 두 출력 모두 자동 추출 결과라 그대로 학습에 넣으면 안 된다.
  절반가량은 신체가 배경일 뿐 실제 주제가 돈·관계인 문장이다.
  ("아픈 데가 많아지니 치료비가 부담이네" — 이건 신체불편이 아니다)
  사람이 검수해 통과한 것만 data/somatic_seed.txt로 옮겨야 한다.

사용법:
    python extract_somatic.py
"""
import csv
import io
import re
import zipfile
from pathlib import Path

from config import TrainConfig

# 본인의 신체 상태를 가리키는 표현
_BODY = re.compile(
    r"(몸이\s*(아프|안\s*좋|무겁)|허리가|무릎이|관절이|어깨가|다리가\s*(아프|저리)"
    r"|소화가\s*안|속이\s*쓰|잠이\s*안\s*(와|오)|불면|기운이\s*없|기력이\s*없"
    r"|입맛이\s*없|어지러|숨이\s*차|눈이\s*(침침|안\s*보)|귀가\s*안\s*들"
    r"|거동이|몸살|아파서|아프니|아픈\s*데|건강이\s*안\s*좋)"
)

# 타인의 아픔은 본인 신체 신호가 아니다. "아들이 수술을 했어"는 불안이지
# 신체불편이 아니다.
_OTHER = re.compile(
    r"(아들|딸|남편|아내|손녀|손자|엄마|아빠|어머니|아버지|친구|가족|자식"
    r"|며느리|사위|동생|형|누나|언니|오빠|주변\s*사람|회장|의사)"
)

# 정서어가 문장을 주도하면 그건 우울·불안 라벨이 맞다. 신체불편으로
# 가져오면 원래 라벨을 훼손한다.
_EMOTION = re.compile(
    r"(우울|슬프|슬퍼|외로|불안|화가\s*나|짜증|속상|서운|억울|기쁘|행복|무섭|두렵|답답해)"
)

# 부정·회복 표현. 신체불편이 아니라 긍정 라벨 후보로 따로 뺀다.
#
# 주의: 여기서 부정되는 대상이 반드시 "아픔"이어야 한다. 처음에 `없어`를 그냥
# 넣었다가 "돈이 없어", "사람이 없어", "관심이 없어"를 전부 긁어와서 68건 중
# 57건이 오염됐다. 아픈 문장이 긍정 라벨로 들어가면 학습이 정면으로 망가진다.
_NEGATED = re.compile(
    r"(아픈\s*(데|곳|구석)\s*(가|는|이)?\s*없"      # 아픈 데(가) 없다
    r"|이상(은|이)?\s*없"                          # 큰 이상은 없다
    r"|안\s*아파|아프지\s*않"                       # 안 아파 / 아프지 않다
    r"|괜찮|멀쩡|완치|다\s*나았|나아서|건강해져)"
)

# 이중부정 함정: "안 아픈 데가 없다"는 부정이 아니라 "다 아프다"는 뜻이다.
# _NEGATED보다 먼저 걸러야 한다.
_DOUBLE_NEGATIVE = re.compile(r"(안\s*아픈|아프지\s*않은|안아픈)\s*(데|곳|구석)?\s*(가|이)?\s*없")

_MIN_LEN, _MAX_LEN = 6, 60


def _split_sentences(config: TrainConfig) -> list[str]:
    with zipfile.ZipFile(config.aihub_train_zip) as z:
        content = z.open("train.csv").read().decode("utf-8-sig")

    sentences: list[str] = []
    for row in csv.DictReader(io.StringIO(content)):
        for i in range(1, 5):
            text = row.get(f"사람문장{i}", "").strip()
            for s in re.split(r"(?<=[.!?])\s+", text):
                s = s.strip()
                if _MIN_LEN <= len(s) <= _MAX_LEN:
                    sentences.append(s)
    return sentences


def extract(config: TrainConfig = TrainConfig()) -> tuple[list[str], list[str]]:
    """(신체불편 후보, 긍정(부정·회복) 후보)를 반환한다."""
    sentences = _split_sentences(config)
    print(f"문장 {len(sentences):,}개로 분해 (중복 포함)")

    matched = [
        s for s in sorted(set(sentences))
        if _BODY.search(s) and not _OTHER.search(s) and not _EMOTION.search(s)
    ]
    # 이중부정("안 아픈 데가 없다" = 다 아프다)은 부정이 아니라 신체불편 쪽이다
    negated = [s for s in matched
               if _NEGATED.search(s) and not _DOUBLE_NEGATIVE.search(s)]
    somatic = [s for s in matched if s not in set(negated)]

    print(f"본인 신체 문장 {len(matched)}건 → 신체불편 후보 {len(somatic)}건 "
          f"/ 부정·회복 후보 {len(negated)}건")
    return somatic, negated


_HEADER = """# {title}
#
# REVIEWED: no
#   ↑ 검수를 마치면 이 값을 yes로 바꾼다. 그전까지 학습에 들어가지 않는다.
#     (train/data_loader.py의 _is_reviewed 참고)
#
# train/extract_somatic.py가 AI Hub 감성분류 데이터에서 자동 추출한 후보다.
# 절반가량은 신체가 배경일 뿐 실제 주제가 돈·관계인 문장이다.
#
# 검수 방법
#   - 거부: 줄 맨 앞에 #을 붙인다 (삭제하지 말 것 — 왜 뺐는지 이력이 남는다)
#   - 수정: 문장을 직접 고쳐도 된다. 단 두 가지를 지킬 것
#       1) 너무 깔끔하게 다듬지 말 것. 실제 입력은 STT가 받아적은 구어체다.
#          문어체로 정제하면 학습과 실전이 어긋난다.
#       2) 다듬다 보면 서로 다른 문장이 같은 표현으로 수렴한다. 파일이 가나다순이라
#          비슷한 것끼리 뭉쳐 있으니, 같은 꼴이 여럿이면 대표 1~2개만 남길 것.
#
# 판정 기준 한 줄: 통화에서 "몸이 어디 불편하세요?"에 대한 답으로 자연스러운가?
#
# 총 {count}건
"""


def main(config: TrainConfig = TrainConfig()) -> None:
    somatic, negated = extract(config)
    data_dir = Path(config.somatic_seed_file).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    for name, lines, title, target in [
        ("somatic_candidates.txt", somatic, "신체불편 라벨 후보 (검수 대기)", "data/somatic_seed.txt"),
        ("somatic_negative_candidates.txt", negated, "긍정 라벨 후보 — 신체 부정·회복 표현 (검수 대기)", "긍정 라벨 시드"),
    ]:
        path = data_dir / name
        header = _HEADER.format(title=title, target=target, count=len(lines))
        path.write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
        print(f"저장: {path} ({len(lines)}건)")


if __name__ == "__main__":
    main()
