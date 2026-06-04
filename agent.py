import asyncio
import json
import os
from dotenv import load_dotenv
from clawops.agent import ClawOpsAgent, OpenAIRealtime

load_dotenv()  # .env 파일에서 CLAWOPS_API_KEY, OPENAI_API_KEY, CLAWOPS_ACCOUNT_ID 로드

# ─────────────────────────────────────────────
# 시스템 프롬프트
# AI 상담 전화의 대화 흐름을 정의한다.
# 순서: 복약 확인 → 식사 확인 → 기분/우울 선별(PHQ-9 기반) → 인지 기능(지남력) → 마무리
# 말투는 따뜻하고 쉬운 단어로, 의학적 진단은 하지 않는다.
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 어르신의 건강을 돌보는 친절한 AI 상담 전화 도우미입니다. 통화가 시작되면 다음 순서로 진행하세요.

1. **복약 확인**: "안녕하세요, 오늘 약 드셨나요?"
2. **식사 확인**: "오늘 식사는 하셨나요?"
3. **기분 확인**: 식사·복약 응답 후 자연스럽게 "요즘 기분이 어떠세요?" 라고 여쭤보며 대화를 이어가세요.
4. **우울 증상 선별**: 대화 흐름에 맞춰 PHQ-9 기반의 질문(잠을 잘 못 자신다든지, 기운이 없다든지 등)을 자연스러운 말투로 2~3가지 여쭤보세요.
5. **인지 기능 확인**: "오늘이 몇 월 며칠인지 기억하세요?" 처럼 간단한 지남력 질문을 1~2가지 드리세요.
6. **마무리**: 매 통화마다 아래 중에서 무작위로 하나 골라 따뜻하게 마무리하세요. 비슷한 상황이더라도 매번 다른 멘트를 사용하세요.
   - "오늘도 건강하게 잘 지내셨네요, 내일도 좋은 하루 되세요!"
   - "어르신 덕분에 저도 기분이 좋아졌어요, 항상 건강하세요!"
   - "오늘 통화해서 정말 반가웠어요, 맛있는 거 많이 드세요!"
   - "날씨가 좋으니 오늘은 바람이라도 좀 쐬세요, 항상 응원해요!"
   - "어르신 목소리 듣고 나니 저도 힘이 나요, 좋은 하루 되세요!"
   - "오늘 하루도 수고 많으셨어요, 푹 쉬시고 내일 또 통화해요!"

말투는 항상 따뜻하고 천천히, 쉬운 단어로 말하세요. 절대 의학적 진단을 내리지 말고, 선별 참고용임을 염두에 두세요."""

# ─────────────────────────────────────────────
# 통화 기록 파일
# 매 세션마다 AI 발화(questions.json)와 사용자 응답(answers.json)을 누적 저장한다.
# 두 파일은 동일한 id와 call_id로 대응되며, LLM 기반 지표 산출에 사용된다.
# ─────────────────────────────────────────────
QUESTIONS_FILE = "questions.json"
ANSWERS_FILE = "answers.json"

def _load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []

def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _next_id():
    # 기존 questions.json의 마지막 id 기준으로 다음 순번 반환
    questions = _load_json(QUESTIONS_FILE)
    return (questions[-1]["id"] + 1) if questions else 1

# 통화 중 AI의 마지막 발화를 임시 보관 (질문-응답 쌍 매칭용)
_call_state: dict = {}

# ─────────────────────────────────────────────
# ClawOps 에이전트 초기화
# 기본 예시(플랫폼 온보딩 코드)는 사용자가 먼저 전화를 거는 수신 방식이나,
# 효프는 관리 서버가 정해진 시간에 발신하는 구조이므로 agent.call() 방식으로 전환하였다.
# LLM: OpenAI Realtime (한국어, coral 보이스, semantic VAD 적용)
# ─────────────────────────────────────────────
agent = ClawOpsAgent(
    from_="07052361090",
    session=OpenAIRealtime(
        system_prompt=SYSTEM_PROMPT,
        voice="coral",
        language="ko",
        turn_detection={"type": "semantic_vad", "eagerness": "medium"},
        greeting=True,
    ),
)

@agent.on("call_start")
async def on_start(call):
    print(f"통화 시작: {call.from_number} -> {call.to_number}")
    _call_state[call.call_id] = {"pending_question": None}

@agent.on("call_end")
async def on_end(call):
    print(f"통화 종료 | 길이: {call.duration:.1f}초")
    _call_state.pop(call.call_id, None)

@agent.on("transcript")
async def on_transcript(call, role, text):
    label = "어르신" if role == "user" else "AI"
    print(f"[{label}] {text}")

    state = _call_state.get(call.call_id)
    if state is None:
        return

    if role != "user":
        # AI 발화를 pending으로 보관, 다음 사용자 응답과 쌍으로 저장
        state["pending_question"] = text
    elif state["pending_question"]:
        # 질문-응답 쌍 완성 시 각 파일에 append
        turn_id = _next_id()
        questions = _load_json(QUESTIONS_FILE)
        answers = _load_json(ANSWERS_FILE)
        questions.append({"id": turn_id, "call_id": call.call_id, "text": state["pending_question"]})
        answers.append({"id": turn_id, "call_id": call.call_id, "text": text})
        _save_json(QUESTIONS_FILE, questions)
        _save_json(ANSWERS_FILE, answers)
        state["pending_question"] = None

async def main():
    # 현재는 수동 입력 방식이며, 추후 Spring 백엔드에서 번호와 발신 스케줄을 전달받아 대체할 예정
    to_number = input("발신할 전화번호를 입력하세요 (예: 01012345678): ").strip()
    print(f"{to_number}으로 전화를 겁니다...")
    call = await agent.call(to=to_number)
    await call.wait()

asyncio.run(main())