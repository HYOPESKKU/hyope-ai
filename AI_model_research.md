# HYOPE 특화 모델 리서치 정리

> 작성일: 2026-06-21  
> 목적: 독거노인 통화 STT 텍스트 기반 상태 평가를 위한 파인튜닝 모델 도입 방향 조사

---

## 요약 (5줄)

1. **인지 저하는 AI가 판단하지 않는다** — VoiceML에 MMSE식 지남력 질문을 넣고 정답/오답을 규칙 기반으로 채점. AI가 추론하는 게 아니라 "오늘이 몇 년도예요?"에 틀리면 점수가 깎이는 구조.
2. **감정/정서 플래깅은 파인튜닝 모델이 담당** — "뭔가 이상한데" 수준의 신호 감지. 진단이 아닌 참고용. Gemini가 뽑은 점수를 보정하는 역할.
3. **쓸 수 있는 데이터는 6개** — AI Hub 2개(노년층 감성 분류, 고령자 구술) + 공개 감정 데이터 4개(KOTE 50k·KEmoFact 16.5k·KorEmpatheticDialogues 24.8k·GoEmotions-Korean 58k). 노인 특화는 AI Hub 두 개가 유일하고, 나머지는 감정 분류 보강용.
4. **모델 서빙은 base_url 한 줄** — KoELECTRA(소형 분류기)는 FastAPI 안에 직접 올리고, 생성 모델(EXAONE 등)은 Ollama/vLLM으로 서빙 후 base_url만 교체. Spring 파이프라인은 건드릴 필요 없음.
5. **모든 지표는 참고용** — 보호자에게 "확인해보세요" 알림을 보내는 트리거일 뿐, 임상 진단이 아님을 명시해야 함.

---

## 1. 전체 평가 구조

```
VoiceML 통화
 ├── [안부·건강 대화]
 │     → Gemini 1차 분석 (emotionalStability, depressionScore, healthRisk)
 │     → KoELECTRA 감정 분류기로 보정 (참고용 플래깅)
 │
 └── [지남력 질문 — MMSE식]
       → 규칙 기반 채점 (정답/오답 판단, AI 추론 없음)
       → cognitiveScore 산출 (참고용)

최종 → Spring /call-result 전송 → 보호자 앱 알림
```

---

## 2. 지남력 평가 (규칙 기반, AI 아님)

인지 저하는 AI가 대화 내용을 보고 추론하는 방식이 아니라,  
**정답이 있는 질문을 직접 물어보고 맞는지 틀리는지 판단**하는 방식으로 처리.

### 질문 체계

#### 시간 지남력 (5점)

| 질문 (VoiceML TTS) | 채점 기준 | 배점 |
|---|---|---|
| "오늘이 몇 년도인지 아세요?" | 실제 연도와 일치 | 1점 |
| "지금이 몇 월인가요?" | 실제 월과 일치 | 1점 |
| "오늘이 며칠이에요?" | ±1일 오차 허용 | 1점 |
| "오늘 무슨 요일이에요?" | 실제 요일과 일치 | 1점 |
| "지금 무슨 계절이에요?" | 봄(3~5월) / 여름(6~8월) / 가을(9~11월) / 겨울(12~2월) | 1점 |

#### 장소 지남력 (3점)

| 질문 | 채점 기준 | 배점 |
|---|---|---|
| "지금 어디 계세요? 집에 계신 거죠?" | 현재 위치 인지 여부 | 1점 |
| "사시는 동네 이름이 어떻게 되세요?" | DB 등록 주소와 일치 | 1점 |
| "사시는 시·군·구가 어디세요?" | DB 등록 주소 기준 | 1점 |

#### 인물 지남력 (3점)

| 질문 | 채점 기준 | 배점 |
|---|---|---|
| "성함이 어떻게 되세요?" | DB 등록 이름과 일치 | 1점 |
| "따님/아드님 성함이 어떻게 되세요?" | DB 보호자 이름과 일치 | 1점 |
| "몇 년생이세요?" | DB 등록 생년과 일치 | 1점 |

#### 상황 지남력 (참고만)

| 질문 | 채점 방식 |
|---|---|
| "어제 저녁 식사는 뭐 드셨어요?" | 구체적 답변 가능 여부만 확인 (내용 검증 불가) |
| "요즘 주무실 때 몇 시쯤 주무세요?" | 구체적 시간 답변 여부 |

> 상황 지남력은 정답 검증이 불가능하므로 Gemini가 맥락 참고용으로만 사용.

### 채점 코드

```python
from datetime import date

SEASON_MAP = {
    (3, 4, 5): "봄",
    (6, 7, 8): "여름",
    (9, 10, 11): "가을",
    (12, 1, 2): "겨울",
}

def get_season(month: int) -> str:
    return next(s for months, s in SEASON_MAP.items() if month in months)

def score_orientation(answers: dict, recipient: dict) -> dict:
    today = date.today()
    scores = {
        "year":    1 if answers.get("year") == str(today.year) else 0,
        "month":   1 if answers.get("month") == str(today.month) else 0,
        "day":     1 if abs(int(answers.get("day", -99)) - today.day) <= 1 else 0,
        "weekday": 1 if answers.get("weekday") == today.strftime("%A") else 0,
        "season":  1 if answers.get("season") == get_season(today.month) else 0,
        "address": 1 if answers.get("district") in recipient["address"] else 0,
        "name":    1 if answers.get("name") == recipient["name"] else 0,
        "guardian":1 if answers.get("guardian_name") == recipient["guardian_name"] else 0,
        "birth":   1 if answers.get("birth_year") == str(recipient["birth_year"]) else 0,
    }
    total = sum(scores.values())  # 만점 9점
    return {"orientationScore": total, "maxScore": 9, "detail": scores}
```

> **주의:** STT 오류로 숫자 전사가 틀릴 수 있음 ("이천이십육" → "2026" 변환 로직 필요).

---

## 3. 감정·정서 플래깅 (파인튜닝 모델)

### 사용 가능한 데이터셋

#### 노인 특화 (AI Hub)

| 데이터셋 | 크기 | 내용 | 활용 방향 |
|---|---|---|---|
| **노년층 대상 감성 분류** ([링크](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=453)) | 노인-챗봇 2~4턴 대화 | 감정 6종 레이블 (분노/기쁨/불안/당황/슬픔/상처) | 감정 분류기 파인튜닝 핵심 데이터 |
| **생성형AI 고령자 구술 데이터** ([링크](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71703)) | 143,630건 | 텍스트 전사본 + 우울/불안 점수, 기억 특이성 점수 | 우울·불안 플래깅 보조 |

> 음성 파일(WAV)은 사용하지 않음. 텍스트 전사본과 JSON 레이블만 사용.

#### 공개 감정 데이터 (신청 불필요, 즉시 사용 가능)

| 데이터셋 | 크기 | 감정 레이블 | 라이선스 | 노인 적합도 |
|---|---|---|---|---|
| **KOTE** ([HuggingFace](https://huggingface.co/datasets/searle-j/kote)) | 50,000건 | 44종 (슬픔/절망/불안/힘듦/행복 등) | MIT | ★★★ |
| **KEmoFact** ([PMC 부록](https://pmc.ncbi.nlm.nih.gov/articles/PMC10613607/)) | 16,532건 | 22종 (lonely/afraid/sad/content 등) | CC-BY-4.0 | ★★★ |
| **KorEmpatheticDialogues** ([HuggingFace](https://huggingface.co/datasets/passing2961/KorEmpatheticDialogues)) | 24,800건 | 32종 (lonely/anxious/nostalgic 등) | CC-BY-NC-4.0 | ★★★ |
| **GoEmotions-Korean** ([GitHub](https://github.com/monologg/GoEmotions-Korean)) | 58,000건 | 28종 (grief/fear/sadness/relief 등) | Apache-2.0 | ★★ |

> KOTE 44종 레이블 중 노인 케어 관련: 슬픔, 절망, 불안/걱정, 힘듦/지침, 서러움, 안심/신뢰, 행복, 귀찮음, 아껴주는
> KoELECTRA/KoBERT 파인튜닝에 가장 많이 쓰이는 데이터셋이 KOTE.

#### 쓸 수 없는 것

| 가능 | 불가능 |
|---|---|
| 감정 분류 및 플래깅 | PHQ-9 임상 진단 |
| 우울·불안 수준 참고값 | MCI/치매 분류 |
| Gemini 점수 보정 | 정신건강 의학적 판단 |

### 권장 모델

| 용도 | 모델 | 크기 | 이유 |
|---|---|---|---|
| 감정 분류기 | **KoELECTRA-v3** | 110MB | CPU 가능, FastAPI 직접 탑재, 분류 성능 우수 |
| 감정+정서 생성 보조 | **EXAONE-3.5-7.8B** | 7.8B | 한국어 최강, GPU 서버 필요 |

### 파인튜닝 설정

**KoELECTRA (분류, CPU 가능):**
```python
# 별도 서버 없이 FastAPI 안에 직접 탑재
from transformers import pipeline

emotion_classifier = pipeline(
    "text-classification",
    model="./finetuned-koelectra-emotion"
)

def flag_emotion(conversation_text: str) -> dict:
    result = emotion_classifier(conversation_text)
    return {"emotion": result[0]["label"], "confidence": result[0]["score"]}
```

**EXAONE QLoRA (생성, GPU 필요):**
```python
# 16GB VRAM 기준
quantization = "4-bit NF4"
lora_r = 8
lora_alpha = 16
target_modules = ["q_proj", "v_proj"]
batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 2e-4
epochs = 3
max_seq_length = 512
```

---

## 4. 서버에 모델 띄우는 방법

현재 Gemini는 Google 서버가 실행하고 우리는 API만 호출.  
파인튜닝 모델은 **직접 서버에서 실행** → 모델 크기에 따라 전략이 다름.

### KoELECTRA (분류기) → FastAPI 안에 직접 포함

별도 서버 불필요. CPU에서도 동작.

```python
# main.py 또는 별도 classifier.py로 분리
from transformers import pipeline

emotion_classifier = pipeline("text-classification", model="./finetuned-koelectra")
```

### EXAONE 등 생성 모델 → 별도 GPU 서버

**Ollama (개발·데모용, 간단):**
```bash
ollama serve
ollama run exaone3.5:7.8b
# http://gpu-server-ip:11434 에서 OpenAI 호환 API 제공
```

**vLLM (프로덕션용):**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./finetuned-exaone --port 8001
# http://gpu-server-ip:8001 에서 OpenAI 호환 API 제공
```

**기존 코드 변경은 base_url 한 줄:**
```python
# llm_analyzer.py
def _get_client():
    return OpenAI(
        api_key="ollama",
        base_url="http://gpu-server-ip:11434/v1",  # 여기만 바꾸면 됨
    )
```

Spring ↔ FastAPI 파이프라인은 전혀 건드릴 필요 없음.

### GPU 서버 선택지

| 옵션 | 비용 | 모델 크기 | 추천 상황 |
|---|---|---|---|
| **Colab Pro+** | 월 ~$50 | 7~13B (A100 40GB) | 개발·학습 |
| **RunPod / Vast.ai** | 시간당 $0.3~1 | 7~70B | 필요할 때만 켜기, 데모용 |
| **학교 GPU 서버** | 무료 | 상황에 따라 | 캡스톤 신청 가능하면 최우선 |
| **AWS/GCP GPU** | 비쌈 | 모든 크기 | 프로덕션 단계 |

---

## 5. 참고 논문

| 논문 | 핵심 내용 |
|---|---|
| [When a chatbot asks "How are you?" (BMC Geriatrics, 2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12924473/) | 네이버 CareCall과 거의 동일한 시스템. HYOPE 개념 증명 역할. 더 나은 NLP 모델의 필요성 직접 언급. |
| [Evaluating LLMs for Depression Detection (Springer, 2025)](https://link.springer.com/chapter/10.1007/978-981-96-7030-7_1) | Mistral-7B + QLoRA 4-bit으로 F1 0.975 달성. 풀 파인튜닝과 품질 동등. |
| [LoRA/QLoRA vs Preference Optimization (arXiv:2604.00773)](https://arxiv.org/pdf/2604.00773) | QLoRA가 풀 파인튜닝과 동등 이상. 캡스톤 파인튜닝 전략 근거. |
| [KMI: Korean Motivational Interviewing (NAACL 2025)](https://arxiv.org/abs/2502.05651) | 한국어 심리 상담 대화 파인튜닝 사례. LLaMA-2-7B 기반. |

---

## 6. 우선순위

1. **AI Hub 데이터 신청** — `노년층 대상 감성 분류` + `생성형AI 고령자 구술 데이터` (승인 1~2주)
2. **VoiceML 지남력 질문 설계** — 시간·장소·인물 지남력 질문 확정 후 TTS 스크립트 작성
3. **KoELECTRA 감정 분류기** — 데이터 수령 후 파인튜닝, FastAPI에 직접 탑재
4. **GPU 확보 후 EXAONE 파인튜닝** — 감정+정서 보정 고도화, Gemini 교체 검토

---

## 7. 참고 링크

**데이터셋**
- [AI Hub — 노년층 대상 감성 분류 (dataSetSn=453)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=453)
- [AI Hub — 생성형AI 고령자 구술 데이터 (dataSetSn=71703)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71703)
- [KOTE — 44종 한국어 감정 (HuggingFace)](https://huggingface.co/datasets/searle-j/kote)
- [KEmoFact — 22종 감정 (PMC 논문 부록)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10613607/)
- [KorEmpatheticDialogues — 32종 감정 (HuggingFace)](https://huggingface.co/datasets/passing2961/KorEmpatheticDialogues)
- [GoEmotions-Korean — 28종 감정 (GitHub)](https://github.com/monologg/GoEmotions-Korean)

**모델**
- [KoELECTRA-v3 (HuggingFace)](https://huggingface.co/monologg/koelectra-base-v3-discriminator)
- [EXAONE-3.5-7.8B (HuggingFace)](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct)
- [HuggingFace PEFT (LoRA/QLoRA)](https://huggingface.co/docs/peft)

**논문**
- [CareCall (BMC Geriatrics 2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12924473/)
- [KOTE 논문 (arXiv:2205.05300)](https://arxiv.org/pdf/2205.05300)
