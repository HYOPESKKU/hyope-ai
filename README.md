# Agent 실행 가이드

## 사전 준비

### 1. ClawOps 계정 정보

[ClawOps](https://clawops.com) 계정 로그인 후 아래 세 가지 정보를 확인한다.

- **계정 ID** — 시리얼 넘버 형태의 고유 식별자
- **API 키** — 발급 시 반드시 별도로 저장해둘 것. 분실 시 재발급 필요
- **발신 번호** — 계정에 할당된 국내 070 번호

> ⚠️ API 키는 발급 시점 이후 다시 확인할 수 없는 경우가 있으므로, 발급 즉시 안전한 곳에 저장해둔다.

---

### 2. LLM 설정

AI 대화 세션 처리에는 **OpenAI Realtime** 또는 **GPT Realtime** 모델을 권장한다. 임포트가 간편하고 성능이 우수하기 때문이다. 본 예시에서는 한국어를 지원하는 **OpenAI Realtime** 모델을 사용한다.

OpenAI는 유료 서비스이므로, [OpenAI 플랫폼](https://platform.openai.com)에서 크레딧 충전 후 Secret Key를 발급받아 아래 환경변수로 넘겨주면 된다.

---

## 실행 방법

`agent.py` 파일이 위치한 디렉토리로 터미널을 이동한 뒤, 아래 커맨드를 순서대로 실행한다.

```bash
export CLAWOPS_API_KEY=<your_api_key>
export OPENAI_API_KEY=<your_openai_key>
export CLAWOPS_ACCOUNT_ID=<your_account_id>
python agent.py
```

실행 후 전화번호를 입력하면 AI 대화 세션이 시작된다.
