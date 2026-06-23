# HYOPE Emotion Evaluation Model

HYOPE의 AI 분석 서버를 보조하기 위한 감정/정서 평가 모델 모듈입니다.

> This module provides auxiliary emotion classification and care signal mapping for HYOPE's AI analysis pipeline.

## Purpose

이 모델은 독거노인과의 ARS/STT 기반 통화 텍스트에서 감정 신호를 분류하고, 보호자 및 복지사의 확인이 필요한 정서적 care signal을 제공하기 위한 보조 모델입니다.

본 모델은 의료적 진단을 수행하지 않으며, Gemini 기반 종합 분석 결과를 대체하지 않습니다.
출력 결과는 보호자 및 복지사의 확인을 돕기 위한 참고 지표로만 사용됩니다.

## Dataset

현재 사용 데이터는 다음과 같습니다.

### Training Data

* `emotionClassification_train.zip`

  * 내부 파일: `train.csv`
  * 라벨 컬럼: `감정_대분류`
  * 텍스트 컬럼:

    * `사람문장1`
    * `사람문장2`
    * `사람문장3`
    * `사람문장4`

### Test Data

* `emotionClassification_test.zip`

  * 내부 파일:

    * `test.csv`
    * `sample_submission.csv`

## Labels

분류 대상 감정은 다음 6개입니다.

| Label | Description              |
| ----- | ------------------------ |
| `기쁨`  | 긍정적 감정 표현                |
| `당황`  | 혼란, 난처함, 예상치 못한 상황 관련 표현 |
| `분노`  | 화남, 짜증, 부정적 반응 관련 표현     |
| `불안`  | 걱정, 긴장, 스트레스 관련 표현       |
| `상처`  | 서운함, 소외감, 관계적 상처 관련 표현   |
| `슬픔`  | 우울감, 외로움, 무기력 관련 표현      |

## Baseline Model

현재 baseline 모델은 다음과 같이 구성되어 있습니다.

* TF-IDF Vectorizer
* Logistic Regression
* Train/Validation split: `8:2`
* Validation Accuracy: `0.6325`
* Macro F1-score: `0.63`

<sub>Baseline 성능은 향후 데이터 정제, feature engineering, KoELECTRA 기반 파인튜닝 등을 통해 개선할 수 있습니다.</sub>

## Care Signal Mapping

각 감정 라벨은 HYOPE 돌봄 플래그로 변환됩니다.

| Emotion | Care Signals                                             |
| ------- | -------------------------------------------------------- |
| `불안`    | `anxiety_related`, `care_check_recommended`              |
| `슬픔`    | `depression_related`, `loneliness_check_recommended`     |
| `상처`    | `depression_related`, `social_support_check_recommended` |
| `분노`    | `negative_emotion`, `care_check_recommended`             |
| `당황`    | `anxiety_related`, `confusion_check_recommended`         |
| `기쁨`    | `positive_emotion`                                       |

## Usage

### Install Dependencies

```bash
pip install -r emotion_evaluation_model/requirements.txt
```

### Train Baseline Model

```bash
python emotion_evaluation_model/train_emotion_classifier.py
```

### Run Prediction

```bash
python emotion_evaluation_model/predict.py --text "약을 너무 많이 먹어야 해서 걱정되고 불안해."
```

### Example Output

```text
Emotion         : 불안
Confidence      : 0.9024
Description     : 불안, 걱정, 스트레스 관련 표현
Care signals    : ['anxiety_related', 'care_check_recommended']
Low confidence  : False
Top emotions    : ['불안', '당황', '슬픔']
```

## Notes

Confidence가 낮은 경우 단일 감정 라벨을 확정적으로 사용하지 않고, `top-3` 감정과 `care signal`을 함께 참고합니다.

이 모델은 HYOPE의 Gemini 분석 결과를 보조하는 감정 플래깅 모델입니다.

향후 KoELECTRA 기반 파인튜닝 모델로 고도화할 수 있습니다.

## Commit

README 저장 후 다음 명령어로 커밋합니다.

```powershell
git add emotion_evaluation_model/README.md
git commit -m "docs: document baseline emotion evaluation model"
git push origin feature/evaluation-model
```
