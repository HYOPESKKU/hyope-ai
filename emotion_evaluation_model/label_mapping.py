from typing import Dict, List


CARE_SIGNAL_MAP: Dict[str, List[str]] = {
    "불안": ["anxiety_related", "care_check_recommended"],
    "슬픔": ["depression_related", "loneliness_check_recommended"],
    "상처": ["depression_related", "social_support_check_recommended"],
    "분노": ["negative_emotion", "care_check_recommended"],
    "당황": ["anxiety_related", "confusion_check_recommended"],
    "기쁨": ["positive_emotion"],
}


KOREAN_LABEL_DESCRIPTION: Dict[str, str] = {
    "불안": "불안, 걱정, 스트레스 관련 표현",
    "슬픔": "슬픔, 우울감, 눈물, 상실감 관련 표현",
    "상처": "서운함, 소외감, 관계적 상처 관련 표현",
    "분노": "화남, 억울함, 짜증 관련 표현",
    "당황": "혼란, 난처함, 예상치 못한 상황 관련 표현",
    "기쁨": "기쁨, 안도, 만족, 긍정 정서 관련 표현",
}


def to_care_signals(label: str) -> List[str]:
    """Convert emotion label to HYOPE care-oriented signal flags."""
    return CARE_SIGNAL_MAP.get(label, ["unknown_emotion"])


def describe_label(label: str) -> str:
    """Return Korean explanation for an emotion label."""
    return KOREAN_LABEL_DESCRIPTION.get(label, "정의되지 않은 감정 라벨")