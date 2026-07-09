# KOTE 44종 → AI Hub 6종 매핑
# None이면 해당 샘플 스킵 (6종에 자연스럽게 매핑되지 않는 중립/복합 감정)
KOTE_TO_6: dict[str, str | None] = {
    "화남/분노": "분노", "짜증": "분노", "증오/혐오": "분노",
    "역겨움/징그러움": "분노", "어이없음": "분노", "한심함": "분노",

    "즐거움/신남": "기쁨", "행복": "기쁨", "기쁨": "기쁨",
    "흐뭇함(귀여움/예쁨)": "기쁨", "뿌듯함": "기쁨",
    "감동/감탄": "기쁨", "환영/호의": "기쁨", "고마움": "기쁨",
    "안심/신뢰": "기쁨", "편안/쾌적": "기쁨",

    "불안/걱정": "불안", "공포/무서움": "불안",

    "당황/난처": "당황", "경악": "당황", "놀람": "당황", "부끄러움": "당황",

    "슬픔": "슬픔", "절망": "슬픔", "힘듦/지침": "슬픔",
    "서러움": "슬픔", "패배/자기혐오": "슬픔", "귀찮음": "슬픔",

    "안타까움/실망": "상처", "재미없음": "상처",
    "불쌍함/연민": "상처", "죄책감": "상처",

    # 명확한 6종 매핑이 없는 항목 → 스킵
    "없음": None, "신기함/관심": None, "비장함": None,
    "깨달음": None, "기대감": None, "존경": None,
    "아껴주는": None, "우쭐댐/무시함": None, "부담/안_내킴": None,
}

# KorEmpatheticDialogues 영문 32종 → AI Hub 6종 매핑
EMPATHETIC_TO_6: dict[str, str | None] = {
    "angry": "분노", "furious": "분노", "disgusted": "분노", "annoyed": "분노",

    "joyful": "기쁨", "excited": "기쁨", "content": "기쁨", "grateful": "기쁨",
    "proud": "기쁨", "hopeful": "기쁨", "trusting": "기쁨", "faithful": "기쁨",
    "impressed": "기쁨", "surprised": "기쁨",

    "afraid": "불안", "anxious": "불안", "terrified": "불안", "apprehensive": "불안",

    "embarrassed": "당황",

    "sad": "슬픔", "lonely": "슬픔", "devastated": "슬픔",
    "sentimental": "슬픔", "nostalgic": "슬픔",

    "disappointed": "상처", "ashamed": "상처", "guilty": "상처", "jealous": "상처",

    # 스킵
    "caring": None, "anticipating": None, "confident": None, "prepared": None,
}
