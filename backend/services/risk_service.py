"""
Risk Escalation Service
Computes a risk score from emotion, keywords, and repetition patterns.
"""
import os
from typing import Tuple, List

ESCALATION_THRESHOLD = float(os.getenv("ESCALATION_THRESHOLD", "0.65"))

# Crisis / high-risk keywords with weights
HIGH_RISK_KEYWORDS = {
    "suicide": 0.9, "kill myself": 0.9, "end my life": 0.9, "want to die": 0.85,
    "self harm": 0.8, "cut myself": 0.8, "overdose": 0.8, "no reason to live": 0.85,
    "hopeless": 0.4, "worthless": 0.4, "give up": 0.3, "can't go on": 0.6,
    "nobody cares": 0.3, "hate myself": 0.5, "hurting myself": 0.7
}

EMOTION_WEIGHTS = {
    "sadness": 0.35, "fear": 0.35, "anger": 0.25,
    "disgust": 0.20, "surprise": 0.10, "joy": 0.0, "neutral": 0.0
}

def compute_risk_score(
    text: str,
    emotion: str,
    confidence: float,
    recent_risk_scores: List[float] = None
) -> Tuple[float, str]:
    """
    Returns (risk_score 0.0-1.0, reason_string)
    """
    score = 0.0
    reasons = []

    # Emotion component
    emotion_weight = EMOTION_WEIGHTS.get(emotion.lower(), 0.0)
    emotion_contribution = emotion_weight * confidence
    score += emotion_contribution
    if emotion_contribution > 0.1:
        reasons.append(f"emotion:{emotion}({confidence:.2f})")

    # Keyword component
    text_lower = text.lower()
    keyword_score = 0.0
    for kw, weight in HIGH_RISK_KEYWORDS.items():
        if kw in text_lower:
            keyword_score = max(keyword_score, weight)
            reasons.append(f"keyword:'{kw}'")
    score += keyword_score * 0.5

    # Repetition component - if user has recent high-risk events
    if recent_risk_scores:
        recent_high = [s for s in recent_risk_scores if s >= 0.5]
        if len(recent_high) >= 2:
            score += 0.15
            reasons.append(f"repeated_risk:count={len(recent_high)}")

    score = min(score, 1.0)
    reason_str = "; ".join(reasons) if reasons else "low_risk"
    return round(score, 4), reason_str

def is_escalation_required(risk_score: float) -> bool:
    return risk_score >= ESCALATION_THRESHOLD

CRISIS_MESSAGE = """
⚠️ **We're concerned about you.**

It sounds like you might be going through something really difficult right now. 
You are not alone, and help is available.

**Immediate Support:**
- 🆘 **National Suicide Prevention Lifeline:** 988 (call or text)
- 💬 **Crisis Text Line:** Text HOME to 741741
- 🌍 **International Association for Suicide Prevention:** https://www.iasp.info/resources/Crisis_Centres/

Please consider reaching out to a counselor or therapist. 
You can also request a therapist contact from this platform below.

*This platform is not a replacement for professional mental health care.*
"""
