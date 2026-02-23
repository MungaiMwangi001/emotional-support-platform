import torch
from transformers import pipeline
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

_emotion_pipeline = None


def get_emotion_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        logger.info("Loading emotion detection model...")
        try:
            _emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=1,
                device=-1  # CPU
            )
            logger.info("Emotion model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            _emotion_pipeline = None
    return _emotion_pipeline


def detect_emotion(text: str) -> Tuple[str, float]:
    """Returns (emotion_label, confidence_score)"""
    pipe = get_emotion_pipeline()
    if pipe is None:
        return "neutral", 0.5

    try:
        result = pipe(text[:512])  # truncate for model limits
        if result and result[0]:
            top = result[0][0]
            return top["label"].lower(), round(top["score"], 4)
    except Exception as e:
        logger.error(f"Emotion detection error: {e}")

    return "neutral", 0.5


# Emotion to risk weight mapping
EMOTION_RISK_WEIGHTS = {
    "anger": 0.5,
    "disgust": 0.3,
    "fear": 0.6,
    "joy": 0.0,
    "neutral": 0.0,
    "sadness": 0.5,
    "surprise": 0.1,
}

HIGH_RISK_KEYWORDS = [
    "suicide", "kill myself", "end my life", "don't want to live",
    "self harm", "hurt myself", "want to die", "no reason to live",
    "hopeless", "worthless", "can't go on", "give up on life",
    "cut myself", "overdose", "nothing to live for"
]


def compute_risk_score(text: str, emotion: str, confidence: float) -> float:
    """Compute a risk score between 0.0 and 1.0"""
    text_lower = text.lower()

    # Emotion weight
    emotion_weight = EMOTION_RISK_WEIGHTS.get(emotion, 0.0) * confidence

    # Keyword weight
    keyword_hits = sum(1 for kw in HIGH_RISK_KEYWORDS if kw in text_lower)
    keyword_weight = min(keyword_hits * 0.25, 0.5)

    # Combine
    risk_score = min(emotion_weight * 0.5 + keyword_weight, 1.0)
    return round(risk_score, 4)
