import logging
from typing import Tuple

logger = logging.getLogger(__name__)

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


class EmotionDetector:
    def __init__(self, model_name: str = "bhadresh-savani/distilbert-base-uncased-emotion"):
        self.model_name = model_name
        self._pipeline = None
        logger.info(f"EmotionDetector initialised (model will load on first use): {model_name}")

    def _get_pipeline(self):
        """Lazy‑load the Hugging Face pipeline."""
        if self._pipeline is None:
            logger.info(f"Loading emotion detection model: {self.model_name}")
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    top_k=1,
                    device=-1  # CPU
                )
                logger.info("Emotion model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load emotion model: {e}")
                self._pipeline = None
        return self._pipeline

    def predict(self, text: str) -> Tuple[str, float]:
        """Returns (emotion_label, confidence_score)."""
        pipe = self._get_pipeline()
        if pipe is None:
            return "neutral", 0.5

        try:
            result = pipe(text[:512])  # truncate for model limits
            if result and result[0]:
                top = result[0][0]  # because top_k=1
                emotion = top["label"].lower()
                confidence = round(top["score"], 4)
                logger.info(f"Emotion: {emotion} ({confidence}) for text: {text[:100]}")
                return emotion, confidence
        except Exception as e:
            logger.error(f"Emotion detection error: {e}")

        return "neutral", 0.5

    def compute_risk_score(self, text: str, emotion: str, confidence: float) -> float:
        """Compute a risk score between 0.0 and 1.0."""
        text_lower = text.lower()

        # Emotion weight
        emotion_weight = EMOTION_RISK_WEIGHTS.get(emotion, 0.0) * confidence

        # Keyword weight
        keyword_hits = sum(1 for kw in HIGH_RISK_KEYWORDS if kw in text_lower)
        keyword_weight = min(keyword_hits * 0.25, 0.5)

        # Combine
        risk_score = min(emotion_weight * 0.5 + keyword_weight, 1.0)
        return round(risk_score, 4)