"""
Emotion Detection Service using HuggingFace Transformers
Uses: j-hartmann/emotion-english-distilroberta-base (local, free)
Outputs: emotion_label, confidence_score
"""
from typing import Tuple
import os

_model = None
_tokenizer = None
_pipeline = None

EMOTION_MODEL = os.getenv("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")

def load_emotion_model():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline
        _pipeline = pipeline(
            "text-classification",
            model=EMOTION_MODEL,
            top_k=1,
            device=-1  # CPU
        )
        print(f"[EmotionDetector] Model loaded: {EMOTION_MODEL}")
    except Exception as e:
        print(f"[EmotionDetector] Warning: Could not load model ({e}). Using fallback.")
        _pipeline = None
    return _pipeline

def detect_emotion_fallback(text: str) -> Tuple[str, float]:
    """Keyword-based fallback if model unavailable"""
    text_lower = text.lower()
    if any(w in text_lower for w in ["suicide", "kill myself", "end my life", "want to die"]):
        return "fear", 0.95
    elif any(w in text_lower for w in ["sad", "depressed", "hopeless", "worthless", "crying"]):
        return "sadness", 0.80
    elif any(w in text_lower for w in ["angry", "furious", "hate", "rage"]):
        return "anger", 0.80
    elif any(w in text_lower for w in ["anxious", "worried", "nervous", "panic", "stress"]):
        return "fear", 0.75
    elif any(w in text_lower for w in ["happy", "great", "excited", "wonderful", "joy"]):
        return "joy", 0.80
    elif any(w in text_lower for w in ["disgusted", "gross", "awful", "horrible"]):
        return "disgust", 0.70
    else:
        return "neutral", 0.60

def detect_emotion(text: str) -> Tuple[str, float]:
    """
    Returns (emotion_label, confidence_score)
    """
    pipeline = load_emotion_model()
    if pipeline is None:
        return detect_emotion_fallback(text)
    try:
        result = pipeline(text[:512])
        top = result[0][0]
        label = top["label"].lower()
        score = round(top["score"], 4)
        return label, score
    except Exception as e:
        print(f"[EmotionDetector] Inference error: {e}")
        return detect_emotion_fallback(text)
