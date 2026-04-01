import logging
from config import settings

logger = logging.getLogger(__name__)

_emotion_detector = None
_rag_pipeline = None

def get_emotion_detector():
    global _emotion_detector
    if _emotion_detector is None:
        from emotion_detector import EmotionDetector
        logger.info("Initializing EmotionDetector (first use)...")
        _emotion_detector = EmotionDetector(model_name=settings.EMOTION_MODEL)
    return _emotion_detector

def get_rag_pipeline():
    global _rag_pipeline
    if _rag_pipeline is None:
        from rag_pipeline import RAGPipeline
        logger.info("Initializing RAGPipeline (first use)...")
        _rag_pipeline = RAGPipeline(
            embedding_model=settings.EMBEDDING_MODEL,
            knowledge_base_dir=settings.KNOWLEDGE_BASE_DIR,
            vector_store_path=settings.VECTOR_STORE_PATH
        )
    return _rag_pipeline