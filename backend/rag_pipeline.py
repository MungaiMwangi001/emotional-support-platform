import os
import logging
from typing import Optional
from config import settings

logger = logging.getLogger(__name__)

_rag_chain = None
_embeddings = None
_vectorstore = None


SYSTEM_PROMPT = """You are a compassionate, empathetic AI support assistant helping students with emotional challenges.

RULES:
- Provide supportive, non-clinical emotional support
- NEVER provide medical diagnoses or prescriptions
- NEVER provide harmful instructions
- Encourage professional help when appropriate
- Keep responses concise (2-4 sentences), warm, and empathetic
- Do not make absolute certainty statements about the person's situation
- Always validate feelings before offering suggestions
- If the student seems in crisis, gently encourage them to seek professional help immediately

You are NOT a replacement for professional mental health care."""


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
    return _embeddings


def load_or_create_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    try:
        from langchain_community.vectorstores import FAISS
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader, DirectoryLoader

        embeddings = get_embeddings()
        if embeddings is None:
            return None

        vector_path = settings.VECTOR_STORE_PATH

        if os.path.exists(f"{vector_path}/index.faiss"):
            logger.info("Loading existing vector store...")
            _vectorstore = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
        else:
            logger.info("Building vector store from knowledge base...")
            kb_dir = settings.KNOWLEDGE_BASE_DIR

            if not os.path.exists(kb_dir):
                logger.warning("Knowledge base directory not found, using minimal fallback.")
                from langchain.schema import Document
                docs = [Document(page_content=FALLBACK_KNOWLEDGE, metadata={"source": "fallback"})]
            else:
                loader = DirectoryLoader(kb_dir, glob="**/*.txt", loader_cls=TextLoader)
                docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)

            _vectorstore = FAISS.from_documents(chunks, embeddings)
            os.makedirs(vector_path, exist_ok=True)
            _vectorstore.save_local(vector_path)
            logger.info("Vector store saved.")

    except Exception as e:
        logger.error(f"Vector store error: {e}")
        return None

    return _vectorstore


FALLBACK_KNOWLEDGE = """
Stress management techniques: Deep breathing exercises can help reduce anxiety. Try breathing in for 4 counts, holding for 4, and exhaling for 4. Regular physical activity, even a short walk, can significantly improve mood and reduce stress hormones.

Academic stress: It's normal to feel overwhelmed by academic pressure. Breaking tasks into smaller steps, using time-blocking techniques, and taking regular breaks using the Pomodoro method (25 minutes work, 5 minute break) can help.

Sleep and mental health: Sleep is crucial for emotional regulation. Aim for 7-9 hours per night. Establishing a consistent sleep schedule, avoiding screens before bed, and creating a calming bedtime routine can improve sleep quality.

Social support: Connecting with friends, family, or a support group can provide emotional relief. You don't have to face challenges alone. Sharing your feelings with trusted people can reduce their intensity.

When to seek help: If you're feeling persistently sad, anxious, or overwhelmed for more than two weeks, or if you're having thoughts of harming yourself, please reach out to a mental health professional or crisis line immediately.

Mindfulness and grounding: When feeling anxious, try the 5-4-3-2-1 technique: notice 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste. This helps ground you in the present moment.

Self-compassion: Be kind to yourself. Everyone struggles sometimes. Treat yourself with the same compassion you'd show a good friend. Mistakes are opportunities for growth, not evidence of failure.
"""


def generate_response(user_message: str, emotion: str) -> str:
    """Generate a supportive response using RAG pipeline or fallback."""
    try:
        vs = load_or_create_vectorstore()
        context = ""

        if vs:
            docs = vs.similarity_search(user_message, k=3)
            context = "\n\n".join([d.page_content for d in docs])

        # Use local generation or template-based fallback
        return _generate_with_context(user_message, emotion, context)

    except Exception as e:
        logger.error(f"RAG generation error: {e}")
        return _fallback_response(emotion)


def _generate_with_context(message: str, emotion: str, context: str) -> str:
    """Try local LLM first, fall back to template responses."""
    # Try Ollama or local LLM
    try:
        import httpx
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": f"{SYSTEM_PROMPT}\n\nContext from knowledge base:\n{context}\n\nStudent (feeling {emotion}): {message}\n\nAssistant:",
                "stream": False,
                "options": {"num_predict": 150, "temperature": 0.7}
            },
            timeout=30.0
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception:
        pass

    # Template-based fallback
    return _template_response(message, emotion, context)


def _template_response(message: str, emotion: str, context: str) -> str:
    """Generate empathetic template response based on emotion."""
    emotion_intros = {
        "sadness": "I hear that you're going through a really difficult time, and it's completely okay to feel this way.",
        "anger": "I can sense your frustration, and those feelings are valid. It's okay to feel angry sometimes.",
        "fear": "Feeling anxious or scared can be really overwhelming. You're not alone in feeling this way.",
        "disgust": "It sounds like you're dealing with something that really bothers you, and that's understandable.",
        "surprise": "It seems like something unexpected has happened. Give yourself a moment to process.",
        "joy": "It's wonderful to hear some positive feelings! Let's keep building on that.",
        "neutral": "Thank you for reaching out. I'm here to listen and support you.",
    }

    intro = emotion_intros.get(emotion, "Thank you for sharing how you're feeling.")

    # Extract a tip from context if available
    tip = ""
    if context:
        lines = context.split(".")
        for line in lines:
            line = line.strip()
            if len(line) > 40 and emotion.lower() in line.lower():
                tip = f" {line}."
                break
        if not tip and lines:
            for line in lines:
                line = line.strip()
                if len(line) > 40:
                    tip = f" {line}."
                    break

    closing = " Remember, it's important to reach out to a mental health professional if these feelings persist."
    return f"{intro}{tip}{closing}"


def _fallback_response(emotion: str) -> str:
    return (
        f"I'm here for you and I want you to know your feelings are valid. "
        f"Whatever you're going through right now, please know you don't have to face it alone. "
        f"If you're struggling, please consider speaking with a counselor or mental health professional who can provide proper support."
    )


def filter_unsafe_response(response: str) -> str:
    """Post-generation safety filter."""
    unsafe_phrases = [
        "you should take medication",
        "you have depression",
        "you are diagnosed",
        "i diagnose",
        "you should harm",
        "you could try hurting",
    ]
    response_lower = response.lower()
    for phrase in unsafe_phrases:
        if phrase in response_lower:
            return _fallback_response("neutral")
    return response
