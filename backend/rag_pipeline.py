import os
import logging
from config import settings

logger = logging.getLogger(__name__)

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

FALLBACK_KNOWLEDGE = """
Stress management techniques: Deep breathing exercises can help reduce anxiety. Try breathing in for 4 counts, holding for 4, and exhaling for 4. Regular physical activity, even a short walk, can significantly improve mood and reduce stress hormones.

Academic stress: It's normal to feel overwhelmed by academic pressure. Breaking tasks into smaller steps, using time-blocking techniques, and taking regular breaks using the Pomodoro method (25 minutes work, 5 minute break) can help.

Sleep and mental health: Sleep is crucial for emotional regulation. Aim for 7-9 hours per night. Establishing a consistent sleep schedule, avoiding screens before bed, and creating a calming bedtime routine can improve sleep quality.

Social support: Connecting with friends, family, or a support group can provide emotional relief. You don't have to face challenges alone. Sharing your feelings with trusted people can reduce their intensity.

When to seek help: If you're feeling persistently sad, anxious, or overwhelmed for more than two weeks, or if you're having thoughts of harming yourself, please reach out to a mental health professional or crisis line immediately.

Mindfulness and grounding: When feeling anxious, try the 5-4-3-2-1 technique: notice 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste. This helps ground you in the present moment.

Self-compassion: Be kind to yourself. Everyone struggles sometimes. Treat yourself with the same compassion you'd show a good friend. Mistakes are opportunities for growth, not evidence of failure.
"""

UNSAFE_PHRASES = [
    "you should take medication",
    "you have depression",
    "you are diagnosed",
    "i diagnose",
    "you should harm",
    "you could try hurting",
]

EMOTION_INTROS = {
    "sadness": "I hear that you're going through a really difficult time, and it's completely okay to feel this way.",
    "anger": "I can sense your frustration, and those feelings are valid. It's okay to feel angry sometimes.",
    "fear": "Feeling anxious or scared can be really overwhelming. You're not alone in feeling this way.",
    "disgust": "It sounds like you're dealing with something that really bothers you, and that's understandable.",
    "surprise": "It seems like something unexpected has happened. Give yourself a moment to process.",
    "joy": "It's wonderful to hear some positive feelings! Let's keep building on that.",
    "neutral": "Thank you for reaching out. I'm here to listen and support you.",
}


class RAGPipeline:
    def __init__(self, embedding_model: str, knowledge_base_dir: str, vector_store_path: str):
        self.embedding_model = embedding_model
        self.knowledge_base_dir = knowledge_base_dir
        self.vector_store_path = vector_store_path
        self._embeddings = None
        self._vectorstore = None
        logger.info("RAGPipeline initialised (models load on first use).")

    # ── Embeddings ────────────────────────────────────────────────────────────
    def _get_embeddings(self):
        if self._embeddings is None:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={"device": "cpu"},
                )
                logger.info("Embeddings loaded.")
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
        return self._embeddings

    # ── Vector store ──────────────────────────────────────────────────────────
    def _get_vectorstore(self):
        if self._vectorstore is not None:
            return self._vectorstore
        try:
            from langchain_community.vectorstores import FAISS
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.document_loaders import TextLoader, DirectoryLoader

            embeddings = self._get_embeddings()
            if embeddings is None:
                return None

            vp = self.vector_store_path
            if os.path.exists(f"{vp}/index.faiss"):
                logger.info("Loading existing vector store...")
                self._vectorstore = FAISS.load_local(
                    vp, embeddings, allow_dangerous_deserialization=True
                )
            else:
                logger.info("Building vector store from knowledge base...")
                kb = self.knowledge_base_dir
                if not os.path.exists(kb):
                    logger.warning("Knowledge base directory not found, using fallback.")
                    from langchain.schema import Document
                    docs = [Document(page_content=FALLBACK_KNOWLEDGE, metadata={"source": "fallback"})]
                else:
                    loader = DirectoryLoader(kb, glob="**/*.txt", loader_cls=TextLoader)
                    docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(docs)
                self._vectorstore = FAISS.from_documents(chunks, embeddings)
                os.makedirs(vp, exist_ok=True)
                self._vectorstore.save_local(vp)
                logger.info("Vector store saved.")
        except Exception as e:
            logger.error(f"Vector store error: {e}")
            return None
        return self._vectorstore

    # ── Public API (called by chat_router) ───────────────────────────────────
    def query(self, message: str, emotion: str) -> str:
        """Generate a supportive response using RAG + fallback."""
        try:
            vs = self._get_vectorstore()
            context = ""
            if vs:
                docs = vs.similarity_search(message, k=3)
                context = "\n\n".join([d.page_content for d in docs])
            return self._generate_with_context(message, emotion, context)
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return self._fallback_response(emotion)

    def filter(self, response: str) -> str:
        """Post-generation safety filter."""
        response_lower = response.lower()
        for phrase in UNSAFE_PHRASES:
            if phrase in response_lower:
                return self._fallback_response("neutral")
        return response

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _generate_with_context(self, message: str, emotion: str, context: str) -> str:
        # Try local Ollama first
        try:
            import httpx
            resp = httpx.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": (
                        f"{SYSTEM_PROMPT}\n\nContext from knowledge base:\n{context}\n\n"
                        f"Student (feeling {emotion}): {message}\n\nAssistant:"
                    ),
                    "stream": False,
                    "options": {"num_predict": 150, "temperature": 0.7},
                },
                timeout=30.0,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception:
            pass
        return self._template_response(message, emotion, context)

    def _template_response(self, message: str, emotion: str, context: str) -> str:
        intro = EMOTION_INTROS.get(emotion, "Thank you for sharing how you're feeling.")
        tip = ""
        if context:
            for line in context.split("."):
                line = line.strip()
                if len(line) > 40:
                    tip = f" {line}."
                    break
        closing = " Remember, it's important to reach out to a mental health professional if these feelings persist."
        return f"{intro}{tip}{closing}"

    def _fallback_response(self, emotion: str) -> str:
        return (
            "I'm here for you and I want you to know your feelings are valid. "
            "Whatever you're going through right now, please know you don't have to face it alone. "
            "If you're struggling, please consider speaking with a counselor or mental health professional."
        )