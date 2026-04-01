import os
import logging
from config import settings

logger = logging.getLogger(__name__)

# Short, empathetic fallback intro
EMOTION_INTROS = {
    "sadness": "I hear you're going through a tough time. That's completely okay.",
    "anger": "I can sense your frustration. It's okay to feel angry.",
    "fear": "Feeling scared is really hard. You're not alone.",
    "disgust": "That sounds really difficult.",
    "surprise": "That sounds like quite a surprise.",
    "joy": "I'm glad you're feeling positive!",
    "neutral": "Thank you for sharing how you're feeling.",
}

UNSAFE_PHRASES = [
    "you should take medication",
    "you have depression",
    "i diagnose",
    "you should harm",
]

class RAGPipeline:
    def __init__(self, embedding_model: str, knowledge_base_dir: str, vector_store_path: str):
        self.embedding_model = embedding_model
        self.knowledge_base_dir = knowledge_base_dir
        self.vector_store_path = vector_store_path
        self._embeddings = None
        self._vectorstore = None
        self._llm = None
        logger.info("RAGPipeline initialised (models load on first use).")

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

    def _get_vectorstore(self):
        if self._vectorstore is not None:
            return self._vectorstore
        try:
            from langchain_community.vectorstores import FAISS
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.document_loaders import TextLoader, DirectoryLoader
            from langchain.schema import Document

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
                    logger.warning("Knowledge base not found, using fallback.")
                    docs = [Document(page_content="", metadata={"source": "fallback"})]
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

    def _get_llm(self):
        if self._llm is None:
            try:
                from transformers import pipeline
                logger.info("Loading local LLM (flan-t5-small)...")
                self._llm = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-small",
                    device=-1,
                    max_new_tokens=80,        # shorter output
                    do_sample=True,
                    temperature=0.7
                )
                logger.info("LLM loaded.")
            except Exception as e:
                logger.error(f"Failed to load LLM: {e}")
                self._llm = None
        return self._llm

    def query(self, message: str, emotion: str) -> str:
        try:
            vs = self._get_vectorstore()
            context = ""
            if vs:
                docs = vs.similarity_search(message, k=2)  # fewer docs
                context = "\n".join([d.page_content[:200] for d in docs])
            return self._generate_with_context(message, emotion, context)
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return self._fallback_response(emotion)

    def filter(self, response: str) -> str:
        response_lower = response.lower()
        for phrase in UNSAFE_PHRASES:
            if phrase in response_lower:
                return self._fallback_response("neutral")
        return response

    def _generate_with_context(self, message: str, emotion: str, context: str) -> str:
        # Short prompt asking for 1-2 sentences
        prompt = f"""You are a compassionate mental health assistant. Respond in 1-2 short sentences. Ask a follow‑up question if appropriate.

User feels {emotion}: {message}

Assistant:"""
        llm = self._get_llm()
        if llm:
            try:
                response = llm(prompt, max_new_tokens=80)[0]['generated_text'].strip()
                if response:
                    logger.info("LLM generated response.")
                    return response
                else:
                    logger.warning("LLM returned empty response.")
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
        # Fallback to short template
        intro = EMOTION_INTROS.get(emotion, "Thank you for sharing how you're feeling.")
        return f"{intro} Would you like to tell me more?"

    def _fallback_response(self, emotion: str) -> str:
        return "I'm here to support you. If you need to talk, I'm listening."