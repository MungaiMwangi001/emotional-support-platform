"""
RAG Pipeline Service
Uses LangChain + FAISS for retrieval-augmented generation.
Local embedding model + local LLM (or fallback template responses).
"""
import os
from typing import Optional

_vectorstore = None
_llm = None
_rag_chain = None

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "")  # Optional: path to local GGUF model
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")

SYSTEM_PROMPT = """You are a compassionate AI emotional support assistant for students.
Your role is to provide empathetic, supportive responses based on the provided context.

Rules you MUST follow:
- Provide supportive but non-clinical advice only
- Do NOT provide medical diagnoses or prescribe treatments
- Do NOT provide harmful, dangerous, or self-harm instructions
- Encourage professional help when appropriate
- Keep responses concise (2-4 paragraphs), warm, and empathetic
- Validate feelings without reinforcing harmful thoughts
- Avoid absolute certainty statements
- If someone expresses crisis, always encourage professional help

Context from knowledge base:
{context}

Student message: {question}

Supportive response:"""

EMOTION_RESPONSES = {
    "sadness": [
        "I can hear that you're going through a really tough time. Feeling sad is a valid and human experience, and it takes courage to acknowledge those feelings. Would you like to talk more about what's been weighing on you?",
        "It sounds like things have been really heavy lately. Your feelings are completely valid. Sometimes just expressing what we're going through can help lighten the load a little.",
    ],
    "fear": [
        "Feeling anxious or scared can be really overwhelming, and I want you to know that what you're experiencing makes sense. Anxiety is your mind's way of trying to protect you, even when it feels out of control.",
        "It's okay to feel scared or worried. Those feelings are telling you something important. Let's think through what might help you feel a bit safer right now.",
    ],
    "anger": [
        "I can tell you're feeling really frustrated and angry. Those feelings are completely valid. It's okay to be angry - what matters is finding healthy ways to process that anger.",
        "Anger can be exhausting to carry around. Your feelings are understandable. Sometimes talking it through helps - what's been making you feel this way?",
    ],
    "joy": [
        "It's wonderful to hear some positive feelings! Moments of joy are worth celebrating, even small ones.",
        "That's really great to hear! Holding onto positive moments, even small ones, can be really helpful for our overall wellbeing.",
    ],
    "neutral": [
        "Thank you for sharing with me. I'm here to listen and support you. How are you really feeling today?",
        "I'm here for you. Sometimes it helps just to talk things through. What's on your mind?",
    ],
    "disgust": [
        "It sounds like something really bothered or upset you. Those feelings of disgust or discomfort are valid responses. Would you like to share more about what happened?",
    ],
    "surprise": [
        "It sounds like something unexpected happened. How are you processing that? I'm here to help you work through it.",
    ]
}

def load_rag_pipeline():
    global _vectorstore, _rag_chain
    if _rag_chain is not None:
        return _rag_chain
    
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader, DirectoryLoader
        import os
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        faiss_index_path = "./faiss_index"
        if os.path.exists(faiss_index_path):
            _vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            print("[RAG] Loaded existing FAISS index")
        else:
            # Load knowledge base documents
            docs = []
            kb_dir = KNOWLEDGE_BASE_DIR
            if os.path.exists(kb_dir):
                for fname in os.listdir(kb_dir):
                    if fname.endswith(".txt"):
                        loader = TextLoader(os.path.join(kb_dir, fname))
                        docs.extend(loader.load())
            
            if docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(docs)
                _vectorstore = FAISS.from_documents(chunks, embeddings)
                _vectorstore.save_local(faiss_index_path)
                print(f"[RAG] Built FAISS index from {len(chunks)} chunks")
            else:
                print("[RAG] No documents found, retrieval disabled")
                return None
        
        _rag_chain = _vectorstore
        return _rag_chain
        
    except Exception as e:
        print(f"[RAG] Could not initialize pipeline: {e}")
        return None

def retrieve_context(query: str, k: int = 3) -> str:
    vectorstore = load_rag_pipeline()
    if vectorstore is None:
        return ""
    try:
        docs = vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([d.page_content for d in docs])
        return context
    except Exception as e:
        print(f"[RAG] Retrieval error: {e}")
        return ""

def generate_response(message: str, emotion: str, context: str = "") -> str:
    """
    Generate supportive response. Uses local LLM if available, else templates.
    """
    # Try local LLM if configured
    if LLM_MODEL and os.path.exists(LLM_MODEL):
        try:
            return _generate_with_llm(message, emotion, context)
        except Exception as e:
            print(f"[RAG] LLM generation failed: {e}")
    
    # Template-based fallback with context injection
    responses = EMOTION_RESPONSES.get(emotion.lower(), EMOTION_RESPONSES["neutral"])
    import random
    base_response = random.choice(responses)
    
    if context:
        resource_hint = "\n\nBased on some resources that might help: " + context[:200] + "..."
        return base_response + resource_hint
    
    return base_response

def _generate_with_llm(message: str, emotion: str, context: str) -> str:
    from llama_cpp import Llama
    llm = Llama(model_path=LLM_MODEL, n_ctx=2048, verbose=False)
    prompt = SYSTEM_PROMPT.format(context=context or "No specific context available.", question=message)
    output = llm(prompt, max_tokens=400, stop=["Student message:", "\n\n\n"])
    return output["choices"][0]["text"].strip()

# Safety filter for post-generation
UNSAFE_PATTERNS = [
    "you should kill", "you can die", "take these pills", "end your life",
    "here's how to harm", "self harm method", "suicide method"
]

SAFE_FALLBACK = (
    "I want you to know that you're not alone in what you're feeling. "
    "It might really help to speak with a counselor or mental health professional "
    "who can provide personalized support. You deserve care and support."
)

def safety_filter(response: str) -> str:
    lower = response.lower()
    for pattern in UNSAFE_PATTERNS:
        if pattern in lower:
            return SAFE_FALLBACK
    return response

def generate_safe_response(message: str, emotion: str) -> str:
    context = retrieve_context(message)
    raw_response = generate_response(message, emotion, context)
    return safety_filter(raw_response)
