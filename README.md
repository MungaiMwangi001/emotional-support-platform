# 🧠 Salama SPace 
# SalamaSPace – AI Student Emotional Support Platform

A secure, ethically compliant AI-powered emotional support system for students. Built with FastAPI, Streamlit, HuggingFace Transformers, and LangChain RAG.

---

## ⚠️ Disclaimer

**SalamaSpace is NOT a replacement for professional mental health care.** It provides informational and emotional support only. In a crisis, contact:
- **Crisis Text Line**: Text HOME to 741741
- **988 Suicide & Crisis Lifeline**: Call or text 988

---

## 🏗️ Architecture

```
┌─────────────────┐     HTTP      ┌──────────────────────────────────────┐
│  Streamlit UI   │ ──────────── │          FastAPI Backend               │
│  (Port 8501)    │              │  ┌─────────┐ ┌──────────┐ ┌────────┐  │
└─────────────────┘              │  │  Auth   │ │  Chat    │ │ Admin  │  │
                                 │  │  (JWT)  │ │  (RAG)   │ │  APIs  │  │
                                 │  └─────────┘ └──────────┘ └────────┘  │
                                 │  ┌──────────────────────────────────┐  │
                                 │  │  AI Pipeline                      │  │
                                 │  │  ┌──────────────┐ ┌───────────┐  │  │
                                 │  │  │ Emotion Model│ │ RAG Chain │  │  │
                                 │  │  │ (DistilBERT) │ │ (FAISS)   │  │  │
                                 │  │  └──────────────┘ └───────────┘  │  │
                                 │  └──────────────────────────────────┘  │
                                 │  ┌──────────────┐                       │
                                 │  │ SQLite DB    │                       │
                                 │  └──────────────┘                       │
                                 └──────────────────────────────────────────┘
```

## 👤 User Roles

| Role | Access |
|------|--------|
| **Student** | Register, login, chat with AI, view emotion, request therapist |
| **Admin** | View anonymized logs, flagged users, system stats |
| **Therapist** | View contact requests, emotional summaries, update status |

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Clone or unzip the project
cd emotional-support-platform

# 2. Copy and configure environment
cp backend/.env.example backend/.env
# Edit backend/.env with your settings

# 3. Build and run
docker-compose up --build

# Access:
#   Frontend: http://localhost:8501
#   Backend API: http://localhost:8000
#   API Docs: http://localhost:8000/docs
```

### Option 2: Local Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env
uvicorn main:app --reload --port 8000
```

**Frontend (in a separate terminal):**
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

---

## 📁 Project Structure

```
emotional-support-platform/
├── backend/
│   ├── main.py               # FastAPI app entrypoint
│   ├── config.py             # Settings & env vars
│   ├── database.py           # SQLAlchemy models & DB init
│   ├── auth.py         
      # JWT auth, password hashing, RBAC
│   ├── emotion_detector.py   # HuggingFace emotion model + risk scoring
│   ├── rag_pipeline.py       # LangChain RAG + response generation
│   ├── routers/
│   │   ├── auth_router.py    # /auth/register, /auth/login
│   │   ├── chat_router.py    # /chat/, /chat/history, /chat/request-therapist
│   │   ├── admin_router.py   # /admin/dashboard, /admin/flagged-users
│   │   └── therapist_router.py # /therapist/requests
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app.py                # Streamlit multi-page application
│   └── requirements.txt
├── knowledge_base/           # Psychoeducation documents for RAG
│   ├── stress_management.txt
│   ├── anxiety.txt
│   ├── depression_support.txt
│   ├── sleep_mental_health.txt
│   └── seeking_help.txt
├── tests/
│   └── test_backend.py       # Unit & integration tests
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
└── README.md
```

---

## 🔐 Security & Ethics

| Feature | Implementation |
|---------|---------------|
| Password storage | bcrypt hashing |
| Authentication | JWT with expiration |
| Authorization | Role-based (student/admin/therapist) |
| Data privacy | Messages anonymized in admin/therapist views |
| Consent | Required on registration |
| Safety filtering | Post-generation output filter |
| Crisis escalation | Auto-triggered above risk threshold |
| Not a medical tool | Disclaimer on every interface |

---

## 🤖 AI Pipeline

### Emotion Detection
- **Model**: `j-hartmann/emotion-english-distilroberta-base` (HuggingFace)
- **Labels**: joy, sadness, anger, fear, disgust, surprise, neutral
- **Output**: emotion label + confidence score

### Risk Scoring
- Combines emotion weight × confidence + keyword detection
- Threshold (default 0.65) triggers crisis escalation
- Crisis resources shown + therapist contact offered

### RAG Response Generation
- **Vector Store**: FAISS (CPU)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Knowledge Base**: 5 curated psychoeducation documents
- **LLM**: Tries local Ollama (Mistral), falls back to empathetic template responses
- **Safety**: Post-generation filter removes diagnostic claims

---

## 🧪 Running Tests

```bash
cd backend
pip install pytest pytest-asyncio
pytest ../tests/test_backend.py -v
```

---

## 🌐 API Documentation

Once running, visit `http://localhost:8000/docs` for interactive Swagger API docs.

### Key Endpoints

| Method | Endpoint | Role | Description |
|--------|----------|------|-------------|
| POST | `/auth/register` | Public | Register new user |
| POST | `/auth/login` | Public | Login, get JWT token |
| POST | `/chat/` | Student | Send message, get AI response |
| GET | `/chat/history` | Student | View chat history |
| POST | `/chat/request-therapist` | Student | Request therapist contact |
| GET | `/admin/dashboard` | Admin | System statistics |
| GET | `/admin/flagged-users` | Admin | View high-risk users |
| GET | `/therapist/requests` | Therapist | View contact requests |
| PUT | `/therapist/requests/{id}` | Therapist | Update request status |

---

## 📊 Evaluation Metrics

The system is designed to be evaluated on:
- **Emotion model**: Accuracy, Precision, Recall, F1 on labeled test set
- **RAG quality**: Response with vs. without RAG context comparison
- **Risk system**: Precision/recall of escalation triggers
- **Performance**: Response time, model load time, memory usage

---

## 🔧 Configuration (`.env`)

```env
SECRET_KEY=your-secret-key-min-32-chars
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
DATABASE_URL=sqlite+aiosqlite:///./emotional_support.db
RISK_ESCALATION_THRESHOLD=0.65
EMOTION_MODEL=j-hartmann/emotion-english-distilroberta-base
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
KNOWLEDGE_BASE_DIR=../knowledge_base
VECTOR_STORE_PATH=./vector_store
```

---

## 📋 Data Retention Policy

- Chat messages are stored with anonymized user IDs
- No real names or personal identifiers are stored beyond username
- Admin/therapist views show only anonymized `user_XXX` identifiers
- Users may request data deletion by contacting administrators
- Data is stored locally and never shared with third-party services
