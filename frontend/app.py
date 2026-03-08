import streamlit as st
import requests
import json
from datetime import datetime

import os
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SalamaSPace",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Background */
    .stApp { background: linear-gradient(135deg, #f0f4ff 0%, #f8f0ff 100%); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3561 0%, #1a1f3a 100%);
        color: white;
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 10px;
        color: white;
        width: 100%;
        padding: 0.6rem;
        transition: all 0.2s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.2);
    }

    /* Cards */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border: 1px solid rgba(0,0,0,0.05);
    }

    /* Emotion Badge */
    .emotion-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: capitalize;
    }

    /* Chat bubbles */
    .chat-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-ai {
        background: white;
        color: #2d3561;
        border-radius: 18px 18px 18px 4px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        max-width: 80%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #eef0ff;
    }

    /* Risk indicator */
    .risk-low { color: #10b981; }
    .risk-medium { color: #f59e0b; }
    .risk-high { color: #ef4444; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }

    /* Disclaimer banner */
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        font-size: 0.85rem;
    }

    /* Status badges */
    .status-pending { background: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 10px; }
    .status-progress { background: #dbeafe; color: #1e40af; padding: 2px 8px; border-radius: 10px; }
    .status-resolved { background: #d1fae5; color: #065f46; padding: 2px 8px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "token" not in st.session_state:
    st.session_state.token = None
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "page" not in st.session_state:
    st.session_state.page = "login"

# ─────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────
def api_post(endpoint, data, auth=False):
    headers = {}
    if auth and st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=data, headers=headers, timeout=300)
        return r
    except requests.exceptions.ConnectionError:
        st.error(" Cannot connect to backend. Please ensure the server is running.")
        return None


def api_get(endpoint):
    headers = {}
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    try:
        r = requests.get(f"{API_BASE}{endpoint}", headers=headers, timeout=300)
        return r
    except requests.exceptions.ConnectionError:
        st.error(" Cannot connect to backend.")
        return None


def api_put(endpoint, data):
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    try:
        r = requests.put(f"{API_BASE}{endpoint}", json=data, headers=headers, timeout=30)
        return r
    except:
        return None


def logout():
    st.session_state.token = None
    st.session_state.role = None
    st.session_state.username = None
    st.session_state.chat_history = []
    st.session_state.page = "login"


# ─────────────────────────────────────────────
# Emotion styling
# ─────────────────────────────────────────────
EMOTION_COLORS = {
    "joy": "#10b981",
    "sadness": "#6366f1",
    "anger": "#ef4444",
    "fear": "#f59e0b",
    "disgust": "#8b5cf6",
    "surprise": "#06b6d4",
    "neutral": "#6b7280",
}

EMOTION_EMOJIS = {
    "joy": "😊", "sadness": "😢", "anger": "😠",
    "fear": "😰", "disgust": "🤢", "surprise": "😲", "neutral": "😐"
}

def emotion_badge(emotion, confidence):
    color = EMOTION_COLORS.get(emotion, "#6b7280")
    emoji = EMOTION_EMOJIS.get(emotion, "😐")
    return f"""<span class="emotion-badge" style="background:{color}22; color:{color}; border: 1px solid {color}66;">
        {emoji} {emotion.capitalize()} ({confidence:.0%})
    </span>"""


# ─────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────

def page_disclaimer():
    st.markdown("""
    <div class="main-header">
        <h1>SalamaSpace</h1>
        <p>AI-Powered Student Emotional Support Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <h3>⚠️ Important Disclaimer</h3>
        <p><strong>        <h1>SalamaSpace</h1>
is NOT a replacement for professional mental health care.</strong></p>
        <ul>
            <li>This platform provides informational and emotional support only.</li>
            <li>It does not provide medical diagnoses or clinical treatment.</li>
            <li>If you are in crisis, please contact emergency services or a crisis hotline immediately.</li>
            <li>Crisis Text Line: Text HOME to 741741 | Lifeline: 988</li>
        </ul>
        <p><strong>By continuing, you acknowledge this disclaimer.</strong></p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ I Understand – Continue to Login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()


def page_login():
    st.markdown("""
    <div class="main-header">
        <h1>SalamaSpace</h1>
        <p>Sign in to access your support space</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            r = requests.post(
                f"{API_BASE}/auth/login",
                data={"username": username, "password": password},
                timeout=30
            )
            if r and r.status_code == 200:
                data = r.json()
                st.session_state.token = data["access_token"]
                st.session_state.role = data["role"]
                st.session_state.username = data["username"]
                st.session_state.page = "chat" if data["role"] == "student" else (
                    "admin" if data["role"] == "admin" else "therapist"
                )
                st.success("✅ Login successful!")
                st.rerun()
            elif r:
                st.error(f"❌ {r.json().get('detail', 'Login failed.')}")

    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("Choose Username")
            new_pass = st.text_input("Choose Password (min 6 chars)", type="password")
            role = st.selectbox("Role", ["student", "therapist", "admin"])
            consent = st.checkbox("I consent to my anonymized data being used to improve the platform.")
            reg_submitted = st.form_submit_button("Register", use_container_width=True)

        if reg_submitted:
            if not consent:
                st.warning("⚠️ You must consent to data usage to register.")
            else:
                r = api_post("/auth/register", {"username": new_user, "password": new_pass, "role": role, "consent": consent})
                if r and r.status_code == 200:
                    st.success("✅ Registered! Please log in.")
                elif r:
                    st.error(f"❌ {r.json().get('detail', 'Registration failed.')}")


def page_chat():
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h2>💬 Your Support Space</h2>
        <p>A safe, private space to share how you're feeling</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        🔒 <strong>Your privacy matters.</strong> Messages are stored anonymously and only used to provide support. 
        This is not a substitute for professional mental health care.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Chat display
    chat_container = st.container()
    with chat_container:
        for item in st.session_state.chat_history:
            st.markdown(f'<div class="chat-user">👤 {item["user"]}</div>', unsafe_allow_html=True)
            badge = emotion_badge(item.get("emotion", "neutral"), item.get("confidence", 0.5))
            st.markdown(badge, unsafe_allow_html=True)

            if item.get("escalated"):
                st.markdown('<div style="color:#ef4444; font-size:0.8rem;">⚠️ High-risk event detected – crisis resources provided</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="chat-ai">🧠 {item["response"]}</div>', unsafe_allow_html=True)
            st.markdown("---")

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("How are you feeling today?", placeholder="Share what's on your mind...", height=100)
        col1, col2 = st.columns([4, 1])
        with col1:
            send = st.form_submit_button("💬 Send", use_container_width=True)
        with col2:
            st.form_submit_button("🗑️ Clear", on_click=lambda: st.session_state.update({"chat_history": []}))

    if send and user_input.strip():
        with st.spinner(" Thinking..."):
            r = api_post("/chat/", {"message": user_input}, auth=True)
            if r and r.status_code == 200:
                data = r.json()
                st.session_state.chat_history.append({
                    "user": user_input,
                    "response": data["reply"],
                    "emotion": data["detected_emotion"],
                    "confidence": data["confidence_score"],
                    "risk_score": data["risk_score"],
                    "escalated": data["escalation_triggered"],
                })
                st.rerun()
            elif r:
                st.error(f"Error: {r.json().get('detail', 'Request failed.')}")

    # Therapist request button
    st.markdown("---")
    if st.button("👩‍⚕️ Request Therapist Contact", use_container_width=False):
        r = api_post("/chat/request-therapist", {}, auth=True)
        if r and r.status_code == 200:
            st.success(r.json()["message"])
        elif r:
            st.warning(r.json().get("detail", "Request failed."))


def page_admin():
    st.markdown("""
    <div class="main-header">
        <h2>🛡️ Admin Dashboard</h2>
        <p>System monitoring and management</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    r = api_get("/admin/dashboard")
    if r and r.status_code == 200:
        data = r.json()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Total Users", data["total_users"])
        col2.metric("💬 Total Chats", data["total_chats"])
        col3.metric("⚠️ Risk Flags", data["total_risk_flags"])
        col4.metric("📋 Pending Requests", data["pending_therapist_requests"])

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([" Flagged Users", "Chat Logs", "📝 System Logs"])

    with tab1:
        r = api_get("/admin/flagged-users")
        if r and r.status_code == 200:
            flags = r.json()
            if flags:
                import pandas as pd
                df = pd.DataFrame(flags)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No risk flags found.")

    with tab2:
        r = api_get("/admin/chat-logs")
        if r and r.status_code == 200:
            logs = r.json()
            if logs:
                import pandas as pd
                df = pd.DataFrame(logs)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No chat logs found.")

    with tab3:
        r = api_get("/admin/system-logs")
        if r and r.status_code == 200:
            logs = r.json()
            if logs:
                import pandas as pd
                df = pd.DataFrame(logs)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No system logs found.")


def page_therapist():
    st.markdown("""
    <div class="main-header">
        <h2>👩‍⚕️ Therapist Portal</h2>
        <p>Manage student contact requests and emotional summaries</p>
    </div>
    """, unsafe_allow_html=True)

    r = api_get("/therapist/requests")
    if r and r.status_code == 200:
        requests_data = r.json()
        if not requests_data:
            st.info("No therapist requests at this time.")
            return

        for req in requests_data:
            with st.expander(f"📋 Request #{req['id']} | {req['user_id']} | Status: {req['status'].upper()}"):
                st.write(f"**Created:** {req['created_at'][:19]}")
                st.write(f"**Current Status:** {req['status']}")
                if req.get('notes'):
                    st.write(f"**Notes:** {req['notes']}")

                # Get emotional summary
                sum_r = api_get(f"/therapist/emotional-summary/{req['user_id']}")
                if sum_r and sum_r.status_code == 200:
                    summary = sum_r.json()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Sessions", summary.get("total_sessions", 0))
                    col2.metric("Dominant Emotion", summary.get("dominant_emotion", "N/A"))
                    col3.metric("Avg Risk Score", f"{summary.get('average_risk_score', 0):.2f}")

                # Update status
                new_status = st.selectbox(
                    "Update Status",
                    ["pending", "in_progress", "resolved"],
                    index=["pending", "in_progress", "resolved"].index(req['status']),
                    key=f"status_{req['id']}"
                )
                notes = st.text_input("Notes", value=req.get('notes', ''), key=f"notes_{req['id']}")

                if st.button("💾 Update", key=f"update_{req['id']}"):
                    upd = api_put(f"/therapist/requests/{req['id']}", {"status": new_status, "notes": notes})
                    if upd and upd.status_code == 200:
                        st.success("✅ Updated!")
                        st.rerun()


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠SalamaSpace")
    st.markdown("---")

    if st.session_state.token:
        st.markdown(f"**👤 {st.session_state.username}**")
        st.markdown(f"*Role: {st.session_state.role.capitalize()}*")
        st.markdown("---")

        if st.session_state.role == "student":
            if st.button("💬 Chat"):
                st.session_state.page = "chat"
                st.rerun()
            if st.button("📖 Chat History"):
                st.session_state.page = "history"
                st.rerun()

        elif st.session_state.role == "admin":
            if st.button("🛡️ Dashboard"):
                st.session_state.page = "admin"
                st.rerun()

        elif st.session_state.role == "therapist":
            if st.button("👩‍⚕️ Requests"):
                st.session_state.page = "therapist"
                st.rerun()

        st.markdown("---")
        if st.button("🚪 Logout"):
            logout()
            st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem; opacity:0.8;">
        🚨 <strong>Crisis Resources</strong><br>
        Crisis Text: HOME → 741741<br>
        Lifeline: 988
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("Please log in to continue.")
        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem; opacity:0.8;">
        🔒 Secure & Private<br>
        🚫 Not a medical service<br>
        🆘 Crisis: 988
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Route to page
# ─────────────────────────────────────────────
if not st.session_state.token:
    if st.session_state.page == "login":
        page_login()
    else:
        page_disclaimer()
else:
    if st.session_state.page == "chat":
        page_chat()
    elif st.session_state.page == "history":
        # Chat history page
        st.markdown("""<div class="main-header"><h2>📖 Chat History</h2></div>""", unsafe_allow_html=True)
        r = api_get("/chat/history")
        if r and r.status_code == 200:
            history = r.json()
            if not history:
                st.info("No chat history yet.")
            for item in history:
                with st.expander(f"🕐 {item['timestamp'][:19]} | {item['emotion'].capitalize()} | Risk: {item['risk_score']:.2f}"):
                    st.markdown(f"**You:** {item['message']}")
                    st.markdown(f"**AI:** {item['response'][:300]}...")
                    badge = emotion_badge(item['emotion'], item['confidence'])
                    st.markdown(badge, unsafe_allow_html=True)
    elif st.session_state.page == "admin":
        page_admin()
    elif st.session_state.page == "therapist":
        page_therapist()
    else:
        st.session_state.page = "chat" if st.session_state.role == "student" else "admin"
        st.rerun()
