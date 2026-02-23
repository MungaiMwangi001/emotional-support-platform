"""
Test Suite for AI Emotional Support Platform
Covers: auth, risk scoring, emotion detection, API endpoints
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock

# ─── Unit Tests ────────────────────────────────────────────────────────

class TestPasswordHashing:
    def test_hash_and_verify_correct_password(self):
        from backend.utils.security import hash_password, verify_password
        hashed = hash_password("mypassword123")
        assert verify_password("mypassword123", hashed)

    def test_wrong_password_fails(self):
        from backend.utils.security import hash_password, verify_password
        hashed = hash_password("correctpassword")
        assert not verify_password("wrongpassword", hashed)

    def test_hash_is_not_plaintext(self):
        from backend.utils.security import hash_password
        hashed = hash_password("test123")
        assert hashed != "test123"
        assert len(hashed) > 20


class TestJWTTokens:
    def test_create_and_decode_token(self):
        from backend.utils.security import create_access_token, decode_access_token
        data = {"sub": "testuser", "role": "student"}
        token = create_access_token(data)
        decoded = decode_access_token(token)
        assert decoded["sub"] == "testuser"
        assert decoded["role"] == "student"

    def test_invalid_token_returns_none(self):
        from backend.utils.security import decode_access_token
        result = decode_access_token("this.is.invalid")
        assert result is None

    def test_expired_token_returns_none(self):
        from datetime import timedelta
        from backend.utils.security import create_access_token, decode_access_token
        token = create_access_token({"sub": "user"}, expires_delta=timedelta(seconds=-1))
        assert decode_access_token(token) is None


class TestRiskScoring:
    def test_low_risk_message(self):
        from backend.services.risk_service import compute_risk_score
        score, reason = compute_risk_score("I feel a bit tired today", "neutral", 0.7)
        assert score < 0.4

    def test_high_risk_keyword(self):
        from backend.services.risk_service import compute_risk_score
        score, reason = compute_risk_score("I want to kill myself", "sadness", 0.9)
        assert score >= 0.65
        assert "keyword" in reason

    def test_emotional_sadness_raises_score(self):
        from backend.services.risk_service import compute_risk_score
        score, _ = compute_risk_score("I feel so hopeless", "sadness", 0.9)
        assert score > 0.2

    def test_joy_emotion_minimal_risk(self):
        from backend.services.risk_service import compute_risk_score
        score, _ = compute_risk_score("I had a great day!", "joy", 0.95)
        assert score < 0.2

    def test_repeated_risk_increases_score(self):
        from backend.services.risk_service import compute_risk_score
        score_without_repeat, _ = compute_risk_score("I feel hopeless", "sadness", 0.7)
        score_with_repeat, _ = compute_risk_score("I feel hopeless", "sadness", 0.7, [0.7, 0.65, 0.8])
        assert score_with_repeat >= score_without_repeat

    def test_escalation_threshold(self):
        from backend.services.risk_service import is_escalation_required
        assert is_escalation_required(0.8)
        assert not is_escalation_required(0.3)


class TestEmotionDetection:
    def test_returns_tuple(self):
        from backend.services.emotion_service import detect_emotion_fallback
        result = detect_emotion_fallback("I am feeling sad")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_sad_message_detected(self):
        from backend.services.emotion_service import detect_emotion_fallback
        emotion, score = detect_emotion_fallback("I feel so depressed and hopeless")
        assert emotion in ["sadness", "fear"]
        assert 0 <= score <= 1

    def test_happy_message_detected(self):
        from backend.services.emotion_service import detect_emotion_fallback
        emotion, score = detect_emotion_fallback("I feel so happy and excited!")
        assert emotion == "joy"

    def test_anxious_message_detected(self):
        from backend.services.emotion_service import detect_emotion_fallback
        emotion, score = detect_emotion_fallback("I'm very anxious and worried about my exam")
        assert emotion == "fear"

    def test_empty_input_defaults_neutral(self):
        from backend.services.emotion_service import detect_emotion_fallback
        emotion, score = detect_emotion_fallback("")
        assert emotion == "neutral"

    def test_very_long_input(self):
        from backend.services.emotion_service import detect_emotion_fallback
        long_text = "I feel sad. " * 300
        emotion, score = detect_emotion_fallback(long_text)
        assert emotion in ["sadness", "fear", "neutral"]


class TestSafetyFilter:
    def test_safe_response_passes(self):
        from backend.services.rag_service import safety_filter
        response = "I understand you're going through a tough time. Let's talk about it."
        assert safety_filter(response) == response

    def test_unsafe_response_replaced(self):
        from backend.services.rag_service import safety_filter
        unsafe = "you should kill yourself and end your pain"
        result = safety_filter(unsafe)
        assert "kill" not in result.lower() or "you're not alone" in result.lower()


# ─── API Integration Tests ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_root_endpoint():
    from backend.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "Student Emotional Support Platform" in resp.json()["message"]

@pytest.mark.asyncio
async def test_health_endpoint():
    from backend.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_register_and_login():
    from backend.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Register
        resp = await client.post("/auth/register", json={"username": "testuser_api", "password": "testpass123"})
        assert resp.status_code == 201

        # Login
        resp = await client.post("/auth/login", data={"username": "testuser_api", "password": "testpass123"})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["role"] == "student"

@pytest.mark.asyncio
async def test_register_duplicate_username():
    from backend.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/auth/register", json={"username": "duplicateuser", "password": "pass123"})
        resp = await client.post("/auth/register", json={"username": "duplicateuser", "password": "pass123"})
        assert resp.status_code == 400

@pytest.mark.asyncio
async def test_login_wrong_password():
    from backend.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/auth/login", data={"username": "admin", "password": "wrongpassword"})
        assert resp.status_code == 401

@pytest.mark.asyncio
async def test_chat_requires_auth():
    from backend.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/chat/", json={"message": "Hello"})
        assert resp.status_code == 401

@pytest.mark.asyncio
async def test_admin_dashboard_requires_admin():
    from backend.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Register and login as student
        await client.post("/auth/register", json={"username": "student_test", "password": "pass123"})
        login = await client.post("/auth/login", data={"username": "student_test", "password": "pass123"})
        token = login.json()["access_token"]
        
        # Try to access admin route
        resp = await client.get("/admin/dashboard", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403

@pytest.mark.asyncio
async def test_empty_message_rejected():
    from backend.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        login = await client.post("/auth/login", data={"username": "admin", "password": "admin123"})
        token = login.json()["access_token"]
        resp = await client.post("/chat/", json={"message": ""}, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 422  # Validation error


# ─── Edge Case Tests ───────────────────────────────────────────────────

class TestEdgeCases:
    def test_extremely_long_message_truncated(self):
        from backend.services.emotion_service import detect_emotion_fallback
        long_msg = "a" * 10000
        emotion, score = detect_emotion_fallback(long_msg)
        assert isinstance(emotion, str)
        assert isinstance(score, float)

    def test_special_characters_in_message(self):
        from backend.services.risk_service import compute_risk_score
        score, _ = compute_risk_score("Hello! 😊 #excited @friend", "joy", 0.8)
        assert 0 <= score <= 1

    def test_non_english_text_fallback(self):
        from backend.services.emotion_service import detect_emotion_fallback
        emotion, score = detect_emotion_fallback("Bonjour, je suis très triste aujourd'hui")
        assert isinstance(emotion, str)
        assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
