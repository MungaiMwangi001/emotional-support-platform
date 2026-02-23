"""
Unit and API tests for the AI Student Emotional Support Platform.
Run with: pytest tests/test_backend.py -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


# ─────────────────────────────────────────────
# Auth Tests
# ─────────────────────────────────────────────

class TestPasswordHashing:
    def test_hash_and_verify(self):
        from auth import hash_password, verify_password
        plain = "SecurePass123"
        hashed = hash_password(plain)
        assert hashed != plain
        assert verify_password(plain, hashed)

    def test_wrong_password_fails(self):
        from auth import hash_password, verify_password
        hashed = hash_password("correct_password")
        assert not verify_password("wrong_password", hashed)

    def test_different_hashes_for_same_password(self):
        from auth import hash_password
        h1 = hash_password("same_pass")
        h2 = hash_password("same_pass")
        assert h1 != h2  # bcrypt uses random salt


class TestJWTToken:
    def test_create_and_decode(self):
        from auth import create_access_token
        from jose import jwt
        from config import settings
        token = create_access_token({"sub": "testuser", "role": "student"})
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        assert payload["sub"] == "testuser"
        assert payload["role"] == "student"

    def test_token_has_expiration(self):
        from auth import create_access_token
        from jose import jwt
        from config import settings
        token = create_access_token({"sub": "testuser"})
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        assert "exp" in payload


# ─────────────────────────────────────────────
# Emotion Detection Tests
# ─────────────────────────────────────────────

class TestRiskScoring:
    def test_neutral_low_risk(self):
        from emotion_detector import compute_risk_score
        score = compute_risk_score("I had a good day today", "neutral", 0.9)
        assert score < 0.3

    def test_high_risk_keywords(self):
        from emotion_detector import compute_risk_score
        score = compute_risk_score("I want to kill myself", "sadness", 0.9)
        assert score >= 0.4

    def test_sadness_medium_risk(self):
        from emotion_detector import compute_risk_score
        score = compute_risk_score("I feel very sad and hopeless", "sadness", 0.8)
        assert 0.0 < score <= 1.0

    def test_joy_zero_risk(self):
        from emotion_detector import compute_risk_score
        score = compute_risk_score("I'm feeling great today!", "joy", 0.95)
        assert score == 0.0

    def test_score_bounded_0_to_1(self):
        from emotion_detector import compute_risk_score
        score = compute_risk_score(
            "suicide kill myself hurt myself want to die nothing to live for",
            "fear", 1.0
        )
        assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────
# RAG Pipeline Tests
# ─────────────────────────────────────────────

class TestResponseFiltering:
    def test_safe_response_unchanged(self):
        from rag_pipeline import filter_unsafe_response
        safe = "Remember to take care of yourself and reach out for support."
        assert filter_unsafe_response(safe) == safe

    def test_diagnostic_claim_filtered(self):
        from rag_pipeline import filter_unsafe_response
        unsafe = "Based on your symptoms, you have depression and you are diagnosed with anxiety."
        result = filter_unsafe_response(unsafe)
        assert "diagnosed" not in result.lower() or result != unsafe


# ─────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_message_detection(self):
        from emotion_detector import detect_emotion
        # Should not crash on minimal input
        emotion, conf = detect_emotion(".")
        assert isinstance(emotion, str)
        assert 0.0 <= conf <= 1.0

    def test_very_long_input(self):
        from emotion_detector import detect_emotion
        long_msg = "I feel stressed " * 200
        emotion, conf = detect_emotion(long_msg)
        assert isinstance(emotion, str)

    def test_risk_score_empty_string(self):
        from emotion_detector import compute_risk_score
        score = compute_risk_score("", "neutral", 0.5)
        assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────
# Config Tests
# ─────────────────────────────────────────────

class TestConfig:
    def test_settings_loaded(self):
        from config import settings
        assert settings.SECRET_KEY
        assert settings.ALGORITHM == "HS256"
        assert settings.RISK_ESCALATION_THRESHOLD > 0
        assert settings.ACCESS_TOKEN_EXPIRE_MINUTES > 0
