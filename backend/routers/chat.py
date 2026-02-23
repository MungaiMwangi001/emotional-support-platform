"""
Chat Router - Main AI interaction endpoint (student only)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, constr
from datetime import datetime, timedelta
from typing import Optional

from backend.database import get_db, Chat, RiskFlag, TherapistRequest
from backend.utils.auth_deps import require_role
from backend.database import User
from backend.services.emotion_service import detect_emotion
from backend.services.risk_service import compute_risk_score, is_escalation_required, CRISIS_MESSAGE
from backend.services.rag_service import generate_safe_response
from backend.utils.logger import log_event

router = APIRouter()

class ChatRequest(BaseModel):
    message: constr(min_length=1, max_length=2000)

class TherapistRequestCreate(BaseModel):
    notes: Optional[str] = None

@router.post("/")
async def chat(
    req: ChatRequest,
    current_user: User = Depends(require_role("student")),
    db: AsyncSession = Depends(get_db)
):
    if not current_user.consent_given:
        raise HTTPException(status_code=403, detail="Consent required before using the chat feature.")

    # Detect emotion
    emotion_label, confidence = detect_emotion(req.message)

    # Get recent risk scores for repetition tracking
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    recent_chats_result = await db.execute(
        select(Chat.risk_score).where(
            Chat.user_id == current_user.id,
            Chat.created_at >= one_hour_ago
        ).order_by(Chat.created_at.desc()).limit(10)
    )
    recent_risk_scores = [row[0] for row in recent_chats_result.fetchall() if row[0] is not None]

    # Compute risk
    risk_score, risk_reason = compute_risk_score(req.message, emotion_label, confidence, recent_risk_scores)
    escalation = is_escalation_required(risk_score)

    # Generate AI response
    ai_response = generate_safe_response(req.message, emotion_label)
    if escalation:
        ai_response = CRISIS_MESSAGE + "\n\n---\n\n" + ai_response

    # Store chat
    chat_record = Chat(
        user_id=current_user.id,
        message=req.message,
        ai_response=ai_response,
        detected_emotion=emotion_label,
        emotion_confidence=confidence,
        risk_score=risk_score,
        escalation_triggered=escalation
    )
    db.add(chat_record)

    # Store risk flag if escalated
    if escalation:
        flag = RiskFlag(
            user_id=current_user.id,
            risk_score=risk_score,
            escalated=True,
            trigger_reason=risk_reason
        )
        db.add(flag)
        log_event("risk_escalation", f"User {current_user.id} | Score: {risk_score} | Reason: {risk_reason}")

    await db.commit()
    log_event("chat_message", f"User {current_user.id} | Emotion: {emotion_label} | Risk: {risk_score}")

    return {
        "reply": ai_response,
        "detected_emotion": emotion_label,
        "confidence_score": confidence,
        "risk_score": risk_score,
        "escalation_triggered": escalation
    }

@router.get("/history")
async def get_chat_history(
    current_user: User = Depends(require_role("student")),
    db: AsyncSession = Depends(get_db),
    limit: int = 20
):
    result = await db.execute(
        select(Chat).where(Chat.user_id == current_user.id)
        .order_by(Chat.created_at.desc()).limit(limit)
    )
    chats = result.scalars().all()
    return [
        {
            "id": c.id,
            "message": c.message,
            "ai_response": c.ai_response,
            "detected_emotion": c.detected_emotion,
            "confidence_score": c.emotion_confidence,
            "risk_score": c.risk_score,
            "escalation_triggered": c.escalation_triggered,
            "created_at": c.created_at.isoformat()
        }
        for c in chats
    ]

@router.post("/request-therapist")
async def request_therapist(
    req: TherapistRequestCreate,
    current_user: User = Depends(require_role("student")),
    db: AsyncSession = Depends(get_db)
):
    tr = TherapistRequest(user_id=current_user.id, notes=req.notes)
    db.add(tr)
    await db.commit()
    log_event("therapist_request", f"User {current_user.id} requested therapist contact")
    return {"message": "Therapist contact request submitted. A counselor will reach out soon."}
