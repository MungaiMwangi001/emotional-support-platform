from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel
from database import get_db, Chat, RiskFlag, TherapistRequest, User
from auth import get_current_user, require_role
from emotion_detector import detect_emotion, compute_risk_score
from rag_pipeline import generate_response, filter_unsafe_response
from config import settings
from datetime import datetime, timedelta

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str


@router.post("/")
async def chat(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("student"))
):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    message = req.message.strip()[:2000]  # Limit input length

    # Emotion detection
    emotion, confidence = detect_emotion(message)

    # Risk scoring
    risk_score = compute_risk_score(message, emotion, confidence)
    escalation_triggered = risk_score >= settings.RISK_ESCALATION_THRESHOLD

    # Generate RAG response
    raw_response = generate_response(message, emotion)
    response = filter_unsafe_response(raw_response)

    # Add crisis message if escalation triggered
    if escalation_triggered:
        crisis_msg = (
            "\n\n⚠️ **I'm concerned about your wellbeing.** "
            "If you're in crisis, please reach out immediately:\n"
            "- **Crisis Text Line**: Text HOME to 741741\n"
            "- **National Suicide Prevention Lifeline**: 988\n"
            "- **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/\n\n"
            "Would you like me to connect you with a therapist? You can request therapist contact from the menu."
        )
        response += crisis_msg

    # Save to DB
    chat_entry = Chat(
        user_id=current_user.id,
        message=message,
        response=response,
        detected_emotion=emotion,
        emotion_confidence=confidence,
        risk_score=risk_score,
        escalation_triggered=escalation_triggered,
    )
    db.add(chat_entry)

    if escalation_triggered:
        risk_flag = RiskFlag(
            user_id=current_user.id,
            risk_score=risk_score,
            escalated=True,
        )
        db.add(risk_flag)

    await db.commit()

    return {
        "reply": response,
        "detected_emotion": emotion,
        "confidence_score": confidence,
        "risk_score": risk_score,
        "escalation_triggered": escalation_triggered,
    }


@router.get("/history")
async def get_history(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("student"))
):
    result = await db.execute(
        select(Chat).where(Chat.user_id == current_user.id).order_by(Chat.created_at.desc()).limit(50)
    )
    chats = result.scalars().all()
    return [
        {
            "message": c.message,
            "response": c.response,
            "emotion": c.detected_emotion,
            "confidence": c.emotion_confidence,
            "risk_score": c.risk_score,
            "escalated": c.escalation_triggered,
            "timestamp": c.created_at.isoformat(),
        }
        for c in chats
    ]


@router.post("/request-therapist")
async def request_therapist(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("student"))
):
    # Check for existing pending request
    existing = await db.execute(
        select(TherapistRequest).where(
            TherapistRequest.user_id == current_user.id,
            TherapistRequest.status == "pending"
        )
    )
    if existing.scalar_one_or_none():
        return {"message": "You already have a pending therapist request."}

    req = TherapistRequest(user_id=current_user.id, status="pending")
    db.add(req)
    await db.commit()
    return {"message": "Therapist contact request submitted successfully."}
