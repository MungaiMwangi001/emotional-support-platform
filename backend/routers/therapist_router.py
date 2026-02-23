from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from database import get_db, TherapistRequest, Chat, User
from auth import require_role

router = APIRouter(prefix="/therapist", tags=["therapist"])


class StatusUpdate(BaseModel):
    status: str  # pending, in_progress, resolved
    notes: str = ""


@router.get("/requests")
async def get_requests(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_role("therapist", "admin"))
):
    result = await db.execute(
        select(TherapistRequest).order_by(TherapistRequest.created_at.desc())
    )
    requests = result.scalars().all()
    return [
        {
            "id": r.id,
            "user_id": f"user_{r.user_id}",  # Anonymized
            "status": r.status,
            "notes": r.notes,
            "created_at": r.created_at.isoformat(),
        }
        for r in requests
    ]


@router.put("/requests/{request_id}")
async def update_request(
    request_id: int,
    update: StatusUpdate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_role("therapist", "admin"))
):
    if update.status not in ("pending", "in_progress", "resolved"):
        raise HTTPException(status_code=400, detail="Invalid status.")

    result = await db.execute(select(TherapistRequest).where(TherapistRequest.id == request_id))
    req = result.scalar_one_or_none()
    if not req:
        raise HTTPException(status_code=404, detail="Request not found.")

    req.status = update.status
    req.notes = update.notes
    await db.commit()
    return {"message": "Request updated successfully."}


@router.get("/emotional-summary/{user_id_anon}")
async def get_emotional_summary(
    user_id_anon: str,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_role("therapist", "admin"))
):
    """Get anonymized emotional summary for a user."""
    try:
        user_id = int(user_id_anon.replace("user_", ""))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format.")

    result = await db.execute(
        select(Chat).where(Chat.user_id == user_id).order_by(Chat.created_at.desc()).limit(20)
    )
    chats = result.scalars().all()

    if not chats:
        return {"summary": "No chat history found."}

    emotion_counts = {}
    avg_risk = 0
    for c in chats:
        emotion_counts[c.detected_emotion] = emotion_counts.get(c.detected_emotion, 0) + 1
        avg_risk += c.risk_score

    avg_risk = round(avg_risk / len(chats), 3) if chats else 0
    dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "unknown"

    return {
        "total_sessions": len(chats),
        "dominant_emotion": dominant_emotion,
        "emotion_distribution": emotion_counts,
        "average_risk_score": avg_risk,
        "high_risk_events": sum(1 for c in chats if c.escalation_triggered),
    }
