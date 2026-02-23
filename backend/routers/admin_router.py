from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from database import get_db, Chat, User, RiskFlag, SystemLog, TherapistRequest
from auth import require_role
from datetime import datetime, timedelta

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/dashboard")
async def admin_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_role("admin"))
):
    # Total users
    total_users = await db.execute(select(func.count(User.id)))
    total_chats = await db.execute(select(func.count(Chat.id)))
    total_flags = await db.execute(select(func.count(RiskFlag.id)))
    pending_requests = await db.execute(
        select(func.count(TherapistRequest.id)).where(TherapistRequest.status == "pending")
    )

    return {
        "total_users": total_users.scalar(),
        "total_chats": total_chats.scalar(),
        "total_risk_flags": total_flags.scalar(),
        "pending_therapist_requests": pending_requests.scalar(),
    }


@router.get("/flagged-users")
async def get_flagged_users(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_role("admin"))
):
    result = await db.execute(
        select(RiskFlag).order_by(RiskFlag.created_at.desc()).limit(50)
    )
    flags = result.scalars().all()
    return [
        {
            "flag_id": f.id,
            "user_id": f"user_{f.user_id}",  # Anonymized
            "risk_score": f.risk_score,
            "timestamp": f.created_at.isoformat(),
        }
        for f in flags
    ]


@router.get("/chat-logs")
async def get_chat_logs(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_role("admin"))
):
    result = await db.execute(
        select(Chat).order_by(Chat.created_at.desc()).limit(100)
    )
    chats = result.scalars().all()
    return [
        {
            "chat_id": c.id,
            "user_id": f"user_{c.user_id}",  # Anonymized
            "emotion": c.detected_emotion,
            "confidence": c.emotion_confidence,
            "risk_score": c.risk_score,
            "escalated": c.escalation_triggered,
            "timestamp": c.created_at.isoformat(),
            # Do NOT include message content for privacy
        }
        for c in chats
    ]


@router.get("/system-logs")
async def get_system_logs(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_role("admin"))
):
    result = await db.execute(
        select(SystemLog).order_by(SystemLog.timestamp.desc()).limit(100)
    )
    logs = result.scalars().all()
    return [{"event": l.event_type, "timestamp": l.timestamp.isoformat()} for l in logs]
