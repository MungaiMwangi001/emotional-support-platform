"""
Admin Router - Dashboard, logs, analytics (admin only)
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from backend.database import get_db, Chat, RiskFlag, TherapistRequest, User, SystemLog
from backend.utils.auth_deps import require_role

router = APIRouter()

@router.get("/dashboard")
async def admin_dashboard(
    current_user: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db)
):
    # Counts
    total_users = (await db.execute(select(func.count(User.id)).where(User.role == "student"))).scalar()
    total_chats = (await db.execute(select(func.count(Chat.id)))).scalar()
    total_escalations = (await db.execute(select(func.count(RiskFlag.id)).where(RiskFlag.escalated == True))).scalar()
    pending_requests = (await db.execute(select(func.count(TherapistRequest.id)).where(TherapistRequest.status == "pending"))).scalar()

    # Emotion distribution
    emotion_dist_result = await db.execute(
        select(Chat.detected_emotion, func.count(Chat.id).label("count"))
        .group_by(Chat.detected_emotion)
        .order_by(desc("count"))
    )
    emotion_distribution = {row[0]: row[1] for row in emotion_dist_result.fetchall() if row[0]}

    return {
        "total_students": total_users,
        "total_chat_messages": total_chats,
        "total_escalations": total_escalations,
        "pending_therapist_requests": pending_requests,
        "emotion_distribution": emotion_distribution
    }

@router.get("/flagged-users")
async def get_flagged_users(
    current_user: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(RiskFlag).where(RiskFlag.escalated == True)
        .order_by(desc(RiskFlag.created_at)).limit(50)
    )
    flags = result.scalars().all()
    # Anonymize: only show user_id, not username
    return [
        {
            "flag_id": f.id,
            "anonymous_user_ref": f"USER-{f.user_id:04d}",
            "risk_score": f.risk_score,
            "trigger_reason": f.trigger_reason,
            "created_at": f.created_at.isoformat()
        }
        for f in flags
    ]

@router.get("/chat-logs")
async def get_anonymized_logs(
    current_user: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
    limit: int = 50
):
    result = await db.execute(
        select(Chat).order_by(desc(Chat.created_at)).limit(limit)
    )
    chats = result.scalars().all()
    return [
        {
            "id": c.id,
            "anonymous_user_ref": f"USER-{c.user_id:04d}",
            "detected_emotion": c.detected_emotion,
            "confidence_score": c.emotion_confidence,
            "risk_score": c.risk_score,
            "escalation_triggered": c.escalation_triggered,
            "created_at": c.created_at.isoformat()
            # message content intentionally omitted for privacy
        }
        for c in chats
    ]
