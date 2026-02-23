"""
Therapist Router - View and manage contact requests (therapist only)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel
from typing import Optional

from backend.database import get_db, TherapistRequest, Chat, User
from backend.utils.auth_deps import require_role

router = APIRouter()

class UpdateRequestStatus(BaseModel):
    status: str  # pending | in_progress | resolved
    notes: Optional[str] = None

@router.get("/requests")
async def get_therapist_requests(
    current_user: User = Depends(require_role("therapist")),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(TherapistRequest).order_by(desc(TherapistRequest.created_at))
    )
    requests = result.scalars().all()
    return [
        {
            "id": r.id,
            "anonymous_user_ref": f"USER-{r.user_id:04d}",
            "status": r.status,
            "notes": r.notes,
            "created_at": r.created_at.isoformat(),
            "updated_at": r.updated_at.isoformat() if r.updated_at else None
        }
        for r in requests
    ]

@router.put("/requests/{request_id}")
async def update_request_status(
    request_id: int,
    update: UpdateRequestStatus,
    current_user: User = Depends(require_role("therapist")),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(TherapistRequest).where(TherapistRequest.id == request_id))
    req = result.scalar_one_or_none()
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    
    valid_statuses = ["pending", "in_progress", "resolved"]
    if update.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Status must be one of: {valid_statuses}")
    
    req.status = update.status
    if update.notes:
        req.notes = update.notes
    await db.commit()
    return {"message": "Request updated", "status": req.status}

@router.get("/emotional-summary/{user_id}")
async def get_user_emotional_summary(
    user_id: int,
    current_user: User = Depends(require_role("therapist")),
    db: AsyncSession = Depends(get_db)
):
    """Get anonymized emotional summary for a user (no message content)"""
    result = await db.execute(
        select(Chat.detected_emotion, Chat.emotion_confidence, Chat.risk_score, Chat.created_at)
        .where(Chat.user_id == user_id)
        .order_by(desc(Chat.created_at)).limit(20)
    )
    rows = result.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="No data found for this user")
    
    return {
        "anonymous_user_ref": f"USER-{user_id:04d}",
        "sessions": [
            {
                "emotion": r[0],
                "confidence": r[1],
                "risk_score": r[2],
                "timestamp": r[3].isoformat()
            }
            for r in rows
        ]
    }
