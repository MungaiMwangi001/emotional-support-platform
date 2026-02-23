from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from database import get_db, User, SystemLog
from auth import verify_password, hash_password, create_access_token
from datetime import datetime

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "student"
    consent: bool = False


class ConsentUpdate(BaseModel):
    consent: bool


@router.post("/register")
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    if not req.consent:
        raise HTTPException(status_code=400, detail="You must consent to data usage to register.")
    if len(req.username) < 3 or len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Username must be ≥3 chars, password ≥6 chars.")
    if req.role not in ("student", "admin", "therapist"):
        raise HTTPException(status_code=400, detail="Invalid role.")

    existing = await db.execute(select(User).where(User.username == req.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username already taken.")

    user = User(
        username=req.username,
        password_hash=hash_password(req.password),
        role=req.role,
        consent_given=req.consent,
    )
    db.add(user)
    log = SystemLog(event_type="user_registered", details=f"role={req.role}")
    db.add(log)
    await db.commit()
    return {"message": "Registration successful."}


@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == form_data.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.password_hash):
        log = SystemLog(event_type="login_failed", details=f"username={form_data.username}")
        db.add(log)
        await db.commit()
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    token = create_access_token({"sub": user.username, "role": user.role})
    log = SystemLog(event_type="login_success", details=f"username={user.username}")
    db.add(log)
    await db.commit()
    return {"access_token": token, "token_type": "bearer", "role": user.role, "username": user.username}
