"""
Authentication router: register, login, consent
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, constr

from backend.database import get_db, User
from backend.utils.security import hash_password, verify_password, create_access_token
from backend.utils.auth_deps import get_current_user
from backend.utils.logger import log_event

router = APIRouter()

class RegisterRequest(BaseModel):
    username: constr(min_length=3, max_length=50)
    password: constr(min_length=6, max_length=100)

class ConsentRequest(BaseModel):
    consent: bool

@router.post("/register", status_code=201)
async def register(data: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.username == data.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already taken")
    user = User(
        username=data.username,
        password_hash=hash_password(data.password),
        role="student"
    )
    db.add(user)
    await db.commit()
    log_event("user_registered", f"New student registered: {data.username}")
    return {"message": "Registration successful. Please log in and provide consent."}

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == form_data.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(form_data.password, user.password_hash):
        log_event("login_failed", f"Failed login attempt: {form_data.username}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    token = create_access_token({"sub": user.username, "role": user.role, "user_id": user.id})
    log_event("login_success", f"User logged in: {user.username} [{user.role}]")
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role,
        "consent_given": user.consent_given,
        "user_id": user.id
    }

@router.post("/consent")
async def give_consent(data: ConsentRequest, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    current_user.consent_given = data.consent
    await db.commit()
    log_event("consent_updated", f"User {current_user.username} consent: {data.consent}")
    return {"message": "Consent recorded", "consent_given": data.consent}

@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "role": current_user.role,
        "consent_given": current_user.consent_given,
        "user_id": current_user.id
    }
