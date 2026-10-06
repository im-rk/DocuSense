from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from db.session import get_db
from db.models import User
from api.schemas.auth import RegisterRequest, LoginRequest, TokenResponse
from config.settings import settings

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"])

def create_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.utcnow() + timedelta(hours=settings.jwt_expiry_hours)
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")

@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=req.email,
        password_hash=pwd_context.hash(req.password),
        full_name=req.full_name
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {"access_token": create_token(str(user.id)), "token_type": "bearer"}

@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not pwd_context.verify(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {"access_token": create_token(str(user.id)), "token_type": "bearer"}
