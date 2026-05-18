from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List
from datetime import timedelta
import logging

from models.schemas import LoginRequest, CreateAdminRequest, SetPasswordRequest, TokenResponse, AdminResponse
from services.auth_service import AuthService
from services.admin_store import admin_store
from services.email_agent import send_raw_email
from config import app_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Auth"])
security = HTTPBearer()

async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    payload = AuthService.verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token payload")
        
    admin = await admin_store.get_by_email(email)
    if not admin:
        raise HTTPException(status_code=401, detail="Admin not found")
        
    return admin

@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    admin = await admin_store.get_by_email(request.email)
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid email or password")
        
    if not AuthService.verify_password(request.password, admin["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
        
    access_token = AuthService.create_access_token(
        data={"sub": admin["email"]},
        expires_delta=timedelta(hours=24)
    )
    return TokenResponse(access_token=access_token)

@router.get("/me", response_model=AdminResponse)
async def get_me(current_admin: dict = Depends(get_current_admin)):
    return current_admin

@router.post("/create-admin", response_model=AdminResponse)
async def create_admin(
    request: CreateAdminRequest, 
    background_tasks: BackgroundTasks,
    current_admin: dict = Depends(get_current_admin)
):
    existing = await admin_store.get_by_email(request.email)
    if existing:
        raise HTTPException(status_code=400, detail="Admin with this email already exists")
        
    # Set a random temporary password which they will change anyway
    temp_pass = AuthService.get_password_hash("temp-password-123")
    
    new_admin = await admin_store.create_admin(
        name=request.name,
        email=request.email,
        password_hash=temp_pass
    )
    
    # Create setup token valid for 48 hours
    setup_token = AuthService.create_access_token(
        data={"sub": request.email, "purpose": "setup_password"},
        expires_delta=timedelta(hours=48)
    )
    
    setup_url = f"{app_config.frontend_url}/set-password?token={setup_token}"
    
    # Send email
    subject = "Welcome to ColorWhistle Chatbot - Admin Account Created"
    body = f"""
    <p>Hi {request.name},</p>
    <p>An admin account has been created for you on the ColorWhistle Chatbot.</p>
    <p>Please set your password by clicking the link below:</p>
    <p><a href="{setup_url}">{setup_url}</a></p>
    <p>This link will expire in 48 hours.</p>
    """
    
    background_tasks.add_task(
        send_raw_email,
        to_email=request.email,
        subject=subject,
        html_content=body
    )
    
    return new_admin

@router.post("/set-password")
async def set_password(request: SetPasswordRequest):
    payload = AuthService.verify_token(request.token)
    if not payload:
        raise HTTPException(status_code=400, detail="Invalid or expired setup token")
        
    if payload.get("purpose") != "setup_password":
        raise HTTPException(status_code=400, detail="Invalid token purpose")
        
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=400, detail="Invalid token payload")
        
    admin = await admin_store.get_by_email(email)
    if not admin:
        raise HTTPException(status_code=404, detail="Admin not found")
        
    new_hash = AuthService.get_password_hash(request.password)
    success = await admin_store.update_password(email, new_hash)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update password")
        
    return {"message": "Password updated successfully"}

@router.get("/list", response_model=List[AdminResponse])
async def list_admins(current_admin: dict = Depends(get_current_admin)):
    return await admin_store.list_admins()
