# auth.py
import pandas as pd
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta

# Constants
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Load user database
users_df = pd.read_csv("/Users/adityadeshmukh/Desktop/mock_users.csv")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# Permissions by role
ROLE_PERMISSIONS = {
    "rag_user": {
        "can_query_all_docs": True,
        "can_manage_documents": False,
        "can_view_analytics": False,
        "can_manage_users": False,
        "can_configure_rag": False,
    },
    "rag_admin": {
        "can_query_all_docs": True,
        "can_manage_documents": True,
        "can_view_analytics": True,
        "can_manage_users": True,
        "can_configure_rag": True,
    },
    "doc_owner": {
        "can_query_all_docs": True,
        "can_manage_documents": True,
        "can_view_analytics": False,
        "can_manage_users": False,
        "can_configure_rag": False,
    }
}

def authenticate_user(email: str, password: str):
    try:
        user = users_df[users_df['email'] == email].iloc[0].to_dict()
        return user, "Authenticated (Password skipped for demo)"
    except IndexError:
        return None, "User not found"

def create_access_token(user: dict):
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user["email"],
        "exp": expire,
        "role": user["role"],
        "organization": user["organization"],
        "name": user["name"]
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = {
            "email": payload["sub"],
            "role": payload["role"],
            "organization": payload["organization"],
            "name": payload["name"]
        }
        return user
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )

def get_user_permissions(role: str):
    return ROLE_PERMISSIONS.get(role, {})