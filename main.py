'''
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    text:str = None
    isdone:bool = False

app = FastAPI()

items = []

@app.get("/")
def root():
    return{"hello" : "world"}

@app.post("/items")
def create_item(item:Item):
    items.append(item)
    return items

@app.get("/items/{item_id}")
def get_item(item_id:int)->Item:
    item = items[item_id]
    return item

@app.get("/items")
def list_items(limit:int=5):
    return items[0:limit]
'''

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()

# -------------------------
# Data Models
# -------------------------

class UserCredentials(BaseModel):
    email: str
    password: str

class UserProfile(BaseModel):
    name: str
    email: str
    role: str

class Document(BaseModel):
    doc_id: int
    title: str
    content: str

class QueryRequest(BaseModel):
    question: str

# -------------------------
# Hardcoded Mock Data
# -------------------------

mock_users = {
    "user@example.com": {"password": "pass123", "role": "rag_user"},
    "admin@example.com": {"password": "adminpass", "role": "rag_admin"}
}

mock_documents = [
    Document(doc_id=1, title="Quarterly Report Q1", content="Revenue increased by 10%"),
    Document(doc_id=2, title="Engineering Overview", content="We use a RAG model for QA."),
]

mock_profile = UserProfile(name="John Doe", email="user@example.com", role="rag_user")

# -------------------------
# Routes
# -------------------------

@app.post("/signup")
def signup(creds: UserCredentials):
    if creds.email in mock_users:
        raise HTTPException(status_code=400, detail="User already exists")
    mock_users[creds.email] = {"password": creds.password, "role": "rag_user"}
    return {"message": "Signup successful", "email": creds.email}


@app.post("/signin")
def signin(creds: UserCredentials):
    user = mock_users.get(creds.email)
    if not user or user["password"] != creds.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Signin successful", "role": user['role']}


@app.post("/login")
def login(creds: UserCredentials):
    # Same as signin; provided separately to mock different endpoint usage
    return signin(creds)


@app.get("/profile")
def get_profile():
    return mock_profile


@app.get("/documents", response_model=List[Document])
def list_documents():
    return mock_documents


@app.get("/documents/{doc_id}", response_model=Document)
def get_document(doc_id: int):
    for doc in mock_documents:
        if doc.doc_id == doc_id:
            return doc
    raise HTTPException(status_code=404, detail="Document not found")


@app.post("/query")
def query_docs(query: QueryRequest):
    # Hardcoded mock response
    if "revenue" in query.question.lower():
        return {"answer": "Revenue increased by 10% in Q1."}
    elif "model" in query.question.lower():
        return {"answer": "We use a RAG model with vector embeddings."}
    else:
        return {"answer": "No relevant information found."}


@app.post("/logout")
def logout():
    return {"message": "Logged out successfully"}

