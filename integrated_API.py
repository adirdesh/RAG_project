import streamlit as st
from typing import Generator
from groq import Groq
import boto3
from botocore.exceptions import NoCredentialsError
from botocore.exceptions import ClientError
import requests
import uuid
from datetime import datetime

# FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:8000/docs"  # Update with your FastAPI server URL

# === Load AWS Credentials from .streamlit/secrets.toml ===
aws_access_key_id = st.secrets["aws"]["access_key"]
aws_secret_access_key = st.secrets["aws"]["secret_key"]
bucket_name = st.secrets["aws"]["bucket_name"]
region_name = st.secrets["aws"]["region"]
folder_prefix = "download-uploads2/"

st.set_page_config(page_icon="💬", layout="wide", page_title="Groq Testing")

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

def send_document_metadata(file_name: str, file_type: str, file_size: int, s3_key: str, user_email: str):
    """Send document metadata to FastAPI backend"""
    try:
        response = requests.post(f"{BACKEND_URL}/documents/upload", json={
            "file_name": file_name,
            "file_type": file_type,
            "file_size": file_size,
            "s3_key": s3_key,
            "bucket_name": bucket_name,
            "uploaded_by": user_email
        })
        if response.status_code == 200:
            return response.json()["doc_id"]
        else:
            st.error(f"Failed to register document: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error sending metadata: {e}")
        return None

def send_chat_message(session_id: str, role: str, content: str, user_email: str, model_used: str = None):
    """Send chat message to FastAPI backend"""
    try:
        message_data = {
            "message_id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "user_email": user_email,
            "model_used": model_used
        }
        
        response = requests.post(f"{BACKEND_URL}/chat/message", json=message_data)
        if response.status_code != 200:
            st.error(f"Failed to save chat message: {response.text}")
    except Exception as e:
        st.error(f"Error sending chat message: {e}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "user_email" not in st.session_state:
    st.session_state.user_email = "user@example.com"  # In real app, get from login

if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = {}

icon("🤖")
st.subheader("Groq Testing App", divider="rainbow", anchor=False)

# User info section
col_user, col_session = st.columns(2)
with col_user:
    st.info(f"👤 User: {st.session_state.user_email}")
with col_session:
    st.info(f"🔗 Session: {st.session_state.session_id[:8]}...")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Define model details
models = {
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "name": "Meta-Llama-4-scout-17b-16e-instruct", 
        "tokens": 8192, 
        "developer": "Meta"
    }
}

# Layout for model selection and max_tokens slider
col1, col2 = st.columns(2)

with col1:
    model_option = st.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=0
    )

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

max_tokens_range = 8192

with col2:
    max_tokens = st.slider(
        "Max Tokens:",
        min_value=512,
        max_value=max_tokens_range,
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
    )

# File upload with metadata tracking
uploaded_file = st.file_uploader(" ", type=['txt', 'pdf', 'docx', 'ppt'], label_visibility="collapsed")

if uploaded_file:
    # Display upload info
    st.session_state.messages.append({
        "role": "user", 
        "content": f"📎 Uploaded file: {uploaded_file.name}"
    })
    
    # Prepare file metadata
    file_metadata = {
        "name": uploaded_file.name,
        "type": uploaded_file.type or "unknown",
        "size": uploaded_file.size,
        "upload_time": datetime.now().isoformat()
    }

custom_folder = "uploads"
if uploaded_file is not None:
    file_name = f"{custom_folder}/{uploaded_file.name}"
    
    try:
        # Initialize S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        
        # Upload file to S3
        s3.upload_fileobj(uploaded_file, bucket_name, file_name)
        st.success(f"✅ File '{file_name}' uploaded successfully to '{bucket_name}'!")
        
        # Send metadata to FastAPI backend
        doc_id = send_document_metadata(
            file_name=uploaded_file.name,
            file_type=uploaded_file.type or "unknown",
            file_size=uploaded_file.size,
            s3_key=file_name,
            user_email=st.session_state.user_email
        )
        
        if doc_id:
            st.session_state.doc_metadata[uploaded_file.name] = {
                "doc_id": doc_id,
                "s3_key": file_name,
                "metadata": file_metadata
            }
            st.success(f"✅ Document metadata registered with ID: {doc_id}")
    
    except NoCredentialsError:
        st.error("❌ AWS credentials not found.")
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")

# File management section
try:
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)
    if "Contents" in response:
        files = [obj["Key"] for obj in response["Contents"] if not obj["Key"].endswith("/")]
        if files:
            selected_file = st.selectbox("Select a file to delete", files)
            
            if st.button("Delete File"):
                try:
                    s3.delete_object(Bucket=bucket_name, Key=selected_file)
                    st.success(f"✅ '{selected_file}' deleted successfully.")
                except ClientError as e:
                    st.error(f"❌ Delete failed: {e}")
    else:
        st.info("No files found in the specified folder.")
except Exception as e:
    st.error(f"❌ Failed to list files: {e}")

# Display uploaded documents metadata
if st.session_state.doc_metadata:
    with st.expander("📋 Uploaded Documents Metadata"):
        for file_name, doc_info in st.session_state.doc_metadata.items():
            st.json({
                "Document ID": doc_info["doc_id"],
                "File Name": file_name,
                "S3 Key": doc_info["s3_key"],
                "Metadata": doc_info["metadata"]
            })

# Display chat messages from history
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👩‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Chat input and processing
if prompt := st.chat_input("Enter your prompt here..."):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Send user message to backend
    send_chat_message(
        session_id=st.session_state.session_id,
        role="user",
        content=prompt,
        user_email=st.session_state.user_email
    )
    
    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)
    
    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            stream=True
        )
        
        # Generate and display response
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
            
    except Exception as e:
        st.error(e, icon="🚨")
        full_response = f"Error: {str(e)}"
    
    # Add assistant response to session state
    if isinstance(full_response, str):
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Send assistant message to backend
        send_chat_message(
            session_id=st.session_state.session_id,
            role="assistant",
            content=full_response,
            user_email=st.session_state.user_email,
            model_used=model_option
        )
    else:
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append({"role": "assistant", "content": combined_response})
        
        # Send combined response to backend
        send_chat_message(
            session_id=st.session_state.session_id,
            role="assistant",
            content=combined_response,
            user_email=st.session_state.user_email,
            model_used=model_option
        )

# Analytics section
if st.button("📊 Show Analytics"):
    try:
        # Get document analytics
        doc_response = requests.get(f"{BACKEND_URL}/analytics/documents")
        chat_response = requests.get(f"{BACKEND_URL}/analytics/chat")
        
        if doc_response.status_code == 200 and chat_response.status_code == 200:
            doc_analytics = doc_response.json()
            chat_analytics = chat_response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📄 Document Analytics")
                st.json(doc_analytics)
            
            with col2:
                st.subheader("💬 Chat Analytics")
                st.json(chat_analytics)
        else:
            st.error("Failed to fetch analytics")
    
    except Exception as e:
        st.error(f"Error fetching analytics: {e}")