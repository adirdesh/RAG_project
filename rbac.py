import streamlit as st
from typing import Generator
from groq import Groq
import boto3
from botocore.exceptions import NoCredentialsError
import json
from datetime import datetime
import hashlib
import pandas as pd

# Load AWS Credentials from secrets
aws_access_key_id = st.secrets["aws"]["access_key"]
aws_secret_access_key = st.secrets["aws"]["secret_key"]
bucket_name = st.secrets["aws"]["bucket_name"]
region_name = st.secrets["aws"]["region"]

st.set_page_config(page_icon=None, layout="wide", page_title="Multi-Org RAG")

# Load user CSV
@st.cache_data
def load_user_data():
    return pd.read_csv("/Users/adityadeshmukh/Desktop/mock_users.csv")

user_df = load_user_data()


st.subheader("Multi-Organization RAG Application", divider="gray")

client = Groq(api_key=st.secrets["GROQ"]["GROQ_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "user_authenticated" not in st.session_state:
    st.session_state.user_authenticated = False

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

USER_ROLES = {
    "RAG_user": {
        "name": "RAG User",
        "permissions": ["read", "query"]
    },
    "RAG_admin": {
        "name": "RAG Admin",
        "permissions": ["read", "query", "admin", "manage_users"]
    },
    "doc_owner": {
        "name": "Document Owner",
        "permissions": ["read", "query", "upload", "modify", "delete"]
    }
}

DEPARTMENTS = [
    "Finance", "HR", "Legal", "IT", "Operations",
    "Marketing", "Sales", "Research", "Engineering", "Other"
]

models = {
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "name": "Meta-Llama-4-scout-17b-16e-instruct",
        "tokens": 8192,
        "developer": "Meta"
    }
}

def sanitize_folder_name(name):
    return "".join(c for c in name if c.isalnum() or c in ('-', '_')).lower()

def generate_user_session_id():
    return hashlib.md5(datetime.now().isoformat().encode()).hexdigest()[:8]

def create_s3_folder_structure(org, dept):
    return f"organizations/{sanitize_folder_name(org)}/{sanitize_folder_name(dept)}/uploads"

def validate_upload_permissions(role):
    return "upload" in USER_ROLES.get(role, {}).get("permissions", [])

# Sidebar — User Authentication
st.sidebar.header("User Profile")

user_id = st.sidebar.text_input("Email", placeholder="Enter your email")
department = st.sidebar.selectbox("Department", [""] + DEPARTMENTS)

matched_user = user_df[user_df["email"].str.lower() == user_id.lower()].squeeze()

if not user_id:
    st.session_state.user_authenticated = False
    st.sidebar.warning("⚠️ Please enter your email.")
elif matched_user.empty:
    st.session_state.user_authenticated = False
    st.sidebar.error("❌ User not found.")
elif not department:
    st.session_state.user_authenticated = False
    st.sidebar.warning("⚠️ Select a department.")
else:
    name = matched_user["name"]
    organization = matched_user["organization"]
    user_role = matched_user["role"]
    st.session_state.user_authenticated = True
    st.session_state.user_profile = {
        "organization": organization,
        "department": department,
        "role": user_role,
        "user_id": user_id,
        "session_id": generate_user_session_id(),
        "name": name
    }
    st.sidebar.success(f"✅ Welcome, {name}")
    st.sidebar.caption(f"Verified as {user_role} in {organization}")

# Model and Token selection
col1, col2 = st.columns(2)
with col1:
    model_option = st.selectbox("Choose a model:", options=list(models.keys()), format_func=lambda x: models[x]["name"])

if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

with col2:
    max_tokens = st.slider("Max Tokens:", min_value=512, max_value=8192, value=2048, step=512)

# File upload helpers
def create_comprehensive_metadata(file_name, user_profile, file_size=None):
    return {
        "file_info": {
            "original_name": file_name,
            "upload_timestamp": datetime.now().isoformat(),
            "file_size_bytes": file_size
        },
        "user_info": user_profile,
        "access_control": {
            "uploaded_by": user_profile["user_id"],
            "organization_access": user_profile["organization"],
            "department_access": user_profile["department"],
            "role_permissions": USER_ROLES[user_profile["role"]]["permissions"]
        },
        "system_info": {
            "app_version": "1.0",
            "upload_method": "streamlit_web_interface"
        }
    }

def upload_file_to_org_structure(uploaded_file, s3_client, bucket_name, user_profile):
    try:
        folder_path = create_s3_folder_structure(user_profile["organization"], user_profile["department"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = uploaded_file.name.split('.')[-1]
        safe_filename = sanitize_folder_name(uploaded_file.name.rsplit('.', 1)[0])
        unique_filename = f"{safe_filename}_{timestamp}.{file_extension}"
        s3_file_path = f"{folder_path}/{unique_filename}"
        file_size = getattr(uploaded_file, 'size', None)
        metadata = create_comprehensive_metadata(uploaded_file.name, user_profile, file_size)

        s3_client.upload_fileobj(uploaded_file, bucket_name, s3_file_path, ExtraArgs={
            'Metadata': {
                'user-id': user_profile["user_id"],
                'role': user_profile["role"],
                'organization': sanitize_folder_name(user_profile["organization"]),
                'department': sanitize_folder_name(user_profile["department"]),
                'upload-timestamp': datetime.now().isoformat(),
                'original-filename': uploaded_file.name
            }
        })

        metadata_path = f"{folder_path}/metadata/{unique_filename}.metadata.json"
        s3_client.put_object(Bucket=bucket_name, Key=metadata_path, Body=json.dumps(metadata, indent=2), ContentType='application/json')
        return s3_file_path, metadata

    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None, None

# File Upload
st.subheader("Document Upload")

if not st.session_state.user_authenticated:
    st.warning("🔒 Please complete your profile.")
    uploaded_file = None
elif not validate_upload_permissions(user_role):
    st.error(f"❌ Your role ({user_role}) does not have upload permissions.")
    uploaded_file = None
else:
    folder_path = create_s3_folder_structure(organization, department)
    st.info(f"Files will be uploaded to: `{folder_path}/`")
    uploaded_file = st.file_uploader("Choose a file:", type=['txt', 'pdf', 'docx', 'ppt', 'pptx', 'csv', 'xlsx'])

if uploaded_file:
    st.session_state.messages.append({
        "role": "user",
        "content": f"📎 Uploaded file: {uploaded_file.name} ({organization}/{department})"
    })
    try:
        s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)
        file_path, metadata = upload_file_to_org_structure(uploaded_file, s3, bucket_name, st.session_state.user_profile)
        if file_path:
            st.success("✅ File uploaded successfully.")
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")

# Chat
st.subheader("Chat Interface")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

if st.session_state.user_authenticated:
    if prompt := st.chat_input("Enter your prompt here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            chat_completion = client.chat.completions.create(
                model=model_option,
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                max_tokens=max_tokens,
                stream=True
            )
            with st.chat_message("assistant"):
                full_response = st.write_stream(generate_chat_responses(chat_completion))
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(e)

# Session Info
if st.session_state.user_authenticated:
    profile = st.session_state.user_profile
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Organization: {profile['organization']}\nDepartment: {profile['department']}\nUser: {profile['user_id']}\nRole: {profile['role']}")