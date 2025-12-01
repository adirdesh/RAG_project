import streamlit as st
from typing import Generator
from groq import Groq
import boto3
from botocore.exceptions import NoCredentialsError
from botocore.exceptions import ClientError
import json
from datetime import datetime
import hashlib

# === Load AWS Credentials from .streamlit/secrets.toml ===
aws_access_key_id = st.secrets["aws"]["access_key"]
aws_secret_access_key = st.secrets["aws"]["secret_key"]
bucket_name = st.secrets["aws"]["bucket_name"]
region_name = st.secrets["aws"]["region"]

st.set_page_config(page_icon="💬", layout="wide",
                   page_title="Multi-Org Groq Testing")

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🏢")

st.subheader("RAG Testing App", divider="rainbow", anchor=False)

client = Groq(
    api_key=st.secrets["GROQ"]["GROQ_API_KEY"],
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "user_authenticated" not in st.session_state:
    st.session_state.user_authenticated = False

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

# Define the three specific roles
USER_ROLES = {
    "RAG_user": {
        "name": "RAG User",
        "description": "Can query and interact with documents",
        "permissions": ["read", "query"]
    },
    "RAG_admin": {
        "name": "RAG Admin", 
        "description": "Can manage RAG system and user access",
        "permissions": ["read", "query", "admin", "manage_users"]
    },
    "doc_owner": {
        "name": "Document Owner",
        "description": "Can upload, modify and manage documents",
        "permissions": ["read", "query", "upload", "modify", "delete"]
    }
}

# Define common departments
DEPARTMENTS = [
    "Finance", "HR", "Legal", "IT", "Operations", 
    "Marketing", "Sales", "Research", "Engineering", "Other"
]

# Define model details
models = {
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "name": "Meta-Llama-4-scout-17b-16e-instruct", 
        "tokens": 8192, 
        "developer": "Meta"
    }
}

def sanitize_folder_name(name):
    """Sanitize organization/department names for S3 folder structure"""
    return "".join(c for c in name if c.isalnum() or c in ('-', '_')).lower()

def generate_user_session_id():
    """Generate a unique session ID for the user"""
    timestamp = datetime.now().isoformat()
    return hashlib.md5(timestamp.encode()).hexdigest()[:8]

def create_s3_folder_structure(organization, department):
    """Create S3 folder path: organizations/{org}/{dept}/uploads/"""
    org_safe = sanitize_folder_name(organization)
    dept_safe = sanitize_folder_name(department)
    return f"organizations/{org_safe}/{dept_safe}/uploads"

def validate_upload_permissions(role):
    """Check if user role has upload permissions"""
    return "upload" in USER_ROLES.get(role, {}).get("permissions", [])

# Sidebar for User Authentication/Profile
st.sidebar.header("🔐 User Profile")

# Organization Selection
organization = st.sidebar.text_input(
    "Organization Name:",
    placeholder="Enter your organization name",
    help="This will create a separate folder structure for your organization"
)

# Department Selection  
department = st.sidebar.selectbox(
    "Department:",
    [""] + DEPARTMENTS,
    help="Select your department within the organization"
)

# Role Selection
user_role = st.sidebar.selectbox(
    "Select your role:",
    [""] + list(USER_ROLES.keys()),
    format_func=lambda x: USER_ROLES[x]["name"] if x else "Select Role",
    help="Choose your role in the RAG system"
)

# Display role permissions
if user_role:
    role_info = USER_ROLES[user_role]
    st.sidebar.info(f"**{role_info['name']}**\n\n{role_info['description']}")
    permissions = ", ".join(role_info['permissions'])
    st.sidebar.caption(f"Permissions: {permissions}")

# User ID for tracking
user_id = st.sidebar.text_input(
    "User ID/Email:",
    placeholder="Enter your user identifier",
    help="This helps track document ownership and access"
)

# Validate user profile completeness
profile_complete = all([organization, department, user_role, user_id])

if profile_complete:
    st.session_state.user_authenticated = True
    st.session_state.user_profile = {
        "organization": organization,
        "department": department,
        "role": user_role,
        "user_id": user_id,
        "session_id": generate_user_session_id()
    }
    st.sidebar.success("✅ Profile Complete")
else:
    st.session_state.user_authenticated = False
    st.sidebar.warning("⚠️ Complete all fields to proceed")

# Main content area
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
        help=f"Adjust the maximum number of tokens for the model's response. Max: {max_tokens_range}"
    )

def create_comprehensive_metadata(file_name, user_profile, file_size=None):
    """Create comprehensive metadata for uploaded files"""
    return {
        "file_info": {
            "original_name": file_name,
            "upload_timestamp": datetime.now().isoformat(),
            "file_size_bytes": file_size
        },
        "user_info": {
            "user_id": user_profile["user_id"],
            "role": user_profile["role"],
            "organization": user_profile["organization"],
            "department": user_profile["department"],
            "session_id": user_profile["session_id"]
        },
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
    """Upload file to organization-specific folder structure"""
    try:
        # Create folder structure
        folder_path = create_s3_folder_structure(
            user_profile["organization"], 
            user_profile["department"]
        )
        
        # Create unique filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = uploaded_file.name.split('.')[-1] if '.' in uploaded_file.name else ''
        safe_filename = sanitize_folder_name(uploaded_file.name.rsplit('.', 1)[0])
        
        if file_extension:
            unique_filename = f"{safe_filename}_{timestamp}.{file_extension}"
        else:
            unique_filename = f"{safe_filename}_{timestamp}"
            
        s3_file_path = f"{folder_path}/{unique_filename}"
        
        # Get file size if available
        file_size = getattr(uploaded_file, 'size', None)
        
        # Create comprehensive metadata
        metadata = create_comprehensive_metadata(
            uploaded_file.name, 
            user_profile, 
            file_size
        )
        
        # Upload file with basic metadata
        s3_client.upload_fileobj(
            uploaded_file,
            bucket_name,
            s3_file_path,
            ExtraArgs={
                'Metadata': {
                    'user-id': user_profile["user_id"],
                    'role': user_profile["role"],
                    'organization': sanitize_folder_name(user_profile["organization"]),
                    'department': sanitize_folder_name(user_profile["department"]),
                    'upload-timestamp': datetime.now().isoformat(),
                    'original-filename': uploaded_file.name
                }
            }
        )
        
        # Upload detailed metadata as JSON
        metadata_path = f"{folder_path}/metadata/{unique_filename}.metadata.json"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=metadata_path,
            Body=json.dumps(metadata, indent=2),
            ContentType='application/json'
        )
        
        return s3_file_path, metadata
        
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None, None

# File Upload Section
st.subheader("📄 Document Upload")

if not st.session_state.user_authenticated:
    st.warning("🔒 Please complete your profile in the sidebar to upload documents.")
    uploaded_file = None
elif not validate_upload_permissions(user_role):
    st.error(f"❌ Your role ({USER_ROLES[user_role]['name']}) does not have upload permissions.")
    st.info("Only Document Owners can upload files. Contact your RAG Admin for access.")
    uploaded_file = None
else:
    # Show current upload location
    folder_path = create_s3_folder_structure(organization, department)
    st.info(f"📁 Files will be uploaded to: `{folder_path}/`")
    
    uploaded_file = st.file_uploader(
        "Choose a file to upload:",
        type=['txt', 'pdf', 'docx', 'ppt', 'pptx', 'csv', 'xlsx'],
        help="Supported formats: TXT, PDF, DOCX, PPT, PPTX, CSV, XLSX"
    )

# Handle file upload
if uploaded_file and st.session_state.user_authenticated:
    st.session_state.messages.append({
        "role": "user", 
        "content": f"📎 Uploaded file: {uploaded_file.name} ({organization}/{department})"
    })
    
    try:
        # Initialize S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        
        # Upload file
        file_path, metadata = upload_file_to_org_structure(
            uploaded_file, s3, bucket_name, st.session_state.user_profile
        )
        
        if file_path:
            st.success(f"✅ File uploaded successfully!")
            
            # Display upload summary
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **Upload Summary:**
                - **File:** {uploaded_file.name}
                - **Organization:** {organization}
                - **Department:** {department}
                - **Uploaded by:** {user_id} ({USER_ROLES[user_role]['name']})
                """)
            
            with col2:
                st.info(f"""
                **S3 Location:**
                - **Path:** `{file_path}`
                - **Bucket:** {bucket_name}
                - **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """)
                    
    except NoCredentialsError:
        st.error("❌ AWS credentials not found.")
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")

# Display chat messages from history
st.subheader("💬 Chat Interface")

for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👩‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Chat input
if not st.session_state.user_authenticated:
    st.info("🔒 Complete your profile to start chatting.")
else:
    if prompt := st.chat_input("Enter your prompt here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar='👨‍💻'):
            st.markdown(prompt)

        # Fetch response from Groq API
        try:
            chat_completion = client.chat.completions.create(
                model=model_option,
                messages=[
                    {
                        "role": m["role"],
                        "content": m["content"]
                    }
                    for m in st.session_state.messages
                ],
                max_tokens=max_tokens,
                stream=True
            )

            # Use the generator function with st.write_stream
            with st.chat_message("assistant", avatar="🤖"):
                chat_responses_generator = generate_chat_responses(chat_completion)
                full_response = st.write_stream(chat_responses_generator)
        except Exception as e:
            st.error(e, icon="🚨")

        # Append the full response to session_state.messages
        if isinstance(full_response, str):
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response})
        else:
            # Handle the case where full_response is not a string
            combined_response = "\n".join(str(item) for item in full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": combined_response})

# Footer with current session info
if st.session_state.user_authenticated:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Current Session:**")
    profile = st.session_state.user_profile
    st.sidebar.caption(f"""
    🏢 {profile['organization']}  
    🏬 {profile['department']}  
    👤 {profile['user_id']}  
    🎭 {USER_ROLES[profile['role']]['name']}
    """)