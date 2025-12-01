import streamlit as st
from typing import Generator, Optional, Dict, Any
from groq import Groq
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import jwt
import json
import requests
from datetime import datetime
import base64

# === Load AWS Credentials from .streamlit/secrets.toml ===
aws_access_key_id = st.secrets["aws"]["access_key"]
aws_secret_access_key = st.secrets["aws"]["secret_key"]
bucket_name = st.secrets["aws"]["bucket_name"]
region_name = st.secrets["aws"]["region"]

# === Cognito Configuration ===
cognito_user_pool_id = st.secrets["cognito"]["user_pool_id"]
cognito_client_id = st.secrets["cognito"]["client_id"]
cognito_region = st.secrets["cognito"]["region"]

st.set_page_config(page_icon="💬", layout="wide", page_title="Groq Testing with User Auth")

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

class CognitoAuth:
    """Handle Cognito authentication and user attribute extraction"""
    
    def __init__(self, user_pool_id: str, client_id: str, region: str):
        self.user_pool_id = user_pool_id
        self.client_id = client_id
        self.region = region
        self.cognito_client = boto3.client(
            'cognito-idp',
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return tokens"""
        try:
            response = self.cognito_client.admin_initiate_auth(
                UserPoolId=self.user_pool_id,
                ClientId=self.client_id,
                AuthFlow='ADMIN_NO_SRP_AUTH',
                AuthParameters={
                    'USERNAME': username,
                    'PASSWORD': password
                }
            )
            return response['AuthenticationResult']
        except ClientError as e:
            st.error(f"Authentication failed: {e.response['Error']['Message']}")
            return None
    
    def get_user_attributes(self, access_token: str) -> Optional[Dict[str, str]]:
        """Extract user attributes from access token"""
        try:
            response = self.cognito_client.get_user(AccessToken=access_token)
            
            # Convert attributes list to dictionary
            attributes = {}
            for attr in response['UserAttributes']:
                attributes[attr['Name']] = attr['Value']
            
            return {
                'username': response['Username'],
                'email': attributes.get('email', ''),
                'organization': attributes.get('custom:organization', 'default-org'),
                'department': attributes.get('custom:department', 'default-dept'),
                'role': attributes.get('custom:role', 'user'),
                'full_name': attributes.get('name', attributes.get('given_name', '') + ' ' + attributes.get('family_name', '')),
                'user_id': attributes.get('sub', response['Username'])
            }
        except ClientError as e:
            st.error(f"Failed to get user attributes: {e.response['Error']['Message']}")
            return None
    
    def decode_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode JWT token (for additional validation if needed)"""
        try:
            # Note: In production, you should verify the token signature
            # This is a simplified version for demonstration
            decoded = jwt.decode(token, options={"verify_signature": False})
            return decoded
        except jwt.InvalidTokenError:
            return None

def create_folder_path(user_attributes: Dict[str, str], file_name: str) -> str:
    """Create organized folder path based on user attributes"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sanitize folder names (remove special characters)
    org = user_attributes['organization'].replace(' ', '_').replace('/', '_')
    dept = user_attributes['department'].replace(' ', '_').replace('/', '_')
    user_id = user_attributes['user_id'].replace(' ', '_').replace('/', '_')
    
    # Create hierarchical folder structure
    folder_path = f"{org}/{dept}/{user_id}/{timestamp}_{file_name}"
    return folder_path

def upload_file_to_s3(file_obj, file_path: str, metadata: Dict[str, str]) -> bool:
    """Upload file to S3 with metadata"""
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        
        # Upload file with metadata
        s3.upload_fileobj(
            file_obj, 
            bucket_name, 
            file_path,
            ExtraArgs={
                'Metadata': metadata,
                'ServerSideEncryption': 'AES256'  # Optional: encrypt at rest
            }
        )
        return True
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return False

def main():
    icon("🤖")
    st.subheader("Groq Testing App with User Authentication", divider="rainbow", anchor=False)
    
    # Initialize Cognito Auth
    auth = CognitoAuth(cognito_user_pool_id, cognito_client_id, cognito_region)
    
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_attributes" not in st.session_state:
        st.session_state.user_attributes = None
    if "access_token" not in st.session_state:
        st.session_state.access_token = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    
    # Authentication Section
    if not st.session_state.authenticated:
        st.header("🔐 User Authentication")
        
        with st.form("login_form"):
            username = st.text_input("Username/Email")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button and username and password:
                # Authenticate user
                auth_result = auth.authenticate_user(username, password)
                
                if auth_result:
                    access_token = auth_result['AccessToken']
                    
                    # Get user attributes
                    user_attributes = auth.get_user_attributes(access_token)
                    
                    if user_attributes:
                        st.session_state.authenticated = True
                        st.session_state.user_attributes = user_attributes
                        st.session_state.access_token = access_token
                        st.success(f"Welcome, {user_attributes['full_name']}!")
                        st.rerun()
        
        # Stop here if not authenticated
        return
    
    # Display user info
    if st.session_state.authenticated:
        user_attrs = st.session_state.user_attributes
        
        with st.sidebar:
            st.header("👤 User Profile")
            st.write(f"**Name:** {user_attrs['full_name']}")
            st.write(f"**Email:** {user_attrs['email']}")
            st.write(f"**Organization:** {user_attrs['organization']}")
            st.write(f"**Department:** {user_attrs['department']}")
            st.write(f"**Role:** {user_attrs['role']}")
            
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.user_attributes = None
                st.session_state.access_token = None
                st.session_state.messages = []
                st.rerun()
    
    # Groq Client Setup
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    # Model Selection
    models = {
        "meta-llama/llama-4-scout-17b-16e-instruct": {
            "name": "Meta-Llama-4-scout-17b-16e-instruct", 
            "tokens": 8192, 
            "developer": "Meta"
        }
    }
    
    col1, col2 = st.columns(2)
    with col1:
        model_option = st.selectbox(
            "Choose a model:",
            options=list(models.keys()),
            format_func=lambda x: models[x]["name"],
            index=0
        )
    
    # Detect model change and clear chat history
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
    
    # File Upload Section
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload your document", 
        type=['txt', 'pdf', 'docx', 'ppt', 'xlsx', 'csv'],
        help="Upload documents to your organization's secure storage"
    )
    
    if uploaded_file is not None:
        user_attrs = st.session_state.user_attributes
        
        # Create organized folder path
        folder_path = create_folder_path(user_attrs, uploaded_file.name)
        
        # Prepare metadata
        metadata = {
            'organization': user_attrs['organization'],
            'department': user_attrs['department'],
            'uploaded_by': user_attrs['username'],
            'user_id': user_attrs['user_id'],
            'upload_date': datetime.now().isoformat(),
            'file_size': str(uploaded_file.size),
            'content_type': uploaded_file.type or 'application/octet-stream'
        }
        
        # Display upload info
        st.info(f"📂 **File will be saved to:** `{folder_path}`")
        st.json(metadata)
        
        if st.button("Upload File", type="primary"):
            with st.spinner("Uploading file..."):
                if upload_file_to_s3(uploaded_file, folder_path, metadata):
                    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": f"📎 Uploaded file: {uploaded_file.name} to {folder_path}"
                    })
                else:
                    st.error("❌ Upload failed. Please try again.")
    
    # Chat Interface
    st.header("💬 Chat Interface")
    
    # Display chat messages
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
            
            # Generate response
            with st.chat_message("assistant", avatar="🤖"):
                chat_responses_generator = generate_chat_responses(chat_completion)
                full_response = st.write_stream(chat_responses_generator)
                
        except Exception as e:
            st.error(f"Error: {e}", icon="🚨")
        
        # Append response to session state
        if isinstance(full_response, str):
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
        else:
            combined_response = "\n".join(str(item) for item in full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": combined_response}
            )

if __name__ == "__main__":
    main()