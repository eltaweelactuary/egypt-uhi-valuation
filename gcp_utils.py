import os
import json
import streamlit as st
from google.oauth2 import service_account

def load_gcp_secrets(secret_id: str, project_id: str = None):
    """
    Simplified loader: Only checks session state or local file.
    """
    if "uploaded_gcp_json" in st.session_state and st.session_state.uploaded_gcp_json:
        return st.session_state.uploaded_gcp_json
    
    local_path = "service_account.json"
    if os.path.exists(local_path):
        with open(local_path, 'r') as f:
            return json.load(f)
            
    return None

def get_gcp_credentials():
    """
    Initializes GCP credentials strictly from uploaded JSON or local file.
    Uses from_service_account_info for in-memory loading (more robust in cloud).
    """
    # 0. Priority 1: User-uploaded JSON in session state
    if "uploaded_gcp_json" in st.session_state and st.session_state.uploaded_gcp_json:
        try:
            info = st.session_state.uploaded_gcp_json
            # Use in-memory loading instead of temp files
            return service_account.Credentials.from_service_account_info(info)
        except Exception as e:
            st.error(f"Auth Error (Session): {str(e)}")
            return None

    # 1. Priority 2: Local service_account.json (Development)
    local_path = "service_account.json"
    if os.path.exists(local_path):
        try:
            return service_account.Credentials.from_service_account_file(local_path)
        except Exception as e:
            # st.error(f"Auth Error (Local): {str(e)}")
            return None

    return None

def initialize_vertex_ai():
    """
    Standardized Vertex AI Initialization (v2.3).
    Ensures global state is set correctly before any model calls.
    """
    import vertexai
    
    # Check session for existing valid project_id
    project_id = None
    if "uploaded_gcp_json" in st.session_state and st.session_state.uploaded_gcp_json:
        project_id = st.session_state.uploaded_gcp_json.get("project_id")
    
    if not project_id:
        local_path = "service_account.json"
        if os.path.exists(local_path):
            with open(local_path, 'r') as f:
                project_id = json.load(f).get("project_id")
                
    if not project_id:
        return False

    # Force re-init if not already cached in this session to avoid hangs
    if "vertex_init_lock" not in st.session_state or st.session_state.vertex_init_lock != project_id:
        creds = get_gcp_credentials()
        if not creds:
            return False
        try:
            vertexai.init(project=project_id, location="us-central1", credentials=creds)
            st.session_state.vertex_init_lock = project_id
            return True
        except Exception as e:
            st.error(f"Vertex AI Discovery Error: {str(e)}")
            return False
            
    return True

def ask_gemini_actuary(user_query: str, data_summary: str):
    """
    Sends a strategic actuarial query to Gemini.
    Optimized for speed and connection stability.
    """
    from vertexai.generative_models import GenerativeModel
    
    if not initialize_vertex_ai():
        return "⚠️ Gemini is unavailable: Auth credentials could not be initialized."
        
    try:
        # Use a highly responsive model variant
        model = GenerativeModel("gemini-1.5-flash")
        
        # Construct the context-heavy prompt
        full_prompt = f"""
        Role: Senior Executive Actuary for Egypt Health Insurance (Law 2/2018).
        Context Data:
        {data_summary}
        
        Task: Analyze the user query using actuarial logic. Be concise and strategic.
        User Query: {user_query}
        """
        
        # Add a safety timeout via exception handling (implicit in SDK)
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        if "quota" in str(e).lower():
            return "❌ API Quota Exceeded. Please try again in 60 seconds."
        return f"❌ AI Strategic Agent Error: {str(e)}"

def get_gcp_diagnostics():
    """
    Silent diagnostics.
    """
    return {"status": "Silent Mode (JSON Only)", "checks": []}
