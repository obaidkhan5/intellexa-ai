import os
import warnings
import transformers

# =========================================================
# SUPPRESS WARNINGS
# =========================================================
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from dotenv import load_dotenv
from docx import Document
import pandas as pd

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Intellexa Workspace",
    layout="wide",
    initial_sidebar_state="auto" # Auto adjusts for mobile/desktop
)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "app_state" not in st.session_state:
    st.session_state.app_state = "upload" # Can be "upload" or "chat"

# =========================================================
# CUSTOM CSS (Ultra-Clean, Responsive SaaS UI)
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant Dark Theme */
    .stApp {
        background-color: #0A0A0A;
        color: #EDEDED;
    }
    
    /* Hide Defaults */
    #MainMenu, footer, header { visibility: hidden; }

    /* Typography */
    .hero-title {
        font-size: clamp(2rem, 5vw, 3.5rem); /* Responsive font size */
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: clamp(0.9rem, 2vw, 1.1rem);
        color: #A1A1AA;
        text-align: center;
        margin-bottom: 3rem;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #121212;
        border-right: 1px solid #27272A;
    }
    
    .sidebar-brand {
        font-size: 1.25rem;
        font-weight: 600;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        padding-bottom: 1rem;
        border-bottom: 1px solid #27272A;
        margin-bottom: 1.5rem;
    }

    /* Centered Uploader Area */
    .upload-container {
        max-width: 700px;
        margin: 0 auto;
        padding: 2rem;
        background: #121212;
        border-radius: 16px;
        border: 1px solid #27272A;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }

    /* File Uploader Customization */
    [data-testid="stFileUploadDropzone"] {
        background-color: #0A0A0A;
        border: 1px dashed #3F3F46;
        border-radius: 8px;
        padding: 2rem;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #FFFFFF;
        background-color: #171717;
    }
    
    /* Primary Button */
    .stButton > button {
        background-color: #FFFFFF;
        color: #0A0A0A;
        border-radius: 6px;
        border: none;
        width: 100%;
        font-weight: 500;
        padding: 0.6rem 1rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #E5E5E5;
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* Processed Files Pill */
    .file-pill {
        background: #171717;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #D4D4D8;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        border: 1px solid #27272A;
        word-break: break-all;
    }
    
    .file-pill svg {
        margin-right: 8px;
        flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# ENV VARIABLES & PINECONE INIT
# =========================================================
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "intellexa-ai"

if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource(show_spinner="Initializing engine...")
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def extract_text(file):
    text = ""
    try:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted: text += extracted + "\n"
        elif file.name.endswith(".txt"):
            text = file.getvalue().decode("utf-8")
        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
            text = df.to_string() + "\n"
        elif file.name.endswith(".docx"):
            doc = Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
    return text

def process_documents(uploaded_files):
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if not new_files:
        st.info("Selected files are already processed.")
        return
        
    with st.spinner(f"Processing {len(new_files)} documents..."):
        all_text = ""
        for file in new_files:
            all_text += extract_text(file)
            st.session_state.processed_files.add(file.name)
        
        if all_text.strip():
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(all_text)
            
            embeddings = model.encode(chunks)
            vectors = []
            for i, embedding in enumerate(embeddings):
                vectors.append({
                    "id": f"vec_{len(st.session_state.processed_files)}_{i}",
                    "values": embedding.tolist(),
                    "metadata": {"text": chunks[i]}
                })
            
            # Batch upsert
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                index.upsert(vectors=vectors[i:i+batch_size])
                
            st.session_state.app_state = "chat"
            st.rerun()
        else:
            st.warning("No readable text found in the uploaded files.")

# =========================================================
# SIDEBAR (Visible only after first upload or if opened manually)
# =========================================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Intellexa Workspace</div>', unsafe_allow_html=True)
    
    # Show uploader in sidebar ONLY if we are in chat mode
    if st.session_state.app_state == "chat":
        st.markdown('<p style="color: #A1A1AA; font-size: 0.9rem; margin-bottom: 0.5rem;">Add more documents</p>', unsafe_allow_html=True)
        more_files = st.file_uploader("Add more documents", accept_multiple_files=True, type=['pdf', 'txt', 'csv', 'docx'], label_visibility="collapsed")
        if st.button("Process New Files") and more_files:
            process_documents(more_files)
            
        st.markdown("<hr style='border-color: #27272A;'>", unsafe_allow_html=True)
    
    st.markdown('<p style="color: #A1A1AA; font-size: 0.9rem; margin-bottom: 1rem;">Active Knowledge Base</p>', unsafe_allow_html=True)
    
    if st.session_state.processed_files:
        for file_name in st.session_state.processed_files:
            # Clean SVG Icon for files
            file_icon = '''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>'''
            st.markdown(f'<div class="file-pill">{file_icon} {file_name}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color: #52525B; font-size: 0.85rem;">No documents processed yet.</p>', unsafe_allow_html=True)

# =========================================================
# MAIN APP AREA
# =========================================================

# STATE 1: UPLOAD (Centered Mid-Screen)
if st.session_state.app_state == "upload":
    # Empty space to push content to middle vertically
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    st.markdown('<div class="hero-title">Intellexa</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Secure, private knowledge retrieval. Upload your documents to begin initializing the workspace.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        label="Drag and drop files here", 
        accept_multiple_files=True, 
        type=['pdf', 'txt', 'csv', 'docx'],
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Initialize Workspace") and uploaded_files:
        process_documents(uploaded_files)
        
    st.markdown('</div>', unsafe_allow_html=True)

# STATE 2: CHAT INTERFACE
elif st.session_state.app_state == "chat":
    st.markdown('<div style="padding-bottom: 2rem; border-bottom: 1px solid #27272A; margin-bottom: 2rem;"><span style="font-size: 1.5rem; font-weight: 600;">Workspace Chat</span></div>', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if query := st.chat_input("Ask a question based on your documents..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context..."):
                query_embedding = model.encode(query).tolist()
                results = index.query(
                    vector=query_embedding,
                    top_k=3,
                    include_metadata=True
                )
                
                if results["matches"] and results["matches"][0]["score"] > 0.1:
                    response = ""
                    for i, match in enumerate(results["matches"]):
                        response += f"{match['metadata']['text']}\n\n"
                        if i < len(results["matches"]) - 1:
                            response += "---\n\n"
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    response = "I couldn't find a highly relevant answer in the uploaded documents. Try rephrasing or upload more context."
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})