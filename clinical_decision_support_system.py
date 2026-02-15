import os
import streamlit as st
from pathlib import Path
import shutil

from loader import load_medical_documents
from rag_core import build_rag

os.chdir("D:/Tushar/Python/CDSS_project")
DOCUMENTS_DIR = os.getcwd()

st.set_page_config(page_title="Clinical RAG Assistant", layout="wide")
st.title("📄 Medical Diagnosis Assistant (RAG)")

# Create a sidebar for file upload
st.sidebar.header("📁 Document Management")
st.sidebar.markdown("---")

# File uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload medical documents (TXT, PDF, XML):",
    type=["txt", "pdf", "xml"],
    accept_multiple_files=True
)

# Handle file uploads
if uploaded_files:
    st.sidebar.info(f"Processing {len(uploaded_files)} file(s)...")
    for uploaded_file in uploaded_files:
        # Save file to DOCUMENTS_DIR
        file_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"✅ Saved: {uploaded_file.name}")
    
    # Force reload of documents
    if "qa_chain" in st.session_state:
        del st.session_state["qa_chain"]
    st.rerun()

# List existing documents
existing_docs = [f for f in os.listdir(DOCUMENTS_DIR) 
                 if f.lower().endswith(('.txt', '.pdf', '.xml'))]
if existing_docs:
    st.sidebar.markdown("### 📚 Loaded Documents")
    for doc in existing_docs:
        st.sidebar.text(f"• {doc}")

st.sidebar.markdown("---")

# Load documents once
if "qa_chain" not in st.session_state:
    with st.spinner("Loading documents and creating index..."):
        docs = load_medical_documents(DOCUMENTS_DIR)
        st.session_state["qa_chain"] = build_rag(docs)
    st.success("✅ Index ready! Ask a question below.")

qa_chain = st.session_state["qa_chain"]

# User query box
st.markdown("### 🔍 Query Section")
query = st.text_area("Enter symptoms or medical query here:")

if st.button("🔎 Get Diagnosis", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Generating response..."):
            result = qa_chain(query)

        # Show the answer
        st.markdown("### 🧠 Suggested Answers")
        st.write(result["result"])

        # Show supporting sources
        st.markdown("### 📄 Supporting Evidence")
        for i, doc in enumerate(result["source_documents"], 1):
            with st.expander(f"📋 Source #{i}"):
                source = doc.metadata.get("source", "Unknown")
                st.write(f"**File:** {source}")
                st.write(doc.page_content)



