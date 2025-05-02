import sys
from pathlib import Path
from patch_streamlit import patch_streamlit_file_watcher
patch_streamlit_file_watcher()


import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"


# Avoid Streamlit inspecting `torch.classes`
if 'torch.classes' in sys.modules:
    del sys.modules['torch.classes']


# Add root directory to sys.path to allow absolute imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
from app.rag_engine import RAGEngine
from app.data_loader import load_pdf

PDF_PATH = "/workspaces/rag-kg/data/Manual example - AeroCraft ACE-900.pdf"

# Streamlit UI setup
st.set_page_config(page_title="AeroCraft ACE-900 RAG", layout="wide")
st.title("AeroCraft ACE-900 - Knowledge Assistant")
st.markdown("Ask any question based on the operations manual PDF.")

@st.cache_data
def setup_rag_engine():
    documents = load_pdf(PDF_PATH)
    full_text = "\n".join([doc.page_content for doc in documents])
    engine = RAGEngine()
    chunks = engine.chunk_text(full_text)
    engine.build_index(chunks)
    return engine

# Load and initialize RAG engine
rag_engine = setup_rag_engine()

# User input
query = st.text_input("🔍 Ask a question:")
if query:
    answer, top_chunks = rag_engine.query(query)
    
    st.subheader("Most Relevant Text Chunks:")
    for i, chunk in enumerate(top_chunks):
        st.markdown(f"**Chunk {i+1}:**")
        st.write(chunk)
    
    st.subheader("Answer:")
    st.success(answer)