# app.py
import streamlit as st
from rag import ResumeRAG
from guardrails import safe_generate
import os

# Page config
st.set_page_config(
    page_title="AI Resume Assistant",
    page_icon="📄",
    layout="wide"
)

# Title & description
st.title(" Multilingual AI Resume Chatbot")
st.markdown("""
Ask questions about my experience, skills, or projects — answers are **grounded in my CV** and include **source snippets**.
Supports English, Spanish, French, German, and 50+ languages.
""")

# Initialize RAG (cached to avoid reloading)
@st.cache_resource
def load_rag():
    return ResumeRAG()

# Check if models exist
if not os.path.exists("models/resume_index.faiss"):
    st.error("FAISS index not found. Please run `python embed.py` first.")
    st.stop()

# Load RAG
rag = load_rag()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Sources from CV"):
                for i, src in enumerate(message["sources"]):
                    st.text_area(f"Snippet {i+1}", src, height=80, key=f"src_{len(st.session_state.messages)}_{i}")

# User input
if prompt := st.chat_input("Ask a question (e.g., 'What ML projects have you done?')"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching CV..."):
            answer, sources = safe_generate(rag, prompt)
        st.markdown(answer)

        # Show sources if available
        if sources:
            with st.expander("📚 Sources from CV"):
                for i, src in enumerate(sources[:3]):  # Show top 3
                    st.text_area(f"Snippet {i+1}", src, height=80, key=f"new_src_{i}")

    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
