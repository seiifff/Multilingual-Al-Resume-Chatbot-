# 🤖 Multilingual AI Resume Chatbot

A recruiter-facing RAG chatbot that answers questions about my CV using **grounded, cited responses** — **zero hallucinations**, full transparency.

✅ **Multilingual**: Supports English, Spanish, French, German, and 50+ languages  
✅ **Grounded**: Every answer cites exact snippets from my CV  
✅ **Safe**: Built-in guardrails prevent fabrication  
✅ **Fast**: FAISS vector search + FLAN-T5 generation  

![Demo](assets/demo-screenshot.png) <!-- Optional: add after generating -->

## 🚀 Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your resume at data/resume.pdf

# 3. Build knowledge base
python ingest.py   # Parse PDF → semantic chunks
python embed.py    # Embed chunks → FAISS index

# 4. Launch interactive UI
streamlit run app.py
