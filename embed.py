# embed.py
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def create_faiss_index(
    chunks_path="models/chunks.pkl",
    index_path="models/resume_index.faiss",
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
):
    # Load chunks
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"❌ Chunks not found at {chunks_path}. Run ingest.py first.")

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"📄 Loaded {len(chunks)} chunks.")

    # Load embedding model (multilingual!)
    print(f"🧠 Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Generate embeddings
    print("🔍 Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    print(f"✅ Generated {embeddings.shape} embeddings.")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance (Euclidean)
    index.add(embeddings.astype('float32'))

    # Save index
    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"💾 FAISS index saved to {index_path}")
    print("\n🎉 Step 2 complete! Now you can retrieve relevant CV snippets.")

if __name__ == "__main__":
    create_faiss_index()
