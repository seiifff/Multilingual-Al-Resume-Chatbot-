# ingest.py
import os
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize

# Set NLTK data path to a writable directory
nltk.data.path.append('/content/nltk_data')

# Download NLTK data (run once)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) # Download punkt_tab resource

def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def chunk_text(text, max_tokens=100):
    """Split text into semantic chunks (by sentence, capped at max_tokens)."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sent in sentences:
        tokens = len(sent.split())
        if current_token_count + tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_token_count = tokens
        else:
            current_chunk.append(sent)
            current_token_count += tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def main():
    # Paths
    pdf_path = "/content/Resume.pdf"  # Corrected path
    output_dir = "models"

    # Validate input
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ Please place your resume at {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Extract and chunk
    print("📄 Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    print(f"✅ Extracted {len(text)} characters.")

    print("✂️ Chunking text...")
    chunks = chunk_text(text, max_tokens=100)
    print(f"✅ Created {len(chunks)} chunks.")

    # Save chunks (embeddings come in Step 2)
    import pickle
    with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print("💾 Saved chunks to models/chunks.pkl")
    print("\n🎉 Step 1 complete! Now run Step 2 to create embeddings.")

if __name__ == "__main__":
    main()
