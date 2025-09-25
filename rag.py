
import os
import pickle
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ResumeRAG:
    def __init__(
        self,
        chunks_path="models/chunks.pkl",
        index_path="models/resume_index.faiss",
        embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        gen_model="google/flan-t5-base"
    ):
        # Load chunks
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load embedding model
        self.embedder = SentenceTransformer(embed_model)

        # Load generation model
        self.tokenizer = AutoTokenizer.from_pretrained(gen_model)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(
            gen_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            self.generator.to("cuda")
        self.generator.eval()


    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Retrieve top-k relevant CV chunks."""
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_emb.astype("float32"), k)
        results = []
        for idx, score in zip(I[0], D[0]):
            results.append({
                "text": self.chunks[idx],
                "score": float(score)
            })
        return results

    def generate_answer(self, query: str, context_chunks: list[dict]) -> tuple[str, list[str]]:
        """Generate answer + return cited snippets."""
        # Build context from top chunks
        context = "\n".join([c["text"] for c in context_chunks])

        # Prompt engineering (critical for grounding)
        input_text = (
            f"Answer based ONLY on the following CV excerpts. "
            f"If the answer isn't in the CV, say 'I don't know based on the CV.'\n\n"
            f"CV Excerpts:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=3,
                early_stopping=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        cited_snippets = [c["text"] for c in context_chunks]
        return answer.strip(), cited_snippets
