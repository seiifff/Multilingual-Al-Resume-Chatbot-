
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def is_answer_supported_by_context(answer: str, context: str, threshold: float = 0.25) -> bool:
    """
    Check if the generated answer is grounded in the retrieved context.
    Uses TF-IDF cosine similarity (lightweight & effective for short texts).
    """
    if not answer.strip() or not context.strip():
        return False

    # Handle very short answers
    if len(answer.split()) < 3:
        return True  # Assume short factual answers are safe

    vectorizer = TfidfVectorizer(stop_words='english').fit([answer, context])
    tfidf_matrix = vectorizer.transform([answer, context])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).item()
    return similarity >= threshold

def is_retrieval_relevant(scores: list[float], min_score: float = 100.0) -> bool:
    """
    Check if the top retrieval score is good enough.
    FAISS L2 scores: lower = better. We use a max distance threshold.
    """
    if not scores:
        return False
    return min(scores) <= min_score  # Tune this based on your data

def safe_generate(rag, query: str, retrieval_k: int = 3):
    """
    Generate answer with hallucination guardrails.
    Returns (answer, cited_snippets) or fallback response.
    """
    # Step 1: Retrieve
    chunks = rag.retrieve(query, k=retrieval_k)
    scores = [c["score"] for c in chunks]

    # Guardrail 1: Check retrieval quality
    if not is_retrieval_relevant(scores, min_score=120.0):
        return "I don't know based on the CV.", []

    # Step 2: Generate
    answer, cited_snippets = rag.generate_answer(query, chunks)

    # Guardrail 2: Check answer grounding
    full_context = "\n".join(cited_snippets)
    if not is_answer_supported_by_context(answer, full_context, threshold=0.25):
        return "I cannot confirm this based on the CV.", []

    return answer, cited_snippets
