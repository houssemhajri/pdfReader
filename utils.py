import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import json

import openai
import os

openai.api_base = "https://api.groq.com/openai/v1"
openai.api_key = "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"
# Divise le texte en petits morceaux
def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Lit le PDF et crée l'index
def process_pdf(file_path):
    with fitz.open(file_path) as doc:
        text = "\n".join(page.get_text() for page in doc)

    chunks = split_text(text)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return chunks, index, model

# Recherche les passages les plus proches
def search_similar_chunks(query, index, model, texts, k=5):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    results = [texts[i] for i in I[0]]
    return "\n---\n".join(results)

# Envoie la question et le contexte à un LLM local (Ollama)
def ask_llm(question, context):
    try:
        response = openai.ChatCompletion.create(
            model="llama3-8b-8192",  # ou mixtral-8x7b-32768
            messages=[
                {"role": "system", "content": "Tu es un assistant qui répond uniquement à partir du contexte fourni."},
                {"role": "user", "content": f"Contexte :\n{context}\n\nQuestion : {question}"}
            ],
            temperature=0.3
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("❌ Erreur Groq:", e)
        return "Erreur lors de la réponse LLM."
