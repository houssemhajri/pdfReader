import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import openai


# Config Groq
openai.api_base = "https://api.groq.com/openai/v1"
openai.api_key = "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"
client = openai.OpenAI(api_key=openai.api_key, base_url=openai.api_base)




def split_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_pdf(file_path):
    with fitz.open(file_path) as doc:
        text = "\n".join(page.get_text() for page in doc)

    chunks = split_text(text)
    model = SentenceTransformer('./my-custom-model')
    embeddings = model.encode(chunks, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    os.makedirs("data", exist_ok=True)

    with open("data/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, "data/index.faiss")

    return True

def search_similar_chunks(query, k=5):
    with open("data/chunks.pkl", "rb") as f:
        texts = pickle.load(f)

    index = faiss.read_index("data/index.faiss")
    model = SentenceTransformer('./my-custom-model')

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    results = [texts[i] for i in I[0]]
    return "\n---\n".join(results)

def ask_llm(question, context):
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "Tu es un assistant qui répond uniquement à partir du contexte fourni."},
                {"role": "user", "content": f"Contexte :\n{context}\n\nQuestion : {question}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Erreur LLM: {str(e)}"
