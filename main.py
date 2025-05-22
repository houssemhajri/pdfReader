from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from utils import process_pdf, search_similar_chunks, ask_llm
import os
from sentence_transformers import SentenceTransformer, models

app = FastAPI()

# CORS pour Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Transformer de base (petit modèle)
word_embedding_model = models.Transformer('distilbert-base-uncased')

# 2. Pooling (pour extraire une seule vector par phrase)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)

# 3. Assemble en SentenceTransformer
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# 4. Sauvegarder le modèle custom
model.save('./my-custom-model')

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        process_pdf(file_path)
    finally:
        os.remove(file_path)

    return {"message": "PDF traité avec succès."}

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    try:
        context = search_similar_chunks(question)
        answer = ask_llm(question, context)
        return {"answer": answer}
    except Exception as e:
        print("❌ Erreur:", e)
        return {"answer": "Erreur serveur. Assurez-vous qu’un document a été uploadé."}
