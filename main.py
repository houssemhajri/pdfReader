from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from utils import process_pdf, search_similar_chunks, ask_llm
import os

app = FastAPI()

# CORS pour Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
