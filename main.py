from fastapi import FastAPI, UploadFile, File, Form,Request
from fastapi.middleware.cors import CORSMiddleware
from utils import process_pdf, search_similar_chunks, ask_llm
import os

app = FastAPI()

# Autoriser CORS pour le frontend Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
documents = []
index = None
embeddings_model = None
texts = []

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global documents, index, embeddings_model, texts
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    texts, index, embeddings_model = process_pdf(file_path)
    os.remove(file_path)
    return {"message": "PDF traité avec succès."}

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    print("✅ Requête reçue ! Question :", question)
    if not index:
        print("⚠️ Aucun PDF n'a été chargé.")
        return {"answer": "Aucun document n'a encore été chargé."}

    context = search_similar_chunks(question, index, embeddings_model, texts)
    print("🧠 Contexte trouvé :", context[:100])
    answer = ask_llm(question, context)
    print("💬 Réponse générée :", answer)
    return {"answer": answer}

@app.post("/ask-stream")
async def ask_stream(request: Request):
    body = await request.json()
    question = body.get("question")
    context = body.get("context")

    async def stream_response():
        response = openai.ChatCompletion.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "Tu es un assistant qui répond uniquement à partir du contexte."},
                {"role": "user", "content": f"Contexte :\n{context}\n\nQuestion : {question}"}
            ],
            stream=True,
        )

        try:
            for chunk in response:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    yield delta["content"]
                    await asyncio.sleep(0)  # permet à FastAPI de stream
        except Exception as e:
            yield f"\n[Erreur] {str(e)}"

    return StreamingResponse(stream_response(), media_type="text/plain")