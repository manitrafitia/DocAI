from fastapi import APIRouter, UploadFile, File
from app.services.llm import llm_service
from app.models.rag import PromptRequest, PromptResponse
import tempfile
from app.utils.file_parser import parse_file
from app.utils.text_chunker import chunk_text
from app.services.vectorstore import vectorstore_service

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Parse fichier
    text = parse_file(tmp_path)

    # Découpe en chunks
    chunks = chunk_text(text)

    # Crée vectorstore
    vectorstore_service.build_store(chunks)
    vectorstore_service.save()

    return {"message": f"Fichier {file.filename} ingéré avec succès", "chunks": len(chunks)}

@router.post("/query")
async def query_rag(request: PromptRequest):
    docs = vectorstore_service.query(request.prompt, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    Contexte :
    {context}

    Question :
    {request.prompt}

    Réponse :
    """
    answer = llm_service.generate(prompt)
    return {"answer": answer, "sources": [doc.metadata for doc in docs]}
