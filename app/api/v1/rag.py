# app/api/v1/rag.py
from fastapi import APIRouter
from app.services.llm import llm_service
from app.models.rag import PromptRequest, PromptResponse

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/ask", response_model=PromptResponse)
async def ask_model(request: PromptRequest):
    result = llm_service.generate(request.prompt)
    return PromptResponse(response=result)
