# app/api/v1/mindmap.py
from fastapi import APIRouter
from app.models.rag import PromptRequest
from app.models.mindmap import MindmapResponse
from app.services.mindmap_service import generate_mindmap

router = APIRouter(prefix="/mindmap", tags=["Mindmap"])

@router.post("/", response_model=MindmapResponse)
async def create_mindmap(request: PromptRequest):
    data = generate_mindmap(request.prompt)
    return data
