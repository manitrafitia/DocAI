from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.rag import router as rag_router
from app.api.v1.mindmap import router as mindmap_router

app = FastAPI(title="MindMap RAG API", version="1.0.0")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure les routeurs
app.include_router(rag_router, prefix="/api/v1")
app.include_router(mindmap_router, prefix="/api/v1")

# Endpoint de santé
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
