# app/main.py
from fastapi import FastAPI
from app.api.v1 import rag

app = FastAPI(title="MindMap RAG API", version="1.0.0")

# Inclure les routes
app.include_router(rag.router)
