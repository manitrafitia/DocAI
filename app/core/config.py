import os

class Settings:
    APP_NAME: str = "MindMap RAG API"
    APP_VERSION: str = "1.0.0"

    # Ollama
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral") 
settings = Settings()
