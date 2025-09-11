from langchain_ollama import OllamaLLM
from app.core.config import settings

class LLMService:
    def __init__(self, model_name: str = None):
        model = model_name or settings.OLLAMA_MODEL
        self.llm = OllamaLLM(model=model)

    def generate(self, prompt: str) -> str:
        """
        Envoie un prompt au modèle et retourne la réponse générée.
        """
        response = self.llm.invoke(prompt)
        return response

# Instance unique du service (singleton)
llm_service = LLMService()
