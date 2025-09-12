from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS

class VectorStoreService:
    def __init__(self, model_name: str = "mistral"):
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vectorstore = None
        try:
            # Try to load existing vectorstore
            self.load()
        except Exception as e:
            # If loading fails, vectorstore will remain None
            print(f"Could not load existing vectorstore: {e}")
            print("Please build the vectorstore first using the /ingest endpoint")

    def build_store(self, texts: list[str]):
        """
        Crée un FAISS vectorstore à partir d’une liste de textes.
        """
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        return self.vectorstore

    def save(self, path: str = "data/faiss_index"):
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load(self, path: str = "data/faiss_index"):
        self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        return self.vectorstore

    def query(self, query: str, k: int = 3):
        if not self.vectorstore:
            raise ValueError("Vectorstore non initialisé.")
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs

# Instance globale
vectorstore_service = VectorStoreService()
