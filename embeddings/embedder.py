import os
from abc import ABC, abstractmethod
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding vector for a text string."""
        pass

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of text strings."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding vector dimensionality."""
        pass

class GeminiEmbedder(BaseEmbedder):
    def __init__(self):
        self.model_name = "text-embedding-004"
        logger.info(f"Initializing GeminiEmbedder using model: {self.model_name}")
        try:
            import google.generativeai as genai
            self.genai = genai
            # If some legacy variable is set
            gemini_key = os.getenv("GEMINI_API_KEY", "")
            if gemini_key:
                self.genai.configure(api_key=gemini_key)
        except ImportError:
            logger.error("google-generativeai is not installed.")
            raise ImportError("google-generativeai package is required for GeminiEmbedder.")

    def embed_text(self, text: str) -> list[float]:
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        try:
            response = self.genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            return response["embedding"]
        except Exception as e:
            logger.error(f"Error generating Gemini embedding: {e}")
            raise

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        try:
            response = self.genai.embed_content(
                model=self.model_name,
                content=texts,
                task_type="retrieval_document"
            )
            return response["embedding"]
        except Exception as e:
            logger.error(f"Error generating Gemini embeddings for batch: {e}")
            raise

    def get_dimension(self) -> int:
        return 768

class LocalEmbedder(BaseEmbedder):
    def __init__(self):
        self.model_name = "BAAI/bge-small-en-v1.5"
        logger.info(f"Initializing LocalEmbedder using SentenceTransformers: {self.model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed.")
            raise ImportError("sentence-transformers is required for local embedding provider.")

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        return 384

# Global instance manager
_embedder_instance = None

def get_embedder() -> BaseEmbedder:
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance

    provider = settings.EMBEDDING_PROVIDER
    if provider == "local":
        try:
            _embedder_instance = LocalEmbedder()
        except ImportError as e:
            logger.error(f"Failed to load LocalEmbedder: {e}. Attempting Gemini fall back...")
            _embedder_instance = GeminiEmbedder()
    else:
        try:
            _embedder_instance = GeminiEmbedder()
        except ImportError:
            logger.warning("Failed to initialize GeminiEmbedder. Defaulting to LocalEmbedder.")
            _embedder_instance = LocalEmbedder()

    return _embedder_instance
