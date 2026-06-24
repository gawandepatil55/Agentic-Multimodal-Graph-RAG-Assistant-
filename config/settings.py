import os
from pathlib import Path
from dotenv import load_dotenv

# Find and load the .env file
base_dir = Path(__file__).resolve().parent.parent
env_path = base_dir / ".env"
load_dotenv(dotenv_path=env_path)

# System Path Configurations
BASE_DIR = base_dir
DEFAULT_CHROMA_PATH = str(base_dir / "chroma_db")

# Neo4j Configurations
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# Groq API Configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY", os.getenv("grok_API_KEY", ""))

# ChromaDB Configurations
CHROMA_PATH = os.getenv("CHROMA_PATH", DEFAULT_CHROMA_PATH)

# Embedding Configurations
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()

# Model Configurations
LLM_MODEL = "llama-3.3-70b-versatile"  # Default Groq LLM for reasoning and text generation
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Groq Vision model for image ingestion and analysis
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Local text embedding model

def validate_config() -> tuple[bool, list[str]]:
    """
    Validates if the required configuration settings are present.
    Returns a tuple of (is_valid, list_of_warnings).
    """
    warnings = []
    
    if not GROQ_API_KEY:
        warnings.append("GROQ_API_KEY is not set. Groq services will fail.")
        
    if not NEO4J_PASSWORD:
        warnings.append("NEO4J_PASSWORD is not set. Connection to Neo4j Graph DB may fail.")
        
    if EMBEDDING_PROVIDER not in ["gemini", "local"]:
        warnings.append(f"Invalid EMBEDDING_PROVIDER '{EMBEDDING_PROVIDER}'. Defaulting to 'local'.")
        
    is_valid = len(warnings) == 0 or not GROQ_API_KEY  # Groq API key is absolutely required
    return is_valid, warnings
