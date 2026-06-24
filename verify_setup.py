import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from config import settings
from config.settings import validate_config
from database.vector_store import VectorStoreManager
from graph.neo4j_client import Neo4jClient
from embeddings.embedder import get_embedder

def run_diagnostics():
    print("==================================================")
    print("    Agentic Multimodal Graph-RAG Diagnostics      ")
    print("                  (Groq Version)                  ")
    print("==================================================")
    
    # 1. Config Validation
    print("\n[Step 1/4] Checking environment configurations...")
    is_valid, warnings = validate_config()
    
    if warnings:
        print("  [!] Configuration Warnings:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("  [OK] Environment configuration looks fully complete.")
        
    if not settings.GROQ_API_KEY:
        print("  [ERROR] Critical Error: GROQ_API_KEY is not set. Groq services are required.")
        return False

    # 2. Embeddings Validation
    print("\n[Step 2/4] Initializing Embedding Engine...")
    try:
        embedder = get_embedder()
        print(f"  [OK] Embedding Provider: {settings.EMBEDDING_PROVIDER.upper()}")
        print(f"  [OK] Dimension: {embedder.get_dimension()}")
        
        # Test embed a short query
        test_vector = embedder.embed_text("Test embedding output")
        if len(test_vector) == embedder.get_dimension():
            print("  [OK] Test embedding generation: Successful")
        else:
            print(f"  [ERROR] Test embedding dimension mismatch: Got {len(test_vector)}, expected {embedder.get_dimension()}")
            return False
    except Exception as e:
        print(f"  [ERROR] Embedding initialization failed: {e}")
        return False

    # 3. Vector Database Validation
    print("\n[Step 3/4] Initializing ChromaDB database...")
    try:
        v_store = VectorStoreManager()
        chunks_count = v_store.chunks_collection.count()
        images_count = v_store.images_collection.count()
        print(f"  [OK] ChromaDB Persistent Path: {settings.CHROMA_PATH}")
        print(f"  [OK] Active collections detected:")
        print(f"    - 'document_chunks': {chunks_count} items")
        print(f"    - 'images': {images_count} items")
    except Exception as e:
        print(f"  [ERROR] ChromaDB initialization failed: {e}")
        return False

    # 4. Neo4j Graph DB Validation
    print("\n[Step 4/4] Testing Neo4j Graph Database connection...")
    try:
        n_client = Neo4jClient()
        if n_client.is_connected:
            print(f"  [OK] Neo4j connection URI: {settings.NEO4J_URI}")
            print("  [OK] Connected successfully to Neo4j.")
            
            # Check node and relationship counts
            ent_count = n_client.run_query("MATCH (e:Entity) RETURN count(e) AS count")
            rel_count = n_client.run_query("MATCH ()-[r]->() RETURN count(r) AS count")
            
            entities = ent_count[0]["count"] if ent_count else 0
            relations = rel_count[0]["count"] if rel_count else 0
            
            print(f"  [OK] Neo4j Database stats:")
            print(f"    - Entity Nodes: {entities}")
            print(f"    - Relationships: {relations}")
        else:
            print("  [WARNING] Neo4j Connection: Disconnected")
            print("    * Note: The RAG engine will run in Vector-Only mode *")
            print("    * To enable Knowledge Graphs, configure Neo4j URI and credentials in .env *")
            print("    * Start local Neo4j Desktop or sign up at Neo4j Aura (free cloud instance) *")
    except Exception as e:
        print(f"  [WARNING] Neo4j connection raised an exception: {e}")
        print("    * Note: The system will operate in Vector-Only mode in this state *")

    print("\n==================================================")
    print("  Diagnostics complete. Ready to run Streamlit app!")
    print("==================================================")
    return True

if __name__ == "__main__":
    run_diagnostics()
