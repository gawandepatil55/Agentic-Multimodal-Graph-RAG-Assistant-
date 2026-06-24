from database.vector_store import VectorStoreManager
from utils.logger import get_logger

logger = get_logger(__name__)

# Initialize VectorStore singleton or instance
try:
    vector_store = VectorStoreManager()
except Exception as e:
    logger.error(f"Failed to initialize VectorStoreManager in vector_retrieval: {e}")
    vector_store = None

def vector_retrieval_node(state: dict) -> dict:
    """
    Retrieves semantically similar document chunks and image descriptions from ChromaDB.
    Runs only if search_mode is 'vector' or 'hybrid'.
    """
    search_mode = state.get("search_mode", "hybrid")
    refined_query = state.get("refined_query", "")
    agent_steps = state.get("agent_steps", [])
    
    vector_results = []
    image_results = []
    
    if search_mode not in ["vector", "hybrid"]:
        log_msg = "Vector Retrieval Agent: Skipped (Not requested by routing)."
        logger.info(log_msg)
        agent_steps.append(log_msg)
        return {
            "vector_results": vector_results,
            "image_results": image_results,
            "agent_steps": agent_steps
        }
        
    log_msg = f"Vector Retrieval Agent: Retrieving document chunks for: '{refined_query}'"
    logger.info(log_msg)
    agent_steps.append(log_msg)
    
    if vector_store:
        try:
            # Query text chunks
            vector_results = vector_store.search_chunks(refined_query, limit=5)
            log_res = f"Vector Retrieval Agent: Retrieved {len(vector_results)} text chunks."
            logger.info(log_res)
            agent_steps.append(log_res)
            
            # Query image descriptions
            image_results = vector_store.search_images(refined_query, limit=3)
            log_img = f"Vector Retrieval Agent: Retrieved {len(image_results)} matching images."
            logger.info(log_img)
            agent_steps.append(log_img)
        except Exception as e:
            err_msg = f"Vector Retrieval Agent Error: Failed to search vector store: {e}"
            logger.error(err_msg)
            agent_steps.append(err_msg)
    else:
        err_msg = "Vector Retrieval Agent Warning: Vector database client not initialized."
        logger.warning(err_msg)
        agent_steps.append(err_msg)
        
    return {
        "vector_results": vector_results,
        "image_results": image_results,
        "agent_steps": agent_steps
    }
