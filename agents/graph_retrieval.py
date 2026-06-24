from graph.neo4j_client import Neo4jClient
from utils.logger import get_logger

logger = get_logger(__name__)

# Initialize Neo4jClient
try:
    neo4j_client = Neo4jClient()
except Exception as e:
    logger.error(f"Failed to initialize Neo4jClient in graph_retrieval: {e}")
    neo4j_client = None

def graph_retrieval_node(state: dict) -> dict:
    """
    Retrieves entities and relationship paths from Neo4j based on keywords.
    Runs only if search_mode is 'graph' or 'hybrid'.
    """
    search_mode = state.get("search_mode", "hybrid")
    keywords = state.get("keywords", [])
    agent_steps = state.get("agent_steps", [])
    
    graph_results = []
    
    if search_mode not in ["graph", "hybrid"]:
        log_msg = "Graph Retrieval Agent: Skipped (Not requested by routing)."
        logger.info(log_msg)
        agent_steps.append(log_msg)
        return {
            "graph_results": graph_results,
            "agent_steps": agent_steps
        }
        
    if not keywords:
        log_msg = "Graph Retrieval Agent: No keywords available for lookup."
        logger.info(log_msg)
        agent_steps.append(log_msg)
        return {
            "graph_results": graph_results,
            "agent_steps": agent_steps
        }
        
    log_msg = f"Graph Retrieval Agent: Searching knowledge graph for keywords: {keywords}"
    logger.info(log_msg)
    agent_steps.append(log_msg)
    
    if neo4j_client and neo4j_client.is_connected:
        try:
            for keyword in keywords:
                # Query related paths
                records = neo4j_client.search_graph_by_keyword(keyword)
                for record in records:
                    fact = {
                        "source": record.get("source"),
                        "source_type": record.get("source_type"),
                        "relationship": record.get("relationship"),
                        "target": record.get("target"),
                        "target_label": record.get("target_label"),
                        "matched_keyword": keyword
                    }
                    # Prevent duplicates in results
                    if fact not in graph_results:
                        graph_results.append(fact)
                        
            log_res = f"Graph Retrieval Agent: Retrieved {len(graph_results)} relationship paths."
            logger.info(log_res)
            agent_steps.append(log_res)
        except Exception as e:
            err_msg = f"Graph Retrieval Agent Error: Failed to execute graph queries: {e}"
            logger.error(err_msg)
            agent_steps.append(err_msg)
    else:
        err_msg = "Graph Retrieval Agent Warning: Neo4j database client is not connected or initialized."
        logger.warning(err_msg)
        agent_steps.append(err_msg)
        
    return {
        "graph_results": graph_results,
        "agent_steps": agent_steps
    }
