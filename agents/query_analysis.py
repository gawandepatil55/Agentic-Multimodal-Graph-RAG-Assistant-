import json
from groq import Groq
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

def query_analysis_node(state: dict) -> dict:
    """
    Analyzes the user's query to decide the retrieval strategy (vector, graph, or hybrid).
    Extracts relevant keywords/entity names for graph lookup and refines the query for vector search.
    """
    query = state.get("query", "")
    agent_steps = state.get("agent_steps", [])
    
    logger.info(f"Query Analysis Agent starting analysis for query: '{query}'")
    step_msg = "Query Analysis Agent: Started query analysis."
    agent_steps.append(step_msg)
    
    if not settings.GROQ_API_KEY:
        logger.warning("Groq API key missing. Defaulting to hybrid search.")
        return {
            "search_mode": "hybrid",
            "keywords": [query],
            "refined_query": query,
            "agent_steps": agent_steps + ["Query Analysis Agent: Groq API Key missing, defaulted to Hybrid search."]
        }

    prompt = f"""
    You are the Query Analysis Agent in an Agentic Multimodal Graph-RAG system.
    Analyze the user's query and decide the best retrieval strategy.
    
    Strategies:
    - "vector": Use when the query asks for general information, summaries, concepts, or semantic similarity (e.g. "What is diabetic retinopathy?", "Explain MRI scan guidelines").
    - "graph": Use when the query involves specific entity lookups, relationships, multi-hop paths, or structural connectivity (e.g. "Which engineers work on Project Medical?", "What symptoms are related to diabetes?").
    - "hybrid": Use when the query requires both semantic text understanding and multi-hop entity traversal (e.g. "Which engineers worked on projects related to medical imaging?").
    
    Extract search keywords (names of people, projects, technologies, conditions, objects, etc.) that can be looked up in the Neo4j Knowledge Graph.
    
    Refine the query to be optimized for semantic vector similarity search.
    
    User Query: "{query}"
    
    Format the output strictly as a JSON object matching this schema:
    {{
        "search_mode": "vector|graph|hybrid",
        "keywords": ["List", "of", "extracted", "entity", "names", "or", "nouns"],
        "refined_query": "A version of the query optimized for vector search",
        "reasoning": "Brief explanation of why this strategy was chosen"
    }}
    """
    
    try:
        client = Groq(api_key=settings.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a query analysis routing agent. Respond strictly in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(response.choices[0].message.content)
        search_mode = analysis.get("search_mode", "hybrid")
        keywords = analysis.get("keywords", [])
        refined_query = analysis.get("refined_query", query)
        reasoning = analysis.get("reasoning", "No reasoning provided.")
        
        log_detail = f"Query Analysis Agent: Selected strategy '{search_mode}' because '{reasoning}'."
        logger.info(log_detail)
        agent_steps.append(log_detail)
        
        return {
            "search_mode": search_mode,
            "keywords": keywords,
            "refined_query": refined_query,
            "agent_steps": agent_steps
        }
    except Exception as e:
        logger.error(f"Error in Query Analysis Agent using Groq: {e}")
        agent_steps.append(f"Query Analysis Agent Error: {e}. Defaulting to hybrid.")
        return {
            "search_mode": "hybrid",
            "keywords": [query],
            "refined_query": query,
            "agent_steps": agent_steps
        }
