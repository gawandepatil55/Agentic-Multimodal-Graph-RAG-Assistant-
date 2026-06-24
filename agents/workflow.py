from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from agents.query_analysis import query_analysis_node
from agents.vector_retrieval import vector_retrieval_node
from agents.graph_retrieval import graph_retrieval_node
from agents.evidence_fusion import evidence_fusion_node
from agents.response_generation import response_generation_node
from utils.logger import get_logger

logger = get_logger(__name__)

class AgentState(TypedDict):
    query: str
    current_image_path: Optional[str]
    search_mode: str
    keywords: List[str]
    refined_query: str
    vector_results: List[Dict[str, Any]]
    image_results: List[Dict[str, Any]]
    graph_results: List[Dict[str, Any]]
    fused_context: str
    response: str
    agent_steps: List[str]

def build_workflow():
    """
    Assembles the LangGraph StateGraph nodes and compiles the state machine.
    """
    logger.info("Assembling Agentic Graph-RAG workflow StateGraph...")
    workflow = StateGraph(AgentState)
    
    # 1. Define nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("vector_retrieval", vector_retrieval_node)
    workflow.add_node("graph_retrieval", graph_retrieval_node)
    workflow.add_node("evidence_fusion", evidence_fusion_node)
    workflow.add_node("response_generation", response_generation_node)
    
    # 2. Define sequential flow
    workflow.set_entry_point("query_analysis")
    workflow.add_edge("query_analysis", "vector_retrieval")
    workflow.add_edge("vector_retrieval", "graph_retrieval")
    workflow.add_edge("graph_retrieval", "evidence_fusion")
    workflow.add_edge("evidence_fusion", "response_generation")
    workflow.add_edge("response_generation", END)
    
    logger.info("StateGraph successfully compiled.")
    return workflow.compile()

def run_rag_agent(query: str, current_image_path: Optional[str] = None) -> dict:
    """
    Entry point to run the compiled Agentic Graph-RAG workflow.
    """
    logger.info(f"Running RAG Agent workflow for query: '{query}'")
    compiled_app = build_workflow()
    
    initial_state = {
        "query": query,
        "current_image_path": current_image_path,
        "search_mode": "hybrid",
        "keywords": [],
        "refined_query": "",
        "vector_results": [],
        "image_results": [],
        "graph_results": [],
        "fused_context": "",
        "response": "",
        "agent_steps": []
    }
    
    try:
        final_state = compiled_app.invoke(initial_state)
        return final_state
    except Exception as e:
        logger.error(f"Error executing agent workflow: {e}")
        return {
            "query": query,
            "current_image_path": current_image_path,
            "search_mode": "hybrid",
            "keywords": [],
            "refined_query": query,
            "vector_results": [],
            "image_results": [],
            "graph_results": [],
            "fused_context": "",
            "response": f"RAG Agent execution failed: {e}",
            "agent_steps": [f"System Error in LangGraph execution: {e}"]
        }
