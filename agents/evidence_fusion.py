from utils.logger import get_logger

logger = get_logger(__name__)

def evidence_fusion_node(state: dict) -> dict:
    """
    Combines retrieved semantic chunks, image descriptions, and graph relationship facts.
    Formats them into a structured context block for the generation model.
    """
    vector_results = state.get("vector_results", [])
    image_results = state.get("image_results", [])
    graph_results = state.get("graph_results", [])
    agent_steps = state.get("agent_steps", [])
    
    logger.info("Evidence Fusion Agent starting context compilation...")
    step_msg = "Evidence Fusion Agent: Started fusing evidence."
    agent_steps.append(step_msg)
    
    context_parts = []
    
    # 1. Fuse Vector text chunks
    if vector_results:
        context_parts.append("### SEMANTIC TEXT CHUNKS FROM DOCUMENTS:")
        for idx, item in enumerate(vector_results):
            meta = item.get("metadata", {})
            doc_name = meta.get("document_name", "Unknown Source")
            page_num = meta.get("page", "?")
            text = item.get("text", "").strip()
            score = item.get("similarity", 0.0)
            
            context_parts.append(
                f"{idx + 1}. [Source: {doc_name} (Page {page_num}), Relevance: {score:.2f}]\n"
                f"   \"{text}\"\n"
            )
    else:
        context_parts.append("### SEMANTIC TEXT CHUNKS FROM DOCUMENTS:\nNo relevant text passages retrieved.\n")

    # 2. Fuse Knowledge Graph relationships
    if graph_results:
        context_parts.append("### KNOWLEDGE GRAPH PATHS & ENTITY RELATIONS:")
        for idx, fact in enumerate(graph_results):
            src = fact.get("source")
            src_type = fact.get("source_type", "Entity")
            rel = fact.get("relationship", "RELATED_TO")
            tgt = fact.get("target")
            tgt_lbl = fact.get("target_label", "Entity")
            
            context_parts.append(
                f"{idx + 1}. ({src}:{src_type}) -[:{rel}]-> ({tgt}:{tgt_lbl})"
            )
        context_parts.append("")  # spacing
    else:
        context_parts.append("### KNOWLEDGE GRAPH PATHS & ENTITY RELATIONS:\nNo relevant graph relationship paths retrieved.\n")

    # 3. Fuse Image metadata and descriptions
    if image_results:
        context_parts.append("### VISUAL EVIDENCE (RELATED IMAGES):")
        for idx, img in enumerate(image_results):
            meta = img.get("metadata", {})
            img_name = img.get("image_name", "Unnamed Image")
            img_path = meta.get("file_path", "")
            desc = img.get("description", "").strip()
            score = img.get("similarity", 0.0)
            
            context_parts.append(
                f"{idx + 1}. Image Reference: '{img_name}' (Similarity: {score:.2f})\n"
                f"   File Path: {img_path}\n"
                f"   Visual Description: \"{desc}\"\n"
            )
    else:
        context_parts.append("### VISUAL EVIDENCE (RELATED IMAGES):\nNo matching images retrieved.\n")

    fused_context = "\n".join(context_parts)
    
    log_detail = (
        f"Evidence Fusion Agent: Fused {len(vector_results)} text chunks, "
        f"{len(graph_results)} graph paths, and {len(image_results)} image items into context."
    )
    logger.info(log_detail)
    agent_steps.append(log_detail)
    
    return {
        "fused_context": fused_context,
        "agent_steps": agent_steps
    }
