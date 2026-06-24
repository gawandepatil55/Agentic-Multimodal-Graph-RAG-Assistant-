import base64
from groq import Groq
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

def encode_image(image_path: str) -> str:
    """Helper function to convert image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def response_generation_node(state: dict) -> dict:
    """
    Synthesizes the final answer using Groq, referencing the fused context
    and any visual input from the user.
    """
    query = state.get("query", "")
    fused_context = state.get("fused_context", "")
    current_image_path = state.get("current_image_path", None)
    agent_steps = state.get("agent_steps", [])
    
    logger.info("Response Generation Agent starting response synthesis...")
    step_msg = "Response Generation Agent: Synthesizing final answer."
    agent_steps.append(step_msg)
    
    if not settings.GROQ_API_KEY:
        error_msg = "Response Generation Agent: Groq API Key missing. Cannot generate response."
        logger.error(error_msg)
        return {
            "response": "Error: GROQ_API_KEY is not configured in the application environment.",
            "agent_steps": agent_steps + [error_msg]
        }
        
    prompt = f"""
    You are the Response Generation Agent in an advanced Agentic Multimodal Graph-RAG system.
    Your goal is to answer the User Query by drawing on both semantic text evidence and knowledge graph relations.
    
    User Query: "{query}"
    
    Below is the retrieved and fused evidence from our hybrid vector & graph indexing systems:
    ---
    {fused_context}
    ---
    
    Instructions:
    1. Answer the query thoroughly, logically, and accurately.
    2. Explicitly cite documents (e.g. "[DocName.pdf (Page X)]") when referencing text chunks.
    3. Explicitly cite knowledge graph relations when answering multi-hop queries (e.g. "We found that Person X works on Project Y, which is built by Team Z").
    4. If related images are mentioned in the context and help answer the query, mention them (e.g., "The related image 'mri_scan.jpg' displays...").
    5. If you do not have enough information to answer, state that clearly, but try to provide any partial information found in the graph or text.
    """
    
    try:
        client = Groq(api_key=settings.GROQ_API_KEY)
        
        # If the user uploaded an image as part of their chat query
        if current_image_path:
            logger.info(f"Encoding user-provided chat image for Groq Vision: {current_image_path}")
            base64_image = encode_image(current_image_path)
            
            log_detail = "Response Generation Agent: Fusing user uploaded image in generation prompt (calling Llama Vision model)."
            logger.info(log_detail)
            agent_steps.append(log_detail)
            
            # Since an image is attached, we must use the vision-capable model
            response = client.chat.completions.create(
                model=settings.VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
        else:
            # If text only, we use the higher-reasoning text-only model
            response = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that synthesizes Graph-RAG responses."},
                    {"role": "user", "content": prompt}
                ]
            )
            
        answer = response.choices[0].message.content
        logger.info("Successfully generated final answer.")
        agent_steps.append("Response Generation Agent: Generated final answer successfully.")
        
        return {
            "response": answer,
            "agent_steps": agent_steps
        }
    except Exception as e:
        err_msg = f"Response Generation Agent Error: Failed to generate response: {e}"
        logger.error(err_msg)
        agent_steps.append(err_msg)
        return {
            "response": f"Failed to generate an answer due to an error in the response generator: {e}",
            "agent_steps": agent_steps
        }
