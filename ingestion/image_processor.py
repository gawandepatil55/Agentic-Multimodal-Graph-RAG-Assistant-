import os
import uuid
import json
import base64
from PIL import Image
from groq import Groq
from config import settings
from database.vector_store import VectorStoreManager
from graph.neo4j_client import Neo4jClient
from utils.logger import get_logger

logger = get_logger(__name__)

def encode_image(image_path: str) -> str:
    """Helper function to convert image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image(image_path: str) -> dict:
    """
    Sends an image to Groq Vision model to extract descriptions, entities, and relationships.
    Returns a dict with keys: 'description', 'entities', 'relationships'
    """
    if not settings.GROQ_API_KEY:
        logger.warning("Groq API Key missing. Skipping image analysis.")
        return {
            "description": "Image uploaded but Groq API key is missing for analysis.",
            "entities": [],
            "relationships": []
        }
        
    try:
        logger.info(f"Encoding image file for Groq Vision analysis: {image_path}")
        base64_image = encode_image(image_path)
        
        prompt = """
        Analyze this image in detail. Extract the following information:
        1. An overall description of the image content.
        2. Identify key entities depicted or mentioned in the image (e.g. Person, Equipment, Disease, Chart, Document, System, etc.).
        3. Identify any relationships between the entities.
        
        Format the output strictly as a JSON object matching this schema:
        {
            "description": "A detailed caption summarizing the visual contents, context, and details of the image.",
            "entities": [
                {
                    "name": "Entity Name",
                    "type": "Equipment|Person|Condition|Technology|etc.",
                    "description": "How the entity relates to the image content"
                }
            ],
            "relationships": [
                {
                    "source": "Entity Name",
                    "target": "Entity Name",
                    "relationship": "DEPICTS|TESTS|DEVELOPS|RELATED_TO|etc."
                }
            ]
        }
        """
        
        logger.info("Calling Groq Vision API...")
        client = Groq(api_key=settings.GROQ_API_KEY)
        
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
            ],
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(response.choices[0].message.content)
        logger.info("Successfully analyzed image via Groq Vision.")
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing image {image_path} with Groq: {e}")
        return {
            "description": f"Failed to analyze image: {os.path.basename(image_path)} due to: {e}",
            "entities": [],
            "relationships": []
        }

def ingest_image(image_path: str, image_name: str, vector_store: VectorStoreManager, neo4j_client: Neo4jClient, doc_id: str = None) -> str:
    """
    Full pipeline to ingest an image.
    """
    image_id = str(uuid.uuid4())
    logger.info(f"Starting ingestion for image '{image_name}' with ID: {image_id}")
    
    analysis = analyze_image(image_path)
    description = analysis.get("description", "")
    
    metadata = {
        "file_path": image_path,
        "doc_id": doc_id or ""
    }
    vector_store.add_image(image_id, image_name, description, metadata=metadata)
    
    if neo4j_client.is_connected:
        neo4j_client.create_image_node(image_id, image_name, description, doc_id=doc_id)
        
        entities = analysis.get("entities", [])
        relationships = analysis.get("relationships", [])
        
        entities_map = {}
        for ent in entities:
            name = ent.get("name")
            ent_type = ent.get("type", "VisualEntity")
            desc = ent.get("description", "")
            
            if name:
                neo4j_client.create_entity_node(name, ent_type, desc)
                neo4j_client.link_image_to_entity(image_id, name)
                entities_map[name.lower()] = name
                
        for rel in relationships:
            src = rel.get("source")
            tgt = rel.get("target")
            rel_type = rel.get("relationship", "RELATED_TO")
            
            if src and tgt:
                src_resolved = entities_map.get(src.lower(), src)
                tgt_resolved = entities_map.get(tgt.lower(), tgt)
                
                neo4j_client.create_entity_node(src_resolved, "VisualEntity")
                neo4j_client.create_entity_node(tgt_resolved, "VisualEntity")
                
                neo4j_client.create_custom_relationship(src_resolved, tgt_resolved, rel_type)
                
    logger.info(f"Successfully finished ingesting image '{image_name}'. ID: {image_id}")
    return image_id
