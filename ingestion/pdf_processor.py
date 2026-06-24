import os
import fitz  # PyMuPDF
import json
import uuid
from groq import Groq
from config import settings
from database.vector_store import VectorStoreManager
from graph.neo4j_client import Neo4jClient
from utils.logger import get_logger

logger = get_logger(__name__)

def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Extracts text from a PDF file page by page.
    Returns a list of dicts: [{"page": page_num, "text": text}]
    """
    logger.info(f"Extracting text from PDF: {file_path}")
    pages_data = []
    try:
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            pages_data.append({
                "page": i + 1,
                "text": text
            })
        doc.close()
        logger.info(f"Extracted {len(pages_data)} pages from {file_path}.")
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
    return pages_data

def recursive_chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Splits text recursively by paragraphs, sentences, and words to maintain semantic units.
    """
    if not text:
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []
    
    def split_recursive(text_to_split: str, current_sep_idx: int) -> list[str]:
        if len(text_to_split) <= chunk_size:
            return [text_to_split]
            
        if current_sep_idx >= len(separators):
            return [text_to_split[i:i + chunk_size] for i in range(0, len(text_to_split), chunk_size)]
            
        sep = separators[current_sep_idx]
        parts = text_to_split.split(sep) if sep else list(text_to_split)
        
        compiled_parts = []
        current_chunk = ""
        
        for part in parts:
            if len(part) > chunk_size:
                if current_chunk:
                    compiled_parts.append(current_chunk)
                    current_chunk = ""
                sub_parts = split_recursive(part, current_sep_idx + 1)
                compiled_parts.extend(sub_parts)
                continue
                
            join_str = sep if current_chunk else ""
            if len(current_chunk) + len(join_str) + len(part) <= chunk_size:
                current_chunk += join_str + part
            else:
                compiled_parts.append(current_chunk)
                current_chunk = part
                
        if current_chunk:
            compiled_parts.append(current_chunk)
            
        return compiled_parts

    raw_chunks = split_recursive(text, 0)
    
    overlapped_chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            overlapped_chunks.append(chunk)
            continue
            
        prev_chunk = raw_chunks[i - 1]
        overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
        overlapped_chunks.append(overlap_text + " " + chunk)
        
    return overlapped_chunks

def extract_kb_from_text(text: str) -> dict:
    """
    Calls Groq API to extract entities and relationships in JSON format.
    """
    if not settings.GROQ_API_KEY:
        logger.warning("Groq API Key missing. Skipping entity extraction.")
        return {"entities": [], "relationships": []}

    prompt = f"""
    You are an AI assistant specialized in information extraction for Knowledge Graphs.
    Given the following text chunk, extract key entities and relationships between them.
    
    Guidelines:
    1. Identify key entities: Person, Organization, Location, Project, Product, Skill, Technology, Disease, Symptom, MedicalProcedure, Document, etc.
    2. Extract semantic relationships between these entities (e.g. WORKS_ON, BUILT_BY, DEVELOPED, RELATED_TO, HAS_SYMPTOM, DIAGNOSED_WITH, IS_A).
    3. Provide a concise, clear description for each entity if it's mentioned.
    4. Keep entity names normalized (e.g., use "MRI Scan" instead of "the scan", "John Doe" instead of "Mr. Doe").
    
    Text Chunk:
    ---
    {text}
    ---
    
    Format the output strictly as a JSON object matching this schema:
    {{
        "entities": [
            {{
                "name": "Entity Name",
                "type": "Person|Organization|Project|Technology|etc.",
                "description": "Short description of the entity based on text"
            }}
        ],
        "relationships": [
            {{
                "source": "Entity Name (Must match an entity name in entities list)",
                "target": "Entity Name (Must match an entity name in entities list)",
                "relationship": "WORKS_ON|BUILT_BY|RELATED_TO|etc."
            }}
        ]
    }}
    """
    
    try:
        client = Groq(api_key=settings.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise data extractor that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Failed to extract KB entities from chunk using Groq: {e}")
        return {"entities": [], "relationships": []}

def ingest_pdf(file_path: str, doc_name: str, vector_store: VectorStoreManager, neo4j_client: Neo4jClient) -> str:
    """
    Full pipeline to ingest a PDF document.
    """
    doc_id = str(uuid.uuid4())
    logger.info(f"Starting ingestion for document '{doc_name}' with ID: {doc_id}")
    
    pages_data = extract_text_from_pdf(file_path)
    if not pages_data:
        logger.error(f"No content extracted from {file_path}. Ingestion failed.")
        return ""
        
    if neo4j_client.is_connected:
        neo4j_client.create_document_node(doc_id, doc_name, "PDF")
        
    all_chunks = []
    chunk_index = 0
    
    for page in pages_data:
        page_num = page["page"]
        page_text = page["text"]
        
        page_chunks = recursive_chunk_text(page_text, chunk_size=800, overlap=100)
        
        for p_chunk in page_chunks:
            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            all_chunks.append({
                "id": chunk_id,
                "text": p_chunk,
                "metadata": {
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "chunk_index": chunk_index,
                    "page": page_num,
                    "type": "text"
                }
            })
            chunk_index += 1
            
    if not all_chunks:
        logger.warning(f"No chunks created from PDF: {file_path}")
        return ""
        
    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [c["id"] for c in all_chunks]
    
    vector_store.add_chunks(texts, metadatas, ids)
    
    for chunk in all_chunks:
        c_id = chunk["id"]
        c_text = chunk["text"]
        c_meta = chunk["metadata"]
        
        if neo4j_client.is_connected:
            neo4j_client.create_chunk_node(c_id, c_text, c_meta["chunk_index"], doc_id)
            
            logger.info(f"Extracting KG elements from chunk {c_meta['chunk_index']}...")
            kb_data = extract_kb_from_text(c_text)
            
            entities_map = {}
            for ent in kb_data.get("entities", []):
                ent_name = ent.get("name")
                ent_type = ent.get("type", "General")
                ent_desc = ent.get("description", "")
                
                if ent_name:
                    neo4j_client.create_entity_node(ent_name, ent_type, ent_desc)
                    neo4j_client.link_chunk_to_entity(c_id, ent_name)
                    entities_map[ent_name.lower()] = ent_name
                    
            for rel in kb_data.get("relationships", []):
                src = rel.get("source")
                tgt = rel.get("target")
                rel_type = rel.get("relationship", "RELATED_TO")
                
                if src and tgt:
                    src_resolved = entities_map.get(src.lower(), src)
                    tgt_resolved = entities_map.get(tgt.lower(), tgt)
                    
                    neo4j_client.create_entity_node(src_resolved, "General")
                    neo4j_client.create_entity_node(tgt_resolved, "General")
                    
                    neo4j_client.create_custom_relationship(src_resolved, tgt_resolved, rel_type)
                    
    logger.info(f"Successfully finished ingesting document '{doc_name}'. Total chunks: {chunk_index}")
    return doc_id
