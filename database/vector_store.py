import chromadb
from chromadb.config import Settings as ChromaSettings
from config import settings
from embeddings.embedder import get_embedder
from utils.logger import get_logger

logger = get_logger(__name__)

class VectorStoreManager:
    def __init__(self):
        self.chroma_path = settings.CHROMA_PATH
        logger.info(f"Initializing ChromaDB persistent client at: {self.chroma_path}")
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collections
        self.chunks_collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        self.images_collection = self.client.get_or_create_collection(
            name="images",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.embedder = get_embedder()

    def add_chunks(self, texts: list[str], metadatas: list[dict], ids: list[str]) -> bool:
        """
        Computes embeddings and adds text chunks to the database.
        """
        if not texts:
            return False
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embedder.embed_texts(texts)
            
            logger.info(f"Adding {len(texts)} chunks to ChromaDB...")
            self.chunks_collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info("Successfully added chunks to vector store.")
            return True
        except Exception as e:
            logger.error(f"Error adding chunks to ChromaDB: {e}")
            return False

    def add_image(self, image_id: str, image_name: str, description: str, metadata: dict = None) -> bool:
        """
        Embeds an image's description and index it in the image collection.
        """
        try:
            meta = {
                "image_id": image_id,
                "image_name": image_name,
                "type": "image",
                **(metadata or {})
            }
            
            logger.info(f"Generating embedding for image description: '{image_name}'...")
            embedding = self.embedder.embed_text(description)
            
            logger.info(f"Adding image '{image_name}' to ChromaDB...")
            self.images_collection.add(
                documents=[description],
                embeddings=[embedding],
                metadatas=[meta],
                ids=[image_id]
            )
            logger.info(f"Successfully added image '{image_name}' to vector store.")
            return True
        except Exception as e:
            logger.error(f"Error adding image to ChromaDB: {e}")
            return False

    def search_chunks(self, query: str, limit: int = 5) -> list[dict]:
        """
        Searches for semantically similar text chunks.
        """
        try:
            query_embedding = self.embedder.embed_text(query)
            results = self.chunks_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
            
            formatted_results = []
            if results and results["documents"]:
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                ids = results["ids"][0]
                distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
                
                for i in range(len(docs)):
                    # Cosine distance is returned; similarity = 1 - distance
                    similarity = 1.0 - distances[i] if distances[i] is not None else 0.0
                    formatted_results.append({
                        "id": ids[i],
                        "text": docs[i],
                        "metadata": metas[i],
                        "similarity": round(float(similarity), 4)
                    })
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching chunks in ChromaDB: {e}")
            return []

    def search_images(self, query: str, limit: int = 3) -> list[dict]:
        """
        Searches for images whose descriptions are semantically similar to the query.
        """
        try:
            query_embedding = self.embedder.embed_text(query)
            results = self.images_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
            
            formatted_results = []
            if results and results["documents"]:
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                ids = results["ids"][0]
                distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
                
                for i in range(len(docs)):
                    similarity = 1.0 - distances[i] if distances[i] is not None else 0.0
                    formatted_results.append({
                        "id": ids[i],
                        "description": docs[i],
                        "metadata": metas[i],
                        "similarity": round(float(similarity), 4)
                    })
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching images in ChromaDB: {e}")
            return []

    def delete_document(self, document_id: str) -> bool:
        """
        Deletes all chunks associated with a document_id.
        """
        try:
            self.chunks_collection.delete(where={"document_id": document_id})
            logger.info(f"Deleted chunks for document_id '{document_id}' from ChromaDB.")
            return True
        except Exception as e:
            logger.error(f"Error deleting document '{document_id}' from ChromaDB: {e}")
            return False

    def clear_all(self) -> bool:
        """
        Clears both collections.
        """
        try:
            self.client.delete_collection("document_chunks")
            self.client.delete_collection("images")
            self.chunks_collection = self.client.get_or_create_collection("document_chunks")
            self.images_collection = self.client.get_or_create_collection("images")
            logger.info("Cleared all collections from ChromaDB.")
            return True
        except Exception as e:
            logger.error(f"Error clearing ChromaDB collections: {e}")
            return False
