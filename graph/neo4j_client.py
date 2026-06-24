from neo4j import GraphDatabase, exceptions
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class Neo4jClient:
    def __init__(self):
        self.uri = settings.NEO4J_URI
        self.username = settings.NEO4J_USERNAME
        self.password = settings.NEO4J_PASSWORD
        self._driver = None
        self.is_connected = False
        
        self.connect()
        if self.is_connected:
            self.setup_constraints()

    def connect(self):
        """Attempts to establish connection to the Neo4j instance."""
        if not self.password:
            logger.warning("NEO4J_PASSWORD is empty. Neo4j client connection skipped.")
            self.is_connected = False
            return
            
        try:
            logger.info(f"Connecting to Neo4j instance at {self.uri} as '{self.username}'...")
            self._driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password),
                connection_timeout=5.0  # short timeout to prevent UI freezes
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            self.is_connected = True
            logger.info("Successfully connected to Neo4j database.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {e}")
            self.is_connected = False
            self._driver = None

    def close(self):
        if self._driver:
            self._driver.close()
            logger.info("Neo4j driver connection closed.")

    def run_query(self, query: str, parameters: dict = None) -> list[dict]:
        """
        Executes a Cypher query and returns the results.
        Fails gracefully if the database is not connected.
        """
        if not self.is_connected or not self._driver:
            logger.warning(f"Neo4j is not connected. Skipping query: {query}")
            return []
            
        parameters = parameters or {}
        try:
            with self._driver.session() as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error running Neo4j query: {query} \nError: {e}")
            return []

    def setup_constraints(self):
        """Creates unique constraints to prevent duplicates."""
        constraints = [
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT image_id_unique IF NOT EXISTS FOR (i:Image) REQUIRE i.id IS UNIQUE"
        ]
        logger.info("Configuring Neo4j database constraints...")
        for constraint in constraints:
            self.run_query(constraint)
        logger.info("Database constraints set up.")

    def create_document_node(self, doc_id: str, name: str, doc_type: str):
        query = """
        MERGE (d:Document {id: $doc_id})
        ON CREATE SET d.name = $name, d.type = $doc_type, d.created_at = timestamp()
        ON MATCH SET d.name = $name, d.type = $doc_type
        RETURN d
        """
        self.run_query(query, {"doc_id": doc_id, "name": name, "doc_type": doc_type})

    def create_chunk_node(self, chunk_id: str, text: str, seq_num: int, doc_id: str):
        query = """
        MERGE (c:Chunk {id: $chunk_id})
        ON CREATE SET c.text = $text, c.seq_num = $seq_num
        ON MATCH SET c.text = $text
        WITH c
        MATCH (d:Document {id: $doc_id})
        MERGE (c)-[:PART_OF]->(d)
        RETURN c
        """
        self.run_query(query, {"chunk_id": chunk_id, "text": text, "seq_num": seq_num, "doc_id": doc_id})

    def create_entity_node(self, name: str, entity_type: str, description: str = ""):
        query = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.type = $entity_type, e.description = $description
        ON MATCH SET e.type = $entity_type, e.description = CASE WHEN $description <> "" THEN $description ELSE e.description END
        RETURN e
        """
        self.run_query(query, {"name": name, "entity_type": entity_type, "description": description})

    def create_image_node(self, image_id: str, name: str, description: str, doc_id: str = None):
        query = """
        MERGE (i:Image {id: $image_id})
        ON CREATE SET i.name = $name, i.description = $description, i.created_at = timestamp()
        ON MATCH SET i.description = $description
        RETURN i
        """
        self.run_query(query, {"image_id": image_id, "name": name, "description": description})
        
        if doc_id:
            link_query = """
            MATCH (i:Image {id: $image_id})
            MATCH (d:Document {id: $doc_id})
            MERGE (i)-[:RELATED_TO]->(d)
            """
            self.run_query(link_query, {"image_id": image_id, "doc_id": doc_id})

    def link_chunk_to_entity(self, chunk_id: str, entity_name: str):
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (e:Entity {name: $entity_name})
        MERGE (c)-[:MENTIONS]->(e)
        """
        self.run_query(query, {"chunk_id": chunk_id, "entity_name": entity_name})

    def link_image_to_entity(self, image_id: str, entity_name: str):
        query = """
        MATCH (i:Image {id: $image_id})
        MATCH (e:Entity {name: $entity_name})
        MERGE (i)-[:DEPICTS]->(e)
        """
        self.run_query(query, {"image_id": image_id, "entity_name": entity_name})

    def create_custom_relationship(self, source_name: str, target_name: str, rel_type: str):
        # Clean relationship type for Cypher safety (replace non-alphanumeric with underscores)
        safe_rel_type = "".join([c if c.isalnum() else "_" for c in rel_type]).upper()
        if not safe_rel_type:
            safe_rel_type = "RELATED_TO"
            
        # Parametrized relationships can't easily set relationship type dynamically,
        # so we format it safely in the query string itself.
        query = f"""
        MATCH (source:Entity {{name: $source_name}})
        MATCH (target:Entity {{name: $target_name}})
        MERGE (source)-[r:{safe_rel_type}]->(target)
        RETURN r
        """
        self.run_query(query, {"source_name": source_name, "target_name": target_name})

    def query_entity_neighborhood(self, entity_name: str, depth: int = 1) -> list[dict]:
        """Retrieves connections around a specific entity up to depth."""
        # A cypher query to get paths
        query = """
        MATCH path = (e:Entity {name: $entity_name})-[r*1..2]-(other)
        RETURN path LIMIT 25
        """
        return self.run_query(query, {"entity_name": entity_name})

    def search_graph_by_keyword(self, keyword: str) -> list[dict]:
        """Searches for entities whose name or description contains the keyword, and returns relationships."""
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($keyword) OR toLower(e.description) CONTAINS toLower($keyword)
        MATCH (e)-[r]-(other)
        RETURN e.name AS source, e.type AS source_type, type(r) AS relationship, other.name AS target, labels(other)[0] AS target_label
        LIMIT 30
        """
        return self.run_query(query, {"keyword": keyword})

    def get_graph_elements(self, limit: int = 100) -> dict:
        """Retrieves all nodes and edges for rendering in a visualization."""
        if not self.is_connected or not self._driver:
            return {"nodes": [], "edges": []}
            
        query_nodes = f"""
        MATCH (n)
        RETURN id(n) AS id, labels(n)[0] AS label, n.name AS name, n.title AS title, n.text AS text, n.description AS description, n.type AS type
        LIMIT {limit}
        """
        
        query_edges = f"""
        MATCH (n)-[r]->(m)
        RETURN id(n) AS source, id(m) AS target, type(r) AS type
        LIMIT {limit * 2}
        """
        
        nodes_res = self.run_query(query_nodes)
        edges_res = self.run_query(query_edges)
        
        nodes = []
        for n in nodes_res:
            name = n.get("name") or n.get("title") or n.get("text") or f"ID: {n['id']}"
            # Truncate text or descriptions if they're too long
            if len(name) > 30:
                name = name[:27] + "..."
            
            nodes.append({
                "id": str(n["id"]),
                "label": n["label"],
                "name": name,
                "type": n.get("type", "Unknown"),
                "full_data": {k: v for k, v in n.items() if v is not None}
            })
            
        edges = []
        for e in edges_res:
            edges.append({
                "source": str(e["source"]),
                "target": str(e["target"]),
                "type": e["type"]
            })
            
        return {"nodes": nodes, "edges": edges}

    def clear_all(self) -> bool:
        """Clears all nodes and relationships from the database."""
        try:
            self.run_query("MATCH (n) DETACH DELETE n")
            logger.info("Cleared all nodes and relationships from Neo4j.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear Neo4j: {e}")
            return False
