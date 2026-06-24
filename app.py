import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components
import json
from pathlib import Path

# Load settings and configure environments
from config import settings
from utils.logger import get_logger
from database.vector_store import VectorStoreManager
from graph.neo4j_client import Neo4jClient
from ingestion.pdf_processor import ingest_pdf
from ingestion.image_processor import ingest_image
from agents.workflow import run_rag_agent

logger = get_logger(__name__)

# Streamlit Page Configurations
st.set_page_config(
    page_title="Agentic Multimodal Graph-RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Custom cards for stats */
    .stat-card {
        background-color: #1e222b;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4c566a;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        text-align: center;
    }
    .stat-val {
        font-size: 32px;
        font-weight: 800;
        color: #88c0d0;
        margin-bottom: 5px;
    }
    .stat-lbl {
        font-size: 14px;
        color: #d8dee9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Agent logs style */
    .agent-step {
        background-color: #2e3440;
        border-left: 4px solid #81a1c1;
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 4px;
        font-family: monospace;
        font-size: 13px;
        color: #d8dee9;
    }
    
    /* Highlighted buttons */
    .stButton>button {
        background-color: #4c566a;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 20px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #5e81ac;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_response_state" not in st.session_state:
    st.session_state.last_response_state = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Chat Assistant"
if "db_stats" not in st.session_state:
    st.session_state.db_stats = {"chunks": 0, "images": 0, "entities": 0, "relations": 0}

# Helper to load and initialize managers safely
@st.cache_resource
def get_db_clients():
    try:
        v_store = VectorStoreManager()
    except Exception as e:
        logger.error(f"Failed to create VectorStoreManager: {e}")
        v_store = None
        
    try:
        n_client = Neo4jClient()
    except Exception as e:
        logger.error(f"Failed to create Neo4jClient: {e}")
        n_client = None
        
    return v_store, n_client

v_store, n_client = get_db_clients()

def update_db_stats():
    """Fetches counts of items in databases for the UI stats panel."""
    stats = {"chunks": 0, "images": 0, "entities": 0, "relations": 0}
    if v_store:
        try:
            stats["chunks"] = v_store.chunks_collection.count()
            stats["images"] = v_store.images_collection.count()
        except Exception:
            pass
            
    if n_client and n_client.is_connected:
        try:
            ent_count = n_client.run_query("MATCH (e:Entity) RETURN count(e) AS count")
            rel_count = n_client.run_query("MATCH ()-[r]->() RETURN count(r) AS count")
            stats["entities"] = ent_count[0]["count"] if ent_count else 0
            stats["relations"] = rel_count[0]["count"] if rel_count else 0
        except Exception:
            pass
            
    st.session_state.db_stats = stats

# Initial statistics fetch
update_db_stats()

# SIDEBAR CONFIGURATIONS
st.sidebar.title("🛠️ Configurations & Connection")

# Credentials Inputs
st.sidebar.subheader("API keys & Database Credentials")
api_key = st.sidebar.text_input("Groq API Key", value=settings.GROQ_API_KEY, type="password")
neo4j_uri = st.sidebar.text_input("Neo4j Bolt URI", value=settings.NEO4J_URI)
neo4j_user = st.sidebar.text_input("Neo4j Username", value=settings.NEO4J_USERNAME)
neo4j_pass = st.sidebar.text_input("Neo4j Password", value=settings.NEO4J_PASSWORD, type="password")

if st.sidebar.button("Save Configurations & Reconnect"):
    # Write variables back to .env file
    env_path = Path(__file__).resolve().parent / ".env"
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(f"GROQ_API_KEY={api_key}\n")
        f.write(f"NEO4J_URI={neo4j_uri}\n")
        f.write(f"NEO4J_USERNAME={neo4j_user}\n")
        f.write(f"NEO4J_PASSWORD={neo4j_pass}\n")
        f.write(f"CHROMA_PATH={settings.CHROMA_PATH}\n")
        f.write(f"EMBEDDING_PROVIDER={settings.EMBEDDING_PROVIDER}\n")
        
    st.sidebar.success("Configurations saved! Reconnecting...")
    # Clear cache to force reload
    get_db_clients.clear()
    st.rerun()

# Display Connection Statuses
st.sidebar.subheader("🔌 Connection Status")
if api_key:
    st.sidebar.markdown("🟢 **Groq API**: Connected")
else:
    st.sidebar.markdown("🔴 **Groq API**: API Key Missing")

if n_client and n_client.is_connected:
    st.sidebar.markdown("🟢 **Neo4j DB**: Connected")
else:
    st.sidebar.markdown("🔴 **Neo4j DB**: Disconnected (Bolt host down or credentials incorrect)")
    
if v_store:
    st.sidebar.markdown(f"🟢 **ChromaDB**: Connected ({settings.EMBEDDING_PROVIDER.upper()} embeddings)")
else:
    st.sidebar.markdown("🔴 **ChromaDB**: Initialization Failed")

# Database operations
st.sidebar.subheader("🧹 System Operations")
if st.sidebar.button("Flush Database (Clear Graph & Vector Data)", type="secondary"):
    flushed = False
    if v_store:
        v_store.clear_all()
        flushed = True
    if n_client and n_client.is_connected:
        n_client.clear_all()
        flushed = True
    if flushed:
        st.sidebar.success("Successfully deleted all databases content!")
        update_db_stats()
        st.rerun()

# ----------------- MAIN APP TABS -----------------
st.title("🤖 Agentic Multimodal Graph-RAG Assistant")
st.markdown("An advanced RAG framework combining semantic vectors, multi-hop knowledge graph queries, and visual indexing.")

# Dashboard Stats Summary Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="stat-card"><div class="stat-val">{st.session_state.db_stats["chunks"]}</div><div class="stat-lbl">Vector Chunks</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-card"><div class="stat-val">{st.session_state.db_stats["images"]}</div><div class="stat-lbl">Indexed Images</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="stat-card"><div class="stat-val">{st.session_state.db_stats["entities"]}</div><div class="stat-lbl">Graph Entities</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="stat-card"><div class="stat-val">{st.session_state.db_stats["relations"]}</div><div class="stat-lbl">Graph Relations</div></div>', unsafe_allow_html=True)

st.write("")

# Define Streamlit Tabs
tab_chat, tab_ingest, tab_viewer, tab_debug = st.tabs([
    "💬 Chat Assistant", 
    "📂 Ingestion Hub", 
    "📊 Knowledge Graph Viewer", 
    "⚙️ Retrieval Debugger"
])

# 1. CHAT ASSISTANT
with tab_chat:
    st.header("💬 Talk to your Graph-RAG Knowledge Base")
    
    # Input files option directly in chat
    chat_image_file = st.file_uploader("Attach Image for Multimodal Reasoning (Optional)", type=["png", "jpg", "jpeg"], key="chat_img")
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
            if chat.get("image"):
                st.image(chat["image"], width=250)
                
    # Chat Input
    user_query = st.chat_input("Ask a question about the documents or uploaded images...")
    
    if user_query:
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": user_query, "image": chat_image_file})
        with st.chat_message("user"):
            st.markdown(user_query)
            if chat_image_file:
                st.image(chat_image_file, width=250)
                
        # Save temp image if uploaded
        temp_img_path = None
        if chat_image_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(chat_image_file.getvalue())
                temp_img_path = tmp.name
                
        # Run agentic workflow
        with st.spinner("Agentic retrieval workflow active. Traversing Vector DB and Neo4j..."):
            try:
                final_state = run_rag_agent(user_query, current_image_path=temp_img_path)
                st.session_state.last_response_state = final_state
                
                # Extract response
                assistant_response = final_state.get("response", "No answer was generated.")
                
                # Append assistant response
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)
            except Exception as e:
                st.error(f"Failed to execute retrieval agent: {e}")
                
            # Clean up temp image
            if temp_img_path and os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                except Exception:
                    pass
        
        # Refresh database stats
        update_db_stats()
        st.rerun()

# 2. INGESTION HUB
with tab_ingest:
    st.header("📂 Ingestion Hub")
    st.markdown("Upload raw files to build embeddings, segment document text, and construct graph connections.")
    
    upload_col_pdf, upload_col_img = st.columns(2)
    
    with upload_col_pdf:
        st.subheader("Document Ingestion (PDFs/Text)")
        uploaded_pdf = st.file_uploader("Choose a PDF or TXT document", type=["pdf", "txt"], key="upload_pdf")
        
        if uploaded_pdf and st.button("Process Document"):
            if not v_store:
                st.error("ChromaDB is not initialized.")
            elif not n_client or not n_client.is_connected:
                st.warning("Neo4j database is not connected. Ingesting text blocks into vector store only.")
                
            with st.spinner("Extracting text, chunking, and compiling entities..."):
                # Save to a temporary location
                suffix = ".pdf" if uploaded_pdf.name.lower().endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_pdf.getvalue())
                    tmp_path = tmp_file.name
                    
                try:
                    doc_id = ingest_pdf(tmp_path, uploaded_pdf.name, v_store, n_client)
                    if doc_id:
                        st.success(f"Successfully processed document: {uploaded_pdf.name} (ID: {doc_id})")
                        update_db_stats()
                    else:
                        st.error(f"Failed to process document {uploaded_pdf.name}.")
                except Exception as e:
                    st.error(f"Error during document processing: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
    with upload_col_img:
        st.subheader("Image Ingestion")
        uploaded_image = st.file_uploader("Choose an Image file", type=["png", "jpg", "jpeg"], key="upload_img")
        
        if uploaded_image and st.button("Process Image"):
            if not v_store:
                st.error("ChromaDB is not initialized.")
                
            with st.spinner("Analyzing image via Groq Vision, compiling embeddings, and linking entity nodes..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(uploaded_image.getvalue())
                    tmp_path = tmp_file.name
                    
                try:
                    img_id = ingest_image(tmp_path, uploaded_image.name, v_store, n_client)
                    if img_id:
                        st.success(f"Successfully ingested image: {uploaded_image.name} (ID: {img_id})")
                        update_db_stats()
                    else:
                        st.error(f"Failed to ingest image {uploaded_image.name}.")
                except Exception as e:
                    st.error(f"Error during image ingestion: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

# 3. KNOWLEDGE GRAPH VIEWER
with tab_viewer:
    st.header("📊 Interactive Knowledge Graph Viewer")
    st.markdown("Visualizes current document mappings, text chunks, and LLM-extracted entity nodes/relationships in Neo4j.")
    
    if not n_client or not n_client.is_connected:
        st.info("Please connect to a Neo4j database instance to view graph structures.")
    else:
        # Load graph elements
        with st.spinner("Fetching nodes and edges..."):
            graph_data = n_client.get_graph_elements(limit=80)
            
        if not graph_data["nodes"]:
            st.warning("The knowledge graph is currently empty. Ingest documents or images to build the graph schema.")
        else:
            # Renders a custom d3.js network visualization in iframe
            graph_data_json = json.dumps(graph_data)
            
            d3_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://d3js.org/d3.v7.min.js"></script>
                <style>
                    body {{ margin: 0; background-color: #151921; color: #fafafa; font-family: sans-serif; overflow: hidden; }}
                    .node {{ stroke: #232731; stroke-width: 2px; cursor: pointer; }}
                    .link {{ stroke: #7e8c9f; stroke-opacity: 0.5; stroke-width: 1.5px; }}
                    .label {{ font-size: 11px; fill: #eceff4; pointer-events: none; text-anchor: middle; font-weight: bold; }}
                    #tooltip {{ 
                        position: absolute; 
                        background: rgba(26, 30, 37, 0.95); 
                        padding: 10px; 
                        border: 1px solid #434c5e; 
                        border-radius: 8px; 
                        pointer-events: none; 
                        font-size: 12px; 
                        display: none; 
                        max-width: 280px; 
                        color: #e5e9f0;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.4); 
                        z-index: 1000;
                    }}
                    .title {{ font-weight: bold; font-size: 13px; margin-bottom: 4px; color: #88c0d0; }}
                    .type {{ font-style: italic; color: #a3be8c; font-size: 10px; margin-bottom: 6px; text-transform: uppercase; }}
                </style>
            </head>
            <body>
                <div id="tooltip"></div>
                <svg width="100%" height="600" id="graph-svg"></svg>
                <script>
                    const data = {graph_data_json};
                    const svg = d3.select("#graph-svg");
                    const width = window.innerWidth || 800;
                    const height = 600;
                    
                    // Create simulation
                    const simulation = d3.forceSimulation(data.nodes)
                        .force("link", d3.forceLink(data.edges).id(d => d.id).distance(100))
                        .force("charge", d3.forceManyBody().strength(-200))
                        .force("center", d3.forceCenter(width / 2, height / 2))
                        .force("collision", d3.forceCollide().radius(35));
                        
                    // Drag functions
                    function dragstarted(event, d) {{
                        if (!event.active) simulation.alphaTarget(0.3).restart();
                        d.fx = d.x;
                        d.fy = d.y;
                    }}
                    function dragged(event, d) {{
                        d.fx = event.x;
                        d.fy = event.y;
                    }}
                    function dragended(event, d) {{
                        if (!event.active) simulation.alphaTarget(0);
                        d.fx = null;
                        d.fy = null;
                    }}
                    
                    // Coloring nodes by label/type
                    function getNodeColor(d) {{
                        if (d.label === "Document") return "#81a1c1"; // soft blue
                        if (d.label === "Chunk") return "#5e81ac"; // dark blue
                        if (d.label === "Image") return "#d08770"; // orange
                        // Entity Node
                        const type = (d.type || "").toLowerCase();
                        if (type.includes("person")) return "#b48ead"; // purple
                        if (type.includes("org")) return "#8fbcbb"; // cyan
                        if (type.includes("disease") || type.includes("symptom")) return "#bf616a"; // red
                        return "#a3be8c"; // green
                    }}
                    
                    // Draw edges
                    const link = svg.append("g")
                        .selectAll("line")
                        .data(data.edges)
                        .join("line")
                        .attr("class", "link");
                        
                    // Draw node circles
                    const node = svg.append("g")
                        .selectAll("circle")
                        .data(data.nodes)
                        .join("circle")
                        .attr("class", "node")
                        .attr("r", d => d.label === "Document" ? 22 : (d.label === "Chunk" ? 12 : 16))
                        .attr("fill", getNodeColor)
                        .call(d3.drag()
                            .on("start", dragstarted)
                            .on("drag", dragged)
                            .on("end", dragended));
                            
                    // Draw text labels
                    const labels = svg.append("g")
                        .selectAll("text")
                        .data(data.nodes)
                        .join("text")
                        .attr("class", "label")
                        .text(d => d.name);
                        
                    // Tooltip dynamics
                    const tooltip = d3.select("#tooltip");
                    node.on("mouseover", function(event, d) {{
                        let content = `<div class='title'>${{d.name}}</div>`;
                        content += `<div class='type'>Label: ${{d.label}}</div>`;
                        
                        if (d.full_data) {{
                            if (d.full_data.type) content += `<div><strong>Type:</strong> ${{d.full_data.type}}</div>`;
                            if (d.full_data.description) content += `<div><strong>Desc:</strong> ${{d.full_data.description}}</div>`;
                            if (d.full_data.text) content += `<div><strong>Text:</strong> ${{d.full_data.text.substring(0, 100)}}...</div>`;
                        }}
                        
                        tooltip.html(content)
                            .style("left", (event.pageX + 15) + "px")
                            .style("top", (event.pageY - 15) + "px")
                            .style("display", "block");
                    }})
                    .on("mousemove", function(event) {{
                        tooltip.style("left", (event.pageX + 15) + "px")
                               .style("top", (event.pageY - 15) + "px");
                    }})
                    .on("mouseout", function() {{
                        tooltip.style("display", "none");
                    }});
                    
                    simulation.on("tick", () => {{
                        link
                            .attr("x1", d => d.source.x)
                            .attr("y1", d => d.source.y)
                            .attr("x2", d => d.target.x)
                            .attr("y2", d => d.target.y);
                            
                        node
                            .attr("cx", d => d.x)
                            .attr("cy", d => d.y);
                            
                        labels
                            .attr("x", d => d.x)
                            .attr("y", d => d.y - 25);
                    }});
                </script>
            </body>
            </html>
            """
            
            # Embed HTML
            components.html(d3_html, height=600)
            st.caption("💡 Drag nodes to interact. Hover over nodes to inspect properties.")

# 4. RETRIEVAL DEBUGER
with tab_debug:
    st.header("⚙️ Agent Retrieval Debugger")
    st.markdown("Inspect decisions, raw retrieval elements, and intermediate logs from the last conversation turn.")
    
    state = st.session_state.last_response_state
    
    if not state:
        st.info("Run a query in the Chat Assistant tab to display debugging steps.")
    else:
        # Display strategy decision
        st.subheader("🚦 Agent Decision Analysis")
        st.markdown(f"**Selected Strategy**: `{state.get('search_mode', '').upper()}`")
        st.markdown(f"**Extracted Graph Keywords**: `{state.get('keywords', [])}`")
        st.markdown(f"**Refined Vector Query**: *\"{state.get('refined_query', '')}\"*")
        
        # Display agentic reasoning steps
        st.subheader("📜 Execution Log")
        for step in state.get("agent_steps", []):
            st.markdown(f'<div class="agent-step">{step}</div>', unsafe_allow_html=True)
            
        # Display retrieved vectors
        st.subheader("📚 Retrieved Vector Chunks (ChromaDB)")
        v_res = state.get("vector_results", [])
        if not v_res:
            st.write("No vector chunks retrieved.")
        else:
            for item in v_res:
                meta = item.get("metadata", {})
                st.markdown(f"**Source**: {meta.get('document_name')} | Page: {meta.get('page')} | Similarity Score: **{item.get('similarity', 0.0):.4f}**")
                st.code(item.get("text", ""))
                
        # Display retrieved images
        st.subheader("🖼️ Retrieved Image Metadata")
        i_res = state.get("image_results", [])
        if not i_res:
            st.write("No matching images retrieved.")
        else:
            for item in i_res:
                meta = item.get("metadata", {})
                st.markdown(f"**Image**: `{item.get('image_name')}` | Similarity Score: **{item.get('similarity', 0.0):.4f}**")
                st.markdown(f"**File Path**: `{meta.get('file_path')}`")
                st.info(f"**Visual Caption**: {item.get('description')}")
                
        # Display Graph paths
        st.subheader("🕸️ Retrieved Graph Traversal Paths (Neo4j)")
        g_res = state.get("graph_results", [])
        if not g_res:
            st.write("No graph elements retrieved.")
        else:
            # Create a simple table
            st.table(g_res)
