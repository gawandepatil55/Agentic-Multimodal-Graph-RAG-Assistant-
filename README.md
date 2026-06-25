Agentic Multimodal Graph-RAG Assistant

## Overview

Agentic Multimodal Graph-RAG Assistant is an AI-powered retrieval and reasoning system that combines Knowledge Graph Retrieval (Neo4j), Vector Search (ChromaDB), and Large Language Models (Groq LLM) to answer complex user queries from documents and web content.

The system extracts entities and relationships from unstructured data, constructs a knowledge graph, generates semantic embeddings, and retrieves relevant context through both graph traversal and vector similarity search to provide accurate, context-aware responses.

---

## Key Features

* Hybrid Retrieval using Neo4j and ChromaDB
* Knowledge Graph Construction from documents
* Semantic Search using Sentence Transformers
* Graph-based reasoning and context enrichment
* Context-aware answer generation using Groq LLM
* PDF and Web Content Ingestion
* Interactive Streamlit Interface
* Agentic Workflow for intelligent retrieval and response generation

---

## Architecture

```text
                         User Query
                              │
                              ▼
                     Agentic Retrieval Layer
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
      Neo4j Knowledge Graph         ChromaDB Vector Store
               │                             │
       Graph Traversal              Similarity Search
               │                             │
               └──────────────┬──────────────┘
                              │
                              ▼
                    Context Aggregation
                              │
                              ▼
                         Groq LLM
                              │
                              ▼
                         Final Answer
```

---

## Technology Stack

### Programming Language

* Python

### AI & NLP

* Groq LLM
* Sentence Transformers

### Retrieval Components

* Neo4j
* ChromaDB

### Frameworks

* Streamlit
* LangChain

### Data Processing

* PDF Processing
* Text Chunking
* Entity Extraction

---

## Project Workflow

1. User uploads documents or provides web content.
2. Text is extracted and divided into manageable chunks.
3. Sentence Transformers generate vector embeddings.
4. Embeddings are stored in ChromaDB for semantic retrieval.
5. Entities and relationships are extracted to construct a Neo4j knowledge graph.
6. User submits a query.
7. Relevant context is retrieved through:

   * Vector similarity search (ChromaDB)
   * Knowledge graph traversal (Neo4j)
8. Retrieved context is aggregated.
9. Groq LLM generates a context-aware response.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/gawandepatil55/Agentic-Multimodal-Graph-RAG-Assistant-.git
cd Agentic-Multimodal-Graph-RAG-Assistant-
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Repository Structure

```text
Agentic-Multimodal-Graph-RAG-Assistant/
│
├── app/
├── data/
├── chroma_db/
├── graph/
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Future Enhancements

* Multi-document knowledge graph integration for cross-source reasoning
* Advanced multimodal support for image and document understanding
* Interactive graph visualization and analytics dashboard
* Persistent conversation memory for personalized interactions
* Explainable AI with source attribution and citation tracking
* Scalable cloud deployment with containerized architecture
* Multi-agent orchestration for complex task planning and execution


---

## Author

Ujwal Gawande

GitHub: https://github.com/gawandepatil55

...
