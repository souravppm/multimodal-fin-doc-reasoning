<div align="center">
  <h1>📈 Financial Document Reasoning Engine (RAG)</h1>
  <p><i>An enterprise-grade, hybrid Retrieval-Augmented Generation (RAG) system for complex financial analysis.</i></p>

  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)]()
  [![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker&logoColor=white)]()
  [![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)]()
  [![LLM](https://img.shields.io/badge/LLM-Llama_3.2_%7C_GPT--4o--mini-brightgreen)]()

</div>

---

## 🚀 Overview

The **Financial Document Reasoning Engine** is a specialized RAG system built to extract, process, and accurately answer questions over complex financial documents such as Annual Reports (10-K), Quarterly Earnings (10-Q), and dense financial statements. It intelligently breaks down documents, isolating raw text from complex quantitative tables, to provide precise reasoning.

This system leverages **Intelligent Query Routing** to decide whether a user's question requires a classic semantic search (via Vector DB) or precise numerical aggregation (via Text-to-SQL on structured tables). 

*Note: Vision-based processing has been deprecated in favor of a highly optimized Text/Table hybrid paradigm ensuring robust retrieval stability.*

## 📸 Screenshots

<div align="center">
  <img src="Screenshot/Screenshot%202026-04-03%20152546.png" width="800" alt="UI Screenshot 1"/><br/><br/>
  <img src="Screenshot/Screenshot%202026-04-03%20152626.png" width="800" alt="UI Screenshot 2"/><br/><br/>
  <img src="Screenshot/Screenshot%202026-04-03%20153459.png" width="800" alt="UI Screenshot 3"/><br/><br/>
  <img src="Screenshot/Screenshot%202026-04-03%20153614.png" width="800" alt="UI Screenshot 4"/>
</div>

## ✨ Key Features

- **Hybrid Routing Pipeline:** Automatically routes queries to the **Text Engine** (semantic search) or the **Table Engine** (Text-to-SQL) based on the intent of the prompt.
- **Smart Fallback Mechanism:** Gracefully falls back to Vector Search if a table query fails, ensuring an answer is always attempted.
- **Advanced Ingestion:** Extracts text cleanly using PyMuPDF and parses borderless/complex financial tables accurately using Camelot (`stream` flavor).
- **Streaming UI with ETA:** ChatGPT-style real-time streaming text output, complete with accurate ingestion ETA and page-selection controls.
- **Dual LLM Support:** Run completely locally for maximum privacy using **Ollama (Llama 3.2)**, or leverage the cloud for ultra-high reasoning using **OpenAI (GPT-4o-mini)**.
- **Zero-Setup Containerization:** Fully Dockerized with seamless volume mappings to preserve vector and relational databases across restarts.

## 🏗️ System Architecture

1. **Ingestion & Parsing (`src/ingestion/`)**
   - Documents are ingested with page-range precision. 
   - Texts are chunked and embedded using Sentence Transformers (`all-MiniLM-L6-v2`).
   - Tables are extracted, cleaned, normalized, and stored as relational data.
2. **Storage (`src/storage/`)**
   - **Qdrant:** Pure-Python vector store for text embeddings.
   - **SQLite:** Relational store dynamically generated for tabular financial data.
3. **Query Routing (`src/retrieval/router.py`)**
   - An LLM-based router classifies the incoming query.
4. **Retrieval & Generation (`src/generation/rag_engine.py`)**
   - **Text Route:** Performs semantic search on Qdrant.
   - **Table Route:** Dynamically writes an SQL query, executes it, and passes the tabular slice.
   - The final context is synthesized, and a grounded answer is streamed to the UI.

## 🛠️ Project Structure

```text
multimodal-fin-doc-reasoning/
├── app.py                     # Main Streamlit UI
├── docker-compose.yml         # Compose config for multi-container deployment
├── Dockerfile                 # Docker image building instructions
├── requirements.txt           # Python dependencies
├── auto_test_*.py/test_*.py   # Unit and integration tests
├── populate_qdrant.py         # Utility script to populate the vector store
└── src/                       # Core Application Source Code
    ├── generation/            # RAG synthesis engine and streaming handling
    ├── ingestion/             # PDF parser, chunking, and table extraction
    ├── retrieval/             # Query routing and orchestration
    └── storage/               # Qdrant Vector DB & SQLite Table integrations
```

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- Git
- Docker & Docker Compose (Optional, for containerized run)

### Local Setup (Recommended for Development)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/souravppm/multimodal-fin-doc-reasoning
   cd multimodal-fin-doc-reasoning
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate       # Unix/Mac
   .\venv\Scripts\activate        # Windows
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment Configuration:**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=sk-your_openai_api_key
   OLLAMA_BASE_URL=http://localhost:11434/v1
   ```
5. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

### Docker Deployment
The easiest way to run the entire stack effortlessly.

```bash
docker-compose up --build
```
Access the UI via your browser at [http://localhost:8501](http://localhost:8501).

## 🧪 Testing and Validation

Comprehensive test scripts are provided to ensure the stability of the reasoning engine:
- `python test_rag.py` - End-to-end RAG reasoning tests.
- `python test_router.py` - Evaluates the accuracy of the LLM query router.
- `python test_ingestion.py` - Checks text and table extraction accuracy.
- `python populate_qdrant.py` - Development script for quickly loading sample data.

## 🛡️ Privacy & Security
If running in **Local Mode** with Ollama, no data leaves your machine. Your documents are processed, embedded, and reasoned over entirely locally. `qdrant_storage/`, `data/`, and `.env` files are correctly excluded from version control to prevent accidental data leaks.

## 🔮 Future Roadmap & Vision
We are actively planning to push the boundaries of multimodal logic. Upcoming features include:
- **Vision-Language Integration (Image & Graph Mining):** Extending the Router to parse, understand, and reason over visual financial data such as analytical charts, bar graphs, and infographic timelines using advanced Vision models like GPT-4o.
- **Dynamic Chart Generation in UI:** Extracting numerical data via Text-to-SQL to render beautiful, interactive graphs (via Plotly/ECharts) right inside the Streamlit view instead of just raw text.
- **Citation & Source Highlighting:** Adding the capability for the UI to display bounding boxes over the exact snippet and page of the PDF that backs up the financial claim.
- **Advanced Agentic Workflows:** Moving from a standard RAG to an autonomous Financial Agent capable of compiling full peer-comparison reports across multiple quarters.

---
<div align="center">
  <i>Built for high-stakes financial analysis.</i>
</div>