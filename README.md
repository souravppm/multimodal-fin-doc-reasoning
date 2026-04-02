# Multimodal Financial Document Reasoning System

## 📌 Project Overview
An enterprise-grade, fully local Retrieval-Augmented Generation (RAG) system designed to extract, process, and reason over complex financial documents (e.g., Annual Reports). Unlike standard text-only RAG pipelines, this system leverages a **modality-aware architecture** to process plain text and intelligently execute **Text-to-SQL** queries for structured tabular data.

## 🚀 Key Features
- **100% Local & Privacy-Preserving:** Runs entirely on local hardware using **Ollama (Llama 3.2)** and local embedding models. No external APIs or cloud costs.
- **Multimodal Ingestion:** Extracts text via PyMuPDF and tabular data via Camelot.
- **Intelligent Query Routing:** An LLM-powered router classifies user queries (Text-based, Numerical/Table) and directs them to the appropriate storage layer.
- **Hybrid Storage Engine:** - **Qdrant (Vector DB):** For semantic text search.
  - **SQLite (Relational DB):** For precise numerical queries on tables via autonomous Text-to-SQL generation.
- **Anti-Hallucination:** Strictly grounded responses. The system refuses to answer if the context is missing, ensuring high reliability for financial data.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Document Processing:** PyMuPDF (fitz), Camelot, pandas
- **Vector Storage:** Qdrant (Local), `sentence-transformers/all-MiniLM-L6-v2`
- **Structured Storage:** SQLite
- **LLM Engine:** Ollama (Llama 3.2)
- **Architecture:** Modality-routed RAG

## 🧠 System Architecture (Input → Processing → Output)
1. **Ingestion:** Raw PDFs are parsed. Text is chunked and embedded; tables are cleaned and converted into SQL tables.
2. **Routing:** User query is analyzed by the local LLM to determine if it requires semantic text search or precise numerical calculation.
3. **Retrieval:**
   - *Text Route:* Performs cosine similarity search in Qdrant.
   - *Table Route:* Generates a strict SQL query based on the SQLite schema, executes it, and retrieves the exact numerical values.
4. **Generation:** The context and query are passed to the final LLM to generate a human-readable, highly accurate response.

## ⚙️ How to Run Locally
1. Clone the repository and activate your virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure [Ollama](https://ollama.com/) is installed and the model is pulled: `ollama run llama3.2`
4. Run the Master Engine: `python test_rag.py`

---
*Built with a focus on production-level engineering, modularity, and zero-cost local execution.*