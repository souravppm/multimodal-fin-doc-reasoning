# 📈 Multimodal Financial RAG System

An enterprise-grade, hybrid Retrieval-Augmented Generation (RAG) system designed to extract, process, and reason over complex financial documents (e.g., Annual Reports, Q4 Updates). This system intelligently routes queries between **Vector Search** (for text) and **Text-to-SQL** (for numerical tables).

---

## 🚀 Key Features

- **Hybrid LLM Engine:** Choose between **100% Local (Ollama + Llama 3.2)** for privacy or **Cloud (GPT-4o-mini)** for high-precision reasoning.
- **Multimodal Parsing:** Extracts text via PyMuPDF and processes borderless financial tables via Camelot (stream flavor).
- **Intelligent Query Routing:** Automatically detects if a question needs a semantic text answer or a precise numerical calculation from a table.
- **Smart Fallback:** If a table query fails (common in messy PDFs), the system automatically falls back to Vector Search to find the answer.
- **GPU Accelerated:** Optimized for NVIDIA GPUs for lightning-fast embedding generation.
- **Dockerized:** One-command deployment using Docker Compose.

---

## 🛠️ Tech Stack

- **Framework:** Streamlit (UI)
- **Document Processing:** PyMuPDF, Camelot (with Ghostscript)
- **Vector DB:** Qdrant (Local Storage)
- **Relational DB:** SQLite (for dynamic SQL execution)
- **AI Models:** Llama 3.2 (via Ollama) / GPT-4o-mini (OpenAI)
- **Embeddings:** `all-MiniLM-L6-v2` (Sentence-Transformers)

---

## 🧠 System Architecture

1. **Ingestion:** Raw PDFs are parsed. Text is chunked and embedded; tables are cleaned and stored in SQLite.
2. **Routing:** LLM analyzes the query to determine the best retrieval path (Text vs. Table).
3. **Retrieval:**
   - *Table Route:* Generates a strict SQL query, executes it on SQLite, and retrieves data.
   - *Text Route:* Performs semantic search in Qdrant.
4. **Generation:** Context is passed to the final LLM for a grounded, accurate response.

---

## ⚙️ How to Run

### Method 1: Local Setup (Recommended for GPU)
1. **Clone & Setup:**
   ```bash
   git clone <your-repo-link>
   cd multimodal-fin-doc-reasoning
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Environment Variables:**
   Create a `.env` file from the provided template:
   ```env
   OPENAI_API_KEY=your_key_here
   OLLAMA_BASE_URL=http://localhost:11434/v1
   ```
3. **Launch:**
   ```bash
   streamlit run app.py
   ```

### Method 2: Docker Deployment
Ensure Docker Desktop is running, then execute:
```bash
docker-compose up --build
```
The app will be available at `http://localhost:8501`.

---

## 📋 Note on Data Privacy
This system is designed to work fully offline using Ollama. To ensure privacy, the `.env` file and `data/` folders are excluded from version control.

---
*Built for production-level financial analysis and complex document reasoning.*