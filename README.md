# Multimodal Financial Document Reasoning System

## 📌 Project Overview
An enterprise-grade Retrieval-Augmented Generation (RAG) system designed to extract, process, and reason over complex financial documents (e.g., Annual Reports). Unlike standard text-only RAG pipelines, this system leverages a **modality-aware architecture** to process plain text, extract structured data from tables, and interpret visual trends from charts.

## 🚀 Key Features
- **Multimodal Ingestion:** Extracts text via PyMuPDF, tabular data via Camelot, and charts/images.
- **Intelligent Query Routing:** Classifies user queries (Text-based, Numerical, or Trend-based) and routes them to the appropriate storage layer.
- **Hybrid Storage:** Uses **Qdrant** for semantic text search and **PostgreSQL/SQLite** for structured table querying.
- **Accurate Grounding:** Reduces hallucination significantly on numerical and financial data compared to baseline text-only RAG.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Document Processing:** PyMuPDF, Camelot, Unstructured
- **Storage:** Qdrant (Vector DB), SQLite/PostgreSQL (Structured DB)
- **AI/LLM:** GPT-4o / Open-source Vision Models (via HuggingFace/Ollama)

*(More details on setup, architecture, and evaluation metrics will be added as the project progresses)*
