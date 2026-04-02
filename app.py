import streamlit as st
import os
from src.generation.rag_engine import FinancialRAG
from src.ingestion.pdf_parser import FinancialDocumentParser
from src.storage.sqlite_store import SQLiteTableStore
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page Config
st.set_page_config(page_title="Financial RAG System", page_icon="📈", layout="wide")

# Initialize Engine
@st.cache_resource
def get_rag_engine():
    return FinancialRAG()

rag = get_rag_engine()

# Sidebar
with st.sidebar:
    st.title("⚙️ System Configuration")
    
    # Model Selector
    st.subheader("🤖 AI Engine")
    model_choice = st.radio(
        "Select Model:",
        ("Ollama (Local)", "GPT-4o-mini (Cloud)"),
        help="Choose between free local processing or highly accurate cloud processing."
    )
    
    st.divider()
    
    # Document Upload
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a Financial PDF", type="pdf")
    
    if st.button("Process Document", type="primary"):
        if uploaded_file is not None:
            with st.spinner("Processing document... This may take a minute."):
                try:
                    # 1. Save uploaded file
                    os.makedirs("data/raw", exist_ok=True)
                    file_path = f"data/raw/{uploaded_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # 2. Ingestion (Text & Tables)
                    st.info("Extracting tables and text...")
                    parser = FinancialDocumentParser(file_path)
                    doc_map = parser.process_document()
                    
                    import fitz
                    pdf_doc = fitz.open(file_path)
                    text_content = "\n".join([page.get_text("text") for page in pdf_doc])
                    
                    # 3. Load tables to SQLite
                    st.info("Building SQL database...")
                    sqlite_store = SQLiteTableStore("data/processed/financial_tables.db")
                    doc_name = parser.doc_name
                    for page in doc_map.get("pages", []):
                        for table in page.get("tables", []):
                            csv_path = table.get("path")
                            table_id = table.get("table_id")
                            page_num = table.get("page_number")
                            table_name = f"{doc_name}_p{page_num}_{table_id}"
                            
                            if os.path.exists(csv_path):
                                sqlite_store.load_csv_to_table(csv_path, table_name)
                    
                    # 4. Insert text to Qdrant
                    st.info("Building Vector database...")
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
                    chunks = text_splitter.split_text(text_content)
                    
                    from qdrant_client.http import models as qmodels
                    # Ensure collection exists and upsert
                    rag.qc.recreate_collection(
                        collection_name=rag.collection_name,
                        vectors_config=qmodels.VectorParams(size=rag.embedder.get_sentence_embedding_dimension(), distance=qmodels.Distance.COSINE)
                    )
                    
                    import uuid
                    embeddings = rag.embedder.encode(chunks).tolist()
                    points = [
                        qmodels.PointStruct(id=str(uuid.uuid4()), vector=emb, payload={"text": chunk})
                        for chunk, emb in zip(chunks, embeddings)
                    ]
                    rag.qc.upsert(collection_name=rag.collection_name, points=points)
                    
                    st.success("✅ Document processed successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {e}")
        else:
            st.warning("Please upload a PDF first.")
            
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main Chat UI
st.title("📈 Multimodal Financial Document Assistant")
st.caption("Ask questions about revenue, tables, or text from the uploaded report.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("E.g., What was the total revenue in Q3?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking using {model_choice}..."):
            # Pass the model choice to the engine
            response = rag.answer_question(prompt, model_choice=model_choice)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})