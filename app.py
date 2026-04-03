import streamlit as st
import os
from src.generation.rag_engine import FinancialRAG
from src.ingestion.pdf_parser import FinancialDocumentParser
from src.storage.sqlite_store import SQLiteTableStore
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page Config
st.set_page_config(page_title="Financial RAG System", page_icon="📈", layout="wide")

st.markdown("""
<style>
    /* Premium Metric Box Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #1C83E1;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.05);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 10px 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
    
    # Page limit selector
    limit_choice = st.radio("Page Selection:", ["First 50 Pages (Recommended)", "All Pages", "Custom Limit"])
    max_pages = None
    if limit_choice == "First 50 Pages (Recommended)":
        max_pages = 50
    elif limit_choice == "Custom Limit":
        max_pages = st.number_input("Enter number of pages to process:", min_value=1, value=50, step=1)
    
    if st.button("Process Document", type="primary"):
        if uploaded_file is not None:
            # 1. Save uploaded file
            os.makedirs("data/raw", exist_ok=True)
            file_path = f"data/raw/{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            import fitz
            pdf_doc = fitz.open(file_path)
            total_pages = len(pdf_doc)
            
            pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
            
            st.markdown("### Processing Progress")
            progress_bar = st.progress(0.0)
            progress_status = st.empty()
            
            def update_progress(phase, current, total):
                if phase == "text":
                    overall_progress = (current / total) * 0.2
                    progress_status.info(f"📄 Extracting text: Page {current}/{total}...")
                elif phase == "tables":
                    overall_progress = 0.2 + (current / total) * 0.7
                    progress_status.info(f"📊 Extracting tables (slow): Page {current}/{total}...")
                else:
                    overall_progress = min(current / total, 1.0)
                progress_bar.progress(min(max(overall_progress, 0.0), 1.0))

            try:
                # 2. Ingestion (Text & Tables)
                progress_status.info("Initializing parser...")
                parser = FinancialDocumentParser(file_path)
                doc_map = parser.process_document(max_pages=max_pages, progress_callback=update_progress)
                
                text_content = "\n".join([pdf_doc.load_page(i).get_text("text") for i in range(pages_to_process)])
                pdf_doc.close()
                
                # 3. Load tables to SQLite
                progress_bar.progress(0.92)
                progress_status.info("🗄️ Building SQL database for tables...")
                sqlite_store = SQLiteTableStore("data/processed/financial_tables.db")
                doc_name = parser.doc_name
                tables_extracted = 0
                for page in doc_map.get("pages", []):
                    for table in page.get("tables", []):
                        csv_path = table.get("path")
                        table_id = table.get("table_id")
                        page_num = table.get("page_number")
                        table_name = f"{doc_name}_p{page_num}_{table_id}"
                        
                        if os.path.exists(csv_path):
                            sqlite_store.load_csv_to_table(csv_path, table_name)
                            tables_extracted += 1
                
                # 4. Insert text to Qdrant
                progress_bar.progress(0.96)
                progress_status.info("🧠 Building Vector database...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
                chunks = text_splitter.split_text(text_content)
                vectors_created = len(chunks)
                
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
                
                progress_bar.progress(1.0)
                progress_status.success("✅ Document processed successfully!")
                
                st.write("### 📊 Ingestion Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Pages", pages_to_process)
                col2.metric("Tables Extracted", tables_extracted)
                col3.metric("Text Vectors", vectors_created)
                
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
        if "context" in message and message["role"] == "assistant":
            with st.expander("🔍 View Sources & Verification"):
                st.code(message["context"], language="text")

if prompt := st.chat_input("E.g., What was the total revenue in Q3?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking using {model_choice}..."):
            # Pass the model choice to the engine
            response_generator, context = rag.answer_question(prompt, model_choice=model_choice)
            
        # Stream the typing effect to the UI
        response = st.write_stream(response_generator)
        
        # Display the source context
        with st.expander("🔍 View Sources & Verification"):
            st.code(context, language="text")
            
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "context": context
        })