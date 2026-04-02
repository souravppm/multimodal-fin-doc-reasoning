import os
import logging
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.storage.qdrant_store import QdrantVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_pdf(file_path: str) -> str:
    """Reads all text from a given PDF file."""
    logger.info(f"Loading PDF file: {file_path}")
    doc = fitz.open(file_path)
    text_content = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_content.append(page.get_text("text"))
        
    logger.info(f"Extracted {len(text_content)} pages from PDF.")
    return "\n".join(text_content)

def main():
    logger.info("Starting populate_qdrant.py script...")
    
    raw_data_dir = os.path.join("data", "raw")
    if not os.path.exists(raw_data_dir):
        logger.error(f"Directory not found: {raw_data_dir}")
        return

    pdf_files = [f for f in os.listdir(raw_data_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.error(f"No PDF files found in {raw_data_dir}")
        return
        
    target_pdf = "sample_report.pdf"
    if target_pdf not in pdf_files:
        target_pdf = pdf_files[0]
        
    file_path = os.path.join(raw_data_dir, target_pdf)
    
    try:
        full_text = parse_pdf(file_path)
    except Exception as e:
        logger.error(f"Failed to read PDF file {file_path}: {e}")
        return
        
    logger.info("Initializing text splitter (chunk_size=700, chunk_overlap=100)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    
    chunks = text_splitter.split_text(full_text)
    logger.info(f"Text split into {len(chunks)} chunks.")
    
    metadata_list = []
    for i in range(len(chunks)):
        metadata_list.append({
            "source_file": target_pdf,
            "chunk_index": i
        })
        
    logger.info("Initializing QdrantVectorStore...")
    try:
        qdrant_store = QdrantVectorStore()
        collection_name = 'test_financial_reports'
        
        logger.info(f"Creating collection '{collection_name}' if it does not exist...")
        qdrant_store.create_collection(collection_name)
        
        logger.info(f"Upserting {len(chunks)} chunks into Qdrant...")
        qdrant_store.add_text_chunks(
            collection_name=collection_name,
            text_chunks=chunks,
            metadata_list=metadata_list
        )
        logger.info("Data successfully populated in Qdrant!")
        
    except Exception as e:
        logger.error(f"Failed to populate Qdrant: {e}")

if __name__ == "__main__":
    main()
