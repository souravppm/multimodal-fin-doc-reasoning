import os
import time
from src.ingestion.pdf_parser import FinancialDocumentParser

def main():
    pdf_path = "data/raw/sample_report.pdf" 
    
    
    image_dir = "data/processed/images"
    table_dir = "data/processed/tables"
    
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found at {pdf_path}. Please check the path.")
        return

    print(f"🚀 Starting extraction for: {pdf_path}")
    start_time = time.time()
    
    try:
        parser = FinancialDocumentParser(pdf_path)
        
        print("⏳ Extracting text and images (PyMuPDF)...")
        parser.extract_text_and_images(image_dir)
        
        print("⏳ Extracting tables (Camelot)... This might take a while.")
        parser.extract_tables(table_dir)
        
        print(f"✅ Extraction complete in {time.time() - start_time:.2f} seconds!")
        print("👉 Now manually check data/processed/tables/ and data/processed/images/")
        
    except Exception as e:
        print(f"❌ Extraction failed with error: {e}")

if __name__ == "__main__":
    main()