import os
import fitz  # PyMuPDF
import camelot
import json
import logging
import uuid
from typing import List, Dict, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialDocumentParser:
    """
    A production-grade parser for extracting text, tables, and images from financial PDFs.
    """

    def __init__(self, pdf_path: str):
        """
        Initialize the parser with a PDF file path.
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
        self.doc_name = self.pdf_path.stem
        self.document_map = {
            "document_name": self.doc_name,
            "pages": []
        }

    def extract_text_and_images(self, output_image_dir: str, max_pages: int = None, progress_callback=None) -> List[Dict]:
        """
        Extracts text blocks and images from each page using PyMuPDF.
        """
        output_path = Path(output_image_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        page_data = []
        
        try:
            doc = fitz.open(self.pdf_path)
            total_pages = len(doc)
            limit = min(total_pages, max_pages) if max_pages else total_pages
            
            for page_num in range(limit):
                if progress_callback:
                    progress_callback("text", page_num + 1, limit)
                page = doc.load_page(page_num)
                
                # Extract text blocks
                text_blocks = page.get_text("blocks")
                text_content = []
                for b in text_blocks:
                    # b = (x0, y0, x1, y1, "text", block_no, block_type)
                    text_content.append({
                        "text": b[4].strip(),
                        "bbox": (b[0], b[1], b[2], b[3]),
                        "type": "text" if b[6] == 0 else "image/other"
                    })
                
                page_info = {
                    "page_number": page_num + 1,
                    "text_blocks": text_content,
                    "image_paths": [] # Imaging disabled
                }
                page_data.append(page_info)
                
            doc.close()
            logger.info(f"Successfully extracted text from {self.pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error during text/image extraction: {str(e)}")
            raise
            
        return page_data

    def extract_tables(self, output_table_dir: str, max_pages: int = None, progress_callback=None) -> List[Dict]:
        """
        Extracts tables from the PDF using Camelot with a dynamic lattice/stream fallback strategy.
        """
        output_path = Path(output_table_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        table_metadata = []
        global_table_counter = 1
        
        try:
            # Determine total pages using fitz
            doc = fitz.open(self.pdf_path)
            total_pages = len(doc)
            doc.close()
            
            limit = min(total_pages, max_pages) if max_pages else total_pages
            for page_num in range(1, limit + 1):
                if progress_callback:
                    progress_callback("tables", page_num, limit)
                try:
                    # Try lattice (grid-based) first
                    tables = camelot.read_pdf(str(self.pdf_path), pages=str(page_num), flavor='lattice')
                    
                    # If lattice yields no tables or tables with poor accuracy, try stream (whitespace-based)
                    if not tables or all(t.accuracy < 70 for t in tables):
                        stream_tables = camelot.read_pdf(str(self.pdf_path), pages=str(page_num), flavor='stream')
                        
                        if stream_tables:
                            if not tables:
                                tables = stream_tables
                            else:
                                # Pick stream if its average accuracy is significantly better
                                avg_lattice = sum(t.accuracy for t in tables) / len(tables) if tables else 0
                                avg_stream = sum(t.accuracy for t in stream_tables) / len(stream_tables) if stream_tables else 0
                                if avg_stream > avg_lattice + 10:
                                    tables = stream_tables

                    for table in tables:
                        # Skip garbage tables to prevent injecting noise into SQLite RAG
                        if getattr(table, 'accuracy', 0) < 50:
                            continue
                            
                        table_id = f"table_{global_table_counter}"
                        global_table_counter += 1
                        
                        # Save table to CSV
                        table_filename = f"{self.doc_name}_p{page_num}_{table_id}.csv"
                        table_file_path = output_path / table_filename
                        table.to_csv(str(table_file_path))
                        
                        table_metadata.append({
                            "table_id": table_id,
                            "page_number": page_num,
                            "path": str(table_file_path),
                            "confidence": table.accuracy,
                            "parsing_report": table.parsing_report
                        })
                except Exception as e:
                    logger.debug(f"Error parsing tables on page {page_num}: {e}")
                    continue
                    
            logger.info(f"Successfully extracted {len(table_metadata)} tables from {self.pdf_path.name}")
            
        except Exception as e:
            logger.warning(f"Error during table extraction setup (likely Ghostscript issue): {str(e)}")
            # We don't crash the pipeline if table extraction fails completely
            return []
            
        return table_metadata

    def process_document(self, max_pages: int = None, progress_callback=None) -> Dict:
        """
        Orchestrates the extraction process and returns a unified document map.
        """
        logger.info(f"Starting processing for: {self.pdf_path.name}")
        
        # Define output sub-directories based on document name
        base_processed_dir = Path("data/processed") / self.doc_name
        table_dir = base_processed_dir / "tables"
        
        # Run extractions
        page_data = self.extract_text_and_images(str(base_processed_dir), max_pages=max_pages, progress_callback=progress_callback) # Images no longer extracted
        table_data = self.extract_tables(str(table_dir), max_pages=max_pages, progress_callback=progress_callback)
        
        # Merge table pointers into page data
        for page in page_data:
            page_num = page["page_number"]
            page["tables"] = [t for t in table_data if t["page_number"] == page_num]
            
        self.document_map["pages"] = page_data
        
        # Optional: Save the full document map as a JSON for reference
        mapping_file = base_processed_dir / "document_map.json"
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(self.document_map, f, indent=4)
            
        logger.info(f"Document processing complete. Inventory saved to {mapping_file}")
        return self.document_map

if __name__ == "__main__":
    # Example usage:
    # parser = FinancialDocumentParser("data/raw/annual_report.pdf")
    # doc_map = parser.process_document()
    pass
