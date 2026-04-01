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

    def extract_text_and_images(self, output_image_dir: str) -> List[Dict]:
        """
        Extracts text blocks and images from each page using PyMuPDF.
        """
        output_path = Path(output_image_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        page_data = []
        
        try:
            doc = fitz.open(self.pdf_path)
            for page_num in range(len(doc)):
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
                
                # Extract images
                image_list = page.get_images(full=True)
                images_on_page = []
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Generate a unique filename for the image
                    img_filename = f"{self.doc_name}_p{page_num+1}_img{img_index+1}_{uuid.uuid4().hex[:8]}.{image_ext}"
                    img_path = output_path / img_filename
                    
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    
                    images_on_page.append(str(img_path))
                
                page_info = {
                    "page_number": page_num + 1,
                    "text_blocks": text_content,
                    "image_paths": images_on_page
                }
                page_data.append(page_info)
                
            doc.close()
            logger.info(f"Successfully extracted text and {sum(len(p['image_paths']) for p in page_data)} images from {self.pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error during text/image extraction: {str(e)}")
            raise
            
        return page_data

    def extract_tables(self, output_table_dir: str) -> List[Dict]:
        """
        Extracts tables from the PDF using Camelot.
        """
        output_path = Path(output_table_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        table_metadata = []
        
        try:
            # Note: flavor='lattice' for grid-based tables, 'stream' for whitespace-based.
            # We use 'lattice' as default but one could implement auto-detection logic.
            tables = camelot.read_pdf(str(self.pdf_path), pages='all', flavor='lattice')
            
            for i, table in enumerate(tables):
                table_id = f"table_{i+1}"
                page_num = table.page
                
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
                
            logger.info(f"Successfully extracted {len(tables)} tables from {self.pdf_path.name}")
            
        except Exception as e:
            logger.warning(f"Error during table extraction (Camelot): {str(e)}. This might happen if no tables are found or Ghostscript is missing.")
            # We don't necessarily want to crash the whole pipeline if tables fail
            return []
            
        return table_metadata

    def process_document(self) -> Dict:
        """
        Orchestrates the extraction process and returns a unified document map.
        """
        logger.info(f"Starting processing for: {self.pdf_path.name}")
        
        # Define output sub-directories based on document name
        base_processed_dir = Path("data/processed") / self.doc_name
        image_dir = base_processed_dir / "images"
        table_dir = base_processed_dir / "tables"
        
        # Run extractions
        page_data = self.extract_text_and_images(str(image_dir))
        table_data = self.extract_tables(str(table_dir))
        
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
