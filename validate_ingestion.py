import sys
import os
from pathlib import Path

# Add src to the Python path
src_path = str(Path(__file__).parent / "src")
sys.path.append(src_path)

try:
    from ingestion.pdf_parser import FinancialDocumentParser
    print("✅ FinancialDocumentParser imported successfully!")
    
    # Try to instantiate with a dummy path and catch the FileNotFoundError
    try:
        parser = FinancialDocumentParser("data/raw/non_existent.pdf")
    except FileNotFoundError as e:
        print(f"✅ Class instantiation handled missing file as expected: {e}")
        
except ImportError as e:
    print(f"❌ Failed to import FinancialDocumentParser: {e}")
except Exception as e:
    print(f"❌ Unexpected error during validation: {e}")

print("\n--- Validation Complete ---")
