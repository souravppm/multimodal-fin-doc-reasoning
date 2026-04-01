import os
from src.storage.qdrant_store import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    print("🚀 Initializing Qdrant and Local Embedding Model...")
    print("⚠️ Note: The first run may take a minute to download the model (~80MB).")
    
    try:
        vector_store = QdrantVectorStore()
        collection_name = "test_financial_reports"

        print(f"⏳ Creating or connecting to collection '{collection_name}'...")
        vector_store.create_collection(collection_name)

        sample_text = """
        Tesla's automotive revenue for Q3 2023 was $19.62 billion. 
        The company deployed 4 gigawatt-hours of energy storage in the same quarter.
        Apple reported a quarterly revenue of $89.5 billion in Q4 2023, down 1% year over year.
        Microsoft's cloud computing platform, Azure, saw a revenue growth of 29% in the last fiscal year.
        NVIDIA reached a record revenue of $18.12 billion, up 206% from a year ago.
        """

        print("✂️ Chunking text into smaller pieces...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=120,
            chunk_overlap=20,
            length_function=len,
        )
        chunks = text_splitter.split_text(sample_text.strip())
        
        metadata_list = [{"source": "mock_report", "chunk_id": i} for i in range(len(chunks))]

        print(f"💾 Embedding and Upserting {len(chunks)} chunks into Qdrant...")
        vector_store.add_text_chunks(collection_name, chunks, metadata_list)

    
        query = "How much was Tesla's automotive revenue?"
        print(f"\n🔍 Searching for: '{query}'")
        results = vector_store.search(collection_name, query, limit=2)

        print("\n✅ Search Results:")
        for i, res in enumerate(results):
            
            payload = getattr(res, 'payload', {})
            score = getattr(res, 'score', 0.0)
            print(f"[{i+1}] Score: {score:.4f}")
            print(f"    Payload: {payload}\n")

    except Exception as e:
        print(f"❌ Error during Qdrant test: {e}")

if __name__ == "__main__":
    main()