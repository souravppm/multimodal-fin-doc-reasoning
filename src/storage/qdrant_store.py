import logging
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QdrantVectorStore:
    """
    Handles local vector storage using Qdrant (local disk mode).
    Embeds text chunks using the SentenceTransformers model.
    """

    def __init__(self, path: str = "qdrant_storage/"):
        """
        Initializes the Qdrant client and the embedding model.
        """
        logger.info(f"Initializing Qdrant client at path: {path}")
        self.client = QdrantClient(path=path)
        
        logger.info("Loading embedding model 'sentence-transformers/all-MiniLM-L6-v2'...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.vector_size = 384  # Vector size for all-MiniLM-L6-v2

    def create_collection(self, collection_name: str):
        """
        Creates a Qdrant collection if it doesn't already exist.
        """
        collections_response = self.client.get_collections()
        existing_collections = [col.name for col in collections_response.collections]
        
        if collection_name not in existing_collections:
            logger.info(f"Creating collection '{collection_name}' with vector size {self.vector_size}.")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{collection_name}' created successfully.")
        else:
            logger.info(f"Collection '{collection_name}' already exists.")

    def add_text_chunks(self, collection_name: str, text_chunks: List[str], metadata_list: List[Dict[str, Any]]):
        """
        Embeds the text using the local model and upserts them into Qdrant.
        """
        if not text_chunks:
            logger.warning("No text chunks provided for upsert.")
            return
        
        if len(text_chunks) != len(metadata_list):
            raise ValueError("The length of text_chunks must match the length of metadata_list.")

        logger.info(f"Embedding {len(text_chunks)} text chunks...")
        embeddings = self.model.encode(text_chunks, show_progress_bar=False)

        points = []
        for chunk, metadata, embedding in zip(text_chunks, metadata_list, embeddings):
            payload = metadata.copy()
            payload["text"] = chunk  # Store the actual text in the payload for retrieval
            
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload=payload
                )
            )

        logger.info(f"Upserting {len(points)} points into collection '{collection_name}'...")
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info("Upsert completed.")

    def search(self, collection_name: str, query_text: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Converts the query text into an embedding, searches Qdrant, and returns the top matches.
        """
        logger.info(f"Embedding query: '{query_text}'")
        query_vector = self.model.encode(query_text).tolist()

        logger.info(f"Searching for top {limit} matches in collection '{collection_name}'...")
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )

        results = []
        for hit in search_result:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
            
        logger.info(f"Found {len(results)} matches.")
        return results

if __name__ == "__main__":
    # Simple self-test
    store = QdrantVectorStore()
    store.create_collection("test_collection")
    store.add_text_chunks(
        "test_collection",
        ["Financial report for Q1 shows increased revenue.", "The CEO announced a new product line."],
        [{"page_number": 1, "source": "report_q1.pdf"}, {"page_number": 2, "source": "report_q1.pdf"}]
    )
    res = store.search("test_collection", "What happened to the revenue?")
    for r in res:
        print(f"Score: {r['score']:.4f} | Text: {r['payload'].get('text', '')}")
