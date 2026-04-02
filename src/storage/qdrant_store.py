import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class QdrantVectorStore:
    def __init__(self, path="qdrant_storage/"):
        self.client = QdrantClient(path=path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_size = self.model.get_sentence_embedding_dimension()

    def create_collection(self, collection_name: str):
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def add_text_chunks(self, collection_name: str, text_chunks: list, metadata_list: list):
        embeddings = self.model.encode(text_chunks).tolist()
        points = [
            PointStruct(
                id=str(uuid.uuid4()), 
                vector=emb, 
                payload={"text": chunk, **meta}
            )
            for chunk, emb, meta in zip(text_chunks, embeddings, metadata_list)
        ]
        self.client.upsert(collection_name=collection_name, points=points)

    def search(self, collection_name: str, query_text: str, limit: int = 3):
        query_vector = self.model.encode(query_text).tolist()
        # এখানে QdrantClient এর আসল query_points মেথড কল করা হচ্ছে
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit
        )
        return response.points