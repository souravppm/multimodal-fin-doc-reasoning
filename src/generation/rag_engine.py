import os
import logging
from openai import OpenAI
from src.storage.qdrant_store import QdrantVectorStore
from src.storage.sqlite_store import SQLiteTableStore
from src.retrieval.router import QueryRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialRAG:
    def __init__(self):
        logger.info("Initializing components for FinancialRAG...")
        self.qdrant = QdrantVectorStore()
        self.collection_name = 'test_financial_reports'
        self.sqlite = SQLiteTableStore(db_path="data/processed/financial_tables.db")
        self.router = QueryRouter()
        
        # Ollama Setup
        self.llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        logger.info("FinancialRAG initialized successfully.")

    def get_database_schema(self) -> str:
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table';"
            tables_df = self.sqlite.execute_query(query)
            schema = ""
            for table in tables_df['name']:
                col_query = f"PRAGMA table_info({table});"
                cols_df = self.sqlite.execute_query(col_query)
                cols = ", ".join(cols_df['name'].tolist())
                schema += f"Table: {table} | Columns: {cols}\n"
            return schema
        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            return "No schema available."

    def answer_question(self, user_query: str) -> str:
        try:
            # Step 1: Route the Query
            route_decision = self.router.route_query(user_query)
            route = route_decision.get('route', 'text')
            logger.info(f"Query routed to: {route}")
            
            context = ""

            # Step 2: Retrieve Context based on Route
            if route == 'text':
                try:
                    # ✅ FIXED: Using the qdrant wrapper's correct search method
                    results = self.qdrant.search(collection_name=self.collection_name, query_text=user_query, limit=3)
                    
                    # ✅ FIXED: Handling pure Qdrant payload extraction
                    context_chunks = []
                    for res in results:
                         if hasattr(res, 'payload') and isinstance(res.payload, dict):
                              text = res.payload.get('text', '')
                              if text: context_chunks.append(text)
                    context = "\n".join(context_chunks)
                except Exception as e:
                    logger.error(f"Text retrieval failed: {e}")
                    context = "Could not retrieve text context."

            elif route == 'table':
                schema = self.get_database_schema()
                sql_prompt = f"""You are a SQLite expert. Given this schema:
{schema}
Write a strict SQL query to answer this user question: '{user_query}'
Return ONLY the raw SQL query. Do not use markdown blocks like ```sql. Do not add explanations."""
                
                sql_response = self.llm.chat.completions.create(
                    model="llama3.2",
                    messages=[{"role": "user", "content": sql_prompt}],
                    temperature=0
                )
                raw_sql = sql_response.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()
                logger.info(f"Generated SQL: {raw_sql}")
                
                try:
                    result_df = self.sqlite.execute_query(raw_sql)
                    context = f"SQL Result:\n{result_df.to_string()}"
                except Exception as e:
                    logger.error(f"SQL execution failed: {e}")
                    context = f"Failed to execute SQL: {raw_sql}"

            elif route == 'image':
                context = "Visual chart reasoning is not fully implemented. Please refer to the text or tables."

            # Step 3: Final Generation
            system_prompt = "You are an expert financial analyst. Answer the user's question clearly based STRICTLY on the provided Context. If the answer is not in the Context, say 'I don't have enough data to answer this'. Do not hallucinate."
            
            final_response = self.llm.chat.completions.create(
                model="llama3.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
                ],
                temperature=0.2
            )
            return final_response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return "An error occurred while processing your request."