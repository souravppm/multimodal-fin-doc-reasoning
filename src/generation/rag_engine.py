import os
import logging
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from src.storage.sqlite_store import SQLiteTableStore
from src.retrieval.router import QueryRouter
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialRAG:
    def __init__(self):
        logger.info("Initializing components for FinancialRAG...")
        self.qc = QdrantClient(path="qdrant_storage/")
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.collection_name = 'test_financial_reports'
        self.sqlite = SQLiteTableStore(db_path="data/processed/financial_tables.db")
        self.router = QueryRouter()
        
        # Initialize both clients
        ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.ollama_client = OpenAI(base_url=ollama_base_url, api_key="ollama")
        
        # Check if OpenAI API key exists for GPT-4o-mini
        openai_key = os.environ.get("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=openai_key) if openai_key else None
        
        logger.info("FinancialRAG initialized successfully.")

    def get_database_schema(self) -> str:
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table';"
            tables_df = self.sqlite.execute_query(query)
            schema = ""
            for table in tables_df['name']:
                col_query = f"PRAGMA table_info('{table}');"
                cols_df = self.sqlite.execute_query(col_query)
                cols = ", ".join(cols_df['name'].tolist())
                # Provide the table name in double quotes to the LLM so it generates valid SQL if it starts with a number
                schema += f'Table: "{table}" | Columns: {cols}\n'
            return schema
        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            return "No schema available."

    def answer_question(self, user_query: str, model_choice: str = "Ollama (Local)"):
        try:
            # Set the active client and model based on user choice
            if model_choice == "GPT-4o-mini (Cloud)":
                if not self.openai_client:
                    return "Error: OPENAI_API_KEY is not set in the .env file."
                active_client = self.openai_client
                active_model = "gpt-4o-mini"
            else:
                active_client = self.ollama_client
                active_model = "llama3.2"

            logger.info(f"Using Model: {active_model}")

            # Step 1: Route the Query
            route_decision = self.router.route_query(user_query)
            route = route_decision.get('route', 'text')
            logger.info(f"Query routed to: {route}")
            
            context = ""

            # Step 2: Retrieve Context
            if route == 'text':
                try:
                    query_vector = self.embedder.encode(user_query).tolist()
                    response = self.qc.query_points(
                        collection_name=self.collection_name, 
                        query=query_vector, 
                        limit=10
                    )
                    results = response.points
                    context_chunks = [res.payload.get('text', '') for res in results if hasattr(res, 'payload')]
                    context = "\n".join(context_chunks)
                except Exception as e:
                    logger.error(f"Text retrieval failed: {e}")
                    context = "Could not retrieve text context."

            elif route == 'table':
                schema = self.get_database_schema()
                sql_prompt = (f"You are a SQLite expert. Schema:\n{schema}\n"
                              f"Write a strict SQL query for: '{user_query}'. "
                              f"IMPORTANT: If a table name starts with a number, you MUST enclose it in double quotes! "
                              f"HINT FOR PDF TABLES: Financial metrics like 'Revenue', 'Income', or 'Expense' are almost NEVER column names! "
                              f"They are usually row values in the first column (like 'Unnamed_0', 'col_0'). "
                              f"Instead of SELECT Revenue, you should do: SELECT * FROM \"table_name\" WHERE \"Unnamed_0\" LIKE '%Revenue%' OR \"col_0\" LIKE '%Revenue%'; "
                              f"If you are unsure which table contains the data, try to guess the most likely table based on the columns. "
                              f"Return ONLY raw SQL. No markdown blocks.")
                
                sql_response = active_client.chat.completions.create(
                    model=active_model,
                    messages=[{"role": "user", "content": sql_prompt}],
                    temperature=0
                )
                raw_sql = sql_response.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()
                logger.info(f"Generated SQL: {raw_sql}")
                
                try:
                    result_df = self.sqlite.execute_query(raw_sql)
                    if result_df.empty:
                        raise ValueError("SQL returned empty results.")
                    context = f"SQL Result:\n{result_df.to_string()}"
                except Exception as e:
                    logger.warning(f"SQL failed or was empty ({e}). Falling back to Vector Search...")
                    try:
                        # Fallback to Text Vector Search
                        query_vector = self.embedder.encode(user_query).tolist()
                        response = self.qc.query_points(
                            collection_name=self.collection_name, 
                            query=query_vector, 
                            limit=10
                        )
                        results = response.points
                        context_chunks = [res.payload.get('text', '') for res in results if hasattr(res, 'payload')]
                        context = "SQL Table search failed. Using Text context instead:\n" + "\n".join(context_chunks)
                    except Exception as fallback_e:
                        logger.error(f"Fallback Text retrieval failed: {fallback_e}")
                        context = "Could not retrieve context from SQL or Text fallback."


            # Step 3: Final Generation
            system_prompt = "You are an expert financial analyst. Answer the user's question clearly based STRICTLY on the provided Context. If the answer is not in the Context, say 'I don't have enough data to answer this'. Do not hallucinate."
            
            final_response = active_client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
                ],
                temperature=0.2,
                stream=True
            )
            
            def stream_response():
                try:
                    for chunk in final_response:
                        if chunk.choices and chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
                except Exception as stream_e:
                    logger.error(f"Streaming error: {stream_e}")
                    yield f"\n[Streaming error occurred]"
                    
            return stream_response(), context

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            def error_response():
                yield f"An error occurred: {str(e)}"
            return error_response(), "No context available due to error."