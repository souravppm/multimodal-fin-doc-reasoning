import os
import logging
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.storage.qdrant_store import QdrantVectorStore
from src.storage.sqlite_store import SQLiteTableStore
from src.retrieval.router import QueryRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class FinancialRAG:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.logger.warning("OPENAI_API_KEY is not set. Inference will fail.")
            
        self.client = OpenAI(api_key=api_key)
        
        self.logger.info("Initializing components for FinancialRAG...")
        try:
            self.qdrant = QdrantVectorStore(collection_name='test_financial_reports')
        except TypeError:
            # Fallback if the kwarg is actually 'collection'
            self.qdrant = QdrantVectorStore(collection='test_financial_reports')
            
        self.sqlite = SQLiteTableStore()
        self.router = QueryRouter()
        
        self.logger.info("FinancialRAG initialized successfully.")

    def get_database_schema(self) -> str:
        """Helper method that queries the SQLite database to get all table names and column names."""
        try:
            tables_df = self.sqlite.execute_query("SELECT name FROM sqlite_master WHERE type='table';")
            if tables_df is None or tables_df.empty:
                return "No tables found in the database."
                
            schema_info = []
            for table_name in tables_df['name']:
                columns_df = self.sqlite.execute_query(f"PRAGMA table_info('{table_name}');")
                if columns_df is not None and not columns_df.empty:
                    columns = ", ".join(columns_df['name'].tolist())
                    schema_info.append(f"Table: {table_name} | Columns: {columns}")
                else:
                    schema_info.append(f"Table: {table_name} | Columns: None")
            
            return "\n".join(schema_info)
        except Exception as e:
            self.logger.error(f"Error fetching database schema: {e}")
            return "Database schema unavailable."

    def answer_question(self, user_query: str) -> str:
        """Main method that routes query, gathers context, and generates final answer."""
        try:
            self.logger.info(f"Processing query: '{user_query}'")
            
            # Step 1: Call routing
            route_info = self.router.route_query(user_query)
            route = route_info.get("route", "text")
            self.logger.info(f"Query routed to: {route}")
            
            context = ""
            
            # Step 2: Context gathering based on route
            if route == 'text':
                try:
                    # Execute semantic search
                    results = self.qdrant.search(user_query, limit=3)
                    
                    context_pieces = []
                    for res in results:
                        # Extract the payload handling both dict and object types loosely
                        if isinstance(res, dict):
                            context_pieces.append(str(res.get('payload', res)))
                        else:
                            context_pieces.append(str(getattr(res, 'payload', res)))
                            
                    context = "\n---\n".join(context_pieces)
                    self.logger.info("Context gathered from Qdrant vector store.")
                except Exception as e:
                    self.logger.error(f"Qdrant search error: {e}")
                    context = "Error retrieving text context."

            elif route == 'table':
                schema = self.get_database_schema()
                self.logger.debug(f"Retrieved Schema:\n{schema}")
                
                sql_prompt = (
                    "Based on the following database schema, write a pure and exact raw SQL query to answer the user's question. "
                    "You MUST return ONLY the completely valid SQL text without any markdown tags or explanations.\n\n"
                    f"Schema:\n{schema}\n\n"
                    f"Question: {user_query}"
                )
                
                try:
                    sql_response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": sql_prompt}],
                        temperature=0.0
                    )
                    sql_query = sql_response.choices[0].message.content.strip()
                    self.logger.info(f"Generated SQL Query: {sql_query}")
                    
                    df = self.sqlite.execute_query(sql_query)
                    if df is not None and not df.empty:
                        context = df.to_string(index=False)
                        self.logger.info("Successfully executed SQL and converted DataFrame result to context string.")
                    else:
                        context = "Database returned empty result for the executed query."
                except Exception as e:
                    self.logger.error(f"Error generating or executing SQL: {e}")
                    context = "Error executing table data retrieval."

            elif route == 'image':
                context = "Visual chart reasoning is not fully implemented in this context yet."
                self.logger.info("Image routing hit - providing placeholder context.")
                
            else:
                context = f"Unknown route defined: {route}. Unable to fetch context."
                self.logger.warning(context)

            # Step 3: Final LLM Answer Generation
            system_prompt = (
                "You are an expert financial analyst. Answer the user's question based STRICTLY "
                "on the provided context. If the answer is not in the context, say so. Do not hallucinate."
            )
            
            final_user_prompt = f"Context:\n{context}\n\nQuestion:\n{user_query}"
            
            self.logger.info("Calling LLM for final response synthesis.")
            final_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_user_prompt}
                ],
                temperature=0.0
            )
            
            final_answer = final_response.choices[0].message.content.strip()
            self.logger.info("Successfully formulated final answer.")
            return final_answer
            
        except Exception as e:
            self.logger.error(f"Critical error in answer_question execution: {e}")
            return f"An error occurred while answering your question: {e}"
