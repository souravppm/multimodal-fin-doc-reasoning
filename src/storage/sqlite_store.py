import sqlite3
import logging
import pandas as pd
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SQLiteTableStore:
    """
    A class to handle storing tabular data into a structured SQLite database 
    and querying it using SQL. Uses pandas and sqlite3.
    """

    def __init__(self, db_path: str = "data/processed/financial_tables.db"):
        """
        Initializes the SQLiteTableStore and establishes a database connection.

        Args:
            db_path (str): The file path to the SQLite database. Defaults to "data/processed/financial_tables.db".
        """
        self.db_path = db_path
        
        # Ensure the directory exists
        db_dir = Path(self.db_path).parent
        if db_dir.name and str(db_dir) != ".":
            db_dir.mkdir(parents=True, exist_ok=True)
            
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            logger.info(f"Successfully connected to SQLite database at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to SQLite database at {self.db_path}: {e}")
            raise

    def __del__(self):
        """
        Closes the database connection when the object is destroyed.
        """
        if hasattr(self, 'conn') and self.conn:
            try:
                self.conn.close()
            except sqlite3.Error as e:
                logger.error(f"Error closing SQLite database connection: {e}")

    def load_csv_to_table(self, csv_path: str, table_name: str) -> None:
        """
        Reads a CSV file, cleans its column names, and saves it as a table in the SQLite database.

        Args:
            csv_path (str): The file path to the CSV file.
            table_name (str): The name of the table to create/replace in the database.
        """
        try:
            logger.info(f"Attempting to load CSV from {csv_path} into table '{table_name}'")
            
            # Read CSV using pandas
            df = pd.read_csv(csv_path)

            cleaned_columns = []
            for i, col in enumerate(df.columns):
                col_str = str(col).strip()
                col_str = col_str.replace(' ', '_')
                col_str = re.sub(r'[^a-zA-Z0-9_]', '', col_str)
                
                # Prevent empty column names
                if not col_str:
                    col_str = f"col_{i}"
                
                # Prevent duplicate column names
                original_col_str = col_str
                counter = 1
                while col_str in cleaned_columns:
                    col_str = f"{original_col_str}_{counter}"
                    counter += 1
                    
                cleaned_columns.append(col_str)
            
            df.columns = cleaned_columns

            # Save the dataframe to the SQLite database
            df.to_sql(table_name, self.conn, if_exists='replace', index=False)
            logger.info(f"Successfully loaded data into table '{table_name}' with {len(df)} rows.")

        except FileNotFoundError:
            logger.error(f"CSV file not found at {csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV to table '{table_name}': {e}")
            raise

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Executes a raw SQL query against the database and returns the results as a pandas DataFrame.

        Args:
            query (str): The raw SQL query to execute.

        Returns:
            pd.DataFrame: A DataFrame containing the query results.
        """
        try:
            logger.debug(f"Executing query: {query}")
            df = pd.read_sql_query(query, self.conn)
            logger.info(f"Query executed successfully, fetched {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(f"Error executing query '{query}': {e}")
            raise
