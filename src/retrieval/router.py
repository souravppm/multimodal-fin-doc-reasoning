import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class QueryRouter:
    def __init__(self):
        """
        Initialize the OpenAI client using OPENAI_API_KEY from environment variables.
        """
        # Load environment variables
        load_dotenv()
        
        # Get OpenAI API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY environment variable not found. Routing might fail if not provided.")
            
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        logger.info("QueryRouter initialized.")

    def route_query(self, query: str) -> dict:
        """
        Routes the query to the appropriate category: text, table, or image.
        Returns a dictionary like {"route": "text"}.
        """
        logger.info(f"Routing query: '{query}'")
        
        system_prompt = (
            "You are a financial query classifier. You must classify the given user query "
            "into exactly one of these two categories:\n"
            "- \"text\": If the question asks for explanations, policies, summaries, or general text information.\n"
            "- \"table\": If the question asks for exact numerical values, financial metrics, comparisons across quarters, or tabular data.\n\n"
            "HINT: If a query involves charts, trends, or visual representation, always route it to \"text\" so it can be handled by standard retrieval. "
            "You MUST return ONLY a raw JSON string e.g. {\"route\": \"text\"} or {\"route\": \"table\"}. "
            "Do NOT include markdown blocks (no ```json). Do NOT add any extra text or explanation."
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0
            )
            
            raw_response = response.choices[0].message.content.strip()
            logger.debug(f"LLM raw response: {raw_response}")
            
            # Parse the JSON string
            route_dict = json.loads(raw_response)
            
            logger.info(f"Successfully routed query to: {route_dict.get('route', 'unknown')}")
            return route_dict
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON. Raw response: {raw_response}. Error: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while routing the query: {e}")
            raise
