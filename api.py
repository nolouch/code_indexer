import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from knowledgebase.best_practices import BestPracticesKnowledgeBase
from llm.factory import LLMInterface
from llm.embedding import get_text_embedding

# --- Initialization (similar to the notebook) ---
try:
    # Configure your LLM client here
    # Replace with your actual bedrock ARN or other LLM config
    llm_client = LLMInterface(
        "bedrock",
        "arn:aws:bedrock:us-east-1:841162690310:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    )

    # Embedding function
    def embedding_func(text: str) -> np.ndarray:
        # Replace with your actual embedding model ID if different
        return get_text_embedding(text, "text-embedding-3-small")

    # Initialize Knowledge Base
    bp_kb = BestPracticesKnowledgeBase(llm_client, embedding_func)

    # Optional: Load existing data if needed
    # bp_kb.load_data() # Assuming a method exists to load persisted data

except Exception as e:
    print(f"Error during initialization: {e}")
    # Handle initialization errors appropriately
    # For a real application, you might want to exit or use a fallback
    llm_client = None
    bp_kb = None

# --- FastAPI App ---
app = FastAPI(
    title="Best Practices API",
    description="API to query development best practices.",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    results: dict  # Define a more specific model if the structure of results is known


@app.post("/best_practices", response_model=QueryResponse)
async def query_best_practices(request: QueryRequest):
    """
    Accepts a query string and returns relevant best practices
    from the knowledge base.
    """
    if bp_kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")

    try:
        practices = bp_kb.find_best_practices(request.query)
        # Ensure the results are serializable (e.g., list of strings or dicts)
        # You might need to adapt this based on the actual return type of find_best_practices
        return QueryResponse(results=practices)
    except Exception as e:
        # Log the exception for debugging
        print(f"Error processing query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail="Error querying best practices")
