"""OpenAI embedding functions."""
import openai
import json
import logging

logger = logging.getLogger(__name__)

# Initialize the OpenAI client
try:
    embedding_model = openai.OpenAI()
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")
    embedding_model = None


def get_text_embedding(text: str, model="text-embedding-3-small"):
    """Get embedding for a text using OpenAI's API.
    
    Args:
        text (str): Text to embed
        model (str): OpenAI model to use
        
    Returns:
        List[float]: Embedding vector as a list
    """
    if embedding_model is None:
        raise ValueError("OpenAI client not initialized")
        
    text = text.replace("\n", " ")
    try:
        return embedding_model.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        logger.error(f"Error getting OpenAI embedding: {e}")
        raise


def get_entity_description_embedding(name: str, description: str, model="text-embedding-3-small"):
    """Get embedding for an entity with name and description.
    
    Args:
        name (str): Entity name
        description (str): Entity description
        model (str): OpenAI model to use
        
    Returns:
        List[float]: Embedding vector as a list
    """
    combined_text = f"{name}: {description}"
    return get_text_embedding(combined_text, model=model)


def get_entity_metadata_embedding(metadata: dict, model="text-embedding-3-small"):
    """Get embedding for entity metadata.
    
    Args:
        metadata (dict): Metadata dictionary
        model (str): OpenAI model to use
        
    Returns:
        List[float]: Embedding vector as a list
    """
    combined_text = json.dumps(metadata)
    return get_text_embedding(combined_text, model=model)


# Dimension mapping for common OpenAI embedding models
OPENAI_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536
}
