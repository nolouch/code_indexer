"""OpenAI embedding functions."""
import openai
import json
import logging
from tqdm import tqdm

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


def batch_get_text_embedding(texts: list, model="text-embedding-3-small", max_batch_size=16):
    """Get embeddings for multiple texts using OpenAI's API in a single batch request.
    
    Args:
        texts (list): List of texts to embed
        model (str): OpenAI model to use
        max_batch_size (int): Maximum number of texts to process in a single API call
        
    Returns:
        List[List[float]]: List of embedding vectors
    """
    if embedding_model is None:
        raise ValueError("OpenAI client not initialized")
    
    if not texts:
        return []
    
    # Replace newlines with spaces
    processed_texts = [text.replace("\n", " ") if isinstance(text, str) else "" for text in texts]
    
    # Split into batches to avoid OpenAI API limits
    all_embeddings = []
    
    # Calculate total number of batches
    total_batches = (len(processed_texts) + max_batch_size - 1) // max_batch_size
    
    # Process batches with progress bar
    for i in tqdm(range(0, len(processed_texts), max_batch_size), 
                 total=total_batches, 
                 desc=f"Batches (size={max_batch_size})", 
                 unit="batch"):
        batch = processed_texts[i:i + max_batch_size]
        
        try:
            response = embedding_model.embeddings.create(input=batch, model=model)
            # Sort by index to ensure order matches input
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Error getting batch OpenAI embeddings (batch {i//max_batch_size}): {e}")
            raise
    
    return all_embeddings


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
