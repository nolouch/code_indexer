"""Configuration settings for code indexer."""

# Embedding model configuration
EMBEDDING_MODEL = {
    "name": "jinaai/jina-embeddings-v3-base-en",
    "dimension": 512,  # Jina v3 base model dimension
}

# Vector search configuration
VECTOR_SEARCH = {
    "default_limit": 10,
    "similarity_metric": "cosine"  # or "l2"
}

"""Configuration for code graph."""
from typing import Dict, Any

# Default configuration for code graph
CODE_GRAPH_CONFIG = {
    "cache_enabled": True,
    "cache_ttl_seconds": 86400,  # 24 hours
    "batch_size": 32,  # Default batch size for embedding operations
}

def get_config() -> Dict[str, Any]:
    """Get the code graph configuration."""
    return CODE_GRAPH_CONFIG 