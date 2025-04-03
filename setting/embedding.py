"""Embedding model configuration settings."""

# Embedding model configuration
EMBEDDING_MODEL = {
    "name": "all-MiniLM-L6-v2",  # Use MiniLM model which is more stable
    "dimension": 384,            # MiniLM dimension
}

# Vector search configuration
VECTOR_SEARCH = {
    "default_limit": 10,
    "similarity_metric": "cosine"  # or "l2"
}

# Embedding dimensions for different types
CODE_EMBEDDING_DIM = EMBEDDING_MODEL["dimension"]
DOC_EMBEDDING_DIM = EMBEDDING_MODEL["dimension"] 