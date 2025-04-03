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