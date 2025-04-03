"""Embedding model configuration settings."""

# Embedding model configuration
EMBEDDING_MODEL = {
    "provider": "sentence_transformers",  # Options: "sentence_transformers", "openai"
    "name": "all-MiniLM-L6-v2",  # Use MiniLM model which is more stable
    "dimension": 384,            # MiniLM dimension
    # OpenAI embedding model config
    "openai_model": "text-embedding-3-small",  # OpenAI embedding model name
    "openai_dimension": 1536,   # OpenAI embedding dimension (1536 for text-embedding-3-small)
}

# Vector search configuration
VECTOR_SEARCH = {
    "default_limit": 10,
    "similarity_metric": "cosine"  # or "l2"
}

# Embedding dimensions for different types based on provider
CODE_EMBEDDING_DIM = EMBEDDING_MODEL["openai_dimension"] if EMBEDDING_MODEL["provider"] == "openai" else EMBEDDING_MODEL["dimension"]
DOC_EMBEDDING_DIM = EMBEDDING_MODEL["openai_dimension"] if EMBEDDING_MODEL["provider"] == "openai" else EMBEDDING_MODEL["dimension"] 