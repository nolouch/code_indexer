"""Code and documentation embedders package."""
from typing import Optional
from setting.embedding import EMBEDDING_MODEL

def create_embedder(embedder_type: str, embedding_dim: Optional[int] = None):
    """Factory function to create an embedder based on provider setting.
    
    Args:
        embedder_type (str): Type of embedder, 'code' or 'doc'
        embedding_dim (Optional[int]): Optional dimension override
        
    Returns:
        An embedder instance based on the provider configuration
    """
    provider = EMBEDDING_MODEL.get("provider", "sentence_transformers")
    
    if provider == "openai":
        from .openai_embedding import OpenAICodeEmbedder, OpenAIDocEmbedder
        if embedder_type.lower() == "code":
            return OpenAICodeEmbedder(embedding_dim=embedding_dim)
        else:
            return OpenAIDocEmbedder(embedding_dim=embedding_dim)
    else:  # Default to sentence_transformers
        from .code import CodeEmbedder
        from .doc import DocEmbedder
        if embedder_type.lower() == "code":
            return CodeEmbedder(embedding_dim=embedding_dim)
        else:
            return DocEmbedder(embedding_dim=embedding_dim) 