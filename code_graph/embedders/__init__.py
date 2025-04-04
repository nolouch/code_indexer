"""Code and documentation embedders package."""
from typing import Optional
from setting.embedding import EMBEDDING_MODEL
from code_graph.config import get_config

def create_embedder(embedder_type: str, embedding_dim: Optional[int] = None, batch_size: Optional[int] = None):
    """Factory function to create an embedder based on provider setting.
    
    Args:
        embedder_type (str): Type of embedder, 'code' or 'doc'
        embedding_dim (Optional[int]): Optional dimension override
        batch_size (Optional[int]): Optional batch size override
        
    Returns:
        An embedder instance based on the provider configuration
    """
    provider = EMBEDDING_MODEL.get("provider", "sentence_transformers")
    
    # Get batch size from config if not provided
    if batch_size is None:
        config = get_config()
        batch_size = config.get("batch_size", 32)
    
    if provider == "openai":
        from .openai_embedding import OpenAICodeEmbedder, OpenAIDocEmbedder
        if embedder_type.lower() == "code":
            return OpenAICodeEmbedder(embedding_dim=embedding_dim, batch_size=batch_size)
        else:
            return OpenAIDocEmbedder(embedding_dim=embedding_dim, batch_size=batch_size)
    else:  # Default to sentence_transformers
        from .code import CodeEmbedder
        from .doc import DocEmbedder
        if embedder_type.lower() == "code":
            return CodeEmbedder(embedding_dim=embedding_dim, batch_size=batch_size)
        else:
            return DocEmbedder(embedding_dim=embedding_dim, batch_size=batch_size) 