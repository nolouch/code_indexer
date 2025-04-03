"""Code graph analysis module."""
from code_graph.models import Node, Edge, Repository
from code_graph.builder import SemanticGraphBuilder
from code_graph.db_manager import GraphDBManager
from code_graph.embedders import create_embedder

# Simplified initialization function for builder with OpenAI embeddings
def create_openai_builder():
    """Create a semantic graph builder that uses OpenAI embeddings."""
    from setting.embedding import EMBEDDING_MODEL
    
    # Temporarily set provider to OpenAI
    original_provider = EMBEDDING_MODEL.get("provider")
    EMBEDDING_MODEL["provider"] = "openai"
    
    # Create embedders and builder
    code_embedder = create_embedder("code")
    doc_embedder = create_embedder("doc")
    builder = SemanticGraphBuilder(code_embedder, doc_embedder)
    
    # Restore original provider setting
    EMBEDDING_MODEL["provider"] = original_provider
    
    return builder 