"""Documentation embedder implementation."""
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from setting.embedding import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class DocEmbedder:
    """Embeds documentation into vectors."""
    
    def __init__(self, embedding_dim=None):
        """Initialize documentation embedder.
        
        Args:
            embedding_dim (int, optional): Embedding dimension. If not provided,
                uses the dimension from settings.
        """
        self.model = SentenceTransformer(EMBEDDING_MODEL["name"])
        self.embedding_dim = embedding_dim or EMBEDDING_MODEL["dimension"]
        
    def embed(self, doc: str) -> List[float]:
        """Embed a documentation string into a vector.
        
        Args:
            doc (str): Documentation string to embed
            
        Returns:
            List[float]: Embedding vector as a list
        """
        embedding = self.model.encode(doc)
        return embedding.tolist()
        
    def batch_embed(self, docs: List[str]) -> List[List[float]]:
        """Embed multiple documentation strings into vectors.
        
        Args:
            docs (List[str]): List of documentation strings to embed
            
        Returns:
            List[List[float]]: List of embedding vectors as lists
        """
        # Filter out invalid docs
        valid_docs = [doc for doc in docs if doc and isinstance(doc, str)]
        
        if not valid_docs:
            return []
            
        # Clean and truncate docs
        max_length = 2048
        processed_docs = [doc.strip()[:max_length] for doc in valid_docs]
                
        embeddings = self.model.encode(processed_docs)
        return [e.tolist() for e in embeddings]

    def _load_model(self, model_name: str):
        """Load the embedding model."""
        try:
            # Try to import sentence-transformers
            self.model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded embedding model: {model_name}")
            # Update embedding dimension based on model
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except ImportError:
            logger.warning("sentence-transformers not installed. Using fallback random embedding.")
        except Exception as e:
            logger.warning(f"Error loading model {model_name}: {e}")
    
    def _get_random_embedding(self) -> List[float]:
        """Generate a random embedding vector."""
        # Use a fixed seed to ensure repeatability
        rng = np.random.RandomState(42)
        return rng.normal(0, 0.1, self.embedding_dim).tolist()