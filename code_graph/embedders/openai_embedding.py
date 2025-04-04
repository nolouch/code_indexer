"""OpenAI embedder implementation."""
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from llm.embedding import get_text_embedding, batch_get_text_embedding

from setting.embedding import EMBEDDING_MODEL
from code_graph.config import get_config

logger = logging.getLogger(__name__)

class OpenAIEmbedder:
    """Base class for embedding using OpenAI's API."""
    
    def __init__(self, embedding_dim=None, model=None, batch_size=None):
        """Initialize OpenAI embedder.
        
        Args:
            embedding_dim (int, optional): Embedding dimension. If not provided,
                uses the dimension from settings.
            model (str, optional): OpenAI model to use. If not provided,
                uses the model from settings.
            batch_size (int, optional): Maximum number of embeddings to send in one API call.
                If not provided, uses the value from code_graph config.
        """
        self.model = model or EMBEDDING_MODEL.get("openai_model", "text-embedding-3-small")
        self.embedding_dim = embedding_dim or EMBEDDING_MODEL.get("openai_dimension", 1536)
        config = get_config()
        self.batch_size = batch_size or config.get("batch_size", 16)
        
    def embed(self, text: str) -> List[float]:
        """Embed text using OpenAI's API.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector as a list
        """
        try:
            return get_text_embedding(text, model=self.model)
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {e}")
            return self._get_random_embedding()
        
    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts using OpenAI's API in a single batch request.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors as lists
        """
        try:
            # Directly process all texts without validity checks
            if texts:
                return batch_get_text_embedding(texts, model=self.model, max_batch_size=self.batch_size)
            else:
                return []
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            # Return random embeddings as fallback
            return [self._get_random_embedding() for _ in range(len(texts))]
    
    def _get_random_embedding(self) -> List[float]:
        """Generate a random embedding vector."""
        # Use a fixed seed to ensure repeatability
        rng = np.random.RandomState(42)
        return rng.normal(0, 0.1, self.embedding_dim).tolist()


class OpenAICodeEmbedder(OpenAIEmbedder):
    """Embeds code snippets into vectors using OpenAI."""
    pass


class OpenAIDocEmbedder(OpenAIEmbedder):
    """Embeds documentation into vectors using OpenAI."""
    
    def embed(self, doc: str) -> List[float]:
        """Embed a documentation string into a vector.
        
        Args:
            doc (str): Documentation string to embed
            
        Returns:
            List[float]: Embedding vector as a list
        """
        # Clean and truncate doc
        max_length = 8192  # OpenAI has higher token limits
        cleaned_doc = doc.strip()[:max_length] if doc and isinstance(doc, str) else ""
        
        if not cleaned_doc:
            return self._get_random_embedding()
            
        return super().embed(cleaned_doc)
