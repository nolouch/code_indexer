"""OpenAI embedder implementation."""
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from llm.embedding import get_text_embedding

from setting.embedding import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class OpenAIEmbedder:
    """Base class for embedding using OpenAI's API."""
    
    def __init__(self, embedding_dim=None, model=None):
        """Initialize OpenAI embedder.
        
        Args:
            embedding_dim (int, optional): Embedding dimension. If not provided,
                uses the dimension from settings.
            model (str, optional): OpenAI model to use. If not provided,
                uses the model from settings.
        """
        self.model = model or EMBEDDING_MODEL.get("openai_model", "text-embedding-3-small")
        self.embedding_dim = embedding_dim or EMBEDDING_MODEL.get("openai_dimension", 1536)
        
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
        """Embed multiple texts using OpenAI's API.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors as lists
        """
        results = []
        for text in texts:
            if not text or not isinstance(text, str):
                results.append(self._get_random_embedding())
                continue
            try:
                results.append(self.embed(text))
            except Exception as e:
                logger.error(f"Error getting OpenAI batch embedding: {e}")
                results.append(self._get_random_embedding())
        return results
    
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
