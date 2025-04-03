"""Documentation embedder implementation."""
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from setting.embedding import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class DocEmbedder:
    """Embeds documentation into vector space."""
    
    def __init__(self, model_name: str = None):
        """Initialize doc embedder with optional model name."""
        self.model = SentenceTransformer(model_name or EMBEDDING_MODEL["name"])
        self.embedding_dim = EMBEDDING_MODEL["dimension"]  # Use dimension from config
        
        # Try to load embedding model if specified
        if model_name:
            try:
                self._load_model(model_name)
            except Exception as e:
                logger.warning(f"Failed to load embedding model {model_name}: {e}")
                logger.info("Using fallback random embedding")
    
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
    
    def embed(self, doc: str) -> List[float]:
        """Embed documentation into vector space.
        
        Args:
            doc: Documentation string to embed
            
        Returns:
            Vector representation of documentation
        """
        if not doc or not isinstance(doc, str):
            logger.warning(f"Invalid doc to embed: {type(doc)}")
            return self._get_random_embedding()
            
        # Clean up the doc (remove excessive whitespace)
        doc = doc.strip()
        
        # If doc is too long, truncate it
        max_length = 2048
        if len(doc) > max_length:
            logger.debug(f"Truncating doc from {len(doc)} to {max_length} characters")
            doc = doc[:max_length]
            
        # Use the model if available
        if self.model:
            try:
                embedding = self.model.encode(doc)
                # Convert to Python list for serialization
                return embedding.tolist()
            except Exception as e:
                logger.warning(f"Error encoding with model: {e}")
                return self._get_random_embedding()
        else:
            # Fallback to random embedding
            return self._get_random_embedding()
            
    def _get_random_embedding(self) -> List[float]:
        """Generate a random embedding vector."""
        # Use a fixed seed to ensure repeatability
        rng = np.random.RandomState(42)
        return rng.normal(0, 0.1, self.embedding_dim).tolist()
        
    def batch_embed(self, docs: List[str]) -> List[List[float]]:
        """Embed multiple documentation strings at once.
        
        Args:
            docs: List of documentation strings to embed
            
        Returns:
            List of vector representations
        """
        # Filter out invalid docs
        valid_docs = [doc for doc in docs if doc and isinstance(doc, str)]
        
        if not valid_docs:
            return []
            
        # If model is available, use batch encoding
        if self.model:
            try:
                # Clean and truncate docs
                max_length = 2048
                processed_docs = [doc.strip()[:max_length] for doc in valid_docs]
                
                embeddings = self.model.encode(processed_docs)
                return embeddings.tolist()
            except Exception as e:
                logger.warning(f"Error batch encoding with model: {e}")
                return [self._get_random_embedding() for _ in valid_docs]
        else:
            # Fallback to individual random embeddings
            return [self._get_random_embedding() for _ in valid_docs] 