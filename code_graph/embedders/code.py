"""Code embedder implementation."""
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from setting.embedding import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class CodeEmbedder:
    """Embeds code into vector space."""
    
    def __init__(self, model_name: str = None):
        """Initialize code embedder with optional model name."""
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
    
    def embed(self, code: str) -> List[float]:
        """Embed code into vector space.
        
        This method will use the context information if available to create
        a more meaningful embedding than just using the code snippet.
        
        Args:
            code: Code string to embed (can be full context or just a function/class definition)
            
        Returns:
            Vector representation of code
        """
        if not code or not isinstance(code, str):
            logger.warning(f"Invalid code to embed: {type(code)}")
            return self._get_random_embedding()
            
        # Clean up the code (remove excessive whitespace)
        code = code.strip()
        
        # If code is too long, truncate it
        max_length = 2048
        if len(code) > max_length:
            logger.debug(f"Truncating code from {len(code)} to {max_length} characters")
            code = code[:max_length]
            
        # Use the model if available
        if self.model:
            try:
                embedding = self.model.encode(code)
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
        
    def batch_embed(self, codes: List[str]) -> List[List[float]]:
        """Embed multiple code snippets at once.
        
        Args:
            codes: List of code strings to embed
            
        Returns:
            List of vector representations
        """
        # Filter out invalid codes
        valid_codes = [code for code in codes if code and isinstance(code, str)]
        
        if not valid_codes:
            return []
            
        # If model is available, use batch encoding
        if self.model:
            try:
                # Clean and truncate codes
                max_length = 2048
                processed_codes = [code.strip()[:max_length] for code in valid_codes]
                
                embeddings = self.model.encode(processed_codes)
                return embeddings.tolist()
            except Exception as e:
                logger.warning(f"Error batch encoding with model: {e}")
                return [self._get_random_embedding() for _ in valid_codes]
        else:
            # Fallback to individual random embeddings
            return [self._get_random_embedding() for _ in valid_codes] 