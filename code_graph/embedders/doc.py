"""Documentation embedder implementation."""
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from setting.embedding import EMBEDDING_MODEL
from code_graph.config import get_config
from llm.embedding import get_sentence_transformer

logger = logging.getLogger(__name__)

class DocEmbedder:
    """Embeds documentation into vectors."""
    
    def __init__(self, embedding_dim=None, batch_size=None):
        """Initialize documentation embedder.
        
        Args:
            embedding_dim (int, optional): Embedding dimension. If not provided,
                uses the dimension from settings.
            batch_size (int, optional): Maximum number of embeddings to process in one batch.
                If not provided, uses the value from code_graph config.
        """
        self.model = get_sentence_transformer(EMBEDDING_MODEL["name"])
        self.embedding_dim = embedding_dim or EMBEDDING_MODEL["dimension"]
        config = get_config()
        self.batch_size = batch_size or config.get("batch_size", 32)
        
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
        if not docs:
            return []
        
        # Clean and truncate docs if needed, but don't filter
        max_length = 2048
        processed_docs = [doc.strip()[:max_length] if doc and isinstance(doc, str) else "" for doc in docs]
            
        try:
            # Process docs in batches without validity checks
            results = []
            
            # Calculate total number of batches
            total_batches = (len(processed_docs) + self.batch_size - 1) // self.batch_size
            
            # Process batches with progress bar
            for i in tqdm(range(0, len(processed_docs), self.batch_size), 
                         total=total_batches, 
                         desc=f"Batches (size={self.batch_size})", 
                         unit="batch"):
                batch = processed_docs[i:i + self.batch_size]
                batch_embeddings = self.model.encode(batch)
                results.extend([e.tolist() for e in batch_embeddings])
            
            return results
        except Exception as e:
            logger.error(f"Error in batch doc embedding: {e}")
            # Return random embeddings as fallback
            return [self._get_random_embedding() for _ in range(len(docs))]

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