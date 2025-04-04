import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import sys
import os

# Add the parent directory to the path to import the embedding module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.embedding import get_sentence_transformer

logger = logging.getLogger(__name__)

class CodeEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", dim=384):
        """Initialize the code embedder with a specific model
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            dim: Expected embedding dimension
        """
        self.model = get_sentence_transformer(model_name)
        self.dimension = dim
    
    def generate_embedding(self, code_text):
        """Generate embeddings for a given code text"""
        if not code_text or len(code_text.strip()) == 0:
            # Return zero vector for empty content
            return np.zeros(self.dimension)
            
        try:
            embedding = self.model.encode(code_text)
            
            # Ensure the embedding has the correct dimension
            if len(embedding) < self.dimension:
                # Pad with zeros if too short
                embedding = np.pad(embedding, (0, self.dimension - len(embedding)))
            elif len(embedding) > self.dimension:
                # Truncate if too long
                embedding = embedding[:self.dimension]
                
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector on error
            return np.zeros(self.dimension)
    
    def embedding_to_tidb_vector(self, embedding):
        """Convert numpy array embeddings to TiDB vector string format
        
        Args:
            embedding: A numpy array or list of embedding values
            
        Returns:
            A string in TiDB vector format: "[val1,val2,...]"
        """
        if embedding is None:
            # Return zero vector
            zeros = [0.0] * self.dimension
            return f"[{','.join(str(x) for x in zeros)}]"
            
        if isinstance(embedding, str):
            # If already formatted as a vector string, return as is
            if embedding.startswith('[') and embedding.endswith(']'):
                return embedding
            # Otherwise return zero vector
            zeros = [0.0] * self.dimension
            return f"[{','.join(str(x) for x in zeros)}]"
            
        # Format numpy array or list as vector string
        if isinstance(embedding, (list, np.ndarray)):
            return f"[{','.join(str(float(x)) for x in embedding)}]"
            
        # Default fallback
        zeros = [0.0] * self.dimension
        return f"[{','.join(str(x) for x in zeros)}]"
    
    def tidb_vector_to_embedding(self, vector_str):
        """Convert TiDB vector string format back to numpy array
        
        Args:
            vector_str: A string in TiDB vector format: "[val1,val2,...]"
            
        Returns:
            numpy array of embedding values
        """
        if not vector_str or not isinstance(vector_str, str):
            return np.zeros(self.dimension)
            
        if vector_str.startswith('[') and vector_str.endswith(']'):
            try:
                # Parse the vector values
                values = vector_str.strip('[]').split(',')
                result = np.array([float(x.strip()) for x in values if x.strip()])
                
                # Ensure correct dimension
                if len(result) < self.dimension:
                    result = np.pad(result, (0, self.dimension - len(result)))
                elif len(result) > self.dimension:
                    result = result[:self.dimension]
                    
                return result
            except Exception as e:
                logger.error(f"Error parsing vector string: {e}")
                return np.zeros(self.dimension)
                
        return np.zeros(self.dimension) 