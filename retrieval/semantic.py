import numpy as np
from typing import List, Dict, Any
import networkx as nx
from sentence_transformers import SentenceTransformer
from setting.embedding import EMBEDDING_MODEL
from llm.embedding import get_sentence_transformer

class SemanticRetriever:
    """Semantic code search using embeddings and graph structure"""
    
    def __init__(self):
        self.model = get_sentence_transformer(EMBEDDING_MODEL["name"])
        self.graph = None
        
    def setup(self, graph: nx.MultiDiGraph):
        """Initialize with semantic graph"""
        self.graph = graph
        
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for code elements relevant to query"""
        if not self.graph:
            raise ValueError("Retriever not initialized with graph")
            
        # Get query embedding
        query_embedding = self.model.encode(query)
        
        # Score nodes by semantic similarity
        scores = []
        for node, data in self.graph.nodes(data=True):
            score = 0.0
            
            # Code similarity
            if "code_embedding" in data:
                code_sim = self._cosine_similarity(
                    query_embedding,
                    data["code_embedding"]
                )
                score += code_sim
                
            # Doc similarity
            if "doc_embedding" in data:
                doc_sim = self._cosine_similarity(
                    query_embedding,
                    data["doc_embedding"]
                )
                score += doc_sim
                
            scores.append((node, score))
            
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k results
        results = []
        for node, score in scores[:k]:
            data = self.graph.nodes[node]
            result = {
                "node": node,
                "type": data["type"],
                "file": data["file"],
                "line": data.get("line", 1),
                "score": score
            }
            if "docstring" in data:
                result["docstring"] = data["docstring"]
            results.append(result)
            
        return results
        
    def expand(self, node: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Expand search results using graph structure"""
        if not self.graph or node not in self.graph:
            return []
            
        # Get subgraph within max_depth
        nodes = set()
        current = {node}
        
        for _ in range(max_depth):
            next_nodes = set()
            for n in current:
                # Add neighbors
                neighbors = set(self.graph.predecessors(n))
                neighbors.update(self.graph.successors(n))
                next_nodes.update(neighbors)
            nodes.update(current)
            current = next_nodes - nodes
            
        # Convert nodes to results
        results = []
        for n in nodes:
            data = self.graph.nodes[n]
            result = {
                "node": n,
                "type": data["type"],
                "file": data["file"],
                "line": data.get("line", 1)
            }
            if "docstring" in data:
                result["docstring"] = data["docstring"]
            results.append(result)
            
        return results
        
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))