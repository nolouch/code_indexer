from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from sqlalchemy import create_engine, Column, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import text
import networkx as nx

from .models import Base, Node, Repository, GraphMetadata, node_relationships

Base = declarative_base()

class GraphData(Base):
    """Graph data table."""
    __tablename__ = 'graphs'
    
    repo_path = Column(String, primary_key=True)
    nodes = Column(JSON)
    edges = Column(JSON)

class GraphDBManager:
    """Manages graph persistence in database."""
    
    def __init__(self, db_url: str):
        """Initialize database connection.
        
        Args:
            db_url: Database connection URL
        """
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
    def save_graph(self, graph: nx.MultiDiGraph, repo_path: str):
        """Save graph to database.
        
        Args:
            graph: Graph to save
            repo_path: Repository path as identifier
        """
        session = self.Session()
        
        # Convert graph to JSON-serializable format
        data = GraphData(
            repo_path=repo_path,
            nodes=dict(graph.nodes(data=True)),
            edges=[
                {
                    "source": u,
                    "target": v,
                    "key": k,
                    "data": d
                }
                for u, v, k, d in graph.edges(data=True, keys=True)
            ]
        )
        
        # Save to database
        session.merge(data)
        session.commit()
        session.close()
        
    def load_graph(self, repo_path: str) -> nx.MultiDiGraph:
        """Load graph from database.
        
        Args:
            repo_path: Repository path as identifier
            
        Returns:
            Loaded graph or None if not found
        """
        session = self.Session()
        data = session.query(GraphData).filter_by(repo_path=repo_path).first()
        session.close()
        
        if not data:
            return None
            
        # Create new graph
        graph = nx.MultiDiGraph()
        
        # Add nodes
        for node, attrs in data.nodes.items():
            graph.add_node(node, **attrs)
            
        # Add edges
        for edge in data.edges:
            graph.add_edge(
                edge["source"],
                edge["target"],
                key=edge["key"],
                **edge["data"]
            )
            
        return graph
        
    def _get_or_create_repository(self, session: Session, path: str) -> Repository:
        """Get existing repository or create new one"""
        repo = session.query(Repository).filter_by(path=path).first()
        if not repo:
            repo = Repository(
                path=path,
                name=path.split('/')[-1],
                last_indexed=datetime.now().isoformat(),
                repo_data={}
            )
            session.add(repo)
            session.flush()
        return repo
        
    def get_repository_stats(self, repository_path: str) -> Dict[str, Any]:
        """Get statistics about the stored graph
        
        Args:
            repository_path: Path to the code repository
            
        Returns:
            Dictionary containing graph statistics
        """
        session = self.Session()
        try:
            repo = session.query(Repository).filter_by(path=repository_path).first()
            if not repo:
                return {}
                
            metadata = session.query(GraphMetadata).filter_by(repository_id=repo.id).first()
            if not metadata:
                return {}
                
            return {
                'repository': repo.name,
                'nodes': metadata.node_count,
                'edges': metadata.edge_count,
                'last_updated': metadata.last_updated,
                'embedder_config': metadata.embedder_config
            }
            
        finally:
            session.close()