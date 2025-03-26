from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import text
import networkx as nx

from .models import Base, Node, Repository, GraphMetadata, node_relationships

class GraphDBManager:
    """Manages persistence of semantic graphs in MySQL database"""
    
    def __init__(self, connection_url: str):
        """Initialize database connection
        
        Args:
            connection_url: SQLAlchemy connection URL for MySQL
                          e.g. 'mysql+pymysql://user:pass@localhost/dbname'
        """
        self.engine = create_engine(connection_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
    def save_graph(self, graph: nx.MultiDiGraph, repository_path: str) -> None:
        """Save semantic graph to database
        
        Args:
            graph: NetworkX graph to persist
            repository_path: Path to the code repository
        """
        session = self.Session()
        try:
            # Create or get repository
            repo = self._get_or_create_repository(session, repository_path)
            
            # Save nodes
            node_map = {}  # Maps node names to DB IDs
            for node_name, data in graph.nodes(data=True):
                node = Node(
                    name=node_name,
                    type=data.get('type'),
                    language=data.get('language'),
                    file_path=data.get('file'),
                    line_number=data.get('line'),
                    docstring=data.get('docstring'),
                    code_content=data.get('code_content'),
                    code_embedding=data.get('code_embedding'),
                    doc_embedding=data.get('doc_embedding'),
                    metadata=data
                )
                session.add(node)
                session.flush()  # Get node ID
                node_map[node_name] = node.id
                
            # Save edges
            for source, target, data in graph.edges(data=True):
                session.execute(
                    node_relationships.insert().values(
                        source_id=node_map[source],
                        target_id=node_map[target],
                        relationship_type=data.get('type'),
                        weight=data.get('weight', 1.0),
                        metadata=data
                    )
                )
                
            # Update graph metadata
            metadata = GraphMetadata(
                repository_id=repo.id,
                node_count=graph.number_of_nodes(),
                edge_count=graph.number_of_edges(),
                last_updated=datetime.now().isoformat(),
                embedder_config={
                    'code_embedder': 'sentence-transformers',
                    'doc_embedder': 'sentence-transformers'
                },
                metadata={}
            )
            session.add(metadata)
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
            
    def load_graph(self, repository_path: str) -> Optional[nx.MultiDiGraph]:
        """Load semantic graph from database
        
        Args:
            repository_path: Path to the code repository
            
        Returns:
            NetworkX graph if found, None otherwise
        """
        session = self.Session()
        try:
            # Get repository
            repo = session.query(Repository).filter_by(path=repository_path).first()
            if not repo:
                return None
                
            # Create new graph
            graph = nx.MultiDiGraph()
            
            # Load nodes
            nodes = session.query(Node).all()
            for node in nodes:
                graph.add_node(
                    node.name,
                    type=node.type,
                    language=node.language,
                    file=node.file_path,
                    line=node.line_number,
                    docstring=node.docstring,
                    code_content=node.code_content,
                    code_embedding=node.code_embedding,
                    doc_embedding=node.doc_embedding,
                    **node.metadata
                )
                
            # Load edges
            edges = session.execute(text('SELECT * FROM node_relationships')).fetchall()
            node_id_map = {n.id: n.name for n in nodes}
            
            for edge in edges:
                source = node_id_map[edge.source_id]
                target = node_id_map[edge.target_id]
                graph.add_edge(
                    source,
                    target,
                    type=edge.relationship_type,
                    weight=edge.weight,
                    **edge.metadata
                )
                
            return graph
            
        except Exception as e:
            print(f"Error loading graph: {e}")
            return None
        finally:
            session.close()
            
    def _get_or_create_repository(self, session: Session, path: str) -> Repository:
        """Get existing repository or create new one"""
        repo = session.query(Repository).filter_by(path=path).first()
        if not repo:
            repo = Repository(
                path=path,
                name=path.split('/')[-1],
                last_indexed=datetime.now().isoformat(),
                metadata={}
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