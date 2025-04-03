from datetime import datetime
import logging
import numpy as np
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.types import UserDefinedType
from setting.db import Base
from setting.embedding import EMBEDDING_MODEL

# Configure logging
logger = logging.getLogger(__name__)

class VECTOR(UserDefinedType):
    """Custom type for TiDB VECTOR data type."""
    
    def __init__(self, dimensions=None):
        self.dimensions = dimensions
    
    def get_col_spec(self, **kw):
        if self.dimensions is not None:
            return f"VECTOR({self.dimensions})"
        return "VECTOR"
        
    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
                
            # Log what we received for debugging
            logging.debug(f"VECTOR.bind_process received: {type(value)}")
            
            # Ensure value is properly formatted
            if isinstance(value, str):
                if not (value.startswith('[') and value.endswith(']')):
                    if self.dimensions is not None:
                        # Create a zero vector with the right dimension
                        zeros = [0.0] * self.dimensions
                        formatted = f"[{','.join(str(x) for x in zeros)}]"
                        logging.debug(f"VECTOR.bind_process converted string to zeros: {formatted[:30]}...")
                        return formatted
                    logging.debug("VECTOR.bind_process returning empty vector []")
                    return '[]'
                logging.debug(f"VECTOR.bind_process using existing vector string: {value[:30]}...")
                return value
            elif isinstance(value, (list, np.ndarray)):
                # Format as vector string
                formatted = f"[{','.join(str(float(x)) for x in value)}]"
                logging.debug(f"VECTOR.bind_process converted list/array to vector string: {formatted[:30]}...")
                return formatted
                
            logging.debug(f"VECTOR.bind_process returning unmodified value: {type(value)}")
            return value
        return process
        
    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
                
            logging.debug(f"VECTOR.result_processor received: {type(value)}")
            
            if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                try:
                    # Parse the vector values
                    values = value.strip('[]').split(',')
                    result = [float(x.strip()) for x in values if x.strip()]
                    logging.debug(f"VECTOR.result_processor parsed vector string to list: {len(result)} values")
                    return result
                except Exception as e:
                    logging.error(f"VECTOR.result_processor error parsing vector: {e}")
                    return value
                    
            logging.debug("VECTOR.result_processor returning unmodified value")
            return value
        return process

class Repository(Base):
    """Repository table model."""
    __tablename__ = 'repositories'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    path = Column(String(255), unique=True)
    name = Column(String(255))
    language = Column(String(50))
    last_indexed = Column(TIMESTAMP, default=datetime.now)
    nodes_count = Column(Integer, default=0)
    edges_count = Column(Integer, default=0)
    
    nodes = relationship("Node", back_populates="repository", cascade="all, delete-orphan")
    edges = relationship("Edge", back_populates="repository", cascade="all, delete-orphan")

class Node(Base):
    """Node table model."""
    __tablename__ = 'nodes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    repository_id = Column(Integer, ForeignKey('repositories.id'))
    node_id = Column(String(255))
    type = Column(String(50))
    name = Column(String(255))
    file_path = Column(String(1024))
    line_number = Column(Integer)
    code_context = Column(Text)  # Code content and context
    doc_context = Column(Text)   # Documentation and comments
    
    # Use TiDB vector data type for embeddings
    # Vector type with fixed dimension from settings
    code_embedding = Column(VECTOR(EMBEDDING_MODEL["dimension"]))
    doc_embedding = Column(VECTOR(EMBEDDING_MODEL["dimension"]))
    
    repository = relationship("Repository", back_populates="nodes")
    
    # Define indexes
    __table_args__ = (
        Index('idx_repository_type', repository_id, type),
        Index('idx_repository_name', repository_id, name),
        Index('idx_repository_node_id', repository_id, node_id, unique=True),
    )

class Edge(Base):
    """Edge table model."""
    __tablename__ = 'edges'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    repository_id = Column(Integer, ForeignKey('repositories.id'))
    source_id = Column(Integer)  # No ForeignKey constraint
    target_id = Column(Integer)  # No ForeignKey constraint
    relation_type = Column(String(50))
    label = Column(String(255))
    
    repository = relationship("Repository", back_populates="edges")
    
    # Define indexes
    __table_args__ = (
        Index('idx_repository_source', repository_id, source_id),
        Index('idx_repository_target', repository_id, target_id),
        Index('idx_repository_relation', repository_id, relation_type),
    ) 