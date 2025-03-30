from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import networkx as nx
import os
import configparser
from pathlib import Path
import numpy as np  # For vector handling
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, ForeignKey, Index, func, text
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import relationship, Session
import pymysql
import logging

from setting.db import SessionLocal, Base, engine
from setting.base import DATABASE_URI

# Configure logging
logger = logging.getLogger(__name__)

# Define custom VECTOR type for TiDB
from sqlalchemy.types import TypeDecorator, UserDefinedType

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

# Define SQLAlchemy models
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
    file_path = Column(String(255))
    line_number = Column(Integer)
    context = Column(Text)
    
    # Use TiDB vector data type for embeddings
    # Vector type with fixed dimension
    code_embedding = Column(VECTOR(384))
    doc_embedding = Column(VECTOR(384))
    
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

class GraphDBManager:
    """Manages graph persistence in database."""
    
    # Embedding dimensions for the vector embeddings
    CODE_EMBEDDING_DIM = 384
    DOC_EMBEDDING_DIM = 384
    
    def __init__(self, db_url=None):
        """Initialize database manager.
        
        Args:
            db_url (str, optional): Database URL to connect to.
        """
        self.db_url = db_url or DATABASE_URI
        self.available = self.db_url is not None and engine is not None
        
        if self.available:
            logger.info(f"Database support is available with URL: {self.db_url}")
            # Create the tables if they don't exist yet
            try:
                self._setup_tables()
            except Exception as e:
                logger.error(f"Failed to setup database tables: {e}")
                self.available = False
        else:
            logger.warning("Database support is not available - operating in JSON-only mode")
    
    def _setup_tables(self):
        """Create tables if they don't exist."""
        if not self.available or engine is None:
            logger.warning("Cannot setup tables - database is not available")
            return
            
        # SQLAlchemy will create tables through Base.metadata
        Base.metadata.create_all(bind=engine)
        
        # Log tables being created
        logger.info(f"Tables created: {list(Base.metadata.tables.keys())}")
        
        # 检测数据库类型
        db_type = "unknown"
        if self.db_url:
            if 'tidb' in self.db_url:
                db_type = "tidb"
            elif 'mysql' in self.db_url:
                db_type = "mysql"
            elif 'sqlite' in self.db_url:
                db_type = "sqlite"
                
        # 只有 TiDB 和部分 MySQL 支持向量索引
        if db_type in ["tidb", "mysql"]:
            with SessionLocal() as session:
                try:
                    # 首先确保表有 TiFlash 副本，向量索引功能需要 TiFlash
                    session.execute(text("""
                        ALTER TABLE nodes SET TIFLASH REPLICA 1
                    """))
                    logger.info("Created TiFlash replica for nodes table")
                except Exception as e:
                    logger.warning(f"Creating TiFlash replica failed, vector index may not work: {e}")
                
                try:
                    # 使用正确的 TiDB 向量索引语法创建 code_embedding 的向量索引
                    session.execute(text("""
                        CREATE VECTOR INDEX IF NOT EXISTS idx_code_vector_cosine 
                        ON nodes ((VEC_COSINE_DISTANCE(code_embedding))) USING HNSW
                    """))
                    logger.info("Created code vector cosine index")
                    
                    session.execute(text("""
                        CREATE VECTOR INDEX IF NOT EXISTS idx_code_vector_l2 
                        ON nodes ((VEC_L2_DISTANCE(code_embedding))) USING HNSW
                    """))
                    logger.info("Created code vector L2 index")
                except Exception as e:
                    logger.warning(f"Code vector index creation failed: {e}")
                    
                try:
                    # 使用正确的 TiDB 向量索引语法创建 doc_embedding 的向量索引
                    session.execute(text("""
                        CREATE VECTOR INDEX IF NOT EXISTS idx_doc_vector_cosine 
                        ON nodes ((VEC_COSINE_DISTANCE(doc_embedding))) USING HNSW
                    """))
                    logger.info("Created doc vector cosine index")
                    
                    session.execute(text("""
                        CREATE VECTOR INDEX IF NOT EXISTS idx_doc_vector_l2
                        ON nodes ((VEC_L2_DISTANCE(doc_embedding))) USING HNSW
                    """))
                    logger.info("Created doc vector L2 index")
                except Exception as e:
                    logger.warning(f"Doc vector index creation failed: {e}")
        else:
            logger.warning(f"Vector indexes not supported for database type: {db_type}")
    
    def _prepare_vector_embedding(self, embedding, dim):
        """Prepare a vector embedding for database storage by ensuring correct format and dimension.
        
        Args:
            embedding: The embedding data (numpy array, list, string, or None)
            dim: Required dimension for the embedding
            
        Returns:
            Formatted vector string representation for TiDB VECTOR type
        """
        # Handle None or empty embedding
        if embedding is None or (isinstance(embedding, (list, np.ndarray)) and len(embedding) == 0):
            zeros = [0.0] * dim
            return f"[{','.join(str(x) for x in zeros)}]"
            
        # If already a string, check format
        if isinstance(embedding, str):
            # If it doesn't look like a vector, return a zero vector
            if not (embedding.startswith('[') and embedding.endswith(']')):
                zeros = [0.0] * dim
                return f"[{','.join(str(x) for x in zeros)}]"
            return embedding
                
        # Handle list or numpy array type
        if isinstance(embedding, (list, np.ndarray)):
            # Ensure correct dimension
            if len(embedding) < dim:
                # Pad with zeros
                if isinstance(embedding, np.ndarray):
                    embedding = np.pad(embedding, (0, dim - len(embedding)))
                else:
                    embedding = embedding + [0.0] * (dim - len(embedding))
            elif len(embedding) > dim:
                # Truncate
                embedding = embedding[:dim]
                
            # Convert to string format for TiDB VECTOR type
            return f"[{','.join(str(float(x)) for x in embedding)}]"
            
        # If we get here, return a zero vector
        zeros = [0.0] * dim
        return f"[{','.join(str(x) for x in zeros)}]"

    def save_graph(self, graph: nx.MultiDiGraph, repo_path: str):
        """Save graph to database with minimal schema.
        
        Args:
            graph: Graph to save
            repo_path: Repository path as identifier
        """
        with SessionLocal() as session:
            # Get or create repository
            repo = self._get_or_create_repository(session, repo_path)
            repo_id = repo.id
            
            # Determine language (if available in graph)
            language = "unknown"
            for _, attrs in graph.nodes(data=True):
                if "language" in attrs:
                    language = attrs["language"]
                    break
                    
            # Update repository language
            repo.language = language
            
            # Clear existing data
            self._clear_repository_data(session, repo_id)
            
            # Process nodes
            nodes_processed = 0
            nodes_skipped = 0
            
            # Create mapping from graph node ID to database ID
            node_id_to_db_id = {}
            
            for node_id, attrs in graph.nodes(data=True):
                try:
                    node_str_id = str(node_id)
                    node_type = attrs.get('type', 'unknown')
                    name = attrs.get('name', str(node_id))
                    file_path = attrs.get('file', '')
                    line_number = attrs.get('line', 0)
                    
                    # Get embeddings
                    code_embedding = None
                    doc_embedding = None
                    
                    if 'code_embedding' in attrs:
                        code_embedding = attrs['code_embedding']
                    if 'doc_embedding' in attrs:
                        doc_embedding = attrs['doc_embedding']
                    
                    # If only 'embedding' exists, use it for both
                    if 'embedding' in attrs and not (code_embedding or doc_embedding):
                        code_embedding = attrs['embedding']
                        doc_embedding = attrs['embedding']
                    
                    # Create default embeddings if none exists
                    if code_embedding is None:
                        code_embedding = [0.0] * self.CODE_EMBEDDING_DIM
                    if doc_embedding is None:
                        doc_embedding = [0.0] * self.DOC_EMBEDDING_DIM
                    
                    # Store all context data as a single JSON string
                    # Remove embeddings which are too large
                    context_data = {k: v for k, v in attrs.items() 
                                   if k not in ['code_embedding', 'doc_embedding', 'embedding']}
                    
                    # Convert context to JSON string
                    try:
                        context_str = json.dumps(context_data)
                    except Exception as e:
                        logger.warning(f"Error serializing context for node {node_str_id}: {e}")
                        # If JSON serialization fails, create a simplified version
                        simplified_context = {
                            'type': node_type,
                            'name': name,
                            'file': file_path,
                            'line': line_number
                        }
                        context_str = json.dumps(simplified_context)
                    
                    # Check if node already exists (by repository_id and node_id)
                    existing_node = session.query(Node).filter(
                        Node.repository_id == repo_id,
                        Node.node_id == node_str_id
                    ).first()
                    
                    if existing_node:
                        # Update existing node
                        existing_node.type = node_type
                        existing_node.name = name
                        existing_node.file_path = file_path
                        existing_node.line_number = line_number
                        existing_node.context = context_str
                        existing_node.code_embedding = code_embedding
                        existing_node.doc_embedding = doc_embedding
                        
                        # Store node ID mapping
                        node_id_to_db_id[node_str_id] = existing_node.id
                    else:
                        # Insert new node
                        new_node = Node(
                            repository_id=repo_id,
                            node_id=node_str_id,
                            type=node_type,
                            name=name,
                            file_path=file_path,
                            line_number=line_number,
                            context=context_str,
                            code_embedding=code_embedding,
                            doc_embedding=doc_embedding
                        )
                        session.add(new_node)
                        session.flush()  # Flush session to get ID of newly inserted node
                        
                        # Store node ID mapping
                        node_id_to_db_id[node_str_id] = new_node.id
                        
                    nodes_processed += 1
                    
                    # Commit in batches to avoid memory issues with large graphs
                    if nodes_processed % 1000 == 0:
                        session.flush()
                        
                except Exception as e:
                    logger.error(f"Error saving node {str(node_id)}: {e}")
                    nodes_skipped += 1
                    continue
            
            # Process edges (relationships)
            edges_processed = 0
            edges_skipped = 0
            
            for source, target, data in graph.edges(data=True):
                try:
                    source_str = str(source)
                    target_str = str(target)
                    relation_type = data.get('type', 'unknown')
                    label = data.get('label', '')
                    
                    # Check if source and target nodes exist in the mapping
                    if source_str not in node_id_to_db_id:
                        logger.warning(f"Source node {source_str} not in node_id_to_db_id map, skipping edge")
                        edges_skipped += 1
                        continue
                        
                    if target_str not in node_id_to_db_id:
                        logger.warning(f"Target node {target_str} not in node_id_to_db_id map, skipping edge")
                        edges_skipped += 1
                        continue
                    
                    # Use mapped database IDs
                    new_edge = Edge(
                        repository_id=repo_id,
                        source_id=node_id_to_db_id[source_str],
                        target_id=node_id_to_db_id[target_str],
                        relation_type=relation_type,
                        label=label
                    )
                    session.add(new_edge)
                    edges_processed += 1
                    
                    # Commit in batches
                    if edges_processed % 5000 == 0:
                        session.flush()
                        
                except Exception as e:
                    logger.error(f"Error saving edge {str(source)} -> {str(target)}: {e}")
                    edges_skipped += 1
                    continue
            
            # Update node and edge counts
            repo.nodes_count = nodes_processed
            repo.edges_count = edges_processed
            repo.last_indexed = datetime.now()
            
            # Commit all changes
            session.commit()
            
            logger.info(f"Graph saved: {nodes_processed} nodes ({nodes_skipped} skipped), "
                       f"{edges_processed} edges ({edges_skipped} skipped)")
        
    def _clear_repository_data(self, session: Session, repository_id: int):
        """Clear existing data for repository"""
        # Delete edges first
        session.query(Edge).filter(Edge.repository_id == repository_id).delete()
        
        # Delete nodes
        session.query(Node).filter(Node.repository_id == repository_id).delete()
        
    def load_graph(self, repo_path: str) -> nx.MultiDiGraph:
        """Load graph from database.
        
        Args:
            repo_path: Repository path as identifier
            
        Returns:
            Loaded graph or None if not found
        """
        try:
            with SessionLocal() as session:
                # Get repository ID
                repo = session.query(Repository).filter(Repository.path == repo_path).first()
                
                if not repo:
                    logger.warning(f"Repository not found: {repo_path}")
                    return None
                    
                # Create new graph
                graph = nx.MultiDiGraph()
                
                # Add repository information to graph first, before processing nodes
                graph.graph['repository'] = repo_path
                graph.graph['name'] = repo.name
                graph.graph['language'] = repo.language
                
                # Load all nodes with attributes
                nodes = session.query(Node).filter(Node.repository_id == repo.id).all()
                
                if not nodes:
                    logger.warning(f"No nodes found for repository: {repo_path}")
                    # Still return the graph with repository info even if no nodes
                    return graph
                
                # Create a mapping from DB id to node_id for edge lookup
                node_db_id_to_graph_id = {}
                    
                # Process each node
                for node in nodes:
                    try:
                        # Get node ID
                        node_id = node.node_id
                        # Store mapping from DB id to graph node_id
                        node_db_id_to_graph_id[node.id] = node_id
                        
                        # Create basic attributes dictionary
                        attrs = {
                            'type': node.type,
                            'name': node.name,
                            'file': node.file_path,
                            'line': node.line_number
                        }
                        
                        # Parse and add context data
                        if node.context:
                            try:
                                context_data = json.loads(node.context)
                                attrs.update(context_data)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse context for node {node_id}: {e}")
                        
                        # Add embeddings to attributes if available
                        if node.code_embedding:
                            # If the VECTOR type already converted to a list, use it directly
                            if isinstance(node.code_embedding, list):
                                attrs["code_embedding"] = np.array(node.code_embedding)
                            # If it's still a string, parse it
                            elif isinstance(node.code_embedding, str):
                                try:
                                    # Strip brackets and split by commas
                                    embedding_str = node.code_embedding.strip('[]')
                                    values = [float(x.strip()) for x in embedding_str.split(',') if x.strip()]
                                    attrs["code_embedding"] = np.array(values)
                                except Exception as e:
                                    logger.warning(f"Failed to parse code embedding for node {node_id}: {e}")
                        
                        if node.doc_embedding:
                            # If the VECTOR type already converted to a list, use it directly
                            if isinstance(node.doc_embedding, list):
                                attrs["doc_embedding"] = np.array(node.doc_embedding)
                            # If it's still a string, parse it
                            elif isinstance(node.doc_embedding, str):
                                try:
                                    # Strip brackets and split by commas
                                    embedding_str = node.doc_embedding.strip('[]')
                                    values = [float(x.strip()) for x in embedding_str.split(',') if x.strip()]
                                    attrs["doc_embedding"] = np.array(values)
                                except Exception as e:
                                    logger.warning(f"Failed to parse doc embedding for node {node_id}: {e}")
                        
                        # Add node to graph
                        graph.add_node(node_id, **attrs)
                    except Exception as e:
                        logger.error(f"Error processing node {node.node_id}: {e}")
                        continue
                
                # Load all edges with attributes using a join to get the correct node IDs
                # Use a SQL query that joins the edges with the nodes based on ID values
                edge_query = """
                    SELECT e.id, e.relation_type, e.label, 
                           src.node_id as source_node_id, 
                           tgt.node_id as target_node_id
                    FROM edges e
                    JOIN nodes src ON e.source_id = src.id
                    JOIN nodes tgt ON e.target_id = tgt.id
                    WHERE e.repository_id = :repo_id
                """
                
                edges = session.execute(text(edge_query), {"repo_id": repo.id}).fetchall()
                
                # Process each edge
                for edge in edges:
                    try:
                        # Get the graph node IDs from the lookup
                        source_node_id = edge.source_node_id
                        target_node_id = edge.target_node_id
                        
                        # Create attributes dictionary
                        attrs = {
                            'type': edge.relation_type,
                            'label': edge.label if edge.label else ""
                        }
                        
                        # Add edge to graph
                        graph.add_edge(source_node_id, target_node_id, **attrs)
                    except Exception as e:
                        logger.error(f"Error processing edge {edge.source_id} -> {edge.target_id}: {e}")
                        continue
                
                logger.info(f"Loaded graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
                return graph
        except Exception as e:
            logger.error(f"Error loading graph for repository {repo_path}: {e}")
            return None
        
    def semantic_search(self, repo_path: str, query_embedding: List[float], limit: int = 10, use_doc_embedding: bool = False) -> List[Dict[str, Any]]:
        """Search for nodes semantically similar to a query embedding.
        
        Args:
            repo_path: Path of the repository to search in
            query_embedding: Vector embedding to search for
            limit: Maximum number of results to return
            use_doc_embedding: Whether to use the doc embedding instead of code embedding
            
        Returns:
            List of matching nodes with similarity scores
        """
        if not self.available:
            logger.warning("Cannot perform semantic search - database is not available")
            return []
            
        try:
            # Convert to absolute path
            abs_path = str(Path(repo_path).resolve())
            
            # Use _get_repository method instead of _find_repository
            repo = self._get_repository(abs_path)
            
            if not repo:
                logger.warning(f"Repository not found in database: {repo_path}")
                return []
                
            with SessionLocal() as session:
                # Determine which embedding to use
                embedding_field = "doc_embedding" if use_doc_embedding else "code_embedding"
                
                # Ensure query embedding has the right dimension
                if use_doc_embedding:
                    query_embedding = self._normalize_vector_size(query_embedding, self.DOC_EMBEDDING_DIM)
                else:
                    query_embedding = self._normalize_vector_size(query_embedding, self.CODE_EMBEDDING_DIM)
                    
                # Format the query vector for TiDB
                # If already a string in the correct format, use it as is
                if isinstance(query_embedding, str) and query_embedding.startswith('[') and query_embedding.endswith(']'):
                    query_str = query_embedding
                else:
                    query_str = f"[{','.join(str(float(x)) for x in query_embedding)}]"
                
                logger.debug(f"Query vector (first 30 chars): {query_str[:30]}...")
                logger.debug(f"Vector dimension: {len(query_embedding) if not isinstance(query_embedding, str) else 'unknown (string)'}")
                
                # Detect database type
                db_type = "unknown"
                if self.db_url:
                    if 'tidb' in self.db_url:
                        db_type = "tidb"
                    elif 'mysql' in self.db_url:
                        db_type = "mysql"
                    elif 'sqlite' in self.db_url:
                        db_type = "sqlite"
                
                try:
                    # Build query based on database type
                    if db_type in ["tidb", "mysql"]:
                        # Use TiDB vector search syntax
                        # According to documentation, use vector distance function in ORDER BY clause
                        query_sql = f"""
                            SELECT 
                                id, node_id, type, name, file_path, line_number, context
                            FROM 
                                nodes
                            WHERE 
                                repository_id = :repo_id
                            ORDER BY 
                                VEC_COSINE_DISTANCE({embedding_field}, :query_vec) ASC
                            LIMIT :limit
                        """
                    else:
                        # SQLite or other databases without vector search support, fetch basic info
                        query_sql = """
                            SELECT 
                                id, node_id, type, name, file_path, line_number, context
                            FROM 
                                nodes
                            WHERE 
                                repository_id = :repo_id
                            LIMIT :limit
                        """
                    
                    logger.debug(f"Executing vector search with SQL: {query_sql}")
                    logger.debug(f"Parameters: repo_id={repo.id}, limit={limit}")
                    
                    # Execute query
                    results = session.execute(
                        text(query_sql), 
                        {
                            "repo_id": repo.id, 
                            "query_vec": query_str if db_type in ["tidb", "mysql"] else None,
                            "limit": limit
                        }
                    ).fetchall()
                    
                    logger.debug(f"Vector search returned {len(results) if results else 0} results")
                    
                    # Process results
                    processed_results = []
                    for row in results:
                        # For TiDB, query results are sorted by similarity but need to calculate similarity value
                        # For other databases, set a simulated similarity value
                        if db_type in ["tidb", "mysql"]:
                            similarity = max(0.0, min(1.0, 0.95 - 0.05 * len(processed_results)))  # Simulated decreasing similarity
                        else:
                            similarity = 0.5  # Default similarity
                        
                        # Parse context JSON if available
                        context = {}
                        if row.context:
                            try:
                                context = json.loads(row.context)
                            except json.JSONDecodeError:
                                context = {"error": "Invalid JSON in context"}
                        
                        # Combine all data
                        node_data = {
                            "id": row.node_id,
                            "name": row.name,
                            "type": row.type,
                            "file_path": row.file_path,
                            "line": row.line_number if hasattr(row, 'line_number') else 0,
                            "similarity": similarity,
                        }
                        
                        # Add context data
                        for key, value in context.items():
                            if key not in node_data:
                                node_data[key] = value
                                
                        processed_results.append(node_data)
                        
                    logger.info(f"Found {len(processed_results)} similar nodes")
                    return processed_results
                    
                except Exception as e:
                    logger.error(f"Error executing vector search query: {e}")
                    import traceback
                    logger.error(f"Error traceback: {traceback.format_exc()}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            import traceback
            logger.error(f"Error traceback: {traceback.format_exc()}")
            return []
                
    def _normalize_vector_size(self, vector, target_size):
        """Normalize vector to the target size by padding or truncating."""
        if len(vector) < target_size:
            # Pad with zeros
            if isinstance(vector, np.ndarray):
                return np.pad(vector, (0, target_size - len(vector)))
            else:
                return vector + [0.0] * (target_size - len(vector))
        elif len(vector) > target_size:
            # Truncate
            return vector[:target_size]
        return vector
        
    def vector_search(self, repo_path: str, query: str, limit: int = 10, use_doc_embedding: bool = False) -> List[Dict[str, Any]]:
        """Perform a vector search using the query string.
        
        Args:
            repo_path: Path of the repository to search in
            query: The text query to search for
            limit: Maximum number of results to return
            use_doc_embedding: Whether to use doc embeddings instead of code embeddings
            
        Returns:
            List of search results with similarity scores
        """
        if not self.available:
            logger.warning("Cannot perform vector search - database is not available")
            return []
        
        # Generate embedding for the query
        from sentence_transformers import SentenceTransformer
        try:
            # Use sentence-transformers to generate embedding
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = model.encode(query)
            
            # Convert to absolute path
            abs_path = str(Path(repo_path).resolve())
            
            # Perform semantic search with the generated embedding
            return self.semantic_search(abs_path, query_embedding, limit, use_doc_embedding)
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return []

    def _get_or_create_repository(self, session: Session, path: str) -> Repository:
        """Get existing repository or create new one"""
        # Convert to absolute path
        abs_path = str(Path(path).resolve())
        
        repo = session.query(Repository).filter(Repository.path == abs_path).first()
        
        if not repo:
            name = Path(abs_path).name
            
            repo = Repository(
                path=abs_path,
                name=name,
                last_indexed=datetime.now()
            )
            session.add(repo)
            session.flush()
            
        return repo

    def _get_repository(self, path: str) -> Optional[Repository]:
        """Safely get a repository by path without creating one.
        
        Args:
            path: Repository path to lookup
            
        Returns:
            Repository object or None if not found
        """
        try:
            # Convert to absolute path
            abs_path = str(Path(path).resolve())
            
            with SessionLocal() as session:
                return session.query(Repository).filter(Repository.path == abs_path).first()
        except Exception as e:
            logger.error(f"Error getting repository {path}: {e}")
            return None
        
    def get_repository_stats(self, repository_path: str) -> Dict[str, Any]:
        """Get statistics about the stored graph
        
        Args:
            repository_path: Path to the code repository
            
        Returns:
            Dictionary containing graph statistics
        """
        # Convert to absolute path
        abs_path = str(Path(repository_path).resolve())
        
        with SessionLocal() as session:
            # Get repository info
            repo = session.query(Repository).filter(Repository.path == abs_path).first()
            logger.info(f"Repository: {repo}, {repository_path}")
            if not repo:
                return {}
            
            # Get node type counts
            node_types = {}
            type_counts = session.query(
                Node.type, func.count(Node.id)
            ).filter(
                Node.repository_id == repo.id
            ).group_by(Node.type).all()
            
            for node_type, count in type_counts:
                node_types[node_type] = count
                
            # Get relationship type counts
            relation_types = {}
            relation_counts = session.query(
                Edge.relation_type, func.count(Edge.id)
            ).filter(
                Edge.repository_id == repo.id
            ).group_by(Edge.relation_type).all()
            
            for relation_type, count in relation_counts:
                relation_types[relation_type] = count
                
            return {
                'repository': repo.name,
                'path': repo.path,
                'language': repo.language,
                'nodes_count': repo.nodes_count,
                'edges_count': repo.edges_count,
                'last_indexed': repo.last_indexed,
                'node_types': node_types,
                'relation_types': relation_types
            }

    def store_semantic_graph(self, repo_path: str, repository_name: str, graph: nx.DiGraph):
        """Store a semantic graph in the database.
        
        Args:
            repo_path: Path of the repository
            repository_name: Name of the repository
            graph: NetworkX graph to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.available:
            logger.warning("Cannot store graph - database is not available")
            return False
            
        try:
            # Convert to absolute path
            abs_path = str(Path(repo_path).resolve())
            
            with SessionLocal() as session:
                # Create or update repository record
                repository = session.query(Repository).filter(Repository.name == repository_name).first()
                if not repository:
                    repository = Repository(
                        name=repository_name,
                        path=abs_path,  # Use absolute path
                        language="unknown"
                    )
                    session.add(repository)
                    session.flush()  # Get ID for the new repository
                    
                # Clear existing data for this repository
                self._clear_repository_data(session, repository.id)
                
                # Node ID mapping to store the relation between graph node_id and database id
                node_id_map = {}
                
                # Store nodes
                for node_id, node_data in graph.nodes(data=True):
                    node_type = node_data.get('type', 'unknown')
                    name = node_data.get('name', str(node_id))
                    file_path = node_data.get('file_path', '')
                    
                    # Get embedding if available
                    embedding = node_data.get('embedding')
                    if embedding is None:
                        # Try code_embedding
                        embedding = node_data.get('code_embedding')
                        
                    # Format the embedding for TiDB
                    embedding_vec = self._prepare_vector_embedding(embedding, self.CODE_EMBEDDING_DIM)
                    
                    # Convert node data to JSON string for context
                    node_context = {k: v for k, v in node_data.items() 
                                    if k not in ['embedding', 'code_embedding', 'doc_embedding']}
                    
                    # Store the node
                    node = Node(
                        repository_id=repository.id,
                        node_id=str(node_id),
                        type=node_type,
                        name=name,
                        file_path=file_path,
                        context=json.dumps(node_context),
                        code_embedding=embedding_vec,
                        doc_embedding=embedding_vec  # Use same embedding for both
                    )
                    session.add(node)
                    session.flush()  # Get the database ID for the node
                    
                    # Store mapping between graph node_id and database id
                    node_id_map[str(node_id)] = node.id
                
                # Store edges using the node ID mapping
                for source, target, edge_data in graph.edges(data=True):
                    edge_type = edge_data.get('type', 'unknown')
                    label = edge_data.get('label', '')
                    
                    # Get database IDs for the source and target nodes
                    source_str = str(source)
                    target_str = str(target)
                    
                    if source_str not in node_id_map:
                        logger.warning(f"Source node {source_str} not found in node map, skipping edge")
                        continue
                        
                    if target_str not in node_id_map:
                        logger.warning(f"Target node {target_str} not found in node map, skipping edge")
                        continue
                    
                    edge = Edge(
                        repository_id=repository.id,
                        source_id=node_id_map[source_str],
                        target_id=node_id_map[target_str],
                        relation_type=edge_type,
                        label=label
                    )
                    session.add(edge)
                
                # Update repository stats
                repository.nodes_count = len(graph.nodes())
                repository.edges_count = len(graph.edges())
                repository.last_indexed = datetime.now()
                
                # Commit all changes
                session.commit()
                
                logger.info(f"Stored semantic graph with {repository.nodes_count} nodes and {repository.edges_count} edges")
                return True
                
        except Exception as e:
            logger.error(f"Error storing semantic graph: {e}")
            return False