from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import networkx as nx
import os
import configparser
from pathlib import Path
import numpy as np  # For vector handling
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text, func
import traceback

from setting.db import SessionLocal, engine
from setting.base import DATABASE_URI
from setting.embedding import EMBEDDING_MODEL, VECTOR_SEARCH, CODE_EMBEDDING_DIM, DOC_EMBEDDING_DIM
from .models import Base, Repository, Node, Edge, VECTOR

# Configure logging
logger = logging.getLogger(__name__)

class GraphDBManager:
    """Manages graph persistence in database."""
    
    def __init__(self, db_url=None):
        """Initialize database manager.
        
        Args:
            db_url (str, optional): Database URL to connect to.
        """
        self.db_url = db_url or DATABASE_URI
        self.available = self.db_url is not None and engine is not None
        
        # Set embedding dimensions from settings
        self.CODE_EMBEDDING_DIM = CODE_EMBEDDING_DIM
        self.DOC_EMBEDDING_DIM = DOC_EMBEDDING_DIM
        
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
        logger.info("db manager init finished")
    
    def _setup_tables(self):
        """Create tables if they don't exist."""
        if not self.available or engine is None:
            logger.warning("Cannot setup tables - database is not available")
            return
            
        # SQLAlchemy will create tables through Base.metadata
        Base.metadata.create_all(bind=engine)
        
        # Log tables being created
        logger.info(f"Tables created: {list(Base.metadata.tables.keys())}")
        
        # Detect database type
        db_type = "unknown"
        if self.db_url:
            if 'tidb' in self.db_url:
                db_type = "tidb"
            elif 'mysql' in self.db_url:
                db_type = "mysql"
            elif 'sqlite' in self.db_url:
                db_type = "sqlite"
                
        # Only TiDB and some MySQL versions support vector indexes
        if db_type in ["tidb", "mysql"]:
            with SessionLocal() as session:
                try:
                    # First ensure the table has a TiFlash replica, which is required for vector index functionality
                    session.execute(text("""
                        ALTER TABLE nodes SET TIFLASH REPLICA 1
                    """))
                    logger.info("Created TiFlash replica for nodes table")
                except Exception as e:
                    logger.warning(f"Creating TiFlash replica failed, vector index may not work: {e}")
                
                try:
                    # Use the correct TiDB vector index syntax to create vector index for code_embedding
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
                    # Use the correct TiDB vector index syntax to create vector index for doc_embedding
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
        """Prepare a vector embedding for database storage by ensuring correct format and dimension."""
        if embedding is None or (isinstance(embedding, (list, np.ndarray)) and len(embedding) == 0):
            zeros = [0.0] * dim
            return f"[{','.join(str(x) for x in zeros)}]"
            
        if isinstance(embedding, str):
            if not (embedding.startswith('[') and embedding.endswith(']')):
                zeros = [0.0] * dim
                return f"[{','.join(str(x) for x in zeros)}]"
            return embedding
                
        if isinstance(embedding, (list, np.ndarray)):
            if len(embedding) < dim:
                if isinstance(embedding, np.ndarray):
                    embedding = np.pad(embedding, (0, dim - len(embedding)))
                else:
                    embedding = embedding + [0.0] * (dim - len(embedding))
            elif len(embedding) > dim:
                embedding = embedding[:dim]
                
            return f"[{','.join(str(float(x)) for x in embedding)}]"
            
        zeros = [0.0] * dim
        return f"[{','.join(str(x) for x in zeros)}]"

    def save_graph(self, graph: nx.MultiDiGraph, repo_path: str):
        """Save graph to database with minimal schema."""
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
                    code_embedding = attrs.get('code_embedding', None)
                    doc_embedding = attrs.get('doc_embedding', None)
                    
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
                        code_context = json.dumps(context_data)
                        doc_context = json.dumps({"docstring": attrs.get("docstring", "")})
                    except Exception as e:
                        logger.warning(f"Error serializing context for node {node_str_id}: {e}")
                        # If JSON serialization fails, create a simplified version
                        code_context = json.dumps({
                            'type': node_type,
                            'name': name,
                            'file': file_path,
                            'line': line_number
                        })
                        doc_context = json.dumps({"docstring": ""})
                    
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
                        existing_node.code_context = code_context
                        existing_node.doc_context = doc_context
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
                            code_context=code_context,
                            doc_context=doc_context,
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
        """Load graph from database."""
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
                        if node.code_context:
                            try:
                                context_data = json.loads(node.code_context)
                                attrs.update(context_data)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse context for node {node_id}: {e}")
                                
                        if node.doc_context:
                            try:
                                doc_data = json.loads(node.doc_context)
                                attrs.update(doc_data)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse doc context for node {node_id}: {e}")
                        
                        # Add embeddings to attributes if available
                        if node.code_embedding:
                            if isinstance(node.code_embedding, list):
                                attrs["code_embedding"] = np.array(node.code_embedding)
                            elif isinstance(node.code_embedding, str):
                                try:
                                    embedding_str = node.code_embedding.strip('[]')
                                    values = [float(x.strip()) for x in embedding_str.split(',') if x.strip()]
                                    attrs["code_embedding"] = np.array(values)
                                except Exception as e:
                                    logger.warning(f"Failed to parse code embedding for node {node_id}: {e}")
                        
                        if node.doc_embedding:
                            if isinstance(node.doc_embedding, list):
                                attrs["doc_embedding"] = np.array(node.doc_embedding)
                            elif isinstance(node.doc_embedding, str):
                                try:
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

    def vector_search(self, repo_name: str, query: str = None, 
                     limit: int = VECTOR_SEARCH["default_limit"], use_doc_embedding: bool = False) -> List[Dict[str, Any]]:
        """Perform a vector search using the query string.
        
        Args:
            repo_name: Repository name
            query: Search query string
            limit: Maximum number of results to return
            use_doc_embedding: Whether to use document embeddings instead of code embeddings
            
        Returns:
            List of matching nodes with their data
        """
        if not self.available:
            logger.warning("Cannot perform vector search - database is not available")
            return []
        
        if not query:
            logger.warning("Cannot perform vector search - query string is empty")
            return []
            
        if not repo_name:
            logger.warning("Cannot perform vector search - repository name not provided")
            return []
        
        # Generate embedding for the query
        try:
            # Use cached model loader instead of creating a new instance each time
            from llm.embedding import get_sentence_transformer
            model = get_sentence_transformer(EMBEDDING_MODEL["name"])
            query_embedding = model.encode(query)
            
            # Find repository by name
            with SessionLocal() as session:
                repo = session.query(Repository).filter(Repository.name == repo_name).first()
                if not repo:
                    logger.warning(f"Repository not found by name: {repo_name}")
                    return []
                
                # Determine which embedding to use
                embedding_field = "doc_embedding" if use_doc_embedding else "code_embedding"
                
                # Format the query vector for TiDB
                query_str = f"[{','.join(str(float(x)) for x in query_embedding)}]"
                
                # Build query based on database type and configured similarity metric
                metric_func = "VEC_COSINE_DISTANCE" if VECTOR_SEARCH["similarity_metric"] == "cosine" else "VEC_L2_DISTANCE"
                
                query_sql = f"""
                    SELECT 
                        id, node_id, type, name, file_path, line_number, code_context, doc_context
                    FROM 
                        nodes
                    WHERE 
                        repository_id = :repo_id
                    ORDER BY 
                        {metric_func}({embedding_field}, :query_vec) ASC
                    LIMIT :limit
                """
                
                # Execute query
                results = session.execute(
                    text(query_sql), 
                    {
                        "repo_id": repo.id, 
                        "query_vec": query_str,
                        "limit": limit
                    }
                ).fetchall()
                
                # Process results
                processed_results = []
                for row in results:
                    # Parse contexts
                    code_context = {}
                    doc_context = {}
                    
                    if row.code_context:
                        try:
                            code_context = json.loads(row.code_context)
                        except json.JSONDecodeError:
                            code_context = {"error": "Invalid JSON in code_context"}
                            
                    if row.doc_context:
                        try:
                            doc_context = json.loads(row.doc_context)
                        except json.JSONDecodeError:
                            doc_context = {"error": "Invalid JSON in doc_context"}
                    
                    # Combine all data
                    node_data = {
                        "id": row.node_id,
                        "name": row.name,
                        "type": row.type,
                        "file_path": row.file_path,
                        "line": row.line_number,
                        "similarity": 0.95 - 0.05 * len(processed_results),  # Simulated similarity
                    }
                    
                    # Add context data
                    node_data.update(code_context)
                    node_data.update(doc_context)
                    
                    processed_results.append(node_data)
                    
                logger.info(f"Found {len(processed_results)} similar nodes")
                return processed_results
                    
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    def full_text_search(self, repo_name: str, query: str = None, 
                        limit: int = VECTOR_SEARCH["default_limit"]) -> List[Dict[str, Any]]:
        """Perform a full text search using the query string.
        
        Args:
            repo_name: Repository name
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching nodes with their data
        """
        if not self.available:
            logger.warning("Cannot perform full text search - database is not available")
            return []
        
        if not query:
            logger.warning("Cannot perform full text search - query string is empty")
            return []
            
        if not repo_name:
            logger.warning("Cannot perform full text search - repository name not provided")
            return []
        
        try:
            # Find repository by name
            with SessionLocal() as session:
                repo = session.query(Repository).filter(Repository.name == repo_name).first()
                if not repo:
                    logger.warning(f"Repository not found by name: {repo_name}")
                    return []
                
                # Use LIKE for text search in code_context
                like_query = f"%{query}%"
                
                query_sql = """
                    SELECT 
                        id, node_id, type, name, file_path, line_number, code_context, doc_context
                    FROM 
                        nodes
                    WHERE 
                        repository_id = :repo_id
                        AND code_context LIKE :query
                    LIMIT :limit
                """
                
                # Execute query
                results = session.execute(
                    text(query_sql), 
                    {
                        "repo_id": repo.id, 
                        "query": like_query,
                        "limit": limit
                    }
                ).fetchall()
                
                # Process results
                processed_results = []
                for row in results:
                    # Parse contexts
                    code_context = {}
                    doc_context = {}
                    
                    if row.code_context:
                        try:
                            code_context = json.loads(row.code_context)
                        except json.JSONDecodeError:
                            code_context = {"error": "Invalid JSON in code_context"}
                            
                    if row.doc_context:
                        try:
                            doc_context = json.loads(row.doc_context)
                        except json.JSONDecodeError:
                            doc_context = {"error": "Invalid JSON in doc_context"}
                    
                    # Combine all data
                    node_data = {
                        "id": row.node_id,
                        "name": row.name,
                        "type": row.type,
                        "file_path": row.file_path,
                        "line": row.line_number,
                        "similarity": 1.0 - (0.05 * len(processed_results)),  # Simple ranking
                    }
                    
                    # Add context data
                    node_data.update(code_context)
                    node_data.update(doc_context)
                    
                    processed_results.append(node_data)
                    
                logger.info(f"Found {len(processed_results)} matching nodes using full text search")
                return processed_results
                    
        except Exception as e:
            logger.error(f"Error in full text search: {e}")
            logger.error(traceback.format_exc())
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

    def get_repository_stats(self, repository_name: str, repository_path: str = None) -> Dict[str, Any]:
        """Get statistics about the stored graph.
        
        Args:
            repository_name: Name of the repository
            repository_path: Deprecated, kept for backward compatibility
            
        Returns:
            Dictionary of repository statistics
        """
        if not repository_name:
            if repository_path:
                # Try to extract name from path for backward compatibility
                try:
                    repository_name = Path(repository_path).name
                except:
                    logger.warning("Cannot extract repository name from path")
                    return {}
            else:
                logger.warning("Cannot get repository stats - repository name not provided")
                return {}
            
        with SessionLocal() as session:
            # Get repository by name
            repo = session.query(Repository).filter(Repository.name == repository_name).first()
                
            if not repo:
                logger.warning(f"Repository not found by name: {repository_name}")
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