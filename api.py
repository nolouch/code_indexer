import os
import logging
import traceback
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Path
from pydantic import BaseModel, Field
from sqlalchemy import text

from knowledgebase.best_practices import BestPracticesKnowledgeBase
from llm.factory import LLMInterface
from llm.embedding import get_text_embedding, get_sentence_transformer
from code_graph.db_manager import GraphDBManager
from code_graph import create_openai_builder
from setting.base import DATABASE_URI
from setting.db import SessionLocal, Base, engine
from knowledgebase.knowledge_graph import GraphKnowledgeBase
from setting.embedding import EMBEDDING_MODEL
from file_indexer.indexer import CodeIndexer
from file_indexer.database import CodeFile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialization ---
llm_client = None
bp_kb = None
code_graph_db = None
tidb_kg = None
file_indexer = None

try:
    # Initialize database
    logger.info("Checking database configuration...")
    if DATABASE_URI:
        logger.info(f"Using database URI: {DATABASE_URI}")

        # Create tables if they don't exist
        if engine is not None:
            try:
                logger.info("Creating database tables if they don't exist...")
                Base.metadata.create_all(bind=engine)
                logger.info("Database tables are ready")
            except Exception as e:
                logger.error(f"Error creating database tables: {e}")
                logger.error(traceback.format_exc())
    else:
        logger.warning("DATABASE_URI is not set! Database features will be limited.")

    # Pre-load the embedding model to improve performance
    logger.info("Pre-loading embedding model...")
    try:
        embedding_model = get_sentence_transformer(EMBEDDING_MODEL["name"])
        logger.info(f"Successfully pre-loaded embedding model: {EMBEDDING_MODEL['name']}")
    except Exception as e:
        logger.error(f"Error pre-loading embedding model: {e}")
        logger.error(traceback.format_exc())

    # Configure LLM client
    logger.info("Initializing LLM client...")
    try:
        llm_client = LLMInterface(
            "bedrock",
            "arn:aws:bedrock:us-east-1:841162690310:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        )
        logger.info("LLM client initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing LLM client: {e}")
        logger.error(traceback.format_exc())
        logger.warning("Continuing without LLM capabilities")
        llm_client = None

    # Initialize Knowledge Base if LLM client is available
    if llm_client is not None:
        # Embedding function
        try:

            def embedding_func(text: str) -> np.ndarray:
                return get_text_embedding(text, "text-embedding-3-small")

            logger.info("Initializing best practices knowledge base...")
            bp_kb = BestPracticesKnowledgeBase(llm_client, embedding_func)
            logger.info("Knowledge base initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            logger.error(traceback.format_exc())
            bp_kb = None
    else:
        logger.warning(
            "Skipping knowledge base initialization due to missing LLM client"
        )
        bp_kb = None

    # Initialize Code Graph database manager
    logger.info("Initializing code graph database...")
    try:
        code_graph_db = GraphDBManager(db_url=DATABASE_URI)
        if not code_graph_db.available:
            logger.warning("Code graph database connection is not available!")
            logger.warning("Check DATABASE_URI configuration in your settings.")
    except Exception as e:
        logger.error(f"Error initializing code graph database: {e}")
        logger.error(traceback.format_exc())

    logger.info("Initializing tidb knowledge graph...")
    try:
        tidb_kg = GraphKnowledgeBase(
            llm_client,
            "tidb_knowledge_graph.entities",
            "tidb_knowledge_graph.relationships",
            "tidb_knowledge_graph.chunks",
        )
    except Exception as e:
        logger.error(f"Error initializing code graph database: {e}", exc_info=True)

    # Initialize file indexer
    logger.info("Initializing file indexer...")
    try:
        file_indexer = CodeIndexer(db_path=DATABASE_URI)
        logger.info("File indexer initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing file indexer: {e}", exc_info=True)
        file_indexer = None

except Exception as e:
    logger.error(f"Error during initialization: {e}")
    logger.error(traceback.format_exc())

# --- FastAPI App ---
app = FastAPI(
    title="Code Knowledge API",
    description="API to query code knowledge base and best practices",
    version="0.1.0",
    openapi_tags=[
        {
            "name": "Best Practices",
            "description": "Operations related to best practices knowledge base",
        },
        {
            "name": "Code Graph",
            "description": "Operations related to code graph analysis and search",
        },
        {
            "name": "File Indexer",
            "description": "Operations related to file indexing and search",
        },
    ],
)


# Root path handler, shows initialization status
@app.get("/", tags=["Status"])
async def root():
    """Get API status and availability information"""
    db_status = {
        "initialized": code_graph_db is not None,
        "available": code_graph_db is not None and code_graph_db.available
        if code_graph_db
        else False,
        "uri_configured": DATABASE_URI is not None and len(DATABASE_URI) > 0,
    }

    return {
        "status": "running",
        "services": {
            "llm_client": llm_client is not None,
            "best_practices_kb": bp_kb is not None,
            "code_graph_db": db_status,
            "file_indexer": file_indexer is not None,
        },
        "docs": "/docs",
    }


# --- Best Practices Models and Endpoints ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="The search query")


class QueryResponse(BaseModel):
    results: dict = Field(..., description="The search results")


@app.get("/tidb_doc", response_model=QueryResponse, tags=["Best Practices"])
async def query_best_practices(
    query: str = Query(..., description="The search query for best practices"),
    top_k: Optional[int] = Query(default=10, description="Top K results to return"),
    threshold: Optional[float] = Query(
        default=0.5, description="Similarity score threshold"
    ),
):
    """
    Accepts a query string and returns relevant best practices from the knowledge base.

    This endpoint finds best practices that match the query semantically.

    Args:
        query: The search query string
        top_k: Top K results to return (optional)
        threshold: Similarity score threshold (optional)

    Returns:
        A dictionary of best practices relevant to the query.
    """
    if llm_client is None:
        raise HTTPException(
            status_code=503,
            detail="LLM client not initialized - AWS credentials may be missing",
        )

    if tidb_kg is None:
        raise HTTPException(
            status_code=503, detail="TiDB Knowledge Graph not initialized"
        )

    try:
        with SessionLocal() as session:
            res = tidb_kg.retrieve_graph_data(session, query, 3)
        return QueryResponse(results=res.to_dict())
    except Exception as e:
        logger.error(f"Error processing query '{query}': {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error querying tidb knowledge graph: {str(e)}"
        )


@app.get("/best_practices", response_model=QueryResponse, tags=["Best Practices"])
async def query_best_practices(
    query: str = Query(..., description="The search query for best practices"),
    top_k: Optional[int] = Query(default=10, description="Top K results to return"),
    threshold: Optional[float] = Query(
        default=0.5, description="Similarity score threshold"
    ),
):
    """
    Accepts a query string and returns relevant best practices from the knowledge base.

    This endpoint finds best practices that match the query semantically.

    Args:
        query: The search query string
        top_k: Top K results to return (optional)
        threshold: Similarity score threshold (optional)

    Returns:
        A dictionary of best practices relevant to the query.
    """
    if llm_client is None:
        raise HTTPException(
            status_code=503,
            detail="LLM client not initialized - AWS credentials may be missing",
        )

    if bp_kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")

    try:
        practices = bp_kb.find_best_practices(
            query=query,
            top_k=top_k,
        )
        return QueryResponse(results=practices)
    except Exception as e:
        logger.error(f"Error processing query '{query}': {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error querying best practices: {str(e)}"
        )


# --- Code Graph Models and Endpoints ---
class CodeSearchRequest(BaseModel):
    query: str = Field(..., description="The code search query")
    repository_name: str = Field(..., description="Name of the repository")
    limit: Optional[int] = Field(10, description="Maximum number of results to return")
    use_doc_embedding: Optional[bool] = Field(
        False,
        description="Whether to search using documentation embeddings instead of code embeddings",
    )


class CodeNode(BaseModel):
    id: str = Field(..., description="Node identifier")
    name: str = Field(..., description="Node name")
    type: str = Field(..., description="Node type (function, class, module, etc.)")
    file_path: str = Field(..., description="File path containing this node")
    line: int = Field(..., description="Line number in the file")
    similarity: float = Field(..., description="Similarity score to the query")
    code_content: Optional[str] = Field(
        None, description="The actual code content of this node"
    )
    doc_content: Optional[str] = Field(
        None, description="Documentation content for this node"
    )


class CodeSearchResponse(BaseModel):
    results: List[CodeNode] = Field(
        ..., description="List of code nodes matching the query"
    )


@app.post("/code_graph/search", response_model=CodeSearchResponse, tags=["Code Graph"])
async def search_code_graph(request: CodeSearchRequest):
    """
    Search the code graph for nodes matching the query.

    This endpoint performs a vector search to find semantically similar code elements.

    - **query**: The search query text
    - **repository_name**: Name of the repository
    - **limit**: Maximum number of results to return (default: 10)
    - **use_doc_embedding**: Whether to search using documentation embeddings instead of code embeddings

    Returns:
        A list of code nodes matching the query, ordered by relevance.
    """
    if not code_graph_db or not code_graph_db.available:
        logger.error(
            f"Code graph database not initialized or unavailable, cannot execute search query: {request.query}"
        )
        raise HTTPException(
            status_code=503, detail="Code graph database not initialized or unavailable"
        )

    try:
        # Use repository name directly in the search
        logger.info(
            f"Executing code search: {request.query}, repository name: {request.repository_name}"
        )

        # Execute the search with repository name
        results = code_graph_db.vector_search(
            repo_name=request.repository_name,
            query=request.query,
            limit=request.limit,
            use_doc_embedding=request.use_doc_embedding,
        )

        # Convert the results to the response model format
        nodes = []
        for result in results:
            # Extract content and documentation
            code_content = result.get("code_context", None)
            doc_content = result.get("doc_context", None)

            # Fall back to docstring if doc_context is not available
            if doc_content is None:
                doc_content = result.get("docstring", None)

            node = CodeNode(
                id=result["id"],
                name=result["name"],
                type=result["type"],
                file_path=result["file_path"],
                line=result["line"],
                similarity=result["similarity"],
                code_content=code_content,
                doc_content=doc_content,
            )
            nodes.append(node)

        logger.info(f"Found {len(nodes)} matching nodes")
        return CodeSearchResponse(results=nodes)
    except Exception as e:
        logger.error(f"Error processing code search query '{request.query}': {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error searching code graph: {str(e)}"
        )


@app.post("/code_graph/full_text_search", response_model=CodeSearchResponse, tags=["Code Graph"])
async def full_text_search_code_graph(request: CodeSearchRequest):
    """
    Search the code graph for nodes containing the query text using full text search.

    This endpoint finds nodes that contain exact matches of the query text in their code_context using SQL LIKE queries.

    - **query**: The search query text
    - **repository_name**: Name of the repository
    - **limit**: Maximum number of results to return (default: 10)

    Returns:
        A list of code nodes matching the query, ordered by relevance.
    """
    if not code_graph_db or not code_graph_db.available:
        logger.error(
            f"Code graph database not initialized or unavailable, cannot execute full text search query: {request.query}"
        )
        raise HTTPException(
            status_code=503, detail="Code graph database not initialized or unavailable"
        )

    try:
        # Use repository name directly in the search
        logger.info(
            f"Executing full text code search: {request.query}, repository name: {request.repository_name}"
        )

        # Execute the search with repository name
        results = code_graph_db.full_text_search(
            repo_name=request.repository_name,
            query=request.query,
            limit=request.limit,
        )

        # Convert the results to the response model format
        nodes = []
        for result in results:
            # Extract content and documentation
            code_content = result.get("code_context", None)
            doc_content = result.get("doc_context", None)

            # Fall back to docstring if doc_context is not available
            if doc_content is None:
                doc_content = result.get("docstring", None)

            node = CodeNode(
                id=result["id"],
                name=result["name"],
                type=result["type"],
                file_path=result["file_path"],
                line=result["line"],
                similarity=result["similarity"],
                code_content=code_content,
                doc_content=doc_content,
            )
            nodes.append(node)

        logger.info(f"Found {len(nodes)} matching nodes with full text search")
        return CodeSearchResponse(results=nodes)
    except Exception as e:
        logger.error(f"Error processing full text search query '{request.query}': {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error searching code graph with full text: {str(e)}"
        )


@app.get("/code_graph/repositories/stats", tags=["Code Graph"])
async def get_repository_stats(
    repository_name: str = Query(..., description="Name of the code repository")
):
    """
    Get statistics about a code repository stored in the graph database.

    This endpoint returns information about the nodes and relationships in the repository.

    - **repository_name**: Name of the code repository

    Returns:
        Statistics about the repository graph, including node and edge counts.
    """
    if not code_graph_db or not code_graph_db.available:
        logger.error(
            "Code graph database not initialized or unavailable, cannot get repository stats"
        )
        raise HTTPException(
            status_code=503, detail="Code graph database not initialized or unavailable"
        )

    try:
        logger.info(f"Getting repository stats by name: {repository_name}")

        stats = code_graph_db.get_repository_stats(repository_name=repository_name)
        if not stats:
            logger.warning(f"Repository not found: {repository_name}")
            raise HTTPException(
                status_code=404, detail=f"Repository not found: {repository_name}"
            )
        return stats
    except Exception as e:
        logger.error(f"Error getting repository stats: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error retrieving repository stats: {str(e)}"
        )


# Keep the original endpoint for backward compatibility
@app.get(
    "/code_graph/repositories/{repository_path}/stats",
    tags=["Code Graph"],
    deprecated=True,
)
async def get_repository_stats_by_path(
    repository_path: str = Path(..., description="Path to the code repository")
):
    """
    Get statistics about a code repository stored in the graph database, identified by path.

    This endpoint is deprecated. Use /code_graph/repositories/stats instead.

    - **repository_path**: Path to the code repository

    Returns:
        Statistics about the repository graph, including node and edge counts.
    """
    # Extract repository name from path (last part of the path)
    from pathlib import Path as PathLib

    repo_name = PathLib(repository_path).name

    # Use the repository name to get stats
    return await get_repository_stats(repository_name=repo_name)


class RepositoryRequest(BaseModel):
    path: str = Field(..., description="Path to the repository")


@app.get("/code_graph/repositories", tags=["Code Graph"])
async def list_repositories():
    """
    List all repositories stored in the code graph database.

    Returns:
        A list of repository information.
    """
    if not code_graph_db or not code_graph_db.available:
        logger.error(
            "Code graph database not initialized or unavailable, cannot list repositories"
        )
        raise HTTPException(
            status_code=503, detail="Code graph database not initialized or unavailable"
        )

    try:
        logger.info("Listing all repositories")
        from code_graph.models import Repository

        # Use the imported SessionLocal, not from code_graph_db
        if SessionLocal is None:
            logger.error("SessionLocal is not initialized")
            raise HTTPException(
                status_code=503, detail="Database session factory is not initialized"
            )

        with SessionLocal() as session:
            repos = session.query(Repository).all()
            return [
                {
                    "id": repo.id,
                    "name": repo.name,
                    "path": repo.path,
                    "language": repo.language,
                    "nodes_count": repo.nodes_count,
                    "edges_count": repo.edges_count,
                    "last_indexed": repo.last_indexed,
                }
                for repo in repos
            ]
    except Exception as e:
        logger.error(f"Error listing repositories: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error listing repositories: {str(e)}"
        )


@app.get("/db_check", tags=["Status"])
async def check_database():
    """Check database connectivity and configuration"""
    if not DATABASE_URI:
        return {
            "status": "error",
            "message": "DATABASE_URI is not configured in settings",
        }

    if not code_graph_db:
        return {
            "status": "error",
            "message": "Code graph database manager not initialized",
        }

    if engine is None:
        return {
            "status": "error",
            "message": "Database engine could not be initialized",
            "database_uri": DATABASE_URI.replace(DATABASE_URI.split("@")[0], "***")
            if "@" in DATABASE_URI
            else "***",
        }

    try:
        # Try to create a session and run a simple query
        with SessionLocal() as session:
            # Run simple query
            result = session.execute("SELECT 1").scalar()

            # Try to query repositories table if it exists
            try:
                from code_graph.models import Repository

                repo_count = session.query(Repository).count()
                return {
                    "status": "ok",
                    "connection": True,
                    "test_query": result == 1,
                    "repositories_count": repo_count,
                    "database_uri": DATABASE_URI.replace(
                        DATABASE_URI.split("@")[0], "***"
                    )
                    if "@" in DATABASE_URI
                    else "***",
                }
            except Exception as e:
                # Table might not exist yet
                return {
                    "status": "partial",
                    "connection": True,
                    "test_query": result == 1,
                    "tables_error": str(e),
                    "message": "Database connection works but tables might not be initialized",
                    "database_uri": DATABASE_URI.replace(
                        DATABASE_URI.split("@")[0], "***"
                    )
                    if "@" in DATABASE_URI
                    else "***",
                }
    except Exception as e:
        return {
            "status": "error",
            "connection": False,
            "message": f"Database connection failed: {str(e)}",
            "database_uri": DATABASE_URI.replace(DATABASE_URI.split("@")[0], "***")
            if "@" in DATABASE_URI
            else "***",
        }


# --- File Indexer Models and Endpoints ---
class FileIndexerResult(BaseModel):
    id: int = Field(..., description="File ID")
    file_path: str = Field(..., description="Path to the file")
    language: str = Field(..., description="Programming language of the file")
    repo_name: Optional[str] = Field(None, description="Repository name")
    similarity: float = Field(..., description="Similarity score to the query")
    content: Optional[str] = Field(None, description="File content (if requested)")


class FileIndexerResponse(BaseModel):
    results: List[FileIndexerResult] = Field(..., description="List of files matching the query")


class FileSearchRequest(BaseModel):
    query: str = Field(..., description="The search query")
    limit: int = Field(10, description="Maximum number of results to return")
    show_content: bool = Field(True, description="Whether to include file content in results")
    sibling_chunk_num: Optional[int] = Field(1, description="Number of sibling chunks to include around each match (default: 1)")
    repository: Optional[str] = Field(None, description="Repository name to filter results")


@app.post("/file_indexer/vector_search", response_model=FileIndexerResponse, tags=["File Indexer"])
async def vector_search_files(request: FileSearchRequest):
    """
    Search for files similar to the query text using vector (semantic) search.
    
    This endpoint finds files that match the query semantically using TiDB vector search.
    
    Args:
        query: The search query string
        limit: Maximum number of results to return (default: 10)
        show_content: Whether to include file content in results (default: True)
        sibling_chunk_num: Number of sibling chunks to include around each match (default: 1)
        repository: Repository name to filter results (optional)
        
    Returns:
        A list of files matching the query, with optional content.
    """
    if not file_indexer:
        raise HTTPException(
            status_code=503, detail="File indexer not initialized"
        )
    
    try:
        search_results = file_indexer.search_similar(
            query_text=request.query,
            limit=request.limit,
            show_content=request.show_content,
            sibling_chunk_num=request.sibling_chunk_num,
            repository=request.repository,
            search_type="vector"
        )
        
        # Convert to response format
        results = []
        for file, similarity, content in search_results:
            results.append(
                FileIndexerResult(
                    id=file.id,
                    file_path=file.file_path,
                    language=file.language or "unknown",
                    repo_name=file.repo_name,
                    similarity=similarity,
                    content=content if request.show_content else None
                )
            )
        
        return FileIndexerResponse(results=results)
    except ValueError as e:
        # Catch the specific error when TiDB is required for vector search
        if "requires TiDB" in str(e):
            raise HTTPException(
                status_code=400, 
                detail="Vector search requires TiDB. Current database does not support vector operations. Try using full_text_search endpoint instead."
            )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing vector search query '{request.query}': {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error searching files: {str(e)}"
        )


@app.post("/file_indexer/full_text_search", response_model=FileIndexerResponse, tags=["File Indexer"])
async def full_text_search_files(request: FileSearchRequest):
    """
    Search for files containing the query text using full text search.
    
    This endpoint finds files that contain exact matches of the query text using SQL LIKE queries.
    
    Args:
        query: The search query string
        limit: Maximum number of results to return (default: 10)
        show_content: Whether to include file content in results (default: True)
        sibling_chunk_num: Number of sibling chunks to include around each match (default: 1)
        repository: Repository name to filter results (optional)
        
    Returns:
        A list of files matching the query, with optional content.
    """
    if not file_indexer:
        raise HTTPException(
            status_code=503, detail="File indexer not initialized"
        )
    
    print(f"[API] Full text search request: query='{request.query}', limit={request.limit}, repository={request.repository}")
    
    try:
        search_results = file_indexer.search_similar(
            query_text=request.query,
            limit=request.limit,
            show_content=request.show_content,
            sibling_chunk_num=request.sibling_chunk_num,
            repository=request.repository,
            search_type="full_text"
        )
        
        print(f"[API] Full text search returned {len(search_results)} results")
        
        # Convert to response format
        results = []
        for file, similarity, content in search_results:
            results.append(
                FileIndexerResult(
                    id=file.id,
                    file_path=file.file_path,
                    language=file.language or "unknown",
                    repo_name=file.repo_name,
                    similarity=similarity,
                    content=content if request.show_content else None
                )
            )
        
        return FileIndexerResponse(results=results)
    except Exception as e:
        error_message = f"Error processing full text search query '{request.query}': {e}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        print(f"[API-ERROR] {error_message}")
        raise HTTPException(
            status_code=500, detail=error_message
        )


# Keep the legacy GET endpoint for backward compatibility
@app.get("/file_indexer/search", response_model=FileIndexerResponse, tags=["File Indexer"], deprecated=True)
async def search_file_indexer(
    query: str = Query(..., description="The search query"),
    limit: Optional[int] = Query(default=10, description="Maximum number of results to return"),
    show_content: Optional[bool] = Query(default=True, description="Whether to include file content in results"),
    sibling_chunk_num: Optional[int] = Query(default=1, description="Number of sibling chunks to include around each match (default: 1)"),
    repository: Optional[str] = Query(default=None, description="Repository name to filter results"),
    search_type: Optional[str] = Query(default="vector", description="Type of search to perform", enum=["vector", "full_text"]),
):
    """
    [DEPRECATED] Search for files similar to the query text.
    
    This endpoint is deprecated. Please use /file_indexer/vector_search or /file_indexer/full_text_search instead.
    
    Args:
        query: The search query string
        limit: Maximum number of results to return (default: 10)
        show_content: Whether to include file content in results (default: True)
        sibling_chunk_num: Number of sibling chunks to include around each match (default: 1)
        repository: Repository name to filter results (optional)
        search_type: Type of search to perform - "vector" (semantic) or "full_text" (default: "vector")
        
    Returns:
        A list of files matching the query, with optional content.
    """
    if not file_indexer:
        raise HTTPException(
            status_code=503, detail="File indexer not initialized"
        )
    
    try:
        search_results = file_indexer.search_similar(
            query_text=query,
            limit=limit,
            show_content=show_content,
            sibling_chunk_num=sibling_chunk_num,
            repository=repository,
            search_type=search_type
        )
        
        # Convert to response format
        results = []
        for file, similarity, content in search_results:
            results.append(
                FileIndexerResult(
                    id=file.id,
                    file_path=file.file_path,
                    language=file.language or "unknown",
                    repo_name=file.repo_name,
                    similarity=similarity,
                    content=content if show_content else None
                )
            )
        
        return FileIndexerResponse(results=results)
    except ValueError as e:
        # Catch the specific error when TiDB is required for vector search
        if "requires TiDB" in str(e) and search_type == "vector":
            raise HTTPException(
                status_code=400, 
                detail="Vector search requires TiDB. Current database does not support vector operations. Try using search_type=full_text instead."
            )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing file search query '{query}': {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error searching files: {str(e)}"
        )


@app.get("/file_indexer/file", tags=["File Indexer"])
async def get_file_by_path(
    file_path: str = Query(..., description="Path to the file to retrieve", example="/path/to/file.py"),
    sibling_chunk_num: Optional[int] = Query(default=1, description="Number of sibling chunks to include around each match (default: 1)"),
    offset: int = Query(default=0, description="Chunk offset for pagination (0-based)", ge=0),
):
    """
    Get the content of a file by its path.
    
    This endpoint retrieves the content of a file directly by specifying its path.
    
    Args:
        file_path: The path of the file to retrieve
        sibling_chunk_num: Number of sibling chunks to include around each match (default: 1)
        offset: Chunk offset for pagination (0-based)
        
    Returns:
        File content and metadata.
    """
    if not file_indexer:
        raise HTTPException(
            status_code=503, detail="File indexer not initialized"
        )
    
    try:
        # URL decode the file path and clean it
        import urllib.parse
        decoded_path = urllib.parse.unquote(file_path)
        original_path = decoded_path
        
        # Remove potential quotes
        if decoded_path.startswith("'") and decoded_path.endswith("'"):
            decoded_path = decoded_path[1:-1]
        elif decoded_path.startswith('"') and decoded_path.endswith('"'):
            decoded_path = decoded_path[1:-1]
        
        # Clean up any trailing spaces in the path
        decoded_path = decoded_path.strip()
        
        print(f"[API] Looking for file with path: {decoded_path}")
        if original_path != decoded_path:
            print(f"[API] Original path was: {original_path}")
        
        # Try exact match first, then try fuzzy matching if needed
        with SessionLocal() as session:
            try:
                # First check if the file table has data
                file_count = session.execute(text("SELECT COUNT(*) FROM code_files")).scalar()
                chunk_count = session.execute(text("SELECT COUNT(*) FROM file_chunks")).scalar()
                print(f"[API] Database has {file_count} files and {chunk_count} chunks")
                
                if file_count == 0:
                    print("[API-ERROR] The code_files table is empty!")
                    raise HTTPException(
                        status_code=500, detail="No files are indexed in the database"
                    )
                
                # Prepare query strategies
                file = None
                match_method = ""
                
                # 1. Exact match
                file = session.query(CodeFile).filter_by(file_path=decoded_path).first()
                if file:
                    match_method = "exact match"
                
                # 2. If not found, try suffix match (handles relative paths)
                if not file:
                    suffix_query = f"%{decoded_path}"
                    print(f"[API-SQL] Trying suffix match: LIKE '{suffix_query}'")
                    file = session.query(CodeFile).filter(CodeFile.file_path.like(suffix_query)).first()
                    if file:
                        match_method = "suffix match"
                
                # 3. If still not found, try matching by basename only
                if not file:
                    basename = os.path.basename(decoded_path)
                    if basename:
                        basename_query = f"%/{basename}"  # Ensure we match the filename, not a part of the path
                        print(f"[API-SQL] Trying basename match: LIKE '{basename_query}'")
                        file = session.query(CodeFile).filter(CodeFile.file_path.like(basename_query)).first()
                        if file:
                            match_method = "basename match"
                
                # 4. Finally try fuzzy match with the basename
                if not file:
                    fuzzy_basename = f"%{os.path.basename(decoded_path)}%"
                    print(f"[API-SQL] Trying fuzzy basename match: LIKE '{fuzzy_basename}'")
                    candidates = session.query(CodeFile).filter(CodeFile.file_path.like(fuzzy_basename)).all()
                    if candidates:
                        # Choose the file with shortest path (usually the most accurate match)
                        file = min(candidates, key=lambda x: len(x.file_path))
                        match_method = "fuzzy basename match"
                
                if not file:
                    # If all strategies failed, return 404
                    print(f"[API-ERROR] File not found after trying all matching strategies: {decoded_path}")
                    # Print some random samples for diagnostics
                    sample = session.query(CodeFile).limit(3).all()
                    if sample:
                        print("[API] Sample file paths in database:")
                        for s in sample:
                            print(f"  - {s.file_path}")
                    
                    raise HTTPException(
                        status_code=404, detail=f"File not found: {decoded_path}"
                    )
                
                print(f"[API] Found file ID {file.id} at path {file.file_path} via {match_method}")
                
                # Validate offset
                if offset >= file.chunks_count:
                    if file.chunks_count > 0:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Invalid offset: {offset}. File has {file.chunks_count} chunks (0-{file.chunks_count-1})"
                        )
                    else:
                        # If the file has no chunks, we'll handle it below
                        offset = 0
                
                # Get file content with pagination
                chunks_query = text("""
                    SELECT content FROM file_chunks 
                    WHERE file_id = :file_id
                    ORDER BY chunk_index
                    OFFSET :chunk_offset
                    LIMIT :chunk_limit
                """)
                
                # Determine how many chunks to fetch
                chunks_to_fetch = file.chunks_count
                if sibling_chunk_num is not None and sibling_chunk_num < chunks_to_fetch:
                    chunks_to_fetch = sibling_chunk_num
                
                # Log the query
                formatted_query = str(chunks_query) \
                    .replace(":file_id", str(file.id)) \
                    .replace(":chunk_offset", str(offset)) \
                    .replace(":chunk_limit", str(chunks_to_fetch))
                print(f"[API-SQL] Chunks query: {formatted_query}")
                
                # Get file content
                full_content = ""
                
                # Check if file has chunks
                if file.chunks_count == 0:
                    print(f"[API-WARNING] File {file.id} ({file.file_path}) has no chunks")
                    full_content = "[File has no content chunks]"
                else:
                    chunks = session.execute(chunks_query, {
                        "file_id": file.id,
                        "chunk_offset": offset,
                        "chunk_limit": chunks_to_fetch
                    })
                    
                    chunks_loaded = 0
                    start_chunk = offset
                    end_chunk = offset
                    
                    for chunk in chunks:
                        full_content += chunk.content
                        chunks_loaded += 1
                        end_chunk = offset + chunks_loaded - 1
                    
                    print(f"[API] Loaded {chunks_loaded} chunks for file {file.file_path} (chunks {start_chunk}-{end_chunk})")
                    
                    # Add pagination info
                    has_more = (offset + chunks_loaded) < file.chunks_count
                    next_offset = offset + chunks_loaded if has_more else None
                    
                    # Add truncation notice
                    if sibling_chunk_num is not None and sibling_chunk_num < (file.chunks_count - offset):
                        full_content += f"\n\n[Content truncated: showing chunks {start_chunk}-{end_chunk} of {file.chunks_count} total chunks]"
                
                # Return file information and content
                response = {
                    "id": file.id,
                    "file_path": file.file_path,
                    "language": file.language or "unknown",
                    "repo_name": file.repo_name,
                    "file_size": file.file_size,
                    "chunks_count": file.chunks_count,
                    "match_method": match_method,
                    "content": full_content,
                    "pagination": {
                        "offset": offset,
                        "limit": chunks_loaded,
                        "total_chunks": file.chunks_count,
                        "has_more": offset + chunks_loaded < file.chunks_count,
                        "next_offset": next_offset
                    }
                }
                
                return response
            except HTTPException:
                raise
            except Exception as db_error:
                error_message = f"Database error while retrieving file '{decoded_path}': {db_error}"
                logger.error(error_message)
                print(f"[API-ERROR] {error_message}")
                raise HTTPException(status_code=500, detail=error_message)
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        error_message = f"Error retrieving file {file_path}: {e}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        print(f"[API-ERROR] {error_message}")
        raise HTTPException(
            status_code=500, detail=error_message
        )


@app.get("/file_indexer/file/{file_id}", tags=["File Indexer"])
async def get_file_by_id(
    file_id: int = Path(..., description="ID of the file to retrieve", gt=0),
    sibling_chunk_num: Optional[int] = Query(default=1, description="Number of sibling chunks to include around each match (default: 1)"),
    offset: int = Query(default=0, description="Chunk offset for pagination (0-based)", ge=0),
):
    """
    Get the content of a file by its ID.
    
    This endpoint retrieves the content of a file by its unique ID. This is useful when you've already
    found a file through search and want to retrieve its content.
    
    Args:
        file_id: The ID of the file to retrieve
        sibling_chunk_num: Number of sibling chunks to include around each match (default: 1)
        offset: Chunk offset for pagination (0-based)
        
    Returns:
        File content and metadata.
    """
    if not file_indexer:
        raise HTTPException(
            status_code=503, detail="File indexer not initialized"
        )

    try:
        with SessionLocal() as session:
            try:
                # First check if the file table has data
                file_count = session.execute(text("SELECT COUNT(*) FROM code_files")).scalar()
                chunk_count = session.execute(text("SELECT COUNT(*) FROM file_chunks")).scalar()
                print(f"[API] Database has {file_count} files and {chunk_count} chunks")
                
                if file_count == 0:
                    print("[API-ERROR] The code_files table is empty!")
                    raise HTTPException(
                        status_code=500, detail="No files are indexed in the database"
                    )
                
                # Query for the file
                file = session.query(CodeFile).filter_by(id=file_id).first()
                if not file:
                    print(f"[API-ERROR] File with ID {file_id} not found")
                    raise HTTPException(
                        status_code=404, detail=f"File with ID {file_id} not found"
                    )
                
                print(f"[API] Found file ID {file_id} at path {file.file_path}")
                
                # Validate offset
                if offset >= file.chunks_count:
                    if file.chunks_count > 0:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Invalid offset: {offset}. File has {file.chunks_count} chunks (0-{file.chunks_count-1})"
                        )
                    else:
                        # If the file has no chunks, we'll handle it below
                        offset = 0
                
                # Get file content with pagination
                chunks_query = text("""
                    SELECT content FROM file_chunks 
                    WHERE file_id = :file_id
                    ORDER BY chunk_index
                    OFFSET :chunk_offset
                    LIMIT :chunk_limit
                """)
                
                # Determine how many chunks to fetch
                chunks_to_fetch = file.chunks_count
                if sibling_chunk_num is not None and sibling_chunk_num < chunks_to_fetch:
                    chunks_to_fetch = sibling_chunk_num
                
                # Log the query
                formatted_query = str(chunks_query) \
                    .replace(":file_id", str(file.id)) \
                    .replace(":chunk_offset", str(offset)) \
                    .replace(":chunk_limit", str(chunks_to_fetch))
                print(f"[API-SQL] Chunks query: {formatted_query}")
                
                # Get file content
                full_content = ""
                
                # Check if file has chunks
                if file.chunks_count == 0:
                    print(f"[API-WARNING] File {file.id} ({file.file_path}) has no chunks")
                    full_content = "[File has no content chunks]"
                else:
                    chunks = session.execute(chunks_query, {
                        "file_id": file.id,
                        "chunk_offset": offset,
                        "chunk_limit": chunks_to_fetch
                    })
                    
                    chunks_loaded = 0
                    start_chunk = offset
                    end_chunk = offset
                    
                    for chunk in chunks:
                        full_content += chunk.content
                        chunks_loaded += 1
                        end_chunk = offset + chunks_loaded - 1
                    
                    print(f"[API] Loaded {chunks_loaded} chunks for file {file.file_path} (chunks {start_chunk}-{end_chunk})")
                    
                    # Add pagination info
                    has_more = (offset + chunks_loaded) < file.chunks_count
                    next_offset = offset + chunks_loaded if has_more else None
                    
                    # Add truncation notice
                    if sibling_chunk_num is not None and sibling_chunk_num < (file.chunks_count - offset):
                        full_content += f"\n\n[Content truncated: showing chunks {start_chunk}-{end_chunk} of {file.chunks_count} total chunks]"
                
                # Return file information and content
                response = {
                    "id": file.id,
                    "file_path": file.file_path,
                    "language": file.language or "unknown",
                    "repo_name": file.repo_name,
                    "file_size": file.file_size,
                    "chunks_count": file.chunks_count,
                    "content": full_content,
                    "pagination": {
                        "offset": offset,
                        "limit": chunks_loaded,
                        "total_chunks": file.chunks_count,
                        "has_more": offset + chunks_loaded < file.chunks_count,
                        "next_offset": next_offset
                    }
                }
                
                return response
            except HTTPException:
                # Re-raise HTTP exceptions
                raise
            except Exception as db_error:
                error_message = f"Database error while retrieving file ID {file_id}: {db_error}"
                logger.error(error_message)
                print(f"[API-ERROR] {error_message}")
                raise HTTPException(status_code=500, detail=error_message)
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        error_message = f"Error retrieving file with ID {file_id}: {e}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        print(f"[API-ERROR] {error_message}")
        raise HTTPException(
            status_code=500, detail=error_message
        )


# API usage examples
"""
# Curl examples for the API endpoints

# Search code in a repository by name
curl -X POST "http://localhost:8000/code_graph/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "implement transaction",
    "repository_name": "tidb",
    "limit": 5,
    "use_doc_embedding": false
  }'

# Full text search for code in a repository by name
curl -X POST "http://localhost:8000/code_graph/full_text_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "implement transaction",
    "repository_name": "tidb",
    "limit": 5
  }'

# Get repository statistics by name
curl -X GET "http://localhost:8000/code_graph/repositories/stats?repository_name=tidb"

# List all repositories
curl -X GET "http://localhost:8000/code_graph/repositories"

# Check database connection
curl -X GET "http://localhost:8000/db_check"

# Check API status
curl -X GET "http://localhost:8000/"

# File Indexer API Examples

# Vector search (semantic search)
curl -X POST "http://localhost:8000/file_indexer/vector_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "implement transaction",
    "limit": 10,
    "show_content": true,
    "sibling_chunk_num": 2,
    "repository": "tidb"
  }'

# Full text search (exact match)
curl -X POST "http://localhost:8000/file_indexer/full_text_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "createTransaction",
    "limit": 5,
    "show_content": true,
    "repository": "tidb"
  }'

# Legacy GET endpoint (deprecated)
curl -X GET "http://localhost:8000/file_indexer/search?query=implement%20transaction&search_type=vector" \
  -H "Accept: application/json"

# Get file content by path
curl -X GET "http://localhost:8000/file_indexer/file?file_path=%2Fpath%2Fto%2Ffile.py" \
  -H "Accept: application/json"

# Get file content with limited chunks
curl -X GET "http://localhost:8000/file_indexer/file?file_path=%2Fpath%2Fto%2Ffile.py&sibling_chunk_num=2" \
  -H "Accept: application/json"

# Get file content by ID
curl -X GET "http://localhost:8000/file_indexer/file/123" \
  -H "Accept: application/json"

# Get file content by ID with limited chunks
curl -X GET "http://localhost:8000/file_indexer/file/123?sibling_chunk_num=2" \
  -H "Accept: application/json"
"""
