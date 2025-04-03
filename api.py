import os
import logging
import traceback
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Path
from pydantic import BaseModel, Field

from knowledgebase.best_practices import BestPracticesKnowledgeBase
from llm.factory import LLMInterface
from llm.embedding import get_text_embedding
from code_graph.db_manager import GraphDBManager
from code_graph import create_openai_builder
from setting.base import DATABASE_URI
from setting.db import SessionLocal, Base, engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialization ---
llm_client = None
bp_kb = None
code_graph_db = None

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
        },
        "docs": "/docs",
    }


# --- Best Practices Models and Endpoints ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="The search query")


class QueryResponse(BaseModel):
    results: dict = Field(..., description="The search results")


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
    code_content: Optional[str] = Field(None, description="The actual code content of this node")
    doc_content: Optional[str] = Field(None, description="Documentation content for this node")


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
        logger.info(f"Executing code search: {request.query}, repository name: {request.repository_name}")
        
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
                doc_content=doc_content
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
        logger.error("Code graph database not initialized or unavailable, cannot get repository stats")
        raise HTTPException(status_code=503, detail="Code graph database not initialized or unavailable")

    try:
        logger.info(f"Getting repository stats by name: {repository_name}")
        
        stats = code_graph_db.get_repository_stats(
            repository_name=repository_name
        )
        if not stats:
            logger.warning(f"Repository not found: {repository_name}")
            raise HTTPException(status_code=404, detail=f"Repository not found: {repository_name}")
        return stats
    except Exception as e:
        logger.error(f"Error getting repository stats: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving repository stats: {str(e)}")


# Keep the original endpoint for backward compatibility
@app.get("/code_graph/repositories/{repository_path}/stats", tags=["Code Graph"], deprecated=True)
async def get_repository_stats_by_path(repository_path: str = Path(..., description="Path to the code repository")):
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

# Get repository statistics by name
curl -X GET "http://localhost:8000/code_graph/repositories/stats?repository_name=tidb"

# List all repositories
curl -X GET "http://localhost:8000/code_graph/repositories"

# Check database connection
curl -X GET "http://localhost:8000/db_check"

# Check API status
curl -X GET "http://localhost:8000/"
"""
