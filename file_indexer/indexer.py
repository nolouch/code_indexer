import os
from tqdm import tqdm
from pathlib import Path
import logging
from sqlalchemy import text
import sys
import traceback

# Add the parent directory to the path to import the setting module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setting.embedding import EMBEDDING_MODEL, VECTOR_SEARCH
from setting.base import DATABASE_URI

# Use absolute imports
from file_indexer.database import (
    CodeFile, FileChunk, get_session, init_db, get_engine, 
    split_content_into_chunks, CHUNK_SIZE
)
from file_indexer.embeddings import CodeEmbedder
from file_indexer.scanner import CodeScanner

logger = logging.getLogger(__name__)

# Maximum file size to index (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024  # Increased maximum file size due to chunk storage

class CodeIndexer:
    def __init__(self, db_path=None, embedding_model=None, embedding_dim=None, ignore_tests=False):
        """Initialize the code indexer with database connection and tools
        
        Args:
            db_path: Database connection string (default from settings)
            embedding_model: Name of the model to use for embeddings (default from settings)
            embedding_dim: Dimension of the embeddings (default from settings)
            ignore_tests: Whether to ignore test files and directories
        """
        # 使用传入参数或setting中的配置
        self.db_path = db_path or DATABASE_URI or 'mysql+pymysql://root@localhost:4000/code_index'
        self.embedding_model = embedding_model or EMBEDDING_MODEL["name"]
        self.embedding_dim = embedding_dim or EMBEDDING_MODEL["dimension"]
        self.ignore_tests = ignore_tests
        
        self.engine = get_engine(self.db_path)
        init_db(self.engine)
        self.session = get_session(self.engine)
        self.scanner = CodeScanner(ignore_tests=self.ignore_tests)
        self.embedder = CodeEmbedder(model_name=self.embedding_model, dim=self.embedding_dim)
        self.using_tidb = 'tidb' in self.db_path.lower() or 'mysql' in self.db_path.lower()
        
    def index_directory(self, directory_path, generate_embeddings=True, repo_name=None):
        """Index all code files in the specified directory
        
        Args:
            directory_path: Path to the directory to index
            generate_embeddings: Whether to generate embeddings
            repo_name: Repository name (if not provided, will be extracted from the directory name)
        """
        directory_path = Path(directory_path).resolve()
        print(f"Indexing files in {directory_path}...")
        
        # If no repo_name provided, use the directory name
        if repo_name is None:
            repo_name = directory_path.name
        
        print(f"Using repository name: {repo_name}")
        
        # Get all files that match our criteria
        files = list(self.scanner.scan_directory(directory_path))
        
        # Process files with a progress bar
        success_count = 0
        skipped_count = 0
        for file_path in tqdm(files, desc="Indexing files"):
            try:
                self.index_file(file_path, generate_embeddings, repo_name)
                success_count += 1
            except Exception as e:
                logger.error(f"Error indexing file {file_path}: {e}")
                skipped_count += 1
                continue
        
        print(f"Indexed {success_count} files, skipped {skipped_count} files.")
    
    def _safe_file_content(self, file_path, content):
        """Ensure file content is valid for database storage"""
        # Check file size - skip if too large
        if len(content) > MAX_FILE_SIZE:
            logger.warning(f"File too large, truncating: {file_path} ({len(content)} bytes)")
            # Truncate to MAX_FILE_SIZE and add a note
            truncated_content = content[:MAX_FILE_SIZE] + "\n\n[CONTENT TRUNCATED: File too large]"
            return truncated_content
        return content
    
    def _detect_language(self, file_path):
        """Detect file language from extension"""
        file_ext = file_path.suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.scala': 'scala',
            '.swift': 'swift',
            '.kt': 'kotlin',
        }
        return language_map.get(file_ext, 'unknown')
    
    def _extract_repo_name(self, file_path):
        """Extract repository name from file path
        
        This tries to extract a reasonable repository name from the file path.
        It looks for common patterns like /repo/path or user/repo format.
        """
        path_parts = Path(file_path).parts
        
        # If we have at least 2 parts, try to get a reasonable repo name
        if len(path_parts) >= 2:
            # For GitHub-like paths, use the last 2 parts of the path before the file
            # Example: /path/to/user/repo/file.py -> user/repo
            if len(path_parts) >= 4:
                return f"{path_parts[-3]}/{path_parts[-2]}"
            # For simple paths, use the parent directory
            # Example: /path/to/repo/file.py -> repo
            return path_parts[-2]
        
        # Fallback: just use the directory name
        return Path(file_path).parent.name
    
    def index_file(self, file_path, generate_embeddings=True, repo_name=None):
        """Index a single file, optionally generating embeddings
        
        Args:
            file_path: Path to the file to index
            generate_embeddings: Whether to generate embeddings
            repo_name: Repository name (will be extracted from path if not provided)
        """
        file_path_str = str(file_path)
        
        # Detect file language
        language = self._detect_language(file_path)
        
        # Extract repository name if not provided
        if repo_name is None:
            repo_name = self._extract_repo_name(file_path)
        
        # Check if file is already indexed
        existing = self.session.query(CodeFile).filter_by(file_path=file_path_str).first()
        
        # Get file content
        content = self.scanner.get_file_content(file_path)
        if content is None:
            print(f"Skipping {file_path} due to read error")
            return
        
        # Process file content to ensure it's safe for database storage
        content = self._safe_file_content(file_path, content)
        file_size = len(content)
        
        # Split content into chunks
        content_chunks = split_content_into_chunks(content)
        chunks_count = len(content_chunks)
        
        # Generate embeddings if requested
        embedding = None
        if generate_embeddings and content:
            try:
                embedding = self.embedder.generate_embedding(content)
            except Exception as e:
                logger.error(f"Error generating embedding for {file_path}: {e}")
        
        try:
            if existing:
                # Update existing file record
                existing.file_size = file_size
                existing.chunks_count = chunks_count
                existing.language = language
                existing.repo_name = repo_name
                
                if embedding is not None:
                    existing.file_embedding = embedding
                
                # Delete old file chunks
                self.session.query(FileChunk).filter_by(file_id=existing.id).delete()
                
                # Add new file chunks
                for i, chunk_content in enumerate(content_chunks):
                    chunk = FileChunk(
                        file_id=existing.id,
                        chunk_index=i,
                        content=chunk_content
                    )
                    self.session.add(chunk)
                
                # Commit changes
                self.session.commit()
            else:
                # Create new file record
                new_file = CodeFile(
                    file_path=file_path_str,
                    repo_name=repo_name,
                    file_size=file_size,
                    chunks_count=chunks_count,
                    language=language,
                    file_embedding=embedding
                )
                self.session.add(new_file)
                self.session.flush()  # Get new file ID
                
                # Add file chunks
                for i, chunk_content in enumerate(content_chunks):
                    chunk = FileChunk(
                        file_id=new_file.id,
                        chunk_index=i,
                        content=chunk_content
                    )
                    self.session.add(chunk)
                
                # Commit changes
                self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"Database error for file {file_path}: {e}")
            raise
    
    def search_similar(self, query_text, limit=None, show_content=True, max_chunks=None, repository=None, search_type="vector"):
        """Search for files similar to the query text
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            show_content: Whether to include file content in results
            max_chunks: Maximum number of chunks to return (None for all chunks)
            repository: Optional repository name to filter results
            search_type: Type of search to perform ("vector" or "full_text")
        """
        if not query_text:
            return []
        
        # 使用setting中的默认限制
        limit = limit or VECTOR_SEARCH["default_limit"]
        
        # Use full text search if requested
        if search_type == "full_text":
            return self.search_full_text(
                query_text, 
                limit=limit,
                show_content=show_content,
                max_chunks=max_chunks,
                repository=repository
            )
            
        try:
            # Generate query embedding
            query_embedding = self.embedder.generate_embedding(query_text)
            
            # Check if database supports vector search
            if self.using_tidb:
                return self._tidb_vector_search(
                    query_embedding, 
                    limit=limit,
                    show_content=show_content,
                    max_chunks=max_chunks,
                    repository=repository
                )
            else:
                raise ValueError("Vector search requires TiDB. Current database does not support vector operations.")
                
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def _tidb_vector_search(self, query_embedding, limit=10, show_content=True, max_chunks=None, repository=None):
        """Use TiDB's native vector search capability
        
        Args:
            query_embedding: The embedding vector for the query
            limit: Maximum number of results to return
            show_content: Whether to include file content in results
            max_chunks: Maximum number of chunks to return (None for all chunks)
            repository: Optional repository name to filter results
        """
        # Convert embedding to TiDB vector format
        query_vector = self.embedder.embedding_to_tidb_vector(query_embedding)
        
        # Build SQL query with vector search
        base_sql = """
            SELECT 
                f.id, 
                f.file_path, 
                f.language,
                f.repo_name,
                f.chunks_count,
                VEC_COSINE_DISTANCE(f.file_embedding, :query_vector) as score
            FROM code_files f
        """
        
        # Add repository filter if provided
        where_clause = ""
        if repository:
            where_clause = " WHERE f.repo_name = :repo_name"
        
        # Complete the query
        sql_query = text(base_sql + where_clause + " ORDER BY score LIMIT :limit")
        
        # Prepare parameters
        params = {"query_vector": query_vector, "limit": limit}
        if repository:
            params["repo_name"] = repository
        
        # Execute query
        results = []
        with self.engine.connect() as conn:
            for row in conn.execute(sql_query, params):
                file = CodeFile(
                    id=row.id,
                    file_path=row.file_path,
                    language=row.language,
                    repo_name=row.repo_name
                )
                
                file_content = ""
                
                if show_content:
                    # Determine how many chunks to fetch
                    chunks_to_fetch = row.chunks_count
                    if max_chunks is not None and max_chunks < chunks_to_fetch:
                        chunks_to_fetch = max_chunks
                    
                    # Construct chunks query with limit if needed
                    chunks_query = text("""
                        SELECT content FROM file_chunks 
                        WHERE file_id = :file_id
                        ORDER BY chunk_index
                        LIMIT :chunk_limit
                    """)
                    
                    # Get chunks content for this file
                    for chunk_row in conn.execute(chunks_query, {
                        "file_id": row.id, 
                        "chunk_limit": chunks_to_fetch
                    }):
                        file_content += chunk_row.content
                    
                    # Add indication if content is truncated
                    if max_chunks is not None and max_chunks < row.chunks_count:
                        file_content += f"\n\n[Content truncated: showing {max_chunks} of {row.chunks_count} chunks]"
                    
                similarity = 1.0 - float(row.score)  # Convert distance to similarity
                results.append((file, similarity, file_content))
                
        return results
    
    def close(self):
        """Close the database session"""
        self.session.close() 
    
    def search_full_text(self, query_text, limit=10, show_content=True, max_chunks=None, repository=None):
        """Search for files containing the query text using full text search
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            show_content: Whether to include file content in results
            max_chunks: Maximum number of chunks to return (None for all chunks)
            repository: Optional repository name to filter results
        """
        try:
            # First check if tables have data
            with self.engine.connect() as conn:
                files_count = conn.execute(text("SELECT COUNT(*) FROM code_files")).scalar()
                chunks_count = conn.execute(text("SELECT COUNT(*) FROM file_chunks")).scalar()
                print(f"[INFO] Database has {files_count} files and {chunks_count} chunks")
                
                if files_count == 0 or chunks_count == 0:
                    print("[INFO] No data available in database tables")
                    return []
            
            # Prepare query - escape special characters for LIKE pattern
            query_pattern = f"%{query_text.replace('%', '\\%').replace('_', '\\_')}%"
            
            # Base query to search in file chunks
            base_sql = """
                SELECT DISTINCT
                    f.id, 
                    f.file_path, 
                    f.language,
                    f.repo_name,
                    f.chunks_count,
                    1.0 as score
                FROM code_files f
                JOIN file_chunks c ON f.id = c.file_id
                WHERE c.content LIKE :query_pattern
            """
            
            # Add repository filter if provided
            if repository:
                base_sql += " AND f.repo_name = :repo_name"
            
            # Complete the query with limit
            sql_query = text(base_sql + " LIMIT :limit")
            
            # Prepare parameters
            params = {"query_pattern": query_pattern, "limit": limit}
            if repository:
                params["repo_name"] = repository
            
            # Log the SQL query
            formatted_sql = str(sql_query)
            for param_name, param_value in params.items():
                formatted_sql = formatted_sql.replace(f":{param_name}", f"'{param_value}'")
            
            #print(f"[SQL] Main search query: {formatted_sql}")
            
            # Execute query
            results = []
            with self.engine.connect() as conn:
                try:
                    result_rows = list(conn.execute(sql_query, params))
                    print(f"[INFO] Found {len(result_rows)} matching files")
                    
                    for row in result_rows:
                        file = CodeFile(
                            id=row.id,
                            file_path=row.file_path,
                            language=row.language,
                            repo_name=row.repo_name
                        )
                        
                        file_content = ""
                        
                        if show_content:
                            # Prioritize chunks that match the query
                            chunks_query = text("""
                                SELECT content, chunk_index FROM file_chunks 
                                WHERE file_id = :file_id
                                ORDER BY 
                                    CASE WHEN content LIKE :query_pattern THEN 0 ELSE 1 END,
                                    chunk_index
                                LIMIT :chunk_limit
                            """)
                            
                            # Determine how many chunks to fetch
                            chunks_to_fetch = row.chunks_count
                            if max_chunks is not None and max_chunks < chunks_to_fetch:
                                chunks_to_fetch = max_chunks
                            
                            # Execute chunks query without too much logging
                            chunks_params = {
                                "file_id": row.id,
                                "query_pattern": query_pattern,
                                "chunk_limit": chunks_to_fetch
                            }
                            
                            # Log just once per query type
                            if row == result_rows[0]:  # Only for the first file
                                # Format the chunks query for logging
                                formatted_chunks_sql = str(chunks_query)
                                for param_name, param_value in chunks_params.items():
                                    if param_name == "file_id":
                                        formatted_chunks_sql = formatted_chunks_sql.replace(f":{param_name}", f"<file_id>")
                                    else:
                                        formatted_chunks_sql = formatted_chunks_sql.replace(f":{param_name}", f"'{param_value}'")
                                
                                print(f"[SQL] Chunks retrieval query: {formatted_chunks_sql}")
                            
                            # Get chunks content for this file
                            try:
                                chunk_results = list(conn.execute(chunks_query, chunks_params))
                                
                                for chunk_row in chunk_results:
                                    file_content += chunk_row.content
                                
                                # Add indication if content is truncated
                                if max_chunks is not None and max_chunks < row.chunks_count:
                                    file_content += f"\n\n[Content truncated: showing {max_chunks} of {row.chunks_count} chunks]"
                            except Exception as chunk_error:
                                print(f"[ERROR] Failed to retrieve chunks for file ID {row.id}: {chunk_error}")
                        
                        # For full text search, we don't have a meaningful similarity score
                        # Just use 1.0 to indicate a match
                        results.append((file, 1.0, file_content))
                except Exception as query_error:
                    print(f"[ERROR] Failed to execute search query: {query_error}")
                    traceback.print_exc()
            
            print(f"[INFO] Full text search completed with {len(results)} results")
            return results
                
        except Exception as e:
            logger.error(f"Error in full text search: {e}")
            logger.error(traceback.format_exc())
            print(f"[ERROR] Exception in full text search: {e}")
            return []