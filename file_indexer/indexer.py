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
from setting.fileindexer_llm import (
    FILEINDER_LLM_PROVIDER, 
    FILEINDER_LLM_MODEL, 
    FILEINDER_COMMENTS_SYSTEM_PROMPT,
    FILEINDER_COMMENTS_MAX_LENGTH,
    FILEINDER_GENERATE_COMMENTS
)

# Use absolute imports
from file_indexer.database import (
    CodeFile, FileChunk, get_session, init_db, get_engine, 
    split_content_into_chunks
)
from file_indexer.embeddings import CodeEmbedder
from file_indexer.scanner import CodeScanner
from llm.factory import LLMInterface

logger = logging.getLogger(__name__)

# Maximum file size to index (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024  # Increased maximum file size due to chunk storage

class CodeIndexer:
    def __init__(self, db_path=None, embedding_model=None, embedding_dim=None, ignore_tests=False, 
                 llm_provider=None, llm_model=None):
        """Initialize the code indexer with database connection and tools
        
        Args:
            db_path: Database connection string (default from settings)
            embedding_model: Name of the model to use for embeddings (default from settings)
            embedding_dim: Dimension of the embeddings (default from settings)
            ignore_tests: Whether to ignore test files and directories
            llm_provider: LLM provider for generating comments (default from settings)
            llm_model: LLM model for generating comments (default from settings)
        """
        # Use provided parameters or defaults from settings
        self.db_path = db_path or DATABASE_URI or 'mysql+pymysql://root@localhost:4000/code_index'
        self.embedding_model = embedding_model or EMBEDDING_MODEL["name"]
        self.embedding_dim = embedding_dim or EMBEDDING_MODEL["dimension"]
        self.ignore_tests = ignore_tests
        self.llm_provider = llm_provider or FILEINDER_LLM_PROVIDER
        self.llm_model = llm_model or FILEINDER_LLM_MODEL
        
        self.engine = get_engine(self.db_path)
        init_db(self.engine)
        self.session = get_session(self.engine)
        self.scanner = CodeScanner(ignore_tests=self.ignore_tests)
        self.embedder = CodeEmbedder(model_name=self.embedding_model, dim=self.embedding_dim)
        self.using_tidb = 'tidb' in self.db_path.lower() or 'mysql' in self.db_path.lower()
        
        # Initialize LLM for code comment generation
        self.llm = None
        print(f"LLM settings: FILEINDER_GENERATE_COMMENTS={FILEINDER_GENERATE_COMMENTS}, provider={self.llm_provider}, model={self.llm_model}")
        if FILEINDER_GENERATE_COMMENTS and self.llm_provider and self.llm_model:
            try:
                logger.info(f"Initializing LLM with provider: {self.llm_provider}, model: {self.llm_model}")
                print(f"Attempting to initialize LLM with provider: {self.llm_provider}, model: {self.llm_model}")
                self.llm = LLMInterface(provider=self.llm_provider, model=self.llm_model)
                print(f"LLM initialized successfully: {self.llm is not None}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                print(f"Failed to initialize LLM: {e}")
                self.llm = None
        else:
            print(f"Skipping LLM initialization. Generate comments: {FILEINDER_GENERATE_COMMENTS}, Provider: {self.llm_provider}, Model: {self.llm_model}")
        
    def index_directory(self, directory_path, generate_embeddings=True, generate_comments=True, repo_name=None, llm_provider=None, llm_model=None):
        """Index all code files in the specified directory
        
        Args:
            directory_path: Path to the directory to index
            generate_embeddings: Whether to generate embeddings
            generate_comments: Whether to generate LLM comments
            repo_name: Repository name (if not provided, will be extracted from the directory name)
            llm_provider: Override LLM provider
            llm_model: Override LLM model
        """
        # Re-initialize LLM if provider or model is specified and different from current
        if (llm_provider and llm_provider != self.llm_provider) or (llm_model and llm_model != self.llm_model):
            try:
                logger.info(f"Re-initializing LLM with provider: {llm_provider or self.llm_provider}, model: {llm_model or self.llm_model}")
                self.llm_provider = llm_provider or self.llm_provider
                self.llm_model = llm_model or self.llm_model
                self.llm = LLMInterface(provider=self.llm_provider, model=self.llm_model)
            except Exception as e:
                logger.error(f"Failed to re-initialize LLM: {e}")
                self.llm = None
        
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
                self.index_file(
                    file_path, 
                    generate_embeddings=generate_embeddings,
                    generate_comments=generate_comments and self.llm is not None,
                    repo_name=repo_name
                )
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
    
    def index_file(self, file_path, generate_embeddings=True, generate_comments=True, repo_name=None):
        """Index a single file, optionally generating embeddings and comments
        
        Args:
            file_path: Path to the file to index
            generate_embeddings: Whether to generate embeddings
            generate_comments: Whether to generate LLM comments
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
        
        # Split content into chunks by lines
        content_chunks, line_count = split_content_into_chunks(content, lines_per_chunk=200)
        chunks_count = len(content_chunks)
        print(f"File {file_path} has {line_count} lines split into {chunks_count} chunks")
        
        # Generate comments if requested and llm is available
        llm_comments = None
        comments_embedding = None
        if generate_comments and self.llm:
            try:
                print(f"Generating comments for {file_path} using {self.llm_provider}/{self.llm_model}")
                llm_comments = self.generate_code_comments(content, language)
                print(f"Generated comments: {llm_comments is not None}")
                # Generate embeddings for comments if comments were generated
                if llm_comments and generate_embeddings:
                    print(f"Generating embedding for comments")
                    comments_embedding = self.embedder.generate_embedding(llm_comments)
                    print(f"Generated comments embedding: {comments_embedding is not None}")
            except Exception as e:
                logger.error(f"Error generating comments for {file_path}: {e}")
                print(f"Error generating comments: {e}")
        else:
            if not generate_comments:
                print(f"Comments generation is disabled")
            if not self.llm:
                print(f"LLM is not initialized: provider={self.llm_provider}, model={self.llm_model}")
        
        # Generate embeddings for content if requested
        content_embedding = None
        if generate_embeddings and content:
            try:
                print(f"Generating embedding for file content")
                content_embedding = self.embedder.generate_embedding(content)
                print(f"Generated content embedding: {content_embedding is not None}")
            except Exception as e:
                logger.error(f"Error generating embedding for {file_path}: {e}")
                print(f"Error generating embedding: {e}")
        else:
            if not generate_embeddings:
                print(f"Embeddings generation is disabled")
        
        try:
            if existing:
                # Update existing file record
                existing.file_size = file_size
                existing.chunks_count = chunks_count
                existing.line_count = line_count
                existing.language = language
                existing.repo_name = repo_name
                
                if content_embedding is not None:
                    existing.file_embedding = content_embedding
                
                if llm_comments is not None:
                    existing.llm_comments = llm_comments
                    
                if comments_embedding is not None:
                    existing.comments_embedding = comments_embedding
                
                # Delete old file chunks
                self.session.query(FileChunk).filter_by(file_id=existing.id).delete()
                
                # Add new file chunks
                for i, (chunk_content, line_range, start_line, end_line) in enumerate(content_chunks):
                    chunk = FileChunk(
                        file_id=existing.id,
                        chunk_index=i,
                        content=chunk_content,
                        line_range=line_range,
                        start_line=start_line,
                        end_line=end_line
                    )
                    self.session.add(chunk)
                
                # Commit changes immediately for this file
                self.session.commit()
            else:
                # Create new file record
                new_file = CodeFile(
                    file_path=file_path_str,
                    repo_name=repo_name,
                    file_size=file_size,
                    chunks_count=chunks_count,
                    line_count=line_count,
                    language=language,
                    file_embedding=content_embedding,
                    llm_comments=llm_comments,
                    comments_embedding=comments_embedding
                )
                self.session.add(new_file)
                self.session.flush()  # Get new file ID
                
                # Add file chunks
                for i, (chunk_content, line_range, start_line, end_line) in enumerate(content_chunks):
                    chunk = FileChunk(
                        file_id=new_file.id,
                        chunk_index=i,
                        content=chunk_content,
                        line_range=line_range,
                        start_line=start_line,
                        end_line=end_line
                    )
                    self.session.add(chunk)
                
                # Commit changes immediately for this file
                self.session.commit()
                
        except Exception as e:
            self.session.rollback()
            logger.error(f"Database error for file {file_path}: {e}")
            raise
    
    def search_similar(self, query_text, limit=None, show_content=True, repository=None, search_type="vector", max_lines=600):
        """Search for files similar to the query text
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            show_content: Whether to include file content in results
            repository: Optional repository name to filter results
            search_type: Type of search to perform ("vector", "full_text", "combined", or "comments")
            max_lines: Maximum number of lines to show per file (default: 600)
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
                repository=repository,
                max_lines=max_lines
            )
            
        try:
            # Generate query embedding
            query_embedding = self.embedder.generate_embedding(query_text)
            
            # Check if database supports vector search
            if self.using_tidb:
                if search_type == "combined":
                    return self._tidb_combined_search(
                        query_embedding, 
                        limit=limit,
                        show_content=show_content,
                        repository=repository,
                        max_lines=max_lines
                    )
                elif search_type == "comments":
                    return self._tidb_comments_search(
                        query_embedding, 
                        limit=limit,
                        show_content=show_content,
                        repository=repository,
                        max_lines=max_lines
                    )
                else:
                    return self._tidb_vector_search(
                        query_embedding, 
                        limit=limit,
                        show_content=show_content,
                        repository=repository,
                        max_lines=max_lines
                    )
            else:
                raise ValueError("Vector search requires TiDB. Current database does not support vector operations.")
                
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def _tidb_vector_search(self, query_embedding, limit=10, show_content=True, repository=None, max_lines=600):
        """Use TiDB's native vector search capability
        
        Args:
            query_embedding: The embedding vector for the query
            limit: Maximum number of results to return
            show_content: Whether to include file content in results
            repository: Optional repository name to filter results
            max_lines: Maximum number of lines to show per file
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
                f.line_count,
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
                    repo_name=row.repo_name,
                    chunks_count=row.chunks_count,
                    line_count=getattr(row, 'line_count', None)
                )
                
                file_content = ""
                
                if show_content:
                    # Get the average number of lines per chunk in this file
                    avg_lines_per_chunk = 100  # Default fallback value
                    if hasattr(file, 'line_count') and file.line_count is not None and file.chunks_count > 0:
                        avg_lines_per_chunk = file.line_count / file.chunks_count
                    
                    # Calculate how many chunks we need to fetch to get approximately max_lines
                    chunks_needed = int(max_lines / avg_lines_per_chunk) + 1 if avg_lines_per_chunk > 0 else 5
                    
                    # Limit the chunks to fetch
                    chunks_to_fetch = min(chunks_needed, row.chunks_count)
                    
                    # Construct chunks query with limit if needed
                    chunks_query = text("""
                        SELECT content, line_range, start_line, end_line FROM file_chunks 
                        WHERE file_id = :file_id
                        ORDER BY chunk_index
                        LIMIT :chunk_limit
                    """)
                    
                    # Get chunks content for this file
                    start_chunk_line = None
                    end_chunk_line = None
                    
                    for chunk_row in conn.execute(chunks_query, {
                        "file_id": row.id, 
                        "chunk_limit": chunks_to_fetch
                    }):
                        file_content += chunk_row.content
                        
                        # Try to extract line range information if available
                        if hasattr(chunk_row, 'start_line') and chunk_row.start_line:
                            if start_chunk_line is None or chunk_row.start_line < start_chunk_line:
                                start_chunk_line = chunk_row.start_line
                                
                        if hasattr(chunk_row, 'end_line') and chunk_row.end_line:
                            if end_chunk_line is None or chunk_row.end_line > end_chunk_line:
                                end_chunk_line = chunk_row.end_line
                    
                    # Add line range info to the file object if it was found in chunks
                    if start_chunk_line is not None:
                        file.start_line = start_chunk_line
                    if end_chunk_line is not None:
                        file.end_line = end_chunk_line
                        
                    # Add indication if content is truncated
                    # If more lines than requested, add a note
                    if end_chunk_line and start_chunk_line and end_chunk_line - start_chunk_line + 1 > max_lines:
                        file_content += f"\n\n[Content truncated: showing approximately {max_lines} lines. File has {getattr(file, 'line_count', 'unknown')} total lines]"
                    # If chunks were limited, add a note
                    elif chunks_to_fetch < row.chunks_count:
                        total_lines = getattr(file, 'line_count', 'unknown')
                        file_content += f"\n\n[Content truncated: showing {chunks_to_fetch} of {row.chunks_count} chunks, approximately {end_chunk_line - start_chunk_line + 1 if end_chunk_line and start_chunk_line else 'unknown'} lines out of {total_lines} total]"
                
                # Try to fetch LLM comments if they exist
                try:
                    llm_comments_query = text("""
                        SELECT llm_comments FROM code_files 
                        WHERE id = :file_id AND llm_comments IS NOT NULL
                    """)
                    
                    llm_comments_result = conn.execute(llm_comments_query, {"file_id": row.id}).first()
                    if llm_comments_result and llm_comments_result.llm_comments:
                        file.llm_comments = llm_comments_result.llm_comments
                except Exception as e:
                    logger.error(f"Error fetching LLM comments for file {row.id}: {e}")
                
                similarity = 1.0 - float(row.score)  # Convert distance to similarity
                results.append((file, similarity, file_content))
                
        return results
    
    def _tidb_combined_search(self, query_embedding, limit=10, show_content=True, repository=None, max_lines=600):
        """Use TiDB's native vector search capability with combined file and comments embeddings
        
        This method searches both code and comments embeddings and combines the results
        with a weighted score.
        
        Args:
            query_embedding: The embedding vector for the query
            limit: Maximum number of results to return
            show_content: Whether to include file content in results
            repository: Optional repository name to filter results
            max_lines: Maximum number of lines to show per file
        """
        # Convert embedding to TiDB vector format
        query_vector = self.embedder.embedding_to_tidb_vector(query_embedding)
        
        # Build SQL query with vector search for both file and comments embeddings
        base_sql = """
            SELECT 
                f.id, 
                f.file_path, 
                f.language,
                f.repo_name,
                f.chunks_count,
                f.line_count,
                VEC_COSINE_DISTANCE(f.file_embedding, :query_vector) as file_score,
                CASE WHEN f.comments_embedding IS NOT NULL 
                     THEN VEC_COSINE_DISTANCE(f.comments_embedding, :query_vector) 
                     ELSE 1.0 END as comment_score
            FROM code_files f
            WHERE f.file_embedding IS NOT NULL
        """
        
        # Add repository filter if provided
        if repository:
            base_sql += " AND f.repo_name = :repo_name"
        
        # Calculate a combined score with weights (0.7 for code, 0.3 for comments)
        # Lower score = better match
        combined_score_clause = """
            (0.7 * VEC_COSINE_DISTANCE(f.file_embedding, :query_vector) + 
             0.3 * CASE WHEN f.comments_embedding IS NOT NULL 
                        THEN VEC_COSINE_DISTANCE(f.comments_embedding, :query_vector) 
                        ELSE 1.0 END)
        """
        
        # Complete the query with combined score ordering
        sql_query = text(base_sql + f" ORDER BY {combined_score_clause} LIMIT :limit")
        
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
                    repo_name=row.repo_name,
                    chunks_count=row.chunks_count,
                    line_count=getattr(row, 'line_count', None)
                )
                
                file_content = ""
                
                if show_content:
                    # Get the average number of lines per chunk in this file
                    avg_lines_per_chunk = 100  # Default fallback value
                    if hasattr(file, 'line_count') and file.line_count is not None and file.chunks_count > 0:
                        avg_lines_per_chunk = file.line_count / file.chunks_count
                    
                    # Calculate how many chunks we need to fetch to get approximately max_lines
                    chunks_needed = int(max_lines / avg_lines_per_chunk) + 1 if avg_lines_per_chunk > 0 else 5
                    
                    # Limit the chunks to fetch
                    chunks_to_fetch = min(chunks_needed, row.chunks_count)
                    
                    # Construct chunks query with limit if needed
                    chunks_query = text("""
                        SELECT content, line_range, start_line, end_line FROM file_chunks 
                        WHERE file_id = :file_id
                        ORDER BY chunk_index
                        LIMIT :chunk_limit
                    """)
                    
                    # Get chunks content for this file
                    start_chunk_line = None
                    end_chunk_line = None
                    
                    for chunk_row in conn.execute(chunks_query, {
                        "file_id": row.id, 
                        "chunk_limit": chunks_to_fetch
                    }):
                        file_content += chunk_row.content
                        
                        # Try to extract line range information if available
                        if hasattr(chunk_row, 'start_line') and chunk_row.start_line:
                            if start_chunk_line is None or chunk_row.start_line < start_chunk_line:
                                start_chunk_line = chunk_row.start_line
                                
                        if hasattr(chunk_row, 'end_line') and chunk_row.end_line:
                            if end_chunk_line is None or chunk_row.end_line > end_chunk_line:
                                end_chunk_line = chunk_row.end_line
                    
                    # Add line range info to the file object if it was found in chunks
                    if start_chunk_line is not None:
                        file.start_line = start_chunk_line
                    if end_chunk_line is not None:
                        file.end_line = end_chunk_line
                        
                    # Add indication if content is truncated
                    # If more lines than requested, add a note
                    if end_chunk_line and start_chunk_line and end_chunk_line - start_chunk_line + 1 > max_lines:
                        file_content += f"\n\n[Content truncated: showing approximately {max_lines} lines. File has {getattr(file, 'line_count', 'unknown')} total lines]"
                    # If chunks were limited, add a note
                    elif chunks_to_fetch < row.chunks_count:
                        total_lines = getattr(file, 'line_count', 'unknown')
                        file_content += f"\n\n[Content truncated: showing {chunks_to_fetch} of {row.chunks_count} chunks, approximately {end_chunk_line - start_chunk_line + 1 if end_chunk_line and start_chunk_line else 'unknown'} lines out of {total_lines} total]"
                
                # Try to fetch LLM comments if they exist
                try:
                    llm_comments_query = text("""
                        SELECT llm_comments FROM code_files 
                        WHERE id = :file_id AND llm_comments IS NOT NULL
                    """)
                    
                    llm_comments_result = conn.execute(llm_comments_query, {"file_id": row.id}).first()
                    if llm_comments_result and llm_comments_result.llm_comments:
                        file.llm_comments = llm_comments_result.llm_comments
                except Exception as e:
                    logger.error(f"Error fetching LLM comments for file {row.id}: {e}")
                
                # Calculate combined similarity score
                file_score = float(row.file_score)
                comment_score = float(row.comment_score)
                
                # Calculate final similarity (inverse of distance, higher is better)
                combined_score = 0.7 * (1.0 - file_score) + 0.3 * (1.0 - comment_score)
                results.append((file, combined_score, file_content))
                
        return results
    
    def _tidb_comments_search(self, query_embedding, limit=10, show_content=True, repository=None, max_lines=600):
        """Use TiDB's native vector search capability with only comments embeddings
        
        This method searches only in the comments embeddings.
        
        Args:
            query_embedding: The embedding vector for the query
            limit: Maximum number of results to return
            show_content: Whether to include file content in results
            repository: Optional repository name to filter results
            max_lines: Maximum number of lines to show per file
        """
        # Convert embedding to TiDB vector format
        query_vector = self.embedder.embedding_to_tidb_vector(query_embedding)
        
        # Build SQL query with vector search for comments embeddings only
        base_sql = """
            SELECT 
                f.id, 
                f.file_path, 
                f.language,
                f.repo_name,
                f.chunks_count,
                f.line_count,
                VEC_COSINE_DISTANCE(f.comments_embedding, :query_vector) as comment_score
            FROM code_files f
            WHERE f.comments_embedding IS NOT NULL
        """
        
        # Add repository filter if provided
        if repository:
            base_sql += " AND f.repo_name = :repo_name"
        
        # Complete the query with score ordering
        sql_query = text(base_sql + " ORDER BY comment_score LIMIT :limit")
        
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
                    repo_name=row.repo_name,
                    chunks_count=row.chunks_count,
                    line_count=getattr(row, 'line_count', None)
                )
                
                file_content = ""
                
                if show_content:
                    # Get the average number of lines per chunk in this file
                    avg_lines_per_chunk = 100  # Default fallback value
                    if hasattr(file, 'line_count') and file.line_count is not None and file.chunks_count > 0:
                        avg_lines_per_chunk = file.line_count / file.chunks_count
                    
                    # Calculate how many chunks we need to fetch to get approximately max_lines
                    chunks_needed = int(max_lines / avg_lines_per_chunk) + 1 if avg_lines_per_chunk > 0 else 5
                    
                    # Limit the chunks to fetch
                    chunks_to_fetch = min(chunks_needed, row.chunks_count)
                    
                    # Construct chunks query with limit if needed
                    chunks_query = text("""
                        SELECT content, line_range, start_line, end_line FROM file_chunks 
                        WHERE file_id = :file_id
                        ORDER BY chunk_index
                        LIMIT :chunk_limit
                    """)
                    
                    # Get chunks content for this file
                    start_chunk_line = None
                    end_chunk_line = None
                    
                    for chunk_row in conn.execute(chunks_query, {
                        "file_id": row.id, 
                        "chunk_limit": chunks_to_fetch
                    }):
                        file_content += chunk_row.content
                        
                        # Try to extract line range information if available
                        if hasattr(chunk_row, 'start_line') and chunk_row.start_line:
                            if start_chunk_line is None or chunk_row.start_line < start_chunk_line:
                                start_chunk_line = chunk_row.start_line
                                
                        if hasattr(chunk_row, 'end_line') and chunk_row.end_line:
                            if end_chunk_line is None or chunk_row.end_line > end_chunk_line:
                                end_chunk_line = chunk_row.end_line
                    
                    # Add line range info to the file object if it was found in chunks
                    if start_chunk_line is not None:
                        file.start_line = start_chunk_line
                    if end_chunk_line is not None:
                        file.end_line = end_chunk_line
                        
                    # Add indication if content is truncated
                    # If more lines than requested, add a note
                    if end_chunk_line and start_chunk_line and end_chunk_line - start_chunk_line + 1 > max_lines:
                        file_content += f"\n\n[Content truncated: showing approximately {max_lines} lines. File has {getattr(file, 'line_count', 'unknown')} total lines]"
                    # If chunks were limited, add a note
                    elif chunks_to_fetch < row.chunks_count:
                        total_lines = getattr(file, 'line_count', 'unknown')
                        file_content += f"\n\n[Content truncated: showing {chunks_to_fetch} of {row.chunks_count} chunks, approximately {end_chunk_line - start_chunk_line + 1 if end_chunk_line and start_chunk_line else 'unknown'} lines out of {total_lines} total]"
                
                # Try to fetch LLM comments if they exist
                try:
                    llm_comments_query = text("""
                        SELECT llm_comments FROM code_files 
                        WHERE id = :file_id AND llm_comments IS NOT NULL
                    """)
                    
                    llm_comments_result = conn.execute(llm_comments_query, {"file_id": row.id}).first()
                    if llm_comments_result and llm_comments_result.llm_comments:
                        file.llm_comments = llm_comments_result.llm_comments
                except Exception as e:
                    logger.error(f"Error fetching LLM comments for file {row.id}: {e}")
                
                # Calculate final similarity (inverse of distance, higher is better)
                similarity = 1.0 - float(row.comment_score)
                results.append((file, similarity, file_content))
                
        return results
    
    def close(self):
        """Close the database session"""
        self.session.close() 
    
    def search_full_text(self, query_text, limit=10, show_content=True, repository=None, max_lines=600):
        """Search for files containing the query text using full text search
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            show_content: Whether to include file content in results
            repository: Optional repository name to filter results
            max_lines: Maximum number of lines to show per file
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
            
            # Check if we can use TiDB's FULLTEXT search capabilities
            use_tidb_fts = False
            if self.using_tidb:
                # Check if the FULLTEXT index exists
                try:
                    with self.engine.connect() as conn:
                        # Check for the existence of FULLTEXT index
                        index_exists = conn.execute(text("""
                            SELECT COUNT(*) 
                            FROM INFORMATION_SCHEMA.STATISTICS 
                            WHERE TABLE_NAME = 'file_chunks' 
                            AND INDEX_NAME = 'idx_content_fulltext'
                        """)).scalar()
                        
                        # Only use FULLTEXT search if the index exists
                        if index_exists > 0:
                            use_tidb_fts = True
                            print(f"[INFO] FULLTEXT index found on file_chunks.content")
                        else:
                            print(f"[WARNING] FULLTEXT index not found. Using LIKE-based search instead.")
                except Exception as e:
                    print(f"[WARNING] Error checking for FULLTEXT index: {e}")
                    # Fall back to LIKE-based search
                    use_tidb_fts = False
            use_tidb_fts = False # manually disable TiDB fulltext search, TODO remote this
            if use_tidb_fts:
                # Use optimized two-step search approach with TiDB FULLTEXT capabilities
                return self._tidb_fulltext_search(
                    query_text,
                    limit=limit,
                    show_content=show_content,
                    repository=repository,
                    max_lines=max_lines
                )
            else:
                # Fallback to LIKE-based search
                print(f"[INFO] Using LIKE-based search with query: {query_text}")
                
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
                
                print(f"[SQL] Main search query: {formatted_sql}")
                
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
                                repo_name=row.repo_name,
                                chunks_count=row.chunks_count
                            )
                            
                            file_content = ""
                            start_chunk_line = None
                            end_chunk_line = None
                            
                            if show_content:
                                # Get the average number of lines per chunk in this file
                                avg_lines_per_chunk = 100  # Default fallback value
                                if hasattr(file, 'line_count') and file.line_count is not None and file.chunks_count > 0:
                                    avg_lines_per_chunk = file.line_count / file.chunks_count
                                
                                # Calculate how many chunks we need to fetch to get approximately max_lines
                                chunks_needed = int(max_lines / avg_lines_per_chunk) + 1 if avg_lines_per_chunk > 0 else 5
                                
                                # Limit the chunks to fetch
                                chunks_to_fetch = min(chunks_needed, file.chunks_count)
                                
                                # Prioritize chunks that match the query
                                chunks_query = text("""
                                    SELECT content, line_range, start_line, end_line, chunk_index FROM file_chunks 
                                    WHERE file_id = :file_id
                                    ORDER BY 
                                        CASE WHEN content LIKE :query_pattern THEN 0 ELSE 1 END,
                                        chunk_index
                                    LIMIT :chunk_limit
                                """)
                                
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
                                        
                                        # Try to extract line range information if available
                                        if hasattr(chunk_row, 'start_line') and chunk_row.start_line:
                                            if start_chunk_line is None or chunk_row.start_line < start_chunk_line:
                                                start_chunk_line = chunk_row.start_line
                                                
                                        if hasattr(chunk_row, 'end_line') and chunk_row.end_line:
                                            if end_chunk_line is None or chunk_row.end_line > end_chunk_line:
                                                end_chunk_line = chunk_row.end_line
                                    
                                    # Add indication if content is truncated
                                    # If more lines than requested, add a note
                                    if end_chunk_line - start_chunk_line + 1 > max_lines:
                                        file_content += f"\n\n[Content truncated: showing approximately {max_lines} lines. File has {file.line_count or 'unknown'} total lines]"
                                    # If chunks were limited, add a note
                                    elif chunks_to_fetch < file.chunks_count:
                                        total_lines = file.line_count or 'unknown'
                                        file_content += f"\n\n[Content truncated: showing {chunks_to_fetch} of {file.chunks_count} chunks, approximately {end_chunk_line - start_chunk_line + 1} lines out of {total_lines} total]"
                                except Exception as chunk_error:
                                    print(f"[ERROR] Failed to retrieve chunks for file ID {file.id}: {chunk_error}")
                            
                            # Add line range info to the file object if it was found in chunks
                            if start_chunk_line is not None:
                                file.start_line = start_chunk_line
                            if end_chunk_line is not None:
                                file.end_line = end_chunk_line
                                
                            # Try to fetch LLM comments if they exist
                            try:
                                llm_comments_query = text("""
                                    SELECT llm_comments FROM code_files 
                                    WHERE id = :file_id AND llm_comments IS NOT NULL
                                """)
                                
                                llm_comments_result = conn.execute(llm_comments_query, {"file_id": row.id}).first()
                                if llm_comments_result and llm_comments_result.llm_comments:
                                    file.llm_comments = llm_comments_result.llm_comments
                            except Exception as e:
                                logger.error(f"Error fetching LLM comments for file {row.id}: {e}")
                            
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

    def _tidb_fulltext_search(self, query_text, limit=10, show_content=True, repository=None, max_lines=600):
        """Search files using full-text search in TiDB
        
        Args:
            query_text: Search query string
            limit: Maximum number of results to return 
            show_content: Whether to include file content in results
            repository: Optional repository name to filter results
            max_lines: Maximum number of lines to show per file
        """
        # Prepare the full-text search query
        # We'll search in both file_tokens and content fields
        base_sql = """
            SELECT 
                f.id, 
                f.file_path, 
                f.language,
                f.repo_name,
                f.chunks_count,
                f.line_count,
                MATCH(f.file_tokens) AGAINST(:query IN NATURAL LANGUAGE MODE) as score
            FROM code_files f
            WHERE MATCH(f.file_tokens) AGAINST(:query IN NATURAL LANGUAGE MODE)
        """
        
        # Add repository filter if provided
        if repository:
            base_sql += " AND f.repo_name = :repo_name"
            
        # Complete the query with score ordering
        sql_query = text(base_sql + " ORDER BY score DESC LIMIT :limit")
        
        # Prepare parameters
        params = {"query": query_text, "limit": limit}
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
                    repo_name=row.repo_name,
                    chunks_count=row.chunks_count,
                    line_count=getattr(row, 'line_count', None)
                )
                
                file_content = ""
                
                if show_content:
                    # Get the average number of lines per chunk in this file
                    avg_lines_per_chunk = 100  # Default fallback value
                    if hasattr(file, 'line_count') and file.line_count is not None and file.chunks_count > 0:
                        avg_lines_per_chunk = file.line_count / file.chunks_count
                    
                    # Calculate how many chunks we need to fetch to get approximately max_lines
                    chunks_needed = int(max_lines / avg_lines_per_chunk) + 1 if avg_lines_per_chunk > 0 else 5
                    
                    # Limit the chunks to fetch
                    chunks_to_fetch = min(chunks_needed, row.chunks_count)
                    
                    # Construct chunks query with limit if needed
                    chunks_query = text("""
                        SELECT content, line_range, start_line, end_line FROM file_chunks 
                        WHERE file_id = :file_id
                        ORDER BY chunk_index
                        LIMIT :chunk_limit
                    """)
                    
                    # Get chunks content for this file
                    start_chunk_line = None
                    end_chunk_line = None
                    
                    for chunk_row in conn.execute(chunks_query, {
                        "file_id": row.id, 
                        "chunk_limit": chunks_to_fetch
                    }):
                        file_content += chunk_row.content
                        
                        # Try to extract line range information if available
                        if hasattr(chunk_row, 'start_line') and chunk_row.start_line:
                            if start_chunk_line is None or chunk_row.start_line < start_chunk_line:
                                start_chunk_line = chunk_row.start_line
                                
                        if hasattr(chunk_row, 'end_line') and chunk_row.end_line:
                            if end_chunk_line is None or chunk_row.end_line > end_chunk_line:
                                end_chunk_line = chunk_row.end_line
                    
                    # Add line range info to the file object if it was found in chunks
                    if start_chunk_line is not None:
                        file.start_line = start_chunk_line
                    if end_chunk_line is not None:
                        file.end_line = end_chunk_line
                        
                    # Add indication if content is truncated
                    # If more lines than requested, add a note
                    if end_chunk_line and start_chunk_line and end_chunk_line - start_chunk_line + 1 > max_lines:
                        file_content += f"\n\n[Content truncated: showing approximately {max_lines} lines. File has {getattr(file, 'line_count', 'unknown')} total lines]"
                    # If chunks were limited, add a note
                    elif chunks_to_fetch < row.chunks_count:
                        total_lines = getattr(file, 'line_count', 'unknown')
                        file_content += f"\n\n[Content truncated: showing {chunks_to_fetch} of {row.chunks_count} chunks, approximately {end_chunk_line - start_chunk_line + 1 if end_chunk_line and start_chunk_line else 'unknown'} lines out of {total_lines} total]"
                
                # Try to fetch LLM comments if they exist
                try:
                    llm_comments_query = text("""
                        SELECT llm_comments FROM code_files 
                        WHERE id = :file_id AND llm_comments IS NOT NULL
                    """)
                    
                    llm_comments_result = conn.execute(llm_comments_query, {"file_id": row.id}).first()
                    if llm_comments_result and llm_comments_result.llm_comments:
                        file.llm_comments = llm_comments_result.llm_comments
                except Exception as e:
                    logger.error(f"Error fetching LLM comments for file {row.id}: {e}")
                
                # Calculate final similarity (we use the MySQL fulltext match score directly)
                similarity = float(row.score)
                results.append((file, similarity, file_content))
                
        return results

    def generate_code_comments(self, code_content, language=None):
        """Generate comments describing the code using LLM
        
        Args:
            code_content: The source code to describe
            language: Programming language of the code
            
        Returns:
            A string containing the LLM-generated comments
        """
        if not self.llm:
            logger.warning("LLM not initialized, cannot generate comments")
            return None
            
        # Limit content size to avoid token limits
        if len(code_content) > FILEINDER_COMMENTS_MAX_LENGTH:
            logger.warning(f"Code content too large ({len(code_content)} chars), truncating for comment generation")
            code_content = code_content[:FILEINDER_COMMENTS_MAX_LENGTH] + "\n\n[CONTENT TRUNCATED]"
            
        language_info = f"(language: {language})" if language else ""
        
        try:
            # Use system prompt from settings
            system_prompt = FILEINDER_COMMENTS_SYSTEM_PROMPT
            
            prompt = f"Here is some source code {language_info}. Please describe what it does:\n\n```\n{code_content}\n```"
            
            comments = self.llm.generate(prompt=prompt, system_prompt=system_prompt)
            return comments
        except Exception as e:
            logger.error(f"Error generating code comments: {e}")
            return None