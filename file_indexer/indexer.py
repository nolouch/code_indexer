import os
from tqdm import tqdm
from pathlib import Path
import logging
from sqlalchemy import text

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
    def __init__(self, db_path='mysql+pymysql://root@localhost:4000/code_index', embedding_model="all-MiniLM-L6-v2", embedding_dim=384):
        """Initialize the code indexer with database connection and tools
        
        Args:
            db_path: Database connection string (default TiDB)
            embedding_model: Name of the model to use for embeddings
            embedding_dim: Dimension of the embeddings
        """
        self.engine = get_engine(db_path)
        init_db(self.engine)
        self.session = get_session(self.engine)
        self.scanner = CodeScanner()
        self.embedder = CodeEmbedder(model_name=embedding_model, dim=embedding_dim)
        self.using_tidb = 'tidb' in db_path.lower() or 'mysql' in db_path.lower()
        
    def index_directory(self, directory_path, generate_embeddings=True):
        """Index all code files in the specified directory"""
        directory_path = Path(directory_path).resolve()
        print(f"Indexing files in {directory_path}...")
        
        # Get all files that match our criteria
        files = list(self.scanner.scan_directory(directory_path))
        
        # Process files with a progress bar
        success_count = 0
        skipped_count = 0
        for file_path in tqdm(files, desc="Indexing files"):
            try:
                self.index_file(file_path, generate_embeddings)
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
    
    def index_file(self, file_path, generate_embeddings=True):
        """Index a single file, optionally generating embeddings"""
        file_path_str = str(file_path)
        
        # Detect file language
        language = self._detect_language(file_path)
        
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
    
    def search_similar(self, query_text, limit=10):
        """Search for files similar to the query text using vector search
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
        """
        if not query_text:
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.embedder.generate_embedding(query_text)
            
            # Check if database supports vector search
            if self.using_tidb:
                return self._tidb_vector_search(query_embedding, limit)
            else:
                return self._in_memory_vector_search(query_embedding, limit)
                
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def _tidb_vector_search(self, query_embedding, limit=10):
        """Use TiDB's native vector search capability"""
        # Convert embedding to TiDB vector format
        query_vector = self.embedder.embedding_to_tidb_vector(query_embedding)
        
        # Build SQL query with vector search
        sql_query = text("""
            SELECT 
                f.id, 
                f.file_path, 
                f.language,
                VEC_COSINE_DISTANCE(f.file_embedding, :query_vector) as score,
                (SELECT content FROM file_chunks 
                 WHERE file_id = f.id AND chunk_index = 0 LIMIT 1) as first_chunk
            FROM code_files f
            ORDER BY score
            LIMIT :limit
        """)
        
        # Execute query
        results = []
        with self.engine.connect() as conn:
            for row in conn.execute(sql_query, {"query_vector": query_vector, "limit": limit}):
                file = CodeFile(
                    id=row.id,
                    file_path=row.file_path,
                    language=row.language
                )
                
                # Get preview content (first 1000 characters of the first chunk)
                preview = row.first_chunk[:1000] if row.first_chunk else ""
                if len(preview) >= 1000:
                    preview += "..."
                    
                similarity = 1.0 - float(row.score)  # Convert distance to similarity
                results.append((file, similarity, preview))
                
        return results
    
    def _in_memory_vector_search(self, query_embedding, limit=10):
        """Fallback to in-memory vector search for non-TiDB databases"""
        # Get all files' metadata and embeddings
        files = self.session.query(CodeFile).all()
        
        # Calculate similarities
        results = []
        for file in files:
            if file.file_embedding:
                file_embedding = self.embedder.tidb_vector_to_embedding(file.file_embedding)
                similarity = self._cosine_similarity(query_embedding, file_embedding)
                
                # Get first chunk as preview
                first_chunk = self.session.query(FileChunk).filter_by(
                    file_id=file.id, chunk_index=0
                ).first()
                
                preview = ""
                if first_chunk:
                    preview = first_chunk.content[:1000]
                    if len(first_chunk.content) > 1000:
                        preview += "..."
                
                results.append((file, similarity, preview))
        
        # Sort by similarity in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return float(a.dot(b) / (self._magnitude(a) * self._magnitude(b)))
    
    def _magnitude(self, vector):
        """Calculate the magnitude of a vector"""
        return float(vector.dot(vector) ** 0.5)
    
    def close(self):
        """Close the database session"""
        self.session.close() 