from sqlalchemy import Column, Integer, String, Text, LargeBinary, create_engine, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.types import UserDefinedType
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

# Default embedding dimension
EMBEDDING_DIM = 384  # Using the default for sentence-transformers MiniLM model

# Maximum size of each chunk (1MB)
CHUNK_SIZE = 1 * 1024 * 1024

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
                
            # Ensure value is properly formatted
            if isinstance(value, str):
                if not (value.startswith('[') and value.endswith(']')):
                    if self.dimensions is not None:
                        # Create a zero vector with the right dimension
                        zeros = [0.0] * self.dimensions
                        formatted = f"[{','.join(str(x) for x in zeros)}]"
                        return formatted
                    return '[]'
                return value
            elif isinstance(value, (list, np.ndarray)):
                # Format as vector string
                formatted = f"[{','.join(str(float(x)) for x in value)}]"
                return formatted
                
            return value
        return process
        
    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
                
            if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                try:
                    # Parse the vector values
                    values = value.strip('[]').split(',')
                    result = [float(x.strip()) for x in values if x.strip()]
                    return result
                except Exception as e:
                    logger.error(f"Error parsing vector: {e}")
                    return value
                    
            return value
        return process

# Define LONGTEXT type for MySQL/TiDB
class LONGTEXT(UserDefinedType):
    def get_col_spec(self, **kw):
        return "LONGTEXT"
        
    def bind_processor(self, dialect):
        def process(value):
            return value
        return process
        
    def result_processor(self, dialect, coltype):
        def process(value):
            return value
        return process

class CodeFile(Base):
    """File metadata table"""
    __tablename__ = 'code_files'
    
    id = Column(Integer, primary_key=True)
    # Reduce the length of file_path to stay within TiDB's key length limit
    # MySQL/TiDB has a max key length of 3072 bytes
    file_path = Column(String(768), nullable=False, index=True)  # Reduced from 1024
    # Total file size
    file_size = Column(Integer, nullable=False, default=0)
    # Total number of chunks
    chunks_count = Column(Integer, nullable=False, default=0)
    # File language
    language = Column(String(50), nullable=True)
    # Use TiDB vector data type for embedding
    file_embedding = Column(VECTOR(EMBEDDING_DIM))

    # Relationship with file content chunks
    chunks = relationship("FileChunk", back_populates="file", cascade="all, delete-orphan")

    # Don't use a unique constraint on file_path as it's too long
    # Instead, create a separate hash column if uniqueness is needed
    __table_args__ = (
        # Create a non-unique index instead
        Index('idx_file_path', file_path),
    )
    
    def get_full_content(self):
        """Get the complete content of the file"""
        if not self.chunks:
            return ""
            
        # Sort chunks by index and concatenate content
        sorted_chunks = sorted(self.chunks, key=lambda c: c.chunk_index)
        return "".join(chunk.content for chunk in sorted_chunks)

class FileChunk(Base):
    """File content chunks table"""
    __tablename__ = 'file_chunks'
    
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('code_files.id'), nullable=False)
    # Chunk index, starting from 0
    chunk_index = Column(Integer, nullable=False)
    # Chunk content
    content = Column(LONGTEXT, nullable=False)
    
    # Relationship with file
    file = relationship("CodeFile", back_populates="chunks")
    
    __table_args__ = (
        # Ensure file ID and chunk index combination is unique
        Index('idx_file_chunk', file_id, chunk_index, unique=True),
    )

def setup_vector_indexes(engine):
    """Setup TiDB vector indexes for similarity search."""
    from sqlalchemy import text
    
    # Detect database type
    db_url = str(engine.url)
    
    # Only create vector indexes for TiDB
    if 'tidb' in db_url or 'mysql' in db_url:
        with engine.connect() as conn:
            try:
                # Create TiFlash replica for the table
                conn.execute(text("""
                    ALTER TABLE code_files SET TIFLASH REPLICA 1
                """))
                logger.info("Created TiFlash replica for code_files table")
            except Exception as e:
                logger.warning(f"Creating TiFlash replica failed, vector index may not work: {e}")
            
            try:
                # Create vector indexes for cosine distance
                conn.execute(text("""
                    CREATE VECTOR INDEX IF NOT EXISTS idx_file_vector_cosine 
                    ON code_files ((VEC_COSINE_DISTANCE(file_embedding))) USING HNSW
                """))
                logger.info("Created file vector cosine index")
                
                # Commit the transaction
                conn.commit()
            except Exception as e:
                logger.warning(f"Vector index creation failed: {e}")
    else:
        logger.info("Not using TiDB, skipping vector index creation")

def get_engine(db_path='mysql+pymysql://root@localhost:4000/code_index'):
    """Create and return a database engine.
    
    Args:
        db_path: Database connection string. Default is TiDB.
               For SQLite use: 'sqlite:///file_index.db'
    """
    return create_engine(db_path)

def get_session(engine=None):
    """Create and return a session"""
    if engine is None:
        engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def init_db(engine=None):
    """Initialize the database, creating tables if they don't exist"""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    
    # Setup vector indexes for TiDB
    setup_vector_indexes(engine)
    
def split_content_into_chunks(content, chunk_size=CHUNK_SIZE):
    """Split content into chunks of specified size
    
    Args:
        content: File content
        chunk_size: Maximum size of each chunk (bytes)
        
    Returns:
        List containing all content chunks
    """
    chunks = []
    content_length = len(content)
    
    # Calculate how many chunks are needed
    num_chunks = (content_length + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, content_length)
        chunk = content[start:end]
        chunks.append(chunk)
    
    return chunks 