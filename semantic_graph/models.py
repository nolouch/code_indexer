from sqlalchemy import Column, Integer, String, Float, ForeignKey, Table, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

# Association table for node relationships
node_relationships = Table(
    'node_relationships',
    Base.metadata,
    Column('source_id', Integer, ForeignKey('nodes.id'), primary_key=True),
    Column('target_id', Integer, ForeignKey('nodes.id'), primary_key=True),
    Column('relationship_type', String(50)),
    Column('weight', Float),
    Column('attributes', JSON),
)

class Node(Base):
    """Represents a node in the semantic graph"""
    __tablename__ = 'nodes'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True)  # Unique identifier for the node
    type = Column(String(50))  # Type of code element (function, class, etc.)
    language = Column(String(50))  # Programming language
    file_path = Column(String(255))  # Source file path
    line_number = Column(Integer)  # Line number in source file
    docstring = Column(Text)  # Documentation string
    code_content = Column(Text)  # Actual code content
    code_embedding = Column(JSON)  # Code embedding vector
    doc_embedding = Column(JSON)  # Documentation embedding vector
    extra_data = Column(JSON)  # Additional metadata

    # Define relationships
    outgoing_edges = relationship(
        'Node',
        secondary=node_relationships,
        primaryjoin=id==node_relationships.c.source_id,
        secondaryjoin=id==node_relationships.c.target_id,
        backref='incoming_edges'
    )

class Repository(Base):
    """Represents a code repository"""
    __tablename__ = 'repositories'

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    path = Column(String(255), unique=True)
    language = Column(String(50))
    last_indexed = Column(String(50))  # Timestamp of last indexing
    repo_data = Column(JSON)  # Additional repository metadata

class GraphMetadata(Base):
    """Stores metadata about the semantic graph"""
    __tablename__ = 'graph_metadata'

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey('repositories.id'))
    node_count = Column(Integer)
    edge_count = Column(Integer)
    last_updated = Column(String(50))  # Timestamp of last update
    embedder_config = Column(JSON)  # Configuration of embedders
    graph_data = Column(JSON)  # Additional graph metadata