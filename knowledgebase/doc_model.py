import uuid
from typing import Dict, List, Literal, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Text,
    ForeignKey,
    DateTime,
    Enum,
    Index,
    JSON,
    ForeignKeyConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from tidb_vector.sqlalchemy import VectorType

Base = declarative_base()


class BestPractice(Base):
    __tablename__ = "best_practices"

    id = Column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True
    )
    source_id = Column(String(255), nullable=True)
    tag = Column(String(255), nullable=True)
    guideline = Column(JSON, nullable=True)
    guideline_vec = Column(
        VectorType(1536), nullable=True
    )  # Vector column for embeddings
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (Index("idx_source_id", "source_id"),)

    def __repr__(self):
        return (
            f"<BestPractice(id={self.id}, source_id={self.source_id}, tag={self.tag})>"
        )


class Concept(Base):
    """Core concept entity in the knowledge graph"""

    __tablename__ = "concepts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    definition = Column(Text, nullable=True)
    definition_vec = Column(
        VectorType(1536), nullable=True
    )  # Vector column for embeddings
    version = Column(String(50), nullable=True)
    source_ids = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    def __repr__(self):
        return f"<Concept(id={self.id}, name={self.name})>"


class SourceData(Base):
    """Source document entity"""

    __tablename__ = "source_data"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    content = Column(LONGTEXT, nullable=True)
    link = Column(String(512), nullable=True)
    version = Column(String(50), nullable=True)
    data_type = Column(Enum("document", "code"), nullable=False, default="document")
    meta_info = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Relationships
    knowledge_blocks = relationship("KnowledgeBlock", back_populates="source")

    __table_args__ = (Index("idx_source_link", "link", unique=True),)

    def __repr__(self):
        return f"<SourceData(id={self.id}, name={self.name}, type={self.data_type})>"


class KnowledgeBlock(Base):
    """Block of knowledge extracted from a source"""

    __tablename__ = "knowledge_blocks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    knowledge_type = Column(Enum("qa", "paragraph", "synopsis"), nullable=False)
    content = Column(LONGTEXT, nullable=True)
    content_vec = Column(
        VectorType(1536), nullable=True
    )  # Vector column for embeddings
    context = Column(Text, nullable=True)
    context_vec = Column(
        VectorType(1536), nullable=True
    )  # Vector column for embeddings
    source_version = Column(String(50), nullable=True)
    source_id = Column(String(36), ForeignKey("source_data.id"), nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # Relationships
    source = relationship("SourceData", back_populates="knowledge_blocks")

    __table_args__ = (Index("fk_1", "source_id"),)

    def __repr__(self):
        return f"<KnowledgeBlock(id={self.id}, name={self.name})>"


# Define standard relation types
STANDARD_RELATION_TYPES = [
    "EXPLAINS",
    "DEPENDS_ON",
    "REFERENCES",
    "PART_OF",
    "SIMILAR_TO",
]


class Relationship(Base):
    """Relationship between entities in the knowledge graph"""

    __tablename__ = "relationships"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id = Column(String(36), nullable=False)
    source_type = Column(String(50), nullable=False)
    target_id = Column(String(36), nullable=False)
    target_type = Column(String(50), nullable=False)
    relationship_desc = Column(
        Text, nullable=False, default="REFERENCES"
    )  # Changed to Text for natural language descriptions
    attributes = Column(Text, nullable=True)  # Store as JSON string
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    __table_args__ = (
        Index("idx_relationship_source", "source_id", "source_type"),
        Index("idx_relationship_target", "target_id", "target_type"),
    )

    def __repr__(self):
        return f"<Relationship(source={self.source_id}, target={self.target_id}, desc={self.relationship_desc})>"

    @property
    def is_standard_relation(self) -> bool:
        """Check if the relation type is a standard type"""
        return self.relationship_desc in STANDARD_RELATION_TYPES
