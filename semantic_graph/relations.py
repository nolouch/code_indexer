"""Semantic relations between code elements."""
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict

class RelationType(Enum):
    """Types of semantic relations."""
    CALLS = "calls"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    CONTAINS = "contains"
    USES = "uses"
    DEFINES = "defines"

@dataclass
class SemanticRelation:
    """Represents a semantic relation between code elements."""
    source: str
    target: str
    type: RelationType
    weight: float = 1.0
    attributes: Dict[str, Any] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {} 