from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class CodeElement(ABC):
    """Base class for all code elements"""
    name: str
    element_type: str  # function, class, interface, etc.
    language: str
    file: str
    line: int
    docstring: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Function(CodeElement):
    """Represents a function or method"""
    parameters: List[Dict[str, str]] = field(default_factory=list)
    return_types: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    parent_class: Optional[str] = None

@dataclass
class Class(CodeElement):
    """Represents a class or struct"""
    fields: List[Dict[str, str]] = field(default_factory=list)
    methods: List[Function] = field(default_factory=list)
    parent_classes: List[str] = field(default_factory=list)
    implements: List[str] = field(default_factory=list)

@dataclass
class Interface(CodeElement):
    """Represents an interface"""
    methods: List[Function] = field(default_factory=list)
    extends: List[str] = field(default_factory=list)

@dataclass
class Module(CodeElement):
    """Represents a module/package/namespace"""
    imports: List[str] = field(default_factory=list)
    functions: List[Function] = field(default_factory=list)
    classes: List[Class] = field(default_factory=list)
    interfaces: List[Interface] = field(default_factory=list)
    submodules: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)

@dataclass
class Variable(CodeElement):
    """Represents a variable or constant"""
    var_type: str
    is_constant: bool = False
    value: Optional[str] = None

@dataclass
class Annotation(CodeElement):
    """Represents an annotation/decorator"""
    target_type: str  # function, class, field, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CodeRepository:
    """Represents an entire code repository"""
    root_path: str
    language: str
    modules: Dict[str, Module] = field(default_factory=dict)
    global_symbols: Dict[str, CodeElement] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict) 